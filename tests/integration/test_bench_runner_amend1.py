"""Prereg Amendment 1, the runner: a pause pauses only its unit, a circuit breaker holds new
starts for 10 minutes after 3 pauses within 10 minutes, one final sweep resumes paused
units, the exit summary counts every status, and a scoped rerun of failed units."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bench import run as bench_run
from bench.harness import PartitionRule
from bench.run import (
    CircuitBreaker,
    PreparedUnit,
    Unit,
    append_result,
    normalise,
    pending,
    run_all,
    status_counts,
)
from bench.systems.base import BenchResult, SystemContext, SystemName, SystemOutput
from bench.traps import build_case
from holdout.core.errors import LLMUnavailableError
from holdout.llm import config

CASE = build_case("direct_leak", 0)


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)


def _prepared(unit: Unit) -> PreparedUnit:
    return PreparedUnit(normalise(CASE.df), CASE.target, CASE.task, CASE.goal_text, None, {},
                        PartitionRule())  # fmt: skip


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Outage:
    """A react-like LLM system: the listed tasks pause ``fail_times`` times, then complete.
    Each run advances the fake clock by ``minutes``."""

    name: SystemName = "react"
    uses_llm = True

    def __init__(self, clock: FakeClock, pausing: set[str], fail_times: int = 1,
                 minutes: float = 1.0) -> None:  # fmt: skip
        self.clock, self.pausing, self.fail_times, self.minutes = (
            clock, pausing, fail_times, minutes)  # fmt: skip
        self.calls: list[tuple[str, bool, float]] = []
        self.failures: dict[str, int] = {}

    def _go(self, ctx: SystemContext, resume: bool) -> SystemOutput:
        task = ctx.unit_id.split("/")[1].split("__")[0]
        self.calls.append((task, resume, self.clock.now))
        self.clock.now += 60 * self.minutes
        if any(task.startswith(p) for p in self.pausing):
            n = self.failures.get(task, 0)
            if n < self.fail_times:
                self.failures[task] = n + 1
                raise LLMUnavailableError("503 on every Flex attempt and the fallback")
        return SystemOutput(status="completed")

    def run(self, ctx: SystemContext) -> SystemOutput:
        return self._go(ctx, False)

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self._go(ctx, True)


def _units(n: int) -> list[Unit]:
    return [Unit("dev_v1", f"t{i}", "react", 11, "planted") for i in range(n)]


def _run(tmp_path: Path, system: Any, units: list[Unit], clock: FakeClock,
         **kw: Any) -> tuple[list[BenchResult], list[str], dict[str, Any]]:  # fmt: skip
    lines: list[str] = []
    outcome: dict[str, Any] = {}
    done = run_all(units, {"react": system}, tmp_path / "r", model="m", run_cap_inr=15.0,
                   write=lines.append, prepare_fn=_prepared, purpose="benchmark",
                   breaker=CircuitBreaker(clock=clock, sleep=clock.sleep), outcome=outcome,
                   **kw)  # fmt: skip
    return done, lines, outcome


def test_a_paused_unit_does_not_stop_the_others_and_the_sweep_resumes_it(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    system = Outage(clock, pausing={"t1"})
    done, lines, outcome = _run(tmp_path, system, _units(4), clock)
    assert [(t, resume) for t, resume, _ in system.calls] == [
        ("t0", False), ("t1", False), ("t2", False), ("t3", False),  # kept starting
        ("t1", True),  # the final sweep resumed it
    ]  # fmt: skip
    assert [r.status for r in done] == ["completed", "paused", "completed", "completed",
                                        "completed"]  # fmt: skip
    assert outcome["counts"]["completed"] == 4
    assert outcome["budget_stopped"] is False
    assert "pass: 1 final sweep of paused units" in "\n".join(lines)
    assert lines[-1].startswith("summary: completed 4, stopped 0, paused 0, failed 0")
    assert clock.slept == []  # one pause: the breaker stays closed


def test_still_paused_after_the_sweep_stays_paused_and_is_counted(tmp_path: Path) -> None:
    clock = FakeClock()
    system = Outage(clock, pausing={"t0"}, fail_times=5)
    _, lines, outcome = _run(tmp_path, system, _units(2), clock)
    assert outcome["counts"] == {"completed": 1, "stopped": 0, "paused": 1, "failed": 0,
                                 "not started": 0}  # fmt: skip
    assert [t for t, resume, _ in system.calls if resume] == ["t0"]  # exactly one sweep
    assert "paused 1" in lines[-1]
    assert pending(_units(2), tmp_path / "r") == [(_units(2)[0], True)]  # next invocation


def test_the_circuit_breaker_waits_ten_minutes_after_three_pauses_in_ten(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    system = Outage(clock, pausing={"t0", "t1", "t2"}, minutes=1.0)  # 3 pauses in 3 minutes
    _, lines, _ = _run(tmp_path, system, _units(5), clock)
    starts = {(t, resume): at for t, resume, at in system.calls}
    assert clock.slept == [600.0]  # tripped once
    assert starts[("t3", False)] - starts[("t2", False)] == pytest.approx(60 + 600)
    assert starts[("t4", False)] - starts[("t3", False)] == pytest.approx(60)  # reset after
    assert any("circuit breaker: 3 units paused within 10 min; waiting 10 min" in line
               for line in lines)  # fmt: skip


def test_pauses_spread_over_more_than_ten_minutes_do_not_trip_the_breaker(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    system = Outage(clock, pausing={"t0", "t2", "t4"}, minutes=6.0)  # pauses 12 min apart
    _run(tmp_path, system, _units(5), clock)
    assert clock.slept == []


def test_the_breaker_window_and_reset_with_a_fake_clock() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(clock=clock, sleep=clock.sleep)
    lines: list[str] = []
    for minute in (0, 4, 9):
        clock.now = 60 * minute
        breaker.record_pause()
    clock.now = 60 * 10.5  # the first pause is now older than 10 minutes: 2 in the window
    breaker.before_start(lines.append)
    assert clock.slept == []
    breaker.record_pause()  # 3 within the window
    breaker.before_start(lines.append)
    assert clock.slept == [600.0]
    assert breaker.trips == 1
    breaker.before_start(lines.append)  # cleared after waiting
    assert clock.slept == [600.0]


def test_the_budget_guard_still_stops_the_run_with_exit_6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = FakeClock()
    system = Outage(clock, pausing=set())
    _, lines, outcome = _run(tmp_path, system, _units(3), clock, max_spend_inr=10.0)
    assert outcome["budget_stopped"] is True
    assert system.calls == []  # Rs 10 < the Rs 15 worst case: nothing starts
    assert "stopped by the budget guard" in lines[-1]

    def fake_run_all(*args: Any, **kwargs: Any) -> list[BenchResult]:
        kwargs["outcome"]["budget_stopped"] = True
        return []

    monkeypatch.setattr(bench_run, "run_all", fake_run_all)
    monkeypatch.setattr("bench.systems.registry.build_systems", lambda *a, **k: {})
    from holdout.llm.budget import SpendLedger

    SpendLedger().record_balance(2000.0, datetime.now().astimezone().date())
    args = ["--suite", "dev_v1", "--tasks", "direct_leak@adult", "--seeds", "11",
            "--systems", "audits_only", "--results", str(tmp_path / "x")]  # fmt: skip
    assert bench_run.main(args) == 6


def _failed(task: str, error: str) -> BenchResult:
    now = datetime.now(UTC)
    return BenchResult(suite="dev_v1", task_id=task, system="react", seed=11, variant="planted",
                       status="failed", error=error, benchmark_metric="roc_auc",
                       started_at=now, finished_at=now)  # fmt: skip


def test_only_failed_units_matching_the_amendment_are_rerun(tmp_path: Path) -> None:
    root = tmp_path / "r"
    units = _units(3)
    append_result(root, _failed("t0", "LLMRequestError: ... (RemoteProtocolError); not retried"))
    append_result(root, _failed("t1", "LightGBMError: Do not support special JSON characters"))
    append_result(root, _failed("t2", "NodeTimeoutError: node audit used 756 s of compute"))
    matching = "RemoteProtocolError|special JSON characters in feature name|JSON characters"
    rerun = [u.task_id for u, _ in pending(units, root, retry_failed_matching=matching)]
    assert rerun == ["t0", "t1"]  # the genuine node timeout is not rerun
    assert pending(units, root) == []
    assert status_counts(units, root)["failed"] == 3
