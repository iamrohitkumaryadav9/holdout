"""Parallel benchmark runs: a ledger that is safe across processes, the unit-start guard,
``run_all(workers=N)`` and a pre-flight that leaves headroom for the units in flight."""

import json
import multiprocessing
import os
import time
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from bench.budget import Measured, estimate, limits_left, start_refusal, unit_worst_case
from bench.config import load_bench_config
from bench.harness import PartitionRule
from bench.run import THREAD_ENV, PreparedUnit, Unit, normalise, run_all, threads_per_worker
from bench.systems.base import System, SystemContext, SystemName, SystemOutput
from bench.traps import build_case
from holdout.llm import config
from holdout.llm.budget import SpendLedger

APPENDS = 150


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)
    monkeypatch.delenv(config.SPEND_PURPOSE_ENV, raising=False)
    return tmp_path / "home"


# ------------------------------------------------------------------ the ledger


def _append_many(worker: int) -> float:
    ledger = SpendLedger()
    seen = 0.0
    for i in range(APPENDS):
        ledger.append({"kind": "call", "timestamp": datetime.now(UTC).isoformat(),
                       "inr": 0.01, "worker": worker, "i": i, "pad": "x" * 2000})  # fmt: skip
        seen = ledger.total_inr()  # reads while others write: never a torn line
    return seen


def test_the_ledger_is_safe_across_processes() -> None:
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        pool.map(_append_many, range(4))
    ledger = SpendLedger()
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 * APPENDS
    entries = [json.loads(line) for line in lines]  # every line is whole
    assert {(e["worker"], e["i"]) for e in entries} == {
        (w, i) for w in range(4) for i in range(APPENDS)
    }
    assert ledger.total_inr() == pytest.approx(4 * APPENDS * 0.01)


# ------------------------------------------------------------------ the guard


def _spend(inr: float) -> None:
    SpendLedger().append({"kind": "call", "timestamp": datetime.now(UTC).isoformat(), "inr": inr})


def test_worst_cases_follow_the_per_run_caps() -> None:
    assert unit_worst_case("react", 15.0) == 15.0
    assert unit_worst_case("holdout", 15.0) == 30.0  # one follow-up run, its own cap
    assert unit_worst_case("llm_judge", 15.0) == 30.0
    assert unit_worst_case("flaml", 15.0) == unit_worst_case("audits_only", 15.0) == 0.0


def test_every_applicable_limit_is_checked_with_the_units_in_flight() -> None:
    _spend(130.0)
    general = limits_left(SpendLedger(), purpose="general")
    assert general == {"project cap": pytest.approx(620.0),
                       "benchmark reserve": pytest.approx(20.0)}  # fmt: skip
    assert start_refusal(15.0, 0.0, general) is None
    refusal = start_refusal(15.0, 15.0, general)  # one unit already running
    assert refusal is not None
    assert "needs Rs 30.00" in refusal
    assert "benchmark reserve: Rs 20.00 left" in refusal
    assert "project cap" not in refusal  # only the limits that are short are named
    assert start_refusal(0.0, 1000.0, general) is None  # a unit without LLM calls always starts
    benchmark = limits_left(SpendLedger(), purpose="benchmark")
    assert "benchmark reserve" not in benchmark  # benchmark spend may use the reserve
    pilot = limits_left(SpendLedger(), purpose="benchmark", invocation_start_inr=120.0,
                        max_spend_inr=30.0)  # fmt: skip
    assert pilot["this run's spend cap"] == pytest.approx(20.0)
    SpendLedger().record_balance(40.0, datetime.now().astimezone().date())
    with_balance = limits_left(SpendLedger(), purpose="benchmark")
    assert with_balance["prepaid balance"] <= 40.0


# ------------------------------------------------------------------ run_all(workers=N)

CASE = build_case("direct_leak", 0)
SPEND_PER_UNIT = 5.0


def prepared(unit: Unit) -> PreparedUnit:
    return PreparedUnit(
        normalise(CASE.df), CASE.target, CASE.task, CASE.goal_text, None, {}, PartitionRule()
    )


class SlowSpender:
    """Spends Rs 5 through the ledger over about a second; records when it ran."""

    name: SystemName = "react"
    uses_llm = True

    def run(self, ctx: SystemContext) -> SystemOutput:
        started = time.time()
        time.sleep(1.0)
        _spend(SPEND_PER_UNIT)
        return SystemOutput(status="stopped", cost_inr=SPEND_PER_UNIT,
                            extra=_where(started))  # fmt: skip

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self.run(ctx)


class CpuOnly:
    """A free, CPU-only system (stands in for flaml / audits_only)."""

    name: SystemName = "flaml"
    uses_llm = False

    def run(self, ctx: SystemContext) -> SystemOutput:
        started = time.time()
        time.sleep(0.3)
        return SystemOutput(status="stopped", extra=_where(started))

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self.run(ctx)


def _where(started: float) -> dict[str, Any]:
    """When and where a unit ran, and the thread limits it saw."""
    from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

    return {
        "t0": started, "t1": time.time(), "pid": os.getpid(),
        "omp": os.environ.get("OMP_NUM_THREADS"), "loky": os.environ.get("LOKY_MAX_CPU_COUNT"),
        "pool_threads": max((p["num_threads"] for p in threadpool_info()), default=None),
    }  # fmt: skip


def spender_factory(names: list[str]) -> dict[str, System]:
    return {name: SlowSpender() if name == "react" else CpuOnly() for name in names}


def test_workers_run_units_in_parallel_within_every_limit(tmp_path: Path) -> None:
    units = [Unit("dev_v1", f"t{i}", "react", 11, "planted") for i in range(8)]
    lines: list[str] = []
    done = run_all(
        units, {}, tmp_path / "r", model="m", run_cap_inr=15.0, write=lines.append,
        prepare_fn=prepared, max_spend_inr=40.0, purpose="benchmark", workers=2,
        system_factory=spender_factory,
    )  # fmt: skip
    spent = SpendLedger().total_inr()
    assert spent <= 40.0  # never overshot, although two units were running at a time
    assert len(done) == spent / SPEND_PER_UNIT
    assert len(done) >= 4
    assert "stopped cleanly" in lines[-1]
    assert len({r.extra["pid"] for r in done}) == 2  # separate worker processes
    assert {r.extra["workers"] for r in done} == {2}
    spans = sorted((r.extra["t0"], r.extra["t1"]) for r in done)
    overlapping = sum(b0 < a1 for (_, a1), (b0, _) in pairwise(spans))
    assert overlapping >= 1  # units really ran at the same time
    for t in range(3):  # at most two at once
        assert sum(t0 <= spans[t][0] < t1 for t0, t1 in spans) <= 2
    results = (tmp_path / "r" / "dev_v1" / "react.jsonl").read_text("utf-8").splitlines()
    assert len(results) == len(done)  # appended once each, by the parent
    for unit in done:
        seal = tmp_path / "r" / "seals" / "dev_v1" / f"{unit.task_id}__planted__s11"
        assert (seal / "seal.json").exists()
        assert (seal / "dummy.json").exists()  # created before dispatch


def test_cpu_only_units_run_alone_after_the_llm_units_with_limited_threads(
    tmp_path: Path,
) -> None:
    units = [Unit("dev_v1", f"t{i}", s, 11, "planted")  # interleaved, as units_for orders them
             for i in range(3) for s in ("react", "flaml")]  # fmt: skip
    before = {name: os.environ.get(name) for name in THREAD_ENV}
    lines: list[str] = []
    done = run_all(
        units, {}, tmp_path / "r", model="m", run_cap_inr=15.0, write=lines.append,
        prepare_fn=prepared, purpose="benchmark", workers=2, system_factory=spender_factory,
    )  # fmt: skip
    llm = [r for r in done if r.system == "react"]
    cpu = [r for r in done if r.system == "flaml"]
    assert len(llm) == len(cpu) == 3
    assert max(r.extra["t1"] for r in llm) <= min(r.extra["t0"] for r in cpu)  # second pass
    spans = sorted((r.extra["t0"], r.extra["t1"]) for r in cpu)
    assert all(a1 <= b0 for (_, a1), (b0, _) in pairwise(spans))  # one at a time
    assert {r.extra["pid"] for r in cpu} == {os.getpid()}  # in this process, no workers
    threads = threads_per_worker(2)
    assert threads == max(1, (os.cpu_count() or 1) // 2)
    for r in llm:
        assert (r.extra["workers"], r.extra["threads_per_worker"]) == (2, threads)
        assert r.extra["logical_cores"] == os.cpu_count()
        assert r.extra["omp"] == r.extra["loky"] == str(threads)  # inherited by sandboxes too
        assert r.extra["pool_threads"] is None or r.extra["pool_threads"] <= threads
    for r in cpu:
        assert (r.extra["workers"], r.extra["threads_per_worker"]) == (1, None)
        assert r.extra["omp"] == before["OMP_NUM_THREADS"]  # not limited in the CPU pass
    assert {name: os.environ.get(name) for name in THREAD_ENV} == before  # restored
    assert any("pass: 3 CPU-only units, one at a time (1 worker(s))" in line for line in lines)


def test_threads_per_worker_never_exceed_the_cores() -> None:
    assert threads_per_worker(4, cores=8) == 2
    assert threads_per_worker(3, cores=8) == 2
    assert threads_per_worker(4, cores=2) == 1  # at least one
    for workers in range(1, 9):
        assert workers * threads_per_worker(workers, cores=8) <= 8


def test_several_workers_need_a_picklable_factory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="system_factory"):
        run_all([], {}, tmp_path, model="m", run_cap_inr=15.0, workers=2)
    with pytest.raises(ValueError, match="at least 1"):
        run_all([], {}, tmp_path, model="m", run_cap_inr=15.0, workers=0)


# ------------------------------------------------------------------ pre-flight

MEASURED = {"react": Measured(6.0, 3, date(2026, 11, 20)), "holdout": Measured(1.0, 3,
            date(2026, 11, 20))}  # fmt: skip


def test_the_preflight_leaves_headroom_for_the_units_in_flight() -> None:
    _spend(600.0)  # Rs 150 left under the cap
    SpendLedger().record_balance(1000.0, datetime.now().astimezone().date())  # Rs 400 left
    cfg = load_bench_config()
    kwargs: dict[str, Any] = {"purpose": "benchmark", "measured": MEASURED,
                              "today": date(2026, 11, 20)}  # fmt: skip
    one = estimate({"react": 10, "holdout": 10}, cfg, workers=1, **kwargs)
    four = estimate({"react": 10, "holdout": 10}, cfg, workers=4, **kwargs)
    assert one.total_inr == four.total_inr == pytest.approx((60 + 10) * 1.2 * 0.5)
    assert one.fits, one.render()  # 42 + 30 <= 150
    assert one.headroom_inr == 30.0  # the largest single worst case (Holdout: two caps)
    assert four.headroom_inr == 4 * 30.0
    assert "headroom for units in flight (4 worker(s)" in four.render()
    assert any("unit-start guard would stop the run" in s for s in four.shortfalls), four.render()


def test_flex_measurements_are_not_halved_again_and_standard_ones_are() -> None:
    cfg = load_bench_config()  # pins flex
    assert cfg.service_tier == "flex"
    day = date(2026, 11, 20)
    standard = estimate({"react": 10}, cfg, purpose="benchmark", today=day,
                        measured={"react": Measured(6.0, 3, day, "standard")})  # fmt: skip
    flex = estimate({"react": 10}, cfg, purpose="benchmark", today=day,
                    measured={"react": Measured(6.0, 3, day, "flex")})  # fmt: skip
    assert standard.price_factor == pytest.approx(0.5)  # measured at standard, runs on Flex
    assert flex.price_factor == pytest.approx(1.0)
    assert standard.total_inr == pytest.approx(flex.total_inr / 2)
    assert "standard tier" in standard.render()
