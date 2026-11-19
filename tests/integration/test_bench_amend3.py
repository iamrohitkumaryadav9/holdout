"""Amendment 3 infrastructure: per-unit cost attribution through ledger tags (parallel units
must not be charged each other's spend), costs measured from the benchmark's own completed
units, and a run that refuses to start without the API key instead of failing every unit."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bench import run as bench_run
from bench.budget import RUN_MEASURE_MIN_UNITS, measured_from_run, unit_cost_by_tier
from bench.harness import PartitionRule
from bench.run import PreparedUnit, Unit, append_result, normalise, run_unit
from bench.systems.base import BenchResult, SystemContext, SystemName, SystemOutput
from bench.traps import build_case
from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger
from holdout.llm.client import Usage

CASE = build_case("direct_leak", 0)
MODEL = "gemini-3.8-flash"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)


def _prepared(unit: Unit) -> PreparedUnit:
    return PreparedUnit(normalise(CASE.df), CASE.target, CASE.task, CASE.goal_text, None, {},
                        PartitionRule())  # fmt: skip


class _Neighbour:
    """Charges its own call, while 'another worker' spends Rs 50 at the same time."""

    name: SystemName = "react"
    uses_llm = True

    def run(self, ctx: SystemContext) -> SystemOutput:
        budget = Budget(run_id=ctx.unit_id, model=MODEL)
        inr, _ = budget.charge(Usage(100_000, 0, 0, tier="standard"), node="n", prompt_name="p")
        SpendLedger().append({"kind": "call", "timestamp": datetime.now(UTC).isoformat(),
                              "inr": 50.0, "unit": "other/unit/react"})  # fmt: skip
        return SystemOutput(status="stopped", cost_inr=0.0, extra={"own": inr})

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self.run(ctx)


def test_a_unit_is_charged_only_its_own_tagged_spend(tmp_path: Path) -> None:
    unit = Unit("dev_v1", "t", "react", 11, "planted")
    result = run_unit(unit, _Neighbour(), tmp_path / "r", model=MODEL, run_cap_inr=15.0,
                      prepare_fn=_prepared)  # fmt: skip
    assert result.cost_inr == pytest.approx(result.extra["own"])  # not + Rs 50
    (own,) = [c for c in SpendLedger().calls() if c.get("unit") == "dev_v1/t__planted__s11/react"]
    assert own["inr"] == pytest.approx(result.extra["own"])
    import os

    assert os.environ.get("HOLDOUT_LEDGER_UNIT") is None  # cleared after the unit


def _done(i: int, system: SystemName, extra: dict[str, Any] | None = None) -> BenchResult:
    now = datetime.now(UTC)
    return BenchResult(suite="final_v1", task_id=f"t{i}", system=system, seed=101,
                       variant="planted", status="completed", benchmark_metric="roc_auc",
                       extra=extra or {}, started_at=now, finished_at=now)  # fmt: skip


def _line(inr: float, tier: str, **where: str) -> None:
    SpendLedger().append({"kind": "call", "timestamp": datetime.now(UTC).isoformat(),
                          "inr": inr, "tier_billed": tier, **where})  # fmt: skip


def test_costs_are_measured_from_completed_units_and_flex_is_priced_at_standard(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    for i in range(RUN_MEASURE_MIN_UNITS):
        append_result(root, _done(i, "react"))
        unit_id = f"final_v1/t{i}__planted__s101/react"
        if i % 2:  # new lines carry the unit tag
            _line(2.0, "flex", unit=unit_id)
            _line(1.0, "standard", unit=unit_id)
        else:  # older lines: ReAct's run id is the unit id
            _line(2.0, "flex", run_id=unit_id)
            _line(1.0, "standard", run_id=unit_id)
        append_result(root, _done(i, "holdout", {"run_dirs": [f"x/20261118T{i:06d}Z-ab"]}))
        _line(1.5, "standard", run_id=f"20261118T{i:06d}Z-ab")  # a Holdout agent run
    _line(99.0, "standard", run_id="unrelated")
    measured = measured_from_run(root, ["final_v1"], MODEL)
    assert measured["react"].per_run_inr == pytest.approx(2.0 * 2 + 1.0)  # Flex at Standard
    assert measured["holdout"].per_run_inr == pytest.approx(1.5)
    assert measured["react"].source == "benchmark-run"
    assert measured["react"].tier == "standard"
    assert "llm_judge" not in measured  # too few completed units: the pilot is used
    by_tier = unit_cost_by_tier(_done(0, "react"), SpendLedger().calls())
    assert by_tier == {"flex": 2.0, "standard": 1.0}


def test_the_run_refuses_to_start_without_the_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    SpendLedger().record_balance(2000.0, datetime.now().astimezone().date())
    built: list[Any] = []
    monkeypatch.setattr("bench.systems.registry.build_systems", lambda *a, **k: built.append(1))
    root = tmp_path / "r"
    code = bench_run.main(["--suite", "dev_v1", "--tasks", "direct_leak@adult", "--seeds", "11",
                           "--systems", "holdout", "--results", str(root)])  # fmt: skip
    assert code == 2
    assert "GEMINI_API_KEY is not set and 1 LLM unit(s) are pending" in capsys.readouterr().out
    assert built == []  # nothing ran, nothing recorded as failed
    assert not (root / "dev_v1").exists()


def test_the_analysis_reports_ledger_attributed_costs(tmp_path: Path) -> None:
    from bench.analyze import attributed_cost

    inflated = _done(0, "react").model_copy(update={"cost_inr": 28.0})
    _line(3.0, "standard", unit="final_v1/t0__planted__s101/react")
    _line(40.0, "standard", unit="final_v1/t9__planted__s101/react")  # a neighbour's spend
    assert attributed_cost(inflated, SpendLedger().calls()) == pytest.approx(3.0)
    alone = _done(5, "react").model_copy(update={"cost_inr": 2.5})  # no ledger lines here
    assert attributed_cost(alone, SpendLedger().calls()) == pytest.approx(2.5)
