"""Every step or node timeout is recorded per unit and system, and the analysis reports
them; a ReAct step timeout is flagged prominently as a possible handicap."""

import functools
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bench import analyze
from bench.harness import PartitionRule
from bench.run import PreparedUnit, Unit, normalise, run_unit
from bench.systems import react as react_module
from bench.systems.base import BenchResult, SystemContext, SystemName, SystemOutput
from bench.systems.holdout_system import _eda_timeouts
from bench.systems.react import ReactSystem
from bench.systems.sandbox import run_step
from bench.traps import build_case
from holdout.agent.graph import check_node_time
from holdout.core.errors import NodeTimeoutError
from holdout.llm import config as llm_config
from holdout.llm.client import FakeLLM

CASE = build_case("direct_leak", 0)


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(llm_config.TRUST_CACHE_DISCOUNT_ENV, raising=False)


def _prepared(unit: Unit) -> PreparedUnit:
    return PreparedUnit(normalise(CASE.df), CASE.target, CASE.task, CASE.goal_text, None,
                        dict(CASE.plan_overrides), PartitionRule())  # fmt: skip


FINAL = {
    "thought": "Done.", "action": "final",
    "final": {"issues": [], "dropped_columns": [], "split": "stratified",
              "deduplicated": False, "validation_metric": "roc_auc", "validation_score": 0.7,
              "report": "Both attempts ran out of time; nothing was fitted here."},
}  # fmt: skip


def test_react_step_timeouts_are_counted_by_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Short limits stand in for 20 s and 60 s; the classification is the sandbox's own.
    monkeypatch.setattr(react_module, "run_step",
                        functools.partial(run_step, step_seconds=2, fit_seconds=3))  # fmt: skip
    fit = {"thought": "Fit.", "action": "python",
           "code": "def fit_model():\n    while True:\n        pass\nfit_model()"}  # fmt: skip
    spin = {"thought": "Loop.", "action": "python", "code": "while True:\n    pass"}
    fake = FakeLLM(scripts={"react": [fit, spin, FINAL]})
    unit = Unit("dev_v1", "direct_leak@synthetic", "react", 11, "planted")
    result = run_unit(unit, ReactSystem(lambda: fake), tmp_path / "r", model="gemini-3.8-flash",
                      run_cap_inr=15.0, prepare_fn=_prepared)  # fmt: skip
    assert result.timeouts == {"react_fit_step": 1, "react_step": 1}
    assert result.extra["step_statuses"] == ["timeout", "timeout"]


class _SlowNode:
    name: SystemName = "holdout"
    uses_llm = False

    def run(self, ctx: SystemContext) -> SystemOutput:
        check_node_time("plan", elapsed=700.0, api_seconds=10.0, limit=600.0)
        raise AssertionError("unreachable")

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self.run(ctx)


def test_a_node_timeout_fails_the_unit_and_is_recorded(tmp_path: Path) -> None:
    unit = Unit("dev_v1", "direct_leak@synthetic", "holdout", 11, "planted")
    result = run_unit(unit, _SlowNode(), tmp_path / "r", model="m", run_cap_inr=15.0,
                      prepare_fn=_prepared)  # fmt: skip
    assert result.status == "failed"
    assert result.timeouts == {"node:plan": 1}
    assert "NodeTimeoutError" in (result.error or "")
    assert issubclass(NodeTimeoutError, Exception)


def test_holdout_eda_timeouts_are_counted_over_all_runs() -> None:
    def manifest(*statuses: str) -> Any:
        return SimpleNamespace(agent=SimpleNamespace(eda=[SimpleNamespace(status=s)
                                                          for s in statuses]))  # fmt: skip

    def manifest_without_agent() -> Any:
        return SimpleNamespace(agent=None)

    assert _eda_timeouts([manifest("ok", "timeout"), manifest("timeout")]) == {"eda_step": 2}
    assert _eda_timeouts([manifest("ok"), manifest_without_agent()]) == {}


def _result(system: SystemName, timeouts: dict[str, int]) -> BenchResult:
    now = datetime.now(UTC)
    return BenchResult(suite="final_v1", task_id="t", system=system, seed=1, variant="planted",
                       status="completed", benchmark_metric="roc_auc", timeouts=timeouts,
                       started_at=now, finished_at=now)  # fmt: skip


def test_the_analysis_reports_timeouts_and_warns_about_react() -> None:
    results = [
        _result("react", {"react_fit_step": 2}),
        _result("react", {}),
        _result("holdout", {"eda_step": 1, "node:plan": 1}),
        _result("flaml", {}),
    ]
    table = "\n".join(analyze.timeout_table(results))
    assert "| system | units | eda_step | node:plan | react_fit_step |" in table
    assert "| react | 1 of 2 | 0 | 0 | 2 |" in table
    assert "| holdout | 1 of 1 | 1 | 1 | 0 |" in table
    warning = "\n".join(analyze.react_timeout_warning(results))
    assert "**Warning: ReAct sandbox steps timed out** in 1 of 2 ReAct runs" in warning
    assert "2 model-fitting steps at the 60 s limit" in warning
    assert analyze.react_timeout_warning([_result("react", {}),
                                          _result("holdout", {"eda_step": 3})]) == []  # fmt: skip
    assert analyze.timeout_table([_result("react", {})])[-1] == "No step or node timed out."
