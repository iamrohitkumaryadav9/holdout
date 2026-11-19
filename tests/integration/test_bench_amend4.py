"""Amendment 4: an atomic model submit, an unreadable model file that keeps ReAct's final
answer (issues scored, remediation not achieved), HTTP 502 as transient, and reruns that name
the failed line they supersede."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bench.harness import PartitionRule
from bench.run import PreparedUnit, Unit, append_result, normalise, run_all
from bench.score import detected, remediated
from bench.systems.base import BenchResult
from bench.systems.react import ReactSystem
from bench.systems.sandbox import MODEL_FILE, prepare_workdir, run_step
from bench.traps import build_case
from holdout.llm import config
from holdout.llm.client import FakeLLM
from holdout.llm.retry import classify
from tests.integration.test_bench_systems import REACT_CODE, REACT_FINAL

CASE = build_case("direct_leak", 0)


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)


def _prepared(unit: Unit) -> PreparedUnit:
    return PreparedUnit(normalise(CASE.df), CASE.target, CASE.task, CASE.goal_text, None, {},
                        PartitionRule())  # fmt: skip


def test_a_failed_submit_never_truncates_an_earlier_model(tmp_path: Path) -> None:
    prepare_workdir(tmp_path, CASE.df.head(200))
    ok = run_step("from sklearn.dummy import DummyClassifier\n"
                  "m = DummyClassifier().fit(df[['refund_issued']], df['converted'])\n"
                  "submit_model(m)", tmp_path)  # fmt: skip
    assert ok.status == "ok", ok.output
    size = (tmp_path / MODEL_FILE).stat().st_size
    assert size > 0
    # An object that passes the check but cannot be pickled (a lambda attribute).
    bad = run_step("from sklearn.dummy import DummyClassifier\n"
                   "m = DummyClassifier().fit(df[['refund_issued']], df['converted'])\n"
                   "m.hook = lambda x: x\n"
                   "submit_model(m)", tmp_path)  # fmt: skip
    assert bad.status == "error"
    assert (tmp_path / MODEL_FILE).stat().st_size == size  # the valid model is intact
    assert not (tmp_path / f"{MODEL_FILE}.tmp").exists()


def _unit_with_empty_model(tmp_path: Path) -> BenchResult:
    """ReAct submits nothing valid: the model file is left empty (as before the fix)."""
    code = REACT_CODE.replace("submit_model(model)", "pass")
    step = {"thought": "Fit.", "action": "python", "code": code}
    fake = FakeLLM(scripts={"react": [step, REACT_FINAL]})
    unit = Unit("dev_v1", "direct_leak@synthetic", "react", 11, "planted")
    sandbox = tmp_path / "r" / "work" / "dev_v1" / unit.slug / "react" / "sandbox"
    sandbox.mkdir(parents=True)

    class _EmptyModel(ReactSystem):
        def _output(self, ctx: Any, session: Any, final: Any, stop: Any, workdir: Path,
                    *args: Any) -> Any:  # fmt: skip
            (workdir / MODEL_FILE).write_bytes(b"")  # the truncated file the bug left behind
            return super()._output(ctx, session, final, stop, workdir, *args)

    (done,) = run_all([unit], {"react": _EmptyModel(lambda: fake)}, tmp_path / "r", model="m",
                      run_cap_inr=15.0, write=lambda s: None, prepare_fn=_prepared)  # fmt: skip
    return done


def test_an_unreadable_model_keeps_the_final_answer(tmp_path: Path) -> None:
    result = _unit_with_empty_model(tmp_path)
    assert result.status == "completed"  # not failed: the final answer is kept
    assert result.extra["model_file_unreadable"] is True
    assert "no valid model submitted: the model file is unreadable (0 bytes" in (
        result.error or "")  # fmt: skip
    assert {i.category.value for i in result.issues} >= {"target_leakage"}
    assert result.pipeline is None  # remediation is not achieved
    assert result.holdout_deploy is None  # nothing to score on the holdout


def test_detection_counts_and_remediation_does_not(tmp_path: Path) -> None:
    from bench.score import Expectation
    from holdout.audit.types import Category, Severity

    result = _unit_with_empty_model(tmp_path)
    exp = Expectation("direct_leak", (Category.TARGET_LEAKAGE,), Severity.CRITICAL,
                      ("refund_issued",), False, "drop:refund_issued")  # fmt: skip
    assert detected(result, exp) is True
    assert remediated(result, exp) is False


def test_http_502_is_transient() -> None:
    class _BadGatewayError(Exception):
        code = 502

    assert classify(_BadGatewayError("502 Bad Gateway")).transient is True


def test_a_rerun_names_the_failed_line_it_supersedes(tmp_path: Path) -> None:
    root = tmp_path / "r"
    now = datetime.now(UTC)
    unit = Unit("dev_v1", "direct_leak@synthetic", "react", 11, "planted")
    append_result(root, BenchResult(suite="dev_v1", task_id=unit.task_id, system="react",
                                    seed=11, variant="planted", status="failed",
                                    error="EOFError: Ran out of input",
                                    benchmark_metric="roc_auc", started_at=now,
                                    finished_at=now))  # fmt: skip
    fake = FakeLLM(scripts={"react": [REACT_FINAL]})
    (done,) = run_all([unit], {"react": ReactSystem(lambda: fake)}, root, model="m",
                      run_cap_inr=15.0, write=lambda s: None, prepare_fn=_prepared,
                      retry_failed_matching="EOFError: Ran out of input")  # fmt: skip
    assert done.extra["supersedes"]["error"] == "EOFError: Ran out of input"
    lines = (root / "dev_v1" / "react.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 2  # the failed line is kept
    assert '"status":"failed"' in lines[0].replace(" ", "")
