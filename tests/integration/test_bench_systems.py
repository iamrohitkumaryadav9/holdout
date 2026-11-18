"""Every benchmark system end to end through the harness (offline: FakeLLM, no data files),
holdout isolation, and the resumable runner."""

import sys
from pathlib import Path
from typing import Any

import pytest

from bench.harness import PartitionRule
from bench.run import PreparedUnit, Unit, append_result, normalise, pending, run_all, run_unit
from bench.systems.base import System, SystemContext, SystemOutput
from bench.systems.flaml_system import FlamlSystem
from bench.systems.holdout_system import HoldoutSystem
from bench.systems.llm_judge import install
from bench.systems.react import ReactSystem
from bench.traps import build_case
from holdout.core.errors import LLMUnavailableError
from holdout.llm.client import FakeLLM, LLMRequest
from tests.agent_fakes import data
from tests.agent_harness import fake_for, write_case

CASE = build_case("direct_leak", 0)


def _prepared(unit: Unit) -> PreparedUnit:
    return PreparedUnit(
        frame=normalise(CASE.df),
        target=CASE.target,
        task=CASE.task,
        goal=CASE.goal_text,
        entity_scope=None,
        plan_overrides=dict(CASE.plan_overrides),
        rule=PartitionRule(),
    )


# ------------------------------------------------------------------ fakes for the LLM systems


def _judge(request: LLMRequest) -> dict[str, Any]:
    profile = data(request, "profile")
    columns = [c["name"] for c in profile["columns"]]
    issues = []
    if "refund_issued" in columns:
        issues.append(
            {
                "category": "post_outcome_column",
                "severity": "critical",
                "columns": ["refund_issued"],
                "explanation": "refund_issued is recorded after the outcome and leaks it.",
                "fix": {"kind": "drop_columns", "columns": ["refund_issued"]},
            }
        )
    plan = data(request, "plan")
    dropped = {d["column"] for d in plan["drop_columns"]}
    return {"issues": [i for i in issues if not set(i["columns"]) <= dropped]}


REACT_CODE = (
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.compose import make_column_transformer, make_column_selector\n"
    "from sklearn.preprocessing import OneHotEncoder\n"
    "from sklearn.impute import SimpleImputer\n"
    "from sklearn.ensemble import HistGradientBoostingClassifier\n"
    "cats = make_pipeline(SimpleImputer(strategy='most_frequent'), "
    "OneHotEncoder(handle_unknown='ignore'))\n"
    "pre = make_column_transformer(('drop', ['refund_issued']), "
    "(cats, make_column_selector(dtype_include=object)), remainder='passthrough')\n"
    "model = make_pipeline(pre, HistGradientBoostingClassifier(max_iter=50))\n"
    "model.fit(df.drop(columns=['converted']), df['converted'])\n"
    "submit_model(model)\n"
    "print('x' * 5000)\n"
)
REACT_FINAL = {
    "thought": "Done.",
    "action": "final",
    "final": {
        "issues": [
            {
                "category": "target_leakage",
                "severity": "critical",
                "columns": ["refund_issued"],
                "explanation": "refund_issued mirrors the target.",
            }
        ],
        "dropped_columns": ["refund_issued"],
        "split": "stratified",
        "deduplicated": False,
        "validation_metric": "roc_auc",
        "validation_score": 0.71,
        "report": "The model ranks conversions reasonably well; refund_issued was removed.",
    },
}


def _react_fake() -> FakeLLM:
    step = {"thought": "Fit and submit.", "action": "python", "code": REACT_CODE}
    return FakeLLM(scripts={"react": [step, REACT_FINAL]})


def _systems(tmp_path: Path) -> dict[str, System]:
    cooperative = fake_for(CASE, write_case(CASE, tmp_path / "case"))
    judge_fake = FakeLLM(
        default={
            **cooperative.default,
            "judge": _judge,
            "judge_report": lambda r: {"causal_statements": []},
        }
    )
    return {
        "holdout": HoldoutSystem("holdout", client_factory=lambda: cooperative),
        "audits_only": HoldoutSystem("audits_only"),
        "llm_judge": HoldoutSystem(
            "llm_judge", client_factory=lambda: judge_fake, customize=install
        ),
        "react": ReactSystem(_react_fake),
        "flaml": FlamlSystem(time_budget=3),
    }


class _Watched:
    """Wraps a system; records every file the Python process opens during its run."""

    opened: list[str]

    def __init__(self, inner: System) -> None:
        self.inner = inner
        self.name = inner.name
        self.uses_llm = inner.uses_llm
        self.opened = []

    def _watch(self, fn: Any, ctx: SystemContext) -> SystemOutput:
        _WATCHERS.append(self.opened)
        try:
            return fn(ctx)  # type: ignore[no-any-return]
        finally:
            _WATCHERS.pop()

    def run(self, ctx: SystemContext) -> SystemOutput:
        return self._watch(self.inner.run, ctx)

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self._watch(self.inner.resume, ctx)


_WATCHERS: list[list[str]] = []


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and _WATCHERS and args and isinstance(args[0], str | Path):
        _WATCHERS[-1].append(str(args[0]))


sys.addaudithook(_hook)  # audit hooks cannot be removed; it records only while watching


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("holdout", marks=pytest.mark.full_only),
        "audits_only",
        pytest.param("llm_judge", marks=pytest.mark.full_only),
        "react",
        pytest.param("flaml", marks=pytest.mark.full_only),
    ],
)
def test_every_system_runs_and_never_reads_the_holdout(tmp_path: Path, name: str) -> None:
    system = _Watched(_systems(tmp_path)[name])
    unit = Unit("dev_v1", "direct_leak@synthetic", name, 11, "planted")  # type: ignore[arg-type]
    result = run_unit(
        unit, system, tmp_path / "results", model="gemini-3.8-flash", run_cap_inr=15.0,
        prepare_fn=_prepared,
    )  # fmt: skip
    assert result.status == "completed", result.error
    seal_dir = tmp_path / "results" / "seals"
    assert not [p for p in system.opened if Path(p).resolve().is_relative_to(seal_dir.resolve())]
    assert result.holdout_iid is not None
    assert result.holdout_deploy is not None
    assert 0.0 <= result.holdout_deploy <= 1.0
    assert result.dummy_holdout_deploy == pytest.approx(0.5)
    log = [e for e in _seal_log(seal_dir) if e["caller"] == name]
    assert sorted(e["holdout"] for e in log) == ["deploy", "iid"]  # once per holdout
    workdir = tmp_path / "results" / "work"
    for path in workdir.rglob("*"):
        assert "holdout.parquet" not in path.name or "runs" in path.parts  # Holdout's own seal
    if name != "flaml":
        categories = {i.category.value for i in result.issues}
        assert categories & {"target_leakage", "post_outcome_column"}
    if name == "react":
        assert result.extra["observation_chars_raw"] > 5000
        assert result.extra["observation_chars_shown"] <= 2000
        assert result.extra["observation_tokens_sent_estimate"] > 0


def _seal_log(seal_dir: Path) -> list[dict[str, Any]]:
    import json

    entries = []
    for log in seal_dir.rglob("access.jsonl"):
        entries += [json.loads(line) for line in log.read_text().splitlines() if line]
    return entries


@pytest.mark.full_only
def test_the_judge_replaces_the_audit_layer(tmp_path: Path) -> None:
    unit = Unit("dev_v1", "direct_leak@synthetic", "llm_judge", 11, "planted")
    result = run_unit(
        unit, _systems(tmp_path)["llm_judge"], tmp_path / "r", model="gemini-3.8-flash",
        run_cap_inr=15.0, prepare_fn=_prepared,
    )  # fmt: skip
    assert result.status == "completed", result.error
    sources = {i.source for i in result.issues}
    assert sources <= {"A2", "A11"}  # the judge's flags (A2 owns post_outcome_column)
    assert result.pipeline is not None
    assert "refund_issued" in result.pipeline.dropped_columns  # the same fix loop applied it


class _Flaky:
    """Pauses (503) on the first run, completes on resume; counts calls."""

    name = "audits_only"
    uses_llm = False

    def __init__(self) -> None:
        self.inner = HoldoutSystem("audits_only")
        self.runs = 0
        self.resumes = 0

    def run(self, ctx: SystemContext) -> SystemOutput:
        self.runs += 1
        raise LLMUnavailableError("503 on every attempt", status=503, retries=5)

    def resume(self, ctx: SystemContext) -> SystemOutput:
        self.resumes += 1
        return self.inner.run(ctx)


def test_the_runner_resumes_paused_units_skips_done_ones_and_records_failures(
    tmp_path: Path,
) -> None:
    flaky = _Flaky()

    class _Broken:
        name = "flaml"
        uses_llm = False

        def run(self, ctx: SystemContext) -> SystemOutput:
            raise RuntimeError("model crashed")

        def resume(self, ctx: SystemContext) -> SystemOutput:
            return self.run(ctx)

    units = [
        Unit("dev_v1", "t", "audits_only", 11, "planted"),
        Unit("dev_v1", "t", "flaml", 11, "planted"),
    ]
    systems: dict[str, Any] = {"audits_only": flaky, "flaml": _Broken()}
    root = tmp_path / "res"
    lines: list[str] = []
    first = run_all(
        units, systems, root, model="m", run_cap_inr=15.0, write=lines.append, prepare_fn=_prepared
    )
    assert [r.status for r in first] == ["paused"]  # a pause stops the pass
    assert "rerun the same command" in lines[-1]
    second = run_all(
        units, systems, root, model="m", run_cap_inr=15.0, write=lines.append, prepare_fn=_prepared
    )
    assert [r.status for r in second] == ["completed", "failed"]
    assert flaky.resumes == 1
    assert second[1].error is not None
    assert "model crashed" in second[1].error
    assert pending(units, root) == []  # completed and failed are terminal
    assert (
        run_all(
            units,
            systems,
            root,
            model="m",
            run_cap_inr=15.0,
            write=lines.append,
            prepare_fn=_prepared,
        )
        == []
    )
    append_result(root, second[0].model_copy(update={"status": "paused"}))
    assert [u.system for u, resume in pending(units, root) if resume] == ["audits_only"]
