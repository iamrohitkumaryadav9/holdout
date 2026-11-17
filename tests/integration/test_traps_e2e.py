"""Every trap, end to end through ``holdout run``: the expected category fires at or above
the trap's ``min_expected_severity``; its clean twin produces no critical flag."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bench.traps import TRAPS, TrapCase, build_case, clean_twin
from holdout.audit.checks.a11_causal import check_causal_claims
from holdout.audit.types import Category, Severity
from holdout.cli import app
from holdout.data.load import load_dataset
from holdout.data.profile import profile_schema
from holdout.plan.defaults import default_plan
from holdout.plan.schema import ModelPlan

runner = CliRunner()

#: Traps run in the fast tier; the rest run in the full tier only.
FAST_TRAPS = {"direct_leak", "cross_split_duplicates"}


def _tiered(names: set[str]) -> list[Any]:
    return [
        n if n in FAST_TRAPS else pytest.param(n, marks=pytest.mark.full_only)
        for n in sorted(names)
    ]


#: Traps whose detection needs a generated report (A11, wired in Week 3).
REPORT_LEVEL = {"causal_question"}


def _run_case(case: TrapCase, tmp_path: Path) -> dict[str, Any]:
    csv = tmp_path / "data.csv"
    case.df.to_csv(csv, index=False)
    args = ["run", "--csv", str(csv), "--target", case.target, "--goal", case.goal_text]
    if case.plan_overrides:
        profile = profile_schema(load_dataset(csv, case.target).frame, case.target, task=case.task)
        plan = ModelPlan.model_validate(
            {**default_plan(profile, case.task).model_dump(mode="json"), **case.plan_overrides}
        )
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
        args += ["--plan", str(plan_path)]
    out = tmp_path / "runs"
    result = runner.invoke(app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    run_dir = next(p for p in out.iterdir() if p.is_dir())
    manifest: dict[str, Any] = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    return manifest


def _detected(manifest: dict[str, Any], category: Category, minimum: Severity) -> bool:
    return any(
        flag["category"] == category.value and Severity(flag["severity"]).at_least(minimum)
        for flag in manifest["audit"]["flags"]
    )


@pytest.mark.parametrize("name", _tiered(set(TRAPS) - REPORT_LEVEL))
def test_trap_is_detected_end_to_end(name: str, tmp_path: Path) -> None:
    case = build_case(name, 0)
    manifest = _run_case(case, tmp_path)
    assert manifest["audit"]["complete"]
    truth = case.truth
    assert truth.expected_category is not None
    assert _detected(manifest, truth.expected_category, truth.min_expected_severity), (
        f"{name}: expected {truth.expected_category.value} >= {truth.min_expected_severity.value}; "
        f"got {[(f['check'], f['severity'], f['category']) for f in manifest['audit']['flags']]}"
    )


@pytest.mark.parametrize("name", _tiered(set(TRAPS)))
def test_clean_twin_has_no_critical_flag_end_to_end(name: str, tmp_path: Path) -> None:
    manifest = _run_case(clean_twin(build_case(name, 0)), tmp_path)
    criticals = [f for f in manifest["audit"]["flags"] if f["severity"] == "critical"]
    assert criticals == [], [(f["check"], f["message"]) for f in criticals]
    assert manifest["holdout_invalid"] is False


@pytest.mark.full_only
def test_causal_question_is_report_level_in_week_2(tmp_path: Path) -> None:
    case = build_case("causal_question", 0)
    manifest = _run_case(case, tmp_path)
    a11 = next(r for r in manifest["audit"]["checks_run"] if r["check"] == "A11")
    assert a11["status"] == "skipped"
    assert manifest["audit"]["complete"]
    assert manifest["goal_text"] == case.goal_text
    # What Week 3 will check on the generated report:
    flags = check_causal_claims("Being contacted by phone causes customers to convert.")
    assert flags[0].category == case.truth.expected_category
