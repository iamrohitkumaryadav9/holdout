"""run_audits: order, corroboration, deep mode, error isolation, and audit.json."""

from pathlib import Path

import pytest

from bench.traps.synthetic import easy_clean
from bench.traps.types import clean_case
from holdout.audit import runner as audit_runner
from holdout.audit.checks import a01_single_feature, a05_group
from holdout.audit.context import AuditContext
from holdout.audit.thresholds import load_thresholds
from holdout.audit.types import AuditReport, Category, CheckId, Severity
from holdout.core.state import RunState
from tests.audit_harness import context_for, trap_context
from tests.tiers import SEEDS


def audit(ctx: AuditContext, run_state: RunState | None = None) -> AuditReport:
    return audit_runner.run_audits(
        run_state,
        ctx.dev,
        ctx.plan,
        ctx.cv_result,
        baseline=ctx.baseline,
        profile=ctx.profile,
        spec=ctx.spec,
        target=ctx.target,
        overlap_source=ctx.overlap_source,
        goal_text=ctx.goal_text,
    )


def test_direct_leak_is_corroborated_to_critical(tmp_path: Path) -> None:
    state = RunState.create(tmp_path, 0)
    report = audit(trap_context("direct_leak", 0), state)
    a1 = next(f for f in report.flags if f.check is CheckId.A1 and f.columns == ["refund_issued"])
    assert a1.severity is Severity.CRITICAL
    assert a1.evidence.corroborated_by == [CheckId.A2]  # type: ignore[union-attr]
    assert report.complete
    assert report.thresholds_hash == load_thresholds()[1]
    assert [r.check for r in report.checks_run] == list(CheckId)
    assert AuditReport.model_validate_json(state.audit_path.read_text(encoding="utf-8")) == report


def test_noisy_proxy_stays_a_warning() -> None:
    report = audit(trap_context("noisy_proxy", 0))
    target_flags = [f for f in report.flags if f.category is Category.TARGET_LEAKAGE]
    assert [f.severity for f in target_flags] == [Severity.WARNING]


@pytest.mark.parametrize("seed", SEEDS)
def test_easy_but_clean_data_gets_at_most_an_a1_warning(seed: int) -> None:
    frame, target, task = easy_clean(seed=seed)
    report = audit(context_for(clean_case(frame, target, task), seed))
    assert report.at_least(Severity.CRITICAL) == []
    above_info = report.at_least(Severity.WARNING)
    assert {f.check for f in above_info} <= {CheckId.A1}
    assert report.deep_mode  # near-separable: deep mode, but no false critical


def test_skipped_a11_is_recorded_and_does_not_make_the_audit_incomplete() -> None:
    report = audit(trap_context("noisy_proxy", 0))
    a11 = next(r for r in report.checks_run if r.check is CheckId.A11)
    assert a11.status == "skipped"
    assert a11.detail == audit_runner.A11_SKIP_DETAIL
    assert report.complete


def test_a_failing_check_is_isolated_and_marks_the_audit_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(ctx: AuditContext) -> None:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(a05_group, "run", broken)
    report = audit(trap_context("direct_leak", 0))
    a5 = next(r for r in report.checks_run if r.check is CheckId.A5)
    assert (a5.status, a5.detail) == ("error", "RuntimeError: injected failure")
    assert not report.complete
    assert any(f.check is CheckId.A1 for f in report.flags)  # the others still ran


def test_a_failing_a1_still_runs_a2_and_a3(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(ctx: AuditContext) -> None:
        raise ValueError("no features")

    monkeypatch.setattr(a01_single_feature, "measure", broken)
    report = audit(trap_context("direct_leak", 0))
    statuses = {r.check: r.status for r in report.checks_run}
    assert statuses[CheckId.A1] == "error"
    assert statuses[CheckId.A2] == "ran"
    assert statuses[CheckId.A3] == "ran"
    assert not report.complete


def test_group_leakage_makes_the_holdout_invalid() -> None:
    report = audit(trap_context("group_leakage", 0))
    assert report.holdout_invalid
    assert any(f.check is CheckId.A5 and f.invalidates_holdout for f in report.flags)
