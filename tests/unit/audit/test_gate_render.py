"""Unseal gate rules and fix rendering."""

import pytest

from holdout.audit.gate import decide_unseal
from holdout.audit.render import fix_text
from holdout.audit.types import (
    AuditFlag,
    AuditReport,
    Category,
    ChangeMetric,
    ChangeSplit,
    CheckId,
    CheckRun,
    CleanFormats,
    Deduplicate,
    DropColumns,
    NoAction,
    NormaliseCategories,
    ParseNumeric,
    RemoveCausalClaim,
    ReplaceSentinel,
    ScoreEvidence,
    Severity,
)
from holdout.core.schemas import MetricName

REASON = "documented exception approved by the data owner"


def _flag(severity: Severity, invalidates: bool = False) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A1,
        category=Category.TARGET_LEAKAGE,
        severity=severity,
        columns=["x"],
        evidence=ScoreEvidence(
            metric=MetricName.ROC_AUC,
            score=0.9,
            dummy_score=0.5,
            normalized_score=0.8,
            deep_mode=False,
        ),
        message="m",
        fix=NoAction(reason="r"),
        invalidates_holdout=invalidates,
    )


def _report(*flags: AuditFlag, failed: bool = False) -> AuditReport:
    status = "error" if failed else "ran"
    return AuditReport(
        flags=list(flags),
        checks_run=[CheckRun(check=CheckId.A5, status=status, seconds=0.0)],  # type: ignore[arg-type]
        thresholds_hash="h",
        deep_mode=False,
        entity_scope=None,
    )


def test_clean_report_allows_unsealing() -> None:
    decision = decide_unseal(_report(_flag(Severity.INFO)), requested=True)
    assert decision.allowed
    assert decision.blocked_by == []


def test_not_requested_is_never_allowed_but_reports_blockers() -> None:
    decision = decide_unseal(_report(_flag(Severity.CRITICAL)), requested=False)
    assert not decision.allowed
    assert decision.blocked_by


def test_warnings_need_acknowledgement() -> None:
    report = _report(_flag(Severity.WARNING))
    assert not decide_unseal(report, requested=True).allowed
    assert decide_unseal(report, requested=True, ack_warnings=True).allowed


def test_critical_needs_accept_risk_with_a_reason() -> None:
    report = _report(_flag(Severity.CRITICAL))
    assert not decide_unseal(report, requested=True).allowed
    decision = decide_unseal(report, requested=True, accept_risk=REASON)
    assert decision.allowed
    assert decision.accepted_risk == REASON
    assert decision.overridden_criticals == ["A1 x"]
    with pytest.raises(ValueError, match="at least 10 characters"):
        decide_unseal(report, requested=True, accept_risk="ok")


def test_accept_risk_does_not_acknowledge_warnings() -> None:
    report = _report(_flag(Severity.CRITICAL), _flag(Severity.WARNING))
    assert not decide_unseal(report, requested=True, accept_risk=REASON).allowed
    assert decide_unseal(report, requested=True, accept_risk=REASON, ack_warnings=True).allowed


def test_invalid_holdout_cannot_be_overridden() -> None:
    report = _report(_flag(Severity.CRITICAL, invalidates=True))
    decision = decide_unseal(report, requested=True, accept_risk=REASON, ack_warnings=True)
    assert not decision.allowed
    assert "not overridable" in decision.blocked_by[0]
    assert decision.overridden_criticals == []


def test_incomplete_audit_cannot_be_overridden() -> None:
    decision = decide_unseal(
        _report(failed=True), requested=True, accept_risk=REASON, ack_warnings=True
    )
    assert not decision.allowed
    assert decision.blocked_by == ["audit incomplete: A5 failed (not overridable)"]


@pytest.mark.parametrize(
    ("fix", "text"),
    [
        (DropColumns(columns=["a", "b"], reason="r"), "drop 'a', 'b'"),
        (ChangeSplit(new_spec=None, instruction="do X"), "do X"),
        (
            Deduplicate(scope="before_split"),
            "deduplicate the data before splitting, then start a new run",
        ),
        (Deduplicate(scope="dev"), "deduplicate development rows"),
        (
            CleanFormats(column="c", action=ParseNumeric(strip_tokens=[","], percent=True)),
            "parse 'c' as numbers (strip ','; percent)",
        ),
        (
            CleanFormats(column="c", action=NormaliseCategories(mapping={"A": "a"})),
            "normalise 1 spelling variant(s) in 'c'",
        ),
        (
            CleanFormats(column="c", action=ReplaceSentinel(value=-999.0)),
            "treat -999 in 'c' as missing",
        ),
        (ChangeMetric(metric=MetricName.MAE), "use mae"),
        (RemoveCausalClaim(statements=["s"]), "remove the causal claim(s)"),
        (NoAction(reason="nothing to do"), "nothing to do"),
    ],
)
def test_fix_text(fix: object, text: str) -> None:
    assert fix_text(fix) == text  # type: ignore[arg-type]
