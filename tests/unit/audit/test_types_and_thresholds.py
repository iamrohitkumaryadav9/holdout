"""Audit types round-trip through JSON; thresholds load strictly and hash stably."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from holdout.audit.thresholds import default_thresholds_bytes, load_thresholds, parse_thresholds
from holdout.audit.types import (
    AuditFlag,
    AuditReport,
    Category,
    ChangeSplit,
    CheckId,
    CheckRun,
    CleanFormats,
    NoAction,
    ParseNumeric,
    SamplingInfo,
    ScoreEvidence,
    Severity,
)
from holdout.core.schemas import MetricName, SplitStrategy
from holdout.data.split import SplitSpec

SAMPLING = SamplingInfo(
    n_rows_total=10,
    n_rows_used=10,
    row_method="none",
    n_features_total=3,
    n_features_used=3,
    feature_method="all",
)


def _flag(severity: Severity, invalidates: bool = False) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A10,
        category=Category.SUSPICIOUS_SCORE,
        severity=severity,
        columns=[],
        evidence=ScoreEvidence(
            metric=MetricName.ROC_AUC,
            score=0.99,
            dummy_score=0.5,
            normalized_score=0.98,
            deep_mode=True,
        ),
        message="Score is near the ceiling.",
        fix=NoAction(reason="investigate"),
        invalidates_holdout=invalidates,
    )


def test_severity_ordering() -> None:
    assert Severity.CRITICAL.at_least(Severity.WARNING)
    assert Severity.WARNING.at_least(Severity.WARNING)
    assert not Severity.INFO.at_least(Severity.WARNING)


def test_report_summary_and_round_trip() -> None:
    report = AuditReport(
        flags=[_flag(Severity.INFO), _flag(Severity.CRITICAL, invalidates=True)],
        checks_run=[CheckRun(check=CheckId.A10, status="ran", seconds=0.1, sampling=SAMPLING)],
        thresholds_hash="abc",
        deep_mode=True,
        entity_scope=None,
    )
    assert report.counts == {"info": 1, "warning": 0, "critical": 1}
    assert report.complete
    assert report.holdout_invalid
    assert [f.severity for f in report.at_least(Severity.WARNING)] == [Severity.CRITICAL]
    again = AuditReport.model_validate_json(report.model_dump_json())
    assert again == report


def test_errors_make_the_report_incomplete_but_skips_do_not() -> None:
    skipped = CheckRun(check=CheckId.A11, status="skipped", seconds=0.0, detail="no report text")
    failed = CheckRun(check=CheckId.A1, status="error", seconds=0.0, detail="boom")
    base = {"flags": [], "thresholds_hash": "h", "deep_mode": False, "entity_scope": None}
    assert AuditReport(checks_run=[skipped], **base).complete  # type: ignore[arg-type]
    assert not AuditReport(checks_run=[skipped, failed], **base).complete  # type: ignore[arg-type]


def test_fixes_are_typed() -> None:
    fix = ChangeSplit(
        new_spec=SplitSpec(strategy=SplitStrategy.GROUP, group_col="account", seed=1),
        instruction="start a new run with --split group:account",
    )
    assert fix.new_spec is not None
    clean = CleanFormats(
        column="income", action=ParseNumeric(strip_tokens=[",", "Rs"], percent=False)
    )
    assert clean.action.kind == "parse_numeric"
    with pytest.raises(ValidationError):
        CleanFormats.model_validate({"column": "x", "action": {"kind": "magic"}})


def test_packaged_thresholds_load_and_hash() -> None:
    thresholds, digest = load_thresholds()
    assert digest == hashlib.sha256(default_thresholds_bytes()).hexdigest()
    assert thresholds.sampling.max_rows == 20000
    assert thresholds.a04_duplicates.holdout_overlap_invalidate_share == pytest.approx(0.01)
    assert b"FROZEN for the Week 4 benchmark" in default_thresholds_bytes()
    assert b"\r\n" not in default_thresholds_bytes()


def test_thresholds_reject_unknown_and_missing_keys(tmp_path: Path) -> None:
    raw = default_thresholds_bytes().decode()
    with pytest.raises(ValidationError, match="extra"):
        parse_thresholds((raw + "\n[sampling_typo]\nmax_rows = 1\n").encode())
    without = raw.replace("max_features = 200\n", "")
    with pytest.raises(ValidationError, match="max_features"):
        parse_thresholds(without.encode())
    path = tmp_path / "t.toml"
    path.write_bytes(raw.replace("max_rows = 20000", "max_rows = 5000").encode())
    custom, digest = load_thresholds(path)
    assert custom.sampling.max_rows == 5000
    assert digest != load_thresholds()[1]
