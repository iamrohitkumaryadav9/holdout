"""Environment capture and report rendering."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from holdout.audit.gate import UnsealDecision, decide_unseal
from holdout.audit.types import (
    AuditFlag,
    AuditReport,
    Category,
    CheckId,
    CheckRun,
    NoAction,
    ScoreEvidence,
    Severity,
)
from holdout.core.schemas import CVResult, Direction, HoldoutResult, MetricName, SplitStrategy
from holdout.data.examples import make_classification_example
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec, propose_split
from holdout.export import manifest as manifest_module
from holdout.export.manifest import (
    NOT_UNSEALED,
    DataRecord,
    PlanRecord,
    ProfileSummary,
    RunManifest,
    SplitRecord,
    capture_environment,
    render_report,
)
from holdout.plan.defaults import baseline_plan, default_plan


def test_capture_environment_records_versions() -> None:
    env = capture_environment()
    assert env.python.startswith("3.12")
    assert env.packages["scikit-learn"] is not None
    assert env.packages["pandas"] is not None
    assert set(env.packages) == set(manifest_module.TRACKED_PACKAGES)


def test_git_state_is_none_outside_a_repository(tmp_path: Path) -> None:
    assert manifest_module._git_state(tmp_path) == (None, None)


def test_unknown_distribution_version_is_none() -> None:
    assert manifest_module._version("definitely-not-installed-dist") is None


def _cv(label: str, mean: float, metric: MetricName, direction: Direction) -> CVResult:
    return CVResult(
        plan_label=label,
        metric=metric,
        direction=direction,
        mean=mean,
        std=0.01,
        fold_scores=[mean] * 5,
        fit_seconds=0.1,
        n_rows=100,
        n_folds=5,
        splitter="StratifiedKFold",
    )


def _audit(*flags: AuditFlag) -> AuditReport:
    return AuditReport(
        flags=list(flags),
        checks_run=[CheckRun(check=CheckId.A1, status="ran", seconds=0.1)],
        thresholds_hash="t" * 64,
        deep_mode=False,
        entity_scope=None,
    )


def _flag(severity: Severity, invalidates: bool = False) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A5,
        category=Category.GROUP_LEAKAGE,
        severity=severity,
        columns=["account"],
        evidence=ScoreEvidence(
            metric=MetricName.ROC_AUC,
            score=0.9,
            dummy_score=0.5,
            normalized_score=0.8,
            deep_mode=False,
        ),
        message="Grouping by 'account' drops the score | a lot.",
        fix=NoAction(reason="start a new run"),
        invalidates_holdout=invalidates,
    )


def _manifest(
    holdout: HoldoutResult | None,
    direction: Direction = Direction.HIGHER,
    *,
    audit: AuditReport | None = None,
    unseal: UnsealDecision | None = None,
    caveats: list[str] | None = None,
) -> RunManifest:
    frame = make_classification_example(n_rows=200, seed=0)
    profile = profile_schema(frame, "churned")
    plan = default_plan(profile)
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=1)
    proposal = propose_split(profile, profile.task_hint, "tabular", seed=1)  # type: ignore[arg-type]
    metric = MetricName.ROC_AUC if direction is Direction.HIGHER else MetricName.RMSE
    now = datetime(2026, 11, 17, tzinfo=UTC)
    return RunManifest(
        run_id="run-1",
        started_at=now,
        finished_at=now,
        data=DataRecord(
            path="data/x.csv", sha256="f" * 64, n_rows=200, n_cols=9, n_dropped_missing_target=0
        ),
        target="churned",
        task=profile.task_hint,
        metric=metric,
        direction=direction,
        profile=ProfileSummary.from_profile(profile),
        split=SplitRecord.build(
            spec,
            proposal,
            n_dev_rows=160,
            n_holdout_rows=40,
            holdout_sha256="e" * 64,
            reuse_warnings=["reused before"],
        ),
        run_seed=1,
        derived_seeds={"split": 1},
        plans=[
            PlanRecord(
                label="dummy baseline",
                plan=baseline_plan(plan),
                issues=[],
                cv=_cv("dummy baseline", 0.5, metric, direction),
            ),
            PlanRecord(
                label="hist_gb", plan=plan, issues=[], cv=_cv("hist_gb", 0.7, metric, direction)
            ),
        ],
        selected_plan="hist_gb",
        validation_score=0.7,
        goal_text="predict churned",
        entity_scope=None,
        thresholds_hash="t" * 64,
        audit=audit if audit is not None else _audit(),
        unseal=unseal if unseal is not None else decide_unseal(_audit(), requested=False),
        holdout_invalid=(audit or _audit()).holdout_invalid,
        caveats=caveats or [],
        holdout=holdout if holdout is not None else NOT_UNSEALED,
        optimism_gap=None if holdout is None else 0.7 - holdout.score,
        artifacts=["pipeline.py"],
        environment=capture_environment(),
    )


def test_report_for_a_sealed_run() -> None:
    report = render_report(_manifest(None))
    assert "| Holdout (single evaluation) | not unsealed |" in report
    assert "| Optimism gap (validation - holdout) | n/a |" in report
    assert (
        "split chosen by the user instead of the proposed one" in report
    )  # spec differs from the proposal
    assert "- reused before" in report


def test_report_for_an_unsealed_lower_is_better_run() -> None:
    holdout = HoldoutResult(
        metric=MetricName.RMSE,
        score=0.75,
        n_rows=40,
        holdout_sha256="e" * 64,
        evaluation_index=1,
        forced=False,
        caller="test",
        evaluated_at=datetime(2026, 11, 17, tzinfo=UTC),
    )
    manifest = _manifest(holdout, Direction.LOWER)
    report = render_report(manifest)
    assert "| Holdout (single evaluation) | 0.7500 |" in report
    assert manifest.optimism_gap == pytest.approx(-0.05)
    assert "negative* gap means validation was optimistic" in report


def test_report_audit_table_escapes_cells_and_shows_counts() -> None:
    report = render_report(_manifest(None, audit=_audit(_flag(Severity.WARNING))))
    assert "complete; 0 critical, 1 warning, 0 info." in report
    assert (
        "| A5 | warning | `account` | Grouping by 'account' drops the score \\| a lot. |" in report
    )


def test_report_shows_accepted_risk_prominently() -> None:
    audit = _audit(_flag(Severity.CRITICAL))
    reason = "approved by the data owner for a demo"
    unseal = decide_unseal(audit, requested=True, accept_risk=reason)
    report = render_report(_manifest(None, audit=audit, unseal=unseal))
    first_lines = report.splitlines()[:4]
    assert any("RISK ACCEPTED" in line for line in first_lines)
    assert f"*{reason}*" in report
    assert "**Unseal allowed.**" in report


def test_report_shows_invalid_holdout_and_caveats() -> None:
    audit = _audit(_flag(Severity.CRITICAL, invalidates=True))
    unseal = decide_unseal(audit, requested=True)
    report = render_report(_manifest(None, audit=audit, unseal=unseal, caveats=["re-drawn"]))
    assert "HOLDOUT INVALID" in report
    assert "> **Caveat:** re-drawn" in report
    assert "**Unseal refused.**" in report
    assert "not overridable" in report


def test_report_lists_failed_checks() -> None:
    audit = AuditReport(
        flags=[],
        checks_run=[CheckRun(check=CheckId.A6, status="error", seconds=0.0, detail="boom")],
        thresholds_hash="t",
        deep_mode=False,
        entity_scope=None,
    )
    report = render_report(_manifest(None, audit=audit))
    assert "INCOMPLETE" in report
    assert "- A6 failed: boom" in report
