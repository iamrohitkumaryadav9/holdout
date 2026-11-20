"""v1.1 deduplication through the agent graph: proposed as a fix, approved by the user in
interactive mode, applied by code before cross-validation and recorded in the manifest.

Post-benchmark fix from docs/error_analysis.md, section 1. Not part of the pre-registered
evaluation. A random seal of data with real duplicates almost always puts some of them in
the holdout (A4 then invalidates the holdout and asks for a new run), so the audit layer is
replaced by a stub that raises the development-only finding (critical, scope "dev")."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bench.traps import build_case
from bench.traps.types import clean_case
from holdout.agent.driver import run_agent
from holdout.agent.graph import AgentDeps, AgentOptions, AuditInputs
from holdout.agent.schemas import Decision
from holdout.audit.types import (
    AuditFlag,
    AuditReport,
    Category,
    CheckId,
    CheckRun,
    Deduplicate,
    DuplicateEvidence,
    Severity,
)


def _a4_dev() -> AuditFlag:
    return AuditFlag(
        check=CheckId.A4,
        category=Category.DUPLICATE_ROWS,
        severity=Severity.CRITICAL,
        columns=[],
        evidence=DuplicateEvidence(
            normalisation="test",
            n_dev_rows=0,
            dev_redundant_rows=0,
            cross_fold_duplicate_rows=0,
            cross_fold_share=0.05,
            holdout_rows=None,
            holdout_rows_with_dev_duplicate=None,
            holdout_overlap_share=None,
            holdout_overlap_note="test",
        ),
        message="5% of validation rows duplicate a training row of their fold.",
        fix=Deduplicate(scope="dev"),
    )


def _duplicate_auditor(deps: AgentDeps) -> None:
    def audit(inputs: AuditInputs) -> AuditReport:
        return AuditReport(
            flags=[] if inputs.plan.deduplicate else [_a4_dev()],
            checks_run=[CheckRun(check=CheckId.A4, status="ran", seconds=0.0)],
            thresholds_hash="t" * 64,
            deep_mode=False,
            entity_scope=None,
        )

    deps.auditor = audit


class _User:
    """Answers interactively: accept the split, answer the review, decline the unseal."""

    def __init__(self, approve_fixes: bool) -> None:
        self.approve_fixes = approve_fixes
        self.payloads: list[dict[str, Any]] = []

    def ask(self, payload: dict[str, Any]) -> Decision:
        self.payloads.append(payload)
        kind = payload["kind"]
        accepted = self.approve_fixes if kind == "review" else kind != "unseal"
        return Decision(kind=kind, mode="interactive", accepted=accepted, reason="test user")


def _run(tmp_path: Path, user: _User) -> Any:
    base = build_case("direct_leak", 0)
    frame = base.df.drop(columns=["refund_issued"])
    frame = pd.concat([frame, frame.sample(150, random_state=0)], ignore_index=True)
    case = clean_case(frame, base.target, base.task)
    csv = tmp_path / "data.csv"
    case.df.to_csv(csv, index=False)
    return run_agent(
        AgentOptions(csv=str(csv), target=case.target, out_dir=str(tmp_path / "runs"),
                     goal=case.goal_text, mode="interactive", offline=True, use_cache=False),
        asker=user,
        customize=_duplicate_auditor,
    )  # fmt: skip


def test_the_user_approves_deduplication_and_code_applies_it(tmp_path: Path) -> None:
    user = _User(approve_fixes=True)
    outcome = _run(tmp_path, user)
    (review,) = [p for p in user.payloads if p["kind"] == "review"]
    assert not review["current_plan"]["deduplicate"]
    assert review["proposed_plan"]["deduplicate"]
    manifest = outcome.manifest
    assert manifest is not None
    assert manifest.plans[-1].plan.deduplicate
    assert manifest.dev_rows_deduplicated is not None
    assert manifest.dev_rows_deduplicated > 0
    assert (
        manifest.plans[-1].cv.n_rows == manifest.split.n_dev_rows - manifest.dev_rows_deduplicated
    )
    assert manifest.plans[0].cv.n_rows == manifest.plans[-1].cv.n_rows  # the dummy too
    assert manifest.agent is not None
    assert [d.kind for d in manifest.agent.decisions if d.accepted] == ["split", "review"]
    (finding,) = manifest.agent.findings
    assert finding.resolved
    assert finding.outcome == "resolved: duplicate development rows removed"
    report = (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    assert "duplicate development rows removed before cross-validation" in report
    assert "DEDUP_SIGNIFICANT_DIGITS" in (outcome.run_dir / "pipeline.py").read_text("utf-8")


def test_a_declined_deduplication_is_not_applied(tmp_path: Path) -> None:
    outcome = _run(tmp_path, _User(approve_fixes=False))
    manifest = outcome.manifest
    assert manifest is not None
    assert not manifest.plans[-1].plan.deduplicate
    assert manifest.dev_rows_deduplicated is None
    assert manifest.agent is not None
    assert manifest.agent.stop_reason == "the proposed fixes were declined"
    (finding,) = manifest.agent.findings
    assert not finding.resolved


@pytest.mark.parametrize("approve", [True])
def test_deduplication_never_touches_the_holdout(tmp_path: Path, approve: bool) -> None:
    outcome = _run(tmp_path, _User(approve_fixes=approve))
    manifest = outcome.manifest
    assert manifest is not None
    assert manifest.split.n_holdout_rows + manifest.split.n_dev_rows == manifest.data.n_rows
