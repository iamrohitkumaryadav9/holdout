"""v1.1 deduplication as an executable plan action on the development rows.

Post-benchmark fix from docs/error_analysis.md, section 1 (deduplication was only ever a
recommendation: no plan could act on an A4 finding, so every duplicate finding stayed
open). Not part of the pre-registered evaluation."""

import numpy as np
import pandas as pd
import pytest

from holdout.agent.planner import apply_fixes, fix_constraints, missed_fixes, plan_fixes
from holdout.audit.checks import a04_duplicates
from holdout.audit.context import AuditContext
from holdout.audit.runner import run_audits
from holdout.audit.thresholds import load_thresholds, load_thresholds_v11
from holdout.audit.types import (
    AuditFlag,
    Category,
    CheckId,
    Deduplicate,
    DuplicateEvidence,
    Severity,
)
from holdout.core import config, portable
from holdout.core.schemas import SplitStrategy, Task
from holdout.data.profile import profile_schema
from holdout.data.rowhash import row_fingerprints
from holdout.data.split import SplitSpec
from holdout.export.pipeline_export import render_pipeline_script
from holdout.plan.compiler import evaluate_cv, plan_rows
from holdout.plan.defaults import baseline_plan, default_plan
from holdout.plan.schema import ModelPlan

DIGITS = config.ROW_HASH_SIGNIFICANT_DIGITS
SPEC = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=5)


def _frame(n: int = 600, n_copies: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.choice(["x", "y", "z"], n),
        "y": rng.integers(0, 2, n),
    })  # fmt: skip
    copies = base.sample(n_copies, random_state=seed)
    return pd.concat([base, copies], ignore_index=True).sample(frac=1.0, random_state=seed + 1)


def _plan(frame: pd.DataFrame, deduplicate: bool) -> ModelPlan:
    plan = default_plan(profile_schema(frame, "y", task=Task.CLASSIFICATION), Task.CLASSIFICATION)
    return plan.model_copy(update={"deduplicate": deduplicate})


# --------------------------------------------------------------------- the kernel


def test_the_first_of_each_duplicate_is_kept_and_the_label_is_ignored() -> None:
    frame = pd.DataFrame({
        "a": [1.0, 1.0000000001, 2.0, 1.0],
        "b": ["X ", "x", "x", "x"],
        "y": [0, 1, 0, 0],
    }, index=[10, 11, 12, 13])  # fmt: skip
    rows, removed = portable.drop_duplicate_rows(frame, "y", DIGITS)
    assert removed == 2  # rows 11 and 13 match row 10 after normalisation
    assert list(rows.index) == [10, 12]  # original order and index, holdout untouched


def test_after_deduplication_the_audit_sees_no_duplicates() -> None:
    rows, removed = portable.drop_duplicate_rows(_frame(), "y", DIGITS)
    assert removed > 0
    fingerprints = row_fingerprints(rows.drop(columns=["y"]), DIGITS)
    assert len(np.unique(fingerprints)) == len(rows)


def test_a_deduplicating_plan_is_evaluated_on_fewer_rows() -> None:
    frame = _frame()
    profile = profile_schema(frame, "y", task=Task.CLASSIFICATION)
    kept = evaluate_cv(_plan(frame, False), frame, "y", SPEC, profile)
    deduped_plan = _plan(frame, True)
    deduped = evaluate_cv(deduped_plan, frame, "y", SPEC, profile)
    assert kept.n_rows == len(frame)
    assert deduped.n_rows == len(plan_rows(deduped_plan, frame, "y")) < len(frame)
    assert baseline_plan(deduped_plan).deduplicate  # the dummy is scored on the same rows


def test_a_v1_0_plan_does_not_deduplicate() -> None:
    raw = _plan(_frame(), False).model_dump(mode="json")
    raw.pop("deduplicate")
    assert not ModelPlan.model_validate(raw).deduplicate


# --------------------------------------------------------------------- planner fixes


def _a4(scope: str, invalidates: bool = False) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A4,
        category=Category.DUPLICATE_ROWS,
        severity=Severity.CRITICAL,
        columns=[],
        evidence=DuplicateEvidence(
            normalisation="test",
            n_dev_rows=100,
            dev_redundant_rows=10,
            cross_fold_duplicate_rows=8,
            cross_fold_share=0.08,
            holdout_rows=None,
            holdout_rows_with_dev_duplicate=None,
            holdout_overlap_share=None,
            holdout_overlap_note="test",
        ),
        message="8 validation rows duplicate a training row of their fold.",
        fix=Deduplicate(scope=scope),  # type: ignore[arg-type]
        invalidates_holdout=invalidates,
    )


def test_a_dev_deduplication_is_a_plan_fix() -> None:
    plan = _plan(_frame(), False)
    flags = [_a4("dev")]
    assert plan_fixes(flags) == flags
    fixed = apply_fixes(plan, plan_fixes(flags))
    assert fixed.deduplicate
    assert missed_fixes(plan, flags) == ["deduplicate:dev"]
    assert missed_fixes(fixed, flags) == []
    assert "revised_plan.deduplicate MUST be true" in fix_constraints(plan, flags)


def test_a_cross_split_overlap_is_not_a_plan_fix() -> None:
    """Holdout overlap keeps the holdout_invalid rule: it needs a new run, not a plan."""
    assert plan_fixes([_a4("before_split", invalidates=True)]) == []
    assert plan_fixes([_a4("before_split")]) == []


# --------------------------------------------------------------------- A4 scope


class _Overlap:
    """A seal whose holdout overlaps the development rows by ``share``."""

    def __init__(self, share: float, rows: int = 1000) -> None:
        self.share, self.rows = share, rows

    def holdout_overlap(self, X: pd.DataFrame) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(holdout_rows=self.rows,
                               holdout_rows_matching=round(self.share * self.rows))  # fmt: skip


@pytest.mark.parametrize(("v11", "share", "scope"), [
    (False, 0.005, "before_split"),  # v1.0: any overlap asked for a new run
    (True, 0.005, "dev"),  # v1.1: below the invalidation share the plan can act
    (True, 0.02, "before_split"),  # an invalidating overlap still needs a new run
])  # fmt: skip
def test_the_deduplication_scope(v11: bool, share: float, scope: str) -> None:
    frame = _frame()
    plan = _plan(frame, False)
    profile = profile_schema(frame, "y", task=Task.CLASSIFICATION)
    cv = evaluate_cv(baseline_plan(plan), frame, "y", SPEC, profile)
    ctx = AuditContext(dev=frame, target="y", profile=profile, spec=SPEC, plan=plan,
                       cv_result=cv, baseline=cv, thresholds=load_thresholds()[0],
                       overlap_source=_Overlap(share),  # type: ignore[arg-type]
                       thresholds_v11=load_thresholds_v11()[0] if v11 else None)  # fmt: skip
    (flag,) = a04_duplicates.run(ctx).flags
    assert isinstance(flag.fix, Deduplicate)
    assert flag.fix.scope == scope
    assert flag.invalidates_holdout is (share >= 0.01)


def test_a_deduplicating_plan_is_audited_on_its_own_rows() -> None:
    frame = _frame()
    profile = profile_schema(frame, "y", task=Task.CLASSIFICATION)
    for deduplicate, expect_a4 in ((False, True), (True, False)):
        plan = _plan(frame, deduplicate)
        cv = evaluate_cv(plan, frame, "y", SPEC, profile)
        report = run_audits(None, frame, plan, cv, baseline=cv, profile=profile, spec=SPEC,
                            target="y", thresholds_v11=load_thresholds_v11(),
                            checks=frozenset({CheckId.A4}))  # fmt: skip
        assert any(f.check is CheckId.A4 for f in report.flags) is expect_a4


# --------------------------------------------------------------------- export


def test_the_exported_script_deduplicates_only_when_the_plan_does() -> None:
    frame = _frame()
    profile = profile_schema(frame, "y", task=Task.CLASSIFICATION)
    plain = render_pipeline_script(plan=_plan(frame, False), profile=profile, spec=SPEC,
                                   data_sha256="0" * 64)  # fmt: skip
    deduped = render_pipeline_script(plan=_plan(frame, True), profile=profile, spec=SPEC,
                                     data_sha256="0" * 64)  # fmt: skip
    assert "dedup_significant_digits=DEDUP_SIGNIFICANT_DIGITS" not in plain
    assert f"DEDUP_SIGNIFICANT_DIGITS = {DIGITS}" in deduped
    assert "dedup_significant_digits=DEDUP_SIGNIFICANT_DIGITS," in deduped
    compile(deduped, "pipeline.py", "exec")
