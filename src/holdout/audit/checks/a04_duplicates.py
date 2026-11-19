"""A4 duplicates: identical feature rows on both sides of a fold, or of the seal.

Rows are compared by normalised *feature* fingerprints (floats rounded to the seal's
significant digits, strings stripped and lower-cased; the label is not included). Three
quantities:

* validation rows whose duplicate is in the same fold's training rows (inflates CV);
* redundant rows within development data;
* via the seal's counts-only API, holdout rows that duplicate a development row.

Share rule (natural duplicates exist in real data, especially low-cardinality data): a
share at or above the threshold is critical; a smaller positive share is a warning with
the count. A holdout overlap at or above the threshold also invalidates the holdout.
"""

from typing import Literal

import numpy as np

from holdout.audit.checks import CheckOutcome
from holdout.audit.context import AuditContext
from holdout.audit.types import (
    AuditFlag,
    Category,
    CheckId,
    Deduplicate,
    DuplicateEvidence,
    Severity,
)
from holdout.core import config, portable
from holdout.data.rowhash import row_fingerprints
from holdout.plan.compiler import cv_from_split


def run(ctx: AuditContext) -> CheckOutcome:
    t = ctx.thresholds.a04_duplicates
    X, y, groups = portable.cv_inputs(ctx.dev, ctx.target, ctx.spec.to_config())
    fingerprints = row_fingerprints(X, config.ROW_HASH_SIGNIFICANT_DIGITS)
    cross_fold = n_valid = 0
    for train, valid in cv_from_split(ctx.spec).split(X, y, groups):
        seen = set(fingerprints[train].tolist())
        cross_fold += int(np.isin(fingerprints[valid], list(seen)).sum())
        n_valid += len(valid)
    cross_share = cross_fold / n_valid if n_valid else 0.0
    redundant = len(fingerprints) - len(np.unique(fingerprints))

    overlap = ctx.overlap_source.holdout_overlap(X) if ctx.overlap_source is not None else None
    if overlap is None:
        holdout_share = None
        note = "not available (no seal, or a seal written before row digests were recorded)"
    else:
        holdout_share = overlap.holdout_rows_matching / overlap.holdout_rows
        note = "counts from the seal's keyed digests; no holdout values or labels were read"

    invalidates = holdout_share is not None and holdout_share >= t.holdout_overlap_invalidate_share
    critical = invalidates or cross_share >= t.cross_fold_critical_share
    positive = cross_fold > 0 or (overlap is not None and overlap.holdout_rows_matching > 0)
    if not positive:
        return CheckOutcome(detail=f"no duplicates across folds; {redundant} redundant dev rows")

    parts = [
        f"{cross_fold} validation rows ({cross_share:.1%}) duplicate a training row of their fold"
    ]
    if overlap is not None and holdout_share is not None:
        parts.append(
            f"{overlap.holdout_rows_matching} holdout rows ({holdout_share:.1%}) duplicate a "
            "development row"
        )
    message = "; ".join(parts) + "."
    if invalidates:
        message += (
            " The sealed holdout is contaminated: deduplicate before splitting and start a new run."
        )
    flag = AuditFlag(
        check=CheckId.A4,
        category=Category.DUPLICATE_ROWS,
        severity=Severity.CRITICAL if critical else Severity.WARNING,
        columns=[],
        evidence=DuplicateEvidence(
            normalisation=(
                f"features only; floats rounded to {config.ROW_HASH_SIGNIFICANT_DIGITS} "
                "significant digits; strings stripped and lower-cased"
            ),
            n_dev_rows=len(X),
            dev_redundant_rows=int(redundant),
            cross_fold_duplicate_rows=cross_fold,
            cross_fold_share=cross_share,
            holdout_rows=None if overlap is None else overlap.holdout_rows,
            holdout_rows_with_dev_duplicate=None
            if overlap is None
            else overlap.holdout_rows_matching,
            holdout_overlap_share=holdout_share,
            holdout_overlap_note=note,
        ),
        message=message,
        fix=Deduplicate(scope=_scope(ctx, holdout_share, invalidates)),
        invalidates_holdout=invalidates,
    )
    return CheckOutcome(flags=[flag], detail=f"{redundant} redundant dev rows")


def _scope(
    ctx: AuditContext, holdout_share: float | None, invalidates: bool
) -> Literal["dev", "before_split"]:
    """v1.0: any holdout overlap asks for deduplication before the split. v1.1 (post-
    benchmark, error analysis section 1): only an overlap that invalidates the holdout does;
    below that threshold, deduplicating the development rows - an action the plan can take
    (``ModelPlan.deduplicate``) - removes the cross-fold inflation."""
    if ctx.thresholds_v11 is None:
        return "before_split" if holdout_share else "dev"
    return "before_split" if invalidates else "dev"
