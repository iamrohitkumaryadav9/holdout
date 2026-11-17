"""A9 encoding leakage: target statistics computed on the rows they encode.

A target encoder fitted naively encodes each training row with a statistic that includes
that row's own label, which leaks. The plan schema only offers ``target_cv``, compiled to
scikit-learn's ``TargetEncoder``, which cross-fits internally. So A9 normally reports info
("prevented by construction"). It inspects the *compiled* pipeline rather than trusting the
schema, and is critical if a future plan path introduces any target-aware encoder that is
not known to cross-fit.
"""

from collections.abc import Iterator
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from holdout.audit.checks import CheckOutcome
from holdout.audit.context import AuditContext
from holdout.audit.types import (
    AuditFlag,
    Category,
    CheckId,
    EncodingEvidence,
    NoAction,
    Severity,
)
from holdout.plan.compiler import compile_plan

#: Encoders that never look at the target.
TARGET_FREE_ENCODERS: tuple[type, ...] = (OneHotEncoder, OrdinalEncoder)


def walk(estimator: Any) -> Iterator[Any]:
    """Every estimator inside nested pipelines and column transformers."""
    yield estimator
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            yield from walk(step)
    elif isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers:
            yield from walk(transformer)


def encoders(pipeline: Any) -> list[tuple[str, bool]]:
    """(class name, cross-fitted) for every encoder that uses the target.

    Target-free encoders are omitted. ``TargetEncoder`` is cross-fitted when ``cv >= 2``
    (scikit-learn applies cross-fitting in ``fit_transform``, which pipelines call). Any
    other class whose name mentions "target" or "encoder" is assumed target-aware and not
    cross-fitted until it is classified here explicitly.
    """
    found: list[tuple[str, bool]] = []
    for step in walk(pipeline):
        name = type(step).__name__
        if isinstance(step, TARGET_FREE_ENCODERS):
            continue
        if isinstance(step, TargetEncoder):
            found.append((name, int(step.cv) >= 2))
        elif "target" in name.lower() or "encoder" in name.lower():
            found.append((name, False))
    return found


def run(ctx: AuditContext) -> CheckOutcome:
    found = encoders(compile_plan(ctx.plan, ctx.profile, run_seed=ctx.spec.seed))
    if not found:
        return CheckOutcome(detail="no target-aware encoders")
    unsafe = [name for name, cross_fitted in found if not cross_fitted]
    evidence = EncodingEvidence(
        target_encoders=[name for name, _ in found], cross_fitted=not unsafe
    )
    if unsafe:
        flag = AuditFlag(
            check=CheckId.A9,
            category=Category.ENCODING_LEAKAGE,
            severity=Severity.CRITICAL,
            columns=[],
            evidence=evidence,
            message=(
                f"The pipeline uses target-aware encoder(s) {unsafe} that are not known to "
                "cross-fit: training rows are encoded with their own labels."
            ),
            fix=NoAction(reason="use the cross-fitted 'target_cv' encoding"),
        )
    else:
        flag = AuditFlag(
            check=CheckId.A9,
            category=Category.ENCODING_LEAKAGE,
            severity=Severity.INFO,
            columns=[],
            evidence=evidence,
            message=(
                "Target encoding is cross-fitted (TargetEncoder): leakage through encoding is "
                "prevented by construction."
            ),
            fix=NoAction(reason="prevented by construction"),
        )
    return CheckOutcome(flags=[flag], detail=f"{len(found)} target-aware encoder(s)")
