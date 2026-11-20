"""A5 group leakage: does the score depend on seeing the same entity on both sides?

Candidate entity columns are found from structure alone (repeated values, moderate
cardinality, other columns constant within a value). v1.1 (with thresholds_v11): a candidate
must also be key-like (string/categorical, or numeric with a key-like name) and have at
least one entity-constant feature. For each, the plan is
cross-validated with the run's splitter and with GroupKFold on that column, on the same
rows. A large relative drop means the plan memorises entities: critical, and the sealed
holdout - drawn row by row - shares entities with development data, so it is invalid.

Entity scope: if the user declared ``--entities known`` (predictions are for entities
already in the data), a row-level split is legitimate; A5 then records info citing the
declaration and never invalidates the holdout. Unanswered is treated as ``new``.
"""

import pandas as pd
from sklearn.model_selection import GroupKFold

from holdout.audit.checks import CheckOutcome
from holdout.audit.checks.a03_id_like import key_like_name
from holdout.audit.comparison import compare, drop_severity, plan_cv
from holdout.audit.context import AuditContext
from holdout.audit.sampling import sample_rows, sampling_info
from holdout.audit.types import (
    AuditFlag,
    Category,
    ChangeSplit,
    CheckId,
    GroupEvidence,
    NoAction,
    Severity,
    SplitComparison,
)
from holdout.core import config, portable
from holdout.core.schemas import EntityScope, SplitStrategy
from holdout.data.split import SplitSpec
from holdout.data.structure import EntityCandidate, constant_feature_count, entity_candidates
from holdout.plan.compiler import compile_plan, cv_from_split


def candidates(ctx: AuditContext) -> list[EntityCandidate]:
    t = ctx.thresholds.a05_group
    # Datetime columns repeat (several rows per day) but are time keys, handled by A6.
    exclude = (
        [ctx.target]
        + ([ctx.spec.group_col] if ctx.spec.group_col else [])
        + ctx.profile.datetime_candidates
    )
    found = entity_candidates(
        ctx.dev,
        exclude=exclude,
        min_mean_rows_per_value=t.min_mean_rows_per_value,
        min_groups=t.min_groups,
        max_unique_ratio=t.max_unique_ratio,
        max_candidates=len(ctx.features),
    )
    v11 = ctx.thresholds_v11
    if v11 is not None:
        # v1.1 (post-benchmark, error analysis section 3): an entity key must look like a
        # key - a string or categorical column, or an integer with a key-like name - and
        # have entity-constant features. A repeated count (telco's number_vmail_messages)
        # or measurement is not an entity, whatever else happens to be constant with it.
        tokens = v11.a03_id_like.key_name_tokens
        minimum = v11.a05_group.min_entity_constant_features
        plausible = [
            c
            for c in found
            if _key_like(ctx.dev[c.column], c.column, tokens)
            and constant_feature_count(ctx.dev, c.column, exclude) >= minimum
        ]
    else:
        # A repeated integer measurement (age, a count) is not an entity key unless other
        # columns are constant within it; string keys qualify on repetition alone.
        plausible = [
            c
            for c in found
            if c.constant_feature_share > 0.0
            or not pd.api.types.is_numeric_dtype(ctx.dev[c.column])
        ]
    return plausible[: t.deep_max_candidates if ctx.deep_mode else t.max_candidates]


def _key_like(values: pd.Series, name: str, tokens: list[str]) -> bool:
    """A string or categorical column, or a numeric column with a key-like name."""
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return True
    return key_like_name(name, tokens)


def run(ctx: AuditContext) -> CheckOutcome:
    if ctx.spec.strategy is SplitStrategy.TIME:
        return CheckOutcome(skipped=True, detail="run uses a time split; group check not applied")
    found = candidates(ctx)
    if not found:
        return CheckOutcome(detail="no entity-like columns")
    if ctx.spec.entity_scope is EntityScope.KNOWN:
        return CheckOutcome(
            flags=[_known_flag(ctx, c) for c in found], detail="entities declared known"
        )

    t = ctx.thresholds.a05_group
    pipeline = compile_plan(ctx.plan, ctx.profile, run_seed=ctx.spec.seed)
    flags: list[AuditFlag] = []
    notes: list[str] = []
    for candidate in found:
        sample, row_method = sample_rows(ctx, f"a05/{candidate.column}", group_col=candidate.column)
        X, y, groups = portable.cv_inputs(sample, ctx.target, ctx.spec.to_config())
        reference_splitter = cv_from_split(ctx.spec)
        # Same splitter, different shuffle: how much the reference moves by chance alone.
        repeat_splitter = portable.make_cv(
            ctx.spec.strategy.value, config.N_FOLDS, ctx.seed("a05/reference_repeat")
        )
        alternative_splitter = GroupKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=ctx.seed("a05/group_kfold")
        )
        comparison = compare(
            ctx,
            type(reference_splitter).__name__,
            plan_cv(ctx, pipeline, X, y, groups, reference_splitter),
            plan_cv(ctx, pipeline, X, y, groups, repeat_splitter),
            f"GroupKFold({candidate.column})",
            plan_cv(ctx, pipeline, X, y, X[candidate.column], alternative_splitter),
            sampling_info(ctx, sample, row_method, len(ctx.features), "all"),
        )
        severity = drop_severity(
            comparison, t.warn_relative_drop, t.critical_relative_drop, t.min_drop_to_noise
        )
        drop = (
            "n/a"
            if comparison is None or comparison.relative_drop is None
            else f"{comparison.relative_drop:.0%}"
        )
        notes.append(f"{candidate.column}: relative drop {drop}")
        if severity is None or comparison is None:
            continue
        critical = severity is Severity.CRITICAL
        new_spec = SplitSpec(
            strategy=SplitStrategy.GROUP,
            group_col=candidate.column,
            test_size=ctx.spec.test_size,
            seed=ctx.spec.seed,
            entity_scope=EntityScope.NEW,
        )
        message = (
            f"Grouping by {candidate.column!r} ({candidate.n_groups} values) drops the CV score "
            f"from {comparison.reference_score:.3f} to {comparison.alternative_score:.3f} "
            f"({drop} of the lift): the plan memorises entities."
        )
        if critical:
            message += " The sealed holdout shares these entities with development data."
        flags.append(
            AuditFlag(
                check=CheckId.A5,
                category=Category.GROUP_LEAKAGE,
                severity=severity,
                columns=[candidate.column],
                evidence=_evidence(ctx, candidate, comparison),
                message=message,
                fix=ChangeSplit(
                    new_spec=new_spec,
                    instruction=(
                        f"start a new run with a group split by {candidate.column!r} and entity "
                        f"scope new (CLI: --split group:{candidate.column} --entities new), "
                        "or declare entity scope known if predictions are for entities "
                        "already in the data (CLI: --entities known)"
                    ),
                ),
                invalidates_holdout=critical,
            )
        )
    return CheckOutcome(flags=flags, detail="; ".join(notes))


def _evidence(
    ctx: AuditContext, candidate: EntityCandidate, comparison: SplitComparison | None
) -> GroupEvidence:
    return GroupEvidence(
        column=candidate.column,
        n_groups=candidate.n_groups,
        mean_rows_per_group=candidate.mean_rows_per_group,
        constant_feature_share=candidate.constant_feature_share,
        entity_scope=ctx.spec.entity_scope,
        comparison=comparison,
    )


def _known_flag(ctx: AuditContext, candidate: EntityCandidate) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A5,
        category=Category.GROUP_LEAKAGE,
        severity=Severity.INFO,
        columns=[candidate.column],
        evidence=_evidence(ctx, candidate, None),
        message=(
            f"{candidate.column!r} looks like an entity key; you declared predictions are for "
            "known entities (entity scope known; CLI: --entities known), so a row-level "
            "split is legitimate."
        ),
        fix=NoAction(reason="entities declared known by the user"),
    )
