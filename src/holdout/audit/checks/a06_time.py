"""A6 time leakage: does the score depend on training on the future?

Candidate orders, in priority: datetime-parseable columns (whatever their name),
monotonic integer columns, and the row order itself - when the target is autocorrelated
along the rows well beyond a shuffled baseline (data sorted by time with the date column
removed).

Before blaming row order on time, A6 tests the sort confound: if the row order is
explained by sorting on an existing feature (near-monotonic rank correlation with the row
position, or a categorical in contiguous blocks), the file may simply be sorted by that
feature. That yields a warning, never a time-leak critical, and never invalidates.

Comparison: the plan is cross-validated with forward-chaining folds along the inferred
order, against a control with the same folds on a shuffled order (same training sizes, so
the learning-curve effect of small early folds cancels). A large relative drop is
critical, and the sealed holdout, drawn without regard to time, is invalid.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from holdout.audit.checks import CheckOutcome
from holdout.audit.comparison import compare, drop_severity, plan_cv
from holdout.audit.context import AuditContext
from holdout.audit.sampling import row_cap, sampling_info
from holdout.audit.types import (
    AuditFlag,
    Category,
    ChangeSplit,
    CheckId,
    NoAction,
    Severity,
    SplitComparison,
    TimeEvidence,
)
from holdout.core import config, portable
from holdout.core.schemas import SplitStrategy, Task
from holdout.data.split import SplitSpec
from holdout.plan.compiler import compile_plan

OrderSource = Literal["datetime_column", "monotonic_column", "row_order"]


@dataclass(frozen=True)
class Autocorrelation:
    observed: float
    shuffled_mean: float
    z_score: float

    @property
    def effect(self) -> float:
        return self.observed - self.shuffled_mean


def run(ctx: AuditContext) -> CheckOutcome:
    if ctx.spec.strategy is SplitStrategy.TIME:
        return CheckOutcome(skipped=True, detail="run already uses a time split")
    t = ctx.thresholds.a06_time
    limit = t.deep_max_candidates if ctx.deep_mode else t.max_candidates
    present = set(ctx.features)
    orders: list[tuple[OrderSource, str | None]] = [
        ("datetime_column", c) for c in ctx.profile.datetime_candidates if c in present
    ]
    # Only integer sequences count as time indices; a continuous feature is monotonic only
    # because the file is sorted by it, which is the sort confound handled below.
    orders += [
        ("monotonic_column", c)
        for c in ctx.profile.monotonic_candidates
        if c in present and _integer_valued(ctx.dev[c])
    ]
    time_columns = {c for _, c in orders if c is not None}
    if not any(_follows_file_order(ctx, c) for c in time_columns):
        orders.append(("row_order", None))  # else the row order *is* that column's order

    pipeline = compile_plan(ctx.plan, ctx.profile, run_seed=ctx.spec.seed)
    flags: list[AuditFlag] = []
    notes: list[str] = []
    for source, column in orders[:limit]:
        ordered = _ordered(ctx, column)
        autocorrelation = target_autocorrelation(ctx, ordered[ctx.target])
        sorted_by = None
        if source == "row_order":
            significant = (
                autocorrelation.z_score >= t.autocorrelation_min_z
                and autocorrelation.effect >= t.autocorrelation_min_effect
            )
            if not significant:
                notes.append("row order: no target autocorrelation")
                continue
            sorted_by = sorted_by_feature(ctx, ordered, exclude=time_columns)
            if sorted_by is not None:
                flags.append(_sorted_flag(ctx, autocorrelation, sorted_by))
                notes.append(f"row order explained by sorting on {sorted_by!r}")
                continue
        comparison = _compare(ctx, pipeline, ordered, column)
        severity = drop_severity(
            comparison, t.warn_relative_drop, t.critical_relative_drop, t.min_drop_to_noise
        )
        label = column if column is not None else "row order"
        drop = (
            "n/a"
            if comparison is None or comparison.relative_drop is None
            else f"{comparison.relative_drop:.0%}"
        )
        notes.append(f"{label}: relative drop {drop}")
        if severity is None or comparison is None:
            continue
        flags.append(_time_flag(ctx, source, column, autocorrelation, comparison, severity))
    return CheckOutcome(flags=flags, detail="; ".join(notes))


def _integer_valued(values: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return False
    finite = values.dropna().to_numpy(dtype="float64")
    return bool(np.all(np.mod(finite, 1.0) == 0.0))


def _follows_file_order(ctx: AuditContext, column: str) -> bool:
    """Whether sorting by ``column`` reproduces the file's row order (|Spearman| ~ 1)."""
    in_file_order = ctx.dev.sort_index()
    key = portable.time_order_key(in_file_order[column]).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(key):
        key = key.astype("int64").astype("float64").where(key.notna())
    position = pd.Series(np.arange(len(key)), dtype="float64")
    mask = key.notna()
    if mask.sum() < 3:
        return False
    rho = key[mask].astype("float64").corr(position[mask], method="spearman")
    return bool(abs(rho) >= 0.99)


def _ordered(ctx: AuditContext, column: str | None) -> pd.DataFrame:
    """Development rows in the candidate order, systematically sampled to the row cap."""
    if column is None:
        ordered = ctx.dev.sort_index()
    else:
        key = portable.time_order_key(ctx.dev[column])
        ordered = ctx.dev.iloc[
            key.reset_index(drop=True).sort_values(kind="stable").index.to_numpy()
        ]
    limit = row_cap(ctx)
    if len(ordered) > limit:
        ordered = ordered.iloc[np.unique(np.linspace(0, len(ordered) - 1, limit).astype(int))]
    return ordered


def target_autocorrelation(ctx: AuditContext, y: pd.Series) -> Autocorrelation:
    """Lag-1 statistic along the given order vs its distribution under shuffling."""
    values = _codes(y, ctx.task)
    rng = np.random.default_rng(ctx.seed("a06/permutations"))
    observed = _lag_statistic(values, ctx.task)
    shuffled = np.array(
        [
            _lag_statistic(rng.permutation(values), ctx.task)
            for _ in range(ctx.thresholds.a06_time.n_permutations)
        ]
    )
    spread = float(shuffled.std())
    z = (observed - float(shuffled.mean())) / spread if spread > 0 else 0.0
    return Autocorrelation(observed=observed, shuffled_mean=float(shuffled.mean()), z_score=z)


def _codes(y: pd.Series, task: Task) -> np.ndarray:
    if task is Task.CLASSIFICATION:
        return pd.factorize(y.astype(str))[0].astype("float64")
    return y.to_numpy(dtype="float64")


def _lag_statistic(values: np.ndarray, task: Task) -> float:
    if len(values) < 3:
        return 0.0
    if task is Task.CLASSIFICATION:
        return float(np.mean(values[:-1] == values[1:]))  # agreement of consecutive labels
    a, b = values[:-1], values[1:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sorted_by_feature(ctx: AuditContext, ordered: pd.DataFrame, exclude: set[str]) -> str | None:
    """A non-time feature that explains the row order, if any."""
    t = ctx.thresholds.a06_time
    position = pd.Series(np.arange(len(ordered)), dtype="float64")
    for column in ctx.features:
        if column in exclude:
            continue
        values = ordered[column].reset_index(drop=True)
        n_unique = values.nunique()
        if n_unique < 2:
            continue
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            mask = values.notna()
            rho = values[mask].astype("float64").corr(position[mask], method="spearman")
            if abs(rho) >= t.sorted_min_abs_spearman:
                return column
        if n_unique <= len(values) / 2:
            text = values.astype(str)
            runs = int((text != text.shift()).sum())
            if runs <= t.sorted_max_runs_per_value * n_unique:
                return column
    return None


def _compare(
    ctx: AuditContext, pipeline: object, ordered: pd.DataFrame, column: str | None
) -> SplitComparison | None:
    X = ordered.drop(columns=[ctx.target]).reset_index(drop=True)
    y = ordered[ctx.target].reset_index(drop=True)
    rng = np.random.default_rng(ctx.seed("a06/control"))
    shuffle, repeat = rng.permutation(len(X)), rng.permutation(len(X))
    folds = TimeSeriesSplit(n_splits=config.N_FOLDS)
    info = sampling_info(
        ctx,
        ordered,
        "none" if len(ordered) == len(ctx.dev) else "systematic",
        len(ctx.features),
        "all",
    )
    label = column if column is not None else "row order"
    return compare(
        ctx,
        "TimeSeriesSplit(shuffled order)",
        plan_cv(ctx, pipeline, X.iloc[shuffle], y.iloc[shuffle], None, folds),
        plan_cv(ctx, pipeline, X.iloc[repeat], y.iloc[repeat], None, folds),
        f"TimeSeriesSplit({label})",
        plan_cv(ctx, pipeline, X, y, None, folds),
        info,
    )


def _time_flag(
    ctx: AuditContext,
    source: OrderSource,
    column: str | None,
    autocorrelation: Autocorrelation,
    comparison: SplitComparison,
    severity: Severity,
) -> AuditFlag:
    critical = severity is Severity.CRITICAL
    label = repr(column) if column is not None else "the row order"
    message = (
        f"Training only on the past along {label} drops the CV score from "
        f"{comparison.reference_score:.3f} to {comparison.alternative_score:.3f}: the plan "
        "benefits from seeing the future."
    )
    if critical:
        message += " The sealed holdout was drawn without regard to time and is invalid."
    if column is not None:
        fix = ChangeSplit(
            new_spec=SplitSpec(
                strategy=SplitStrategy.TIME,
                time_col=column,
                test_size=ctx.spec.test_size,
                seed=ctx.spec.seed,
                entity_scope=ctx.spec.entity_scope,
            ),
            instruction=f"start a new run with --split time:{column}",
        )
    else:
        fix = ChangeSplit(
            new_spec=None,
            instruction=(
                "the time order exists only as row order: add a row-number column "
                "(e.g. row_number = 1..n in file order) and start a new run with "
                "--split time:row_number"
            ),
        )
    return AuditFlag(
        check=CheckId.A6,
        category=Category.TIME_LEAKAGE,
        severity=severity,
        columns=[] if column is None else [column],
        evidence=TimeEvidence(
            order_source=source,
            column=column,
            autocorrelation=autocorrelation.observed,
            shuffled_mean=autocorrelation.shuffled_mean,
            z_score=autocorrelation.z_score,
            sorted_by=None,
            comparison=comparison,
        ),
        message=message,
        fix=fix,
        invalidates_holdout=critical,
    )


def _sorted_flag(ctx: AuditContext, autocorrelation: Autocorrelation, sorted_by: str) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A6,
        category=Category.TIME_LEAKAGE,
        severity=Severity.WARNING,
        columns=[sorted_by],
        evidence=TimeEvidence(
            order_source="row_order",
            column=None,
            autocorrelation=autocorrelation.observed,
            shuffled_mean=autocorrelation.shuffled_mean,
            z_score=autocorrelation.z_score,
            sorted_by=sorted_by,
            comparison=None,
        ),
        message=(
            f"The target follows the row order, but the file appears sorted by {sorted_by!r}; "
            "row order may not be time. If rows are in time order, add a time column."
        ),
        fix=NoAction(reason=f"verify whether rows sorted by {sorted_by!r} are also in time order"),
    )
