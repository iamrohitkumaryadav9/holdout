"""Bounded work for expensive checks: sample rows the way the run split, cap features.

Rows are sampled so the sample keeps the structure the run's splitter relies on: whole
groups for group splits (and for group-leakage candidates), every k-th row in time order
for time splits, class-stratified for classification. Features beyond the cap are ranked
by a cheap univariate association with the target *on development rows*, so a leaky
column is not dropped from the audit by an arbitrary cut.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from holdout.audit.context import AuditContext
from holdout.audit.types import FeatureMethod, RowMethod, SamplingInfo
from holdout.core import portable
from holdout.core.schemas import SplitStrategy, Task


def row_cap(ctx: AuditContext) -> int:
    t = ctx.thresholds
    return t.a10_too_good.deep_max_rows if ctx.deep_mode else t.sampling.max_rows


def feature_cap(ctx: AuditContext) -> int:
    t = ctx.thresholds
    return t.a10_too_good.deep_max_features if ctx.deep_mode else t.sampling.max_features


def sample_rows(
    ctx: AuditContext, purpose: str, *, group_col: str | None = None, max_rows: int | None = None
) -> tuple[pd.DataFrame, RowMethod]:
    """Return (sample of development rows, method). Order: time order for time splits,
    original row order otherwise."""
    dev = ctx.dev
    limit = row_cap(ctx) if max_rows is None else max_rows
    if len(dev) <= limit:
        return dev, "none"
    seed = ctx.seed(f"sample/{purpose}")
    positions = np.arange(len(dev))
    group = group_col if group_col is not None else ctx.spec.group_col
    if group is not None:
        splitter = GroupShuffleSplit(n_splits=1, train_size=limit / len(dev), random_state=seed)
        chosen, _ = next(splitter.split(positions, groups=dev[group].to_numpy()))
        return dev.iloc[np.sort(chosen)], "grouped"
    if ctx.spec.strategy is SplitStrategy.TIME:
        key = portable.time_order_key(dev[str(ctx.spec.time_col)])
        order = key.reset_index(drop=True).sort_values(kind="stable").index.to_numpy()
        picks = np.unique(np.linspace(0, len(dev) - 1, limit).astype(int))
        return dev.iloc[order[picks]], "systematic"
    if ctx.task is Task.CLASSIFICATION:
        try:
            chosen, _ = train_test_split(
                positions, train_size=limit, random_state=seed, stratify=dev[ctx.target].to_numpy()
            )
            return dev.iloc[np.sort(chosen)], "stratified"
        except ValueError:
            pass  # a class too rare to stratify: fall back to a random sample
    rng = np.random.default_rng(seed)
    return dev.iloc[np.sort(rng.choice(positions, size=limit, replace=False))], "random"


def select_features(
    sample: pd.DataFrame, target: str, task: Task, columns: list[str], max_features: int
) -> tuple[list[str], FeatureMethod]:
    """All ``columns`` when within the cap, else the top ``max_features`` by association."""
    if len(columns) <= max_features:
        return columns, "all"
    strength = {c: association(sample[c], sample[target], task) for c in columns}
    ranked = sorted(columns, key=lambda c: (-strength[c], c))
    keep = set(ranked[:max_features])
    return [c for c in columns if c in keep], "top_univariate"


def association(feature: pd.Series, target: pd.Series, task: Task) -> float:
    """Correlation ratio (eta squared) of the target across feature bins or categories."""
    if pd.api.types.is_numeric_dtype(feature) and not pd.api.types.is_bool_dtype(feature):
        values = feature.astype("float64")
        if values.nunique() > 20:
            groups = pd.qcut(values.rank(method="first"), 10, labels=False)
        else:
            groups = values
    else:
        groups = feature.astype(str)
    groups = groups.fillna(-1) if pd.api.types.is_numeric_dtype(groups) else groups
    if task is Task.CLASSIFICATION:
        indicators = pd.get_dummies(target.astype(str), dtype="float64")
    else:
        indicators = target.astype("float64").to_frame()
    total = float(indicators.var(ddof=0).sum())
    if total <= 0.0:
        return 0.0
    group_means = indicators.groupby(groups.to_numpy()).transform("mean")
    return float(group_means.var(ddof=0).sum() / total)


def sampling_info(
    ctx: AuditContext,
    sample: pd.DataFrame,
    row_method: RowMethod,
    n_features_used: int,
    feature_method: FeatureMethod,
) -> SamplingInfo:
    return SamplingInfo(
        n_rows_total=len(ctx.dev),
        n_rows_used=len(sample),
        row_method=row_method,
        n_features_total=len(ctx.features),
        n_features_used=n_features_used,
        feature_method=feature_method,
    )
