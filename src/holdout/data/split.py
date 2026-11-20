"""Split specification, proposal and application.

The split decides what the holdout can prove. A random split of time-ordered or
entity-grouped data puts near-duplicates of training rows into the holdout, and the
holdout score then confirms the leak instead of exposing it. ``propose_split`` therefore
inspects the (target-blind) profile and argues for a strategy, with reasons and warnings.
The cross-validation splitter is later derived from the same spec, so folds and holdout
always share one notion of "unseen".
"""

import hashlib
import json
from typing import Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from holdout.core import config
from holdout.core.errors import SplitError
from holdout.core.portable import SplitConfig, split_frame, time_order_key
from holdout.core.schemas import DataType, EntityScope, SplitStrategy, Task
from holdout.core.seeds import derive_seed
from holdout.data.profile import DatasetProfile, StructuralEntity

#: Exact warning shown when time-ordered data is about to be split at random.
TIME_ORDER_WARNING: Final = (
    "This looks time-ordered; a random split may leak the future. Switch to a time-based split?"
)


class SplitSpec(BaseModel):
    """How to carve out the holdout. ``seed`` is the *run* seed; the split seed is derived."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: SplitStrategy
    test_size: float = Field(
        default=config.TEST_SIZE_DEFAULT, ge=config.TEST_SIZE_MIN, le=config.TEST_SIZE_MAX
    )
    group_col: str | None = None
    time_col: str | None = None
    seed: int
    entity_scope: EntityScope | None = Field(
        default=None,
        description=(
            "Whether predictions target new entities or entities already in the data; "
            "None means unanswered and is treated as 'new' (the rigorous default)."
        ),
    )

    @model_validator(mode="after")
    def _columns_match_strategy(self) -> "SplitSpec":
        needs_group = self.strategy is SplitStrategy.GROUP
        needs_time = self.strategy is SplitStrategy.TIME
        if needs_group != (self.group_col is not None):
            raise ValueError(
                "group_col is required for a group split and not allowed otherwise"
                f" (strategy={self.strategy.value}, group_col={self.group_col!r})"
            )
        if needs_time != (self.time_col is not None):
            raise ValueError(
                "time_col is required for a time split and not allowed otherwise"
                f" (strategy={self.strategy.value}, time_col={self.time_col!r})"
            )
        return self

    def to_config(self) -> SplitConfig:
        """Translate into the portable kernel's plain config, deriving the split seed."""
        return SplitConfig(
            strategy=self.strategy.value,
            test_size=self.test_size,
            seed=derive_seed(self.seed, "split"),
            group_col=self.group_col,
            time_col=self.time_col,
        )

    def fingerprint(self) -> str:
        """Stable hash of the spec, used to recognise the same holdout across runs.

        ``entity_scope`` is excluded: it is a declaration about deployment, not about which
        rows are held out, so the same rows must keep the same fingerprint.
        """
        payload = json.dumps(self.model_dump(mode="json", exclude={"entity_scope"}), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate_against(self, frame: pd.DataFrame, target: str) -> None:
        """Check that this spec can be applied to ``frame``.

        Raises:
            SplitError: naming the column and the exact problem.
        """
        if target not in frame.columns:
            raise SplitError(f"target column {target!r} not found")
        for role, column in (("group_col", self.group_col), ("time_col", self.time_col)):
            if column is None:
                continue
            if column not in frame.columns:
                raise SplitError(f"{role} {column!r} not found; columns are {list(frame.columns)}")
            if column == target:
                raise SplitError(f"{role} cannot be the target column {target!r}")
        if self.strategy is SplitStrategy.GROUP:
            self._validate_groups(frame[self.group_col])
        elif self.strategy is SplitStrategy.TIME:
            self._validate_time(frame[self.time_col])
        elif self.strategy is SplitStrategy.STRATIFIED:
            smallest = frame[target].value_counts()
            if smallest.min() < 2:
                rare = smallest[smallest < 2].index.tolist()
                raise SplitError(
                    f"stratified split needs at least 2 rows per class; classes {rare} have 1. "
                    "Merge rare classes or use a random split."
                )

    def _validate_groups(self, groups: pd.Series) -> None:
        n_missing = int(groups.isna().sum())
        if n_missing:
            raise SplitError(
                f"group column {groups.name!r} has {n_missing} missing value(s); every row "
                "needs a group, or its entity could appear on both sides of the split"
            )
        n_groups = int(groups.nunique())
        if n_groups < config.N_FOLDS + 1:
            raise SplitError(
                f"group column {groups.name!r} has {n_groups} groups; need at least "
                f"{config.N_FOLDS + 1} (one holdout group plus {config.N_FOLDS} CV folds)"
            )

    def _validate_time(self, values: pd.Series) -> None:
        try:
            key = time_order_key(values)
        except ValueError as exc:
            raise SplitError(str(exc)) from exc
        n_bad = int(key.isna().sum())
        if n_bad:
            raise SplitError(
                f"time column {values.name!r}: {n_bad} of {len(values)} values are missing or "
                "do not parse as datetimes/numbers; a time split needs every row ordered"
            )


class SplitProposal(BaseModel):
    """A proposed split with the reasoning a user (or LLM) can inspect and override."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: SplitSpec
    reasons: list[str]
    warnings: list[str]
    alternatives: list[SplitSpec]


def entity_question(entity: StructuralEntity) -> str:
    """The pre-seal question asked when a column looks like an entity key."""
    return (
        f"{entity.column!r} looks like an entity identifier ({entity.n_groups} values across "
        "all rows, "
        f"~{entity.mean_rows_per_group:.1f} rows each, "
        f"{entity.constant_feature_share:.0%} of other columns constant within it). "
        "Will predictions be made for new entities or for entities already in the data "
        "(entity scope new or known; CLI: --entities)? Unanswered is treated as new."
    )


def propose_split(
    profile: DatasetProfile,
    task: Task,
    user_data_type: DataType,
    *,
    seed: int,
    entity_scope: EntityScope | None = None,
) -> SplitProposal:
    """Propose a split strategy from the target-blind profile.

    Rules, in priority order:
      1. The user says ``timeseries`` and a time column exists: time split.
      2. An entity column exists (by name, or by structure when the user declared
         ``--entities new``) and predictions are not for known entities: group split.
      3. Classification: stratified split (preserves class balance in the holdout).
      4. Otherwise: random split.
    If the data looks time-ordered but the proposal is not a time split, the exact
    :data:`TIME_ORDER_WARNING` is raised and a time-based alternative is offered. A
    structural entity column without a declaration raises :func:`entity_question`.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    alternatives: list[SplitSpec] = []
    time_col = _best_time_column(profile)
    known = entity_scope is EntityScope.KNOWN
    structural = [e.column for e in profile.structural_entities]
    group_candidates = profile.entity_candidates + (
        structural if entity_scope is EntityScope.NEW else []
    )

    def make(
        strategy: SplitStrategy, group_col: str | None = None, time_col: str | None = None
    ) -> SplitSpec:
        return SplitSpec(
            strategy=strategy,
            seed=seed,
            entity_scope=entity_scope,
            group_col=group_col,
            time_col=time_col,
        )

    if user_data_type is DataType.TIMESERIES and time_col is not None:
        spec = make(SplitStrategy.TIME, time_col=time_col)
        reasons.append(
            f"data declared as time series; the latest rows by {time_col!r} are held out"
        )
    else:
        if user_data_type is DataType.TIMESERIES:
            warnings.append(
                "data declared as time series, but no parseable datetime or monotonic index "
                "column was found; choose the time column with a time split (CLI: --split "
                "time:COLUMN)"
            )
        if group_candidates and not known:
            group_col = group_candidates[0]
            spec = make(SplitStrategy.GROUP, group_col=group_col)
            entity = profile.column(group_col)
            reasons.append(
                f"{group_col!r} looks like an entity id ({entity.n_unique} distinct values "
                f"across all {profile.n_rows} rows); keeping each entity on one side "
                "stops the model from memorising entities"
            )
            if task is Task.CLASSIFICATION:
                reasons.append("group split takes precedence over stratification")
        elif task is Task.CLASSIFICATION:
            spec = make(SplitStrategy.STRATIFIED)
            reasons.append("classification target: stratify so the holdout keeps class balance")
        else:
            spec = make(SplitStrategy.RANDOM)
            reasons.append("regression target with no entity or time structure: random split")
        if known and (profile.entity_candidates or structural):
            reasons.append(
                "entities declared known (entity scope known): predictions are for entities "
                "already in the data, so a row-level split is legitimate"
            )

    if entity_scope is None:
        for structural_entity in profile.structural_entities:
            warnings.append(entity_question(structural_entity))
            if spec.strategy is not SplitStrategy.GROUP:
                alternatives.append(make(SplitStrategy.GROUP, group_col=structural_entity.column))

    if profile.looks_time_ordered and spec.strategy is not SplitStrategy.TIME:
        if user_data_type is DataType.TABULAR:
            warnings.append(TIME_ORDER_WARNING)
        if time_col is not None:
            alternatives.append(make(SplitStrategy.TIME, time_col=time_col))
        reasons.append("time-order signals: " + "; ".join(profile.time_order_signals))

    counts = profile.target_class_counts
    if task is Task.CLASSIFICATION and counts and min(counts.values()) < config.MIN_ROWS_PER_CLASS:
        rare = sorted(k for k, v in counts.items() if v < config.MIN_ROWS_PER_CLASS)
        warnings.append(
            f"classes {rare} have fewer than {config.MIN_ROWS_PER_CLASS} rows; "
            "holdout and fold scores for them will be unstable"
        )
    return SplitProposal(spec=spec, reasons=reasons, warnings=warnings, alternatives=alternatives)


def _best_time_column(profile: DatasetProfile) -> str | None:
    if profile.datetime_candidates:
        return profile.datetime_candidates[0]
    if profile.monotonic_candidates:
        return profile.monotonic_candidates[0]
    return None


def apply_split(
    frame: pd.DataFrame, spec: SplitSpec, target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into (development, holdout) according to ``spec``.

    Guarantees (property-tested): the parts partition the rows; a group never appears on
    both sides; for time splits ``max(dev time) <= min(holdout time)``; the same spec and
    data always give the same split.

    Raises:
        SplitError: if the spec does not fit the data, or a side would be too small.
    """
    spec.validate_against(frame, target)
    dev, holdout = split_frame(frame, target, spec.to_config())
    if holdout.empty:
        raise SplitError("the holdout would be empty; the dataset is too small")
    if len(dev) < 2 * config.N_FOLDS:
        raise SplitError(
            f"only {len(dev)} development rows; need at least {2 * config.N_FOLDS} "
            f"for {config.N_FOLDS}-fold cross-validation"
        )
    if spec.group_col is not None and dev[spec.group_col].nunique() < config.N_FOLDS:
        raise SplitError(
            f"only {dev[spec.group_col].nunique()} groups left for development; "
            f"need at least {config.N_FOLDS} for group {config.N_FOLDS}-fold cross-validation"
        )
    return dev, holdout
