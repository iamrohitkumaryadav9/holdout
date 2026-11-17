"""Typed vocabulary of the audit layer: checks, flags, evidence, fixes and the report.

Evidence and fixes are discriminated unions rather than free dictionaries, so every flag
states exactly what was measured and exactly what to change, and a consumer (the CLI, the
report, a future LLM) can act on it without parsing prose.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from holdout.core.schemas import EntityScope, MetricName
from holdout.data.split import SplitSpec


class CheckId(StrEnum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"
    A7 = "A7"
    A8 = "A8"
    A9 = "A9"
    A10 = "A10"
    A11 = "A11"
    A12 = "A12"


CHECK_NAMES: dict[CheckId, str] = {
    CheckId.A1: "single-feature leakage",
    CheckId.A2: "post-outcome column hints",
    CheckId.A3: "ID-like column with signal",
    CheckId.A4: "duplicate rows",
    CheckId.A5: "group leakage",
    CheckId.A6: "time leakage",
    CheckId.A7: "metric fit",
    CheckId.A8: "format issues",
    CheckId.A9: "encoding leakage",
    CheckId.A10: "too good to be true",
    CheckId.A11: "causal claims",
    CheckId.A12: "unstable validation",
}


class Category(StrEnum):
    TARGET_LEAKAGE = "target_leakage"
    POST_OUTCOME_COLUMN = "post_outcome_column"
    ID_SIGNAL = "id_signal"
    DUPLICATE_ROWS = "duplicate_rows"
    GROUP_LEAKAGE = "group_leakage"
    TIME_LEAKAGE = "time_leakage"
    METRIC_MISMATCH = "metric_mismatch"
    FORMAT_ISSUE = "format_issue"
    ENCODING_LEAKAGE = "encoding_leakage"
    SUSPICIOUS_SCORE = "suspicious_score"
    CAUSAL_CLAIM = "causal_claim"
    UNSTABLE_VALIDATION = "unstable_validation"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    def at_least(self, other: "Severity") -> bool:
        return self.rank >= other.rank


_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


RowMethod = Literal["none", "stratified", "grouped", "systematic", "random"]
FeatureMethod = Literal["all", "top_univariate"]


class SamplingInfo(_Frozen):
    """How much of the development data an expensive check actually used."""

    n_rows_total: int
    n_rows_used: int
    row_method: RowMethod
    n_features_total: int
    n_features_used: int
    feature_method: FeatureMethod


# ----------------------------------------------------------------------------- evidence


class SingleFeatureEvidence(_Frozen):
    kind: Literal["single_feature"] = "single_feature"
    metric: MetricName
    single_score: float
    dummy_score: float
    full_score: float = Field(description="the run's CV score of the full plan")
    relative_lift: float | None = Field(description="(single - dummy) / (full - dummy)")
    normalized_score: float = Field(description="share of the way from dummy to perfect")
    used_by_plan: bool
    corroborated_by: list["CheckId"]
    splitter: str
    sampling: SamplingInfo


class HintEvidence(_Frozen):
    kind: Literal["hint"] = "hint"
    provider: str
    reasons: list[str]
    matched_terms: list[str]
    single_feature_relative_lift: float | None
    used_by_plan: bool


class IdLikeEvidence(_Frozen):
    kind: Literal["id_like"] = "id_like"
    signals: list[str]
    unique_ratio: float
    row_order_spearman: float | None
    single_feature_relative_lift: float | None
    single_feature_normalized_score: float | None
    used_by_plan: bool


class DuplicateEvidence(_Frozen):
    kind: Literal["duplicates"] = "duplicates"
    normalisation: str
    n_dev_rows: int
    dev_redundant_rows: int
    cross_fold_duplicate_rows: int
    cross_fold_share: float
    holdout_rows: int | None
    holdout_rows_with_dev_duplicate: int | None
    holdout_overlap_share: float | None
    holdout_overlap_note: str


class SplitComparison(_Frozen):
    """Plan CV under the run's (or a control) splitter vs under a stricter splitter."""

    reference_splitter: str
    reference_score: float
    alternative_splitter: str
    alternative_score: float
    dummy_score: float
    absolute_drop: float = Field(description="how much better the reference looks")
    relative_drop: float | None = Field(description="absolute_drop / reference lift")
    reference_repeat_score: float | None = Field(
        description="the reference splitter re-run with another shuffle (noise estimate)"
    )
    relative_noise: float | None = Field(
        description="|reference - repeat| / reference lift: the drop expected from noise alone"
    )
    n_usable_folds: int
    sampling: SamplingInfo


class GroupEvidence(_Frozen):
    kind: Literal["group"] = "group"
    column: str
    n_groups: int
    mean_rows_per_group: float
    constant_feature_share: float
    entity_scope: EntityScope | None
    comparison: SplitComparison | None


class TimeEvidence(_Frozen):
    kind: Literal["time"] = "time"
    order_source: Literal["datetime_column", "monotonic_column", "row_order"]
    column: str | None
    autocorrelation: float | None
    shuffled_mean: float | None
    z_score: float | None
    sorted_by: str | None
    comparison: SplitComparison | None


class MetricEvidence(_Frozen):
    kind: Literal["metric"] = "metric"
    metric: MetricName
    plan_issue_code: str | None
    minority_class_fraction: float | None
    secondary_scores: dict[str, float] = Field(description="plan CV mean per extra metric")
    secondary_dummy_scores: dict[str, float]
    tail_ratio: float | None


FormatIssue = Literal["numeric_strings", "category_variants", "sentinel"]


class FormatEvidence(_Frozen):
    kind: Literal["format"] = "format"
    issue: FormatIssue
    n_affected_rows: int
    share: float
    examples: list[str]


class EncodingEvidence(_Frozen):
    kind: Literal["encoding"] = "encoding"
    target_encoders: list[str]
    cross_fitted: bool


class ScoreEvidence(_Frozen):
    kind: Literal["score"] = "score"
    metric: MetricName
    score: float
    dummy_score: float
    normalized_score: float
    deep_mode: bool


class CausalEvidence(_Frozen):
    kind: Literal["causal"] = "causal"
    source: Literal["text", "structured"]
    statements: list[str]
    phrases: list[str]


class StabilityEvidence(_Frozen):
    kind: Literal["stability"] = "stability"
    fold_scores: list[float]
    dummy_score: float
    dummy_fold_scores: list[float]
    fold_lifts: list[float] = Field(description="per-fold lift of the plan over the dummy")
    coefficient_of_variation: float | None = Field(description="std / |mean| of fold scores")
    lift_coefficient_of_variation: float | None = Field(
        description="std / |mean - dummy|: spread relative to the lift being claimed"
    )
    test_train_ratio: float = Field(description="mean n_test / n_train over the folds")
    lift_interval_low: float = Field(description="Nadeau-Bengio corrected interval, low end")
    lift_interval_high: float
    confidence: float


Evidence = Annotated[
    SingleFeatureEvidence
    | HintEvidence
    | IdLikeEvidence
    | DuplicateEvidence
    | GroupEvidence
    | TimeEvidence
    | MetricEvidence
    | FormatEvidence
    | EncodingEvidence
    | ScoreEvidence
    | CausalEvidence
    | StabilityEvidence,
    Field(discriminator="kind"),
]


# ----------------------------------------------------------------------------- fixes


class ParseNumeric(_Frozen):
    kind: Literal["parse_numeric"] = "parse_numeric"
    strip_tokens: list[str]
    percent: bool


class NormaliseCategories(_Frozen):
    kind: Literal["normalise_categories"] = "normalise_categories"
    mapping: dict[str, str]


class ReplaceSentinel(_Frozen):
    kind: Literal["replace_sentinel"] = "replace_sentinel"
    value: float
    replacement: Literal["missing"] = "missing"


FormatAction = Annotated[
    ParseNumeric | NormaliseCategories | ReplaceSentinel, Field(discriminator="kind")
]


class DropColumns(_Frozen):
    kind: Literal["drop_columns"] = "drop_columns"
    columns: list[str]
    reason: str


class ChangeSplit(_Frozen):
    """Start a new run with a stricter split.

    ``new_spec`` is None when no column can express the split (e.g. the time order lives
    only in the row order); ``instruction`` then says how to make it expressible.
    """

    kind: Literal["change_split"] = "change_split"
    new_spec: SplitSpec | None
    instruction: str


class Deduplicate(_Frozen):
    kind: Literal["deduplicate"] = "deduplicate"
    scope: Literal["dev", "before_split"]


class CleanFormats(_Frozen):
    kind: Literal["clean_formats"] = "clean_formats"
    column: str
    action: FormatAction


class ChangeMetric(_Frozen):
    kind: Literal["change_metric"] = "change_metric"
    metric: MetricName


class RemoveCausalClaim(_Frozen):
    kind: Literal["remove_causal_claim"] = "remove_causal_claim"
    statements: list[str]


class NoAction(_Frozen):
    kind: Literal["no_action"] = "no_action"
    reason: str


Fix = Annotated[
    DropColumns
    | ChangeSplit
    | Deduplicate
    | CleanFormats
    | ChangeMetric
    | RemoveCausalClaim
    | NoAction,
    Field(discriminator="kind"),
]


# ----------------------------------------------------------------------------- flags


class AuditFlag(_Frozen):
    check: CheckId
    category: Category
    severity: Severity
    columns: list[str]
    evidence: Evidence
    message: str = Field(min_length=1, description="plain language, one or two sentences")
    fix: Fix
    invalidates_holdout: bool = False


class CheckRun(_Frozen):
    """What happened when one check ran. A skip is recorded, not a failure."""

    check: CheckId
    status: Literal["ran", "skipped", "error"]
    seconds: float = Field(ge=0.0)
    sampling: SamplingInfo | None = None
    detail: str | None = None


class AuditReport(BaseModel):
    # extra="ignore": the computed fields below are serialised, and must be recomputed
    # (not rejected) when an audit.json or manifest is read back.
    model_config = ConfigDict(frozen=True, extra="ignore")

    flags: list[AuditFlag]
    checks_run: list[CheckRun]
    thresholds_hash: str
    deep_mode: bool
    entity_scope: EntityScope | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complete(self) -> bool:
        """False when any check errored; an incomplete audit blocks unsealing."""
        return all(run.status != "error" for run in self.checks_run)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def holdout_invalid(self) -> bool:
        return any(flag.invalidates_holdout for flag in self.flags)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts(self) -> dict[str, int]:
        return {s.value: sum(f.severity is s for f in self.flags) for s in Severity}

    def at_least(self, severity: Severity) -> list[AuditFlag]:
        return [f for f in self.flags if f.severity.at_least(severity)]
