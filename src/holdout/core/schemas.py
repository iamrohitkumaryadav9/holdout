"""Shared vocabulary: enums and result records used across layers.

Results live here (rather than next to the code that produces them) because the manifest,
the report and the CLI all consume them; keeping them in ``core`` avoids upward imports.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Task(StrEnum):
    """Supervised learning task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DataType(StrEnum):
    """What the user says the data is; used to sanity-check the split proposal."""

    TABULAR = "tabular"
    TIMESERIES = "timeseries"


class ColumnKind(StrEnum):
    """Semantic kind of a column, inferred without looking at the target."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID_LIKE = "id_like"


class SplitStrategy(StrEnum):
    """How the holdout is carved out (and, consistently, how CV folds are built)."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    TIME = "time"


class MetricName(StrEnum):
    """Evaluation metrics a plan may choose."""

    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    F1 = "f1"
    BALANCED_ACCURACY = "balanced_accuracy"
    ACCURACY = "accuracy"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"


class Direction(StrEnum):
    """Whether a larger metric value is better."""

    HIGHER = "higher"
    LOWER = "lower"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CVResult(_Record):
    """Outcome of cross-validating one plan on the development data."""

    plan_label: str
    metric: MetricName
    direction: Direction
    mean: float
    std: float
    fold_scores: list[float]
    fit_seconds: float = Field(ge=0.0)
    n_rows: int = Field(ge=1)
    n_folds: int = Field(ge=2)
    splitter: str


class HoldoutResult(_Record):
    """Outcome of the (single) evaluation of a fitted pipeline on the sealed holdout."""

    metric: MetricName
    score: float
    n_rows: int = Field(ge=1)
    holdout_sha256: str
    evaluation_index: int = Field(ge=1, description="1 for the first evaluation of the run")
    forced: bool
    caller: str
    evaluated_at: datetime
