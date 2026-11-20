"""``ModelPlan``: the contract a proposer (later, an LLM) must fill, and its validation.

Why a typed plan instead of letting the proposer write modelling code: free-form code can
fit a scaler on all rows, peek at the holdout, or tune on the test fold, and none of that is
visible until it has already happened. A plan can only *choose* among options whose
implementations are known to be leakage-safe; every hyperparameter is bounded; unknown
fields are rejected. Validation happens in two stages:

1. **Shape** (this module's pydantic models): types, enums, bounds, required reasons. Needs
   no data, so a malformed plan fails at parse time.
2. **Context** (:func:`validate_plan`): checks that need the data profile, e.g. accuracy on
   an imbalanced target. Returns typed :class:`PlanIssue` objects with human-readable
   messages, so a proposer can be told exactly what to fix.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from holdout.core import config
from holdout.core.schemas import ColumnKind, MetricName, Task
from holdout.data.profile import ColumnProfile, DatasetProfile
from holdout.plan.metrics import get_metric

#: One short line: rationale and reasons must stay reviewable at a glance.
OneLine = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=200, pattern=r"^[^\r\n]+$"),
]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DropColumn(_Strict):
    """A column excluded from modelling, with a mandatory reason."""

    column: str = Field(min_length=1)
    reason: OneLine


class NumericPreprocessing(_Strict):
    impute: Literal["median", "mean"] = "median"
    scale: Literal["none", "standard"] = "none"


class CategoricalPreprocessing(_Strict):
    impute: Literal["most_frequent", "constant"] = "most_frequent"
    encode: Literal["onehot", "ordinal", "target_cv"] = "onehot"


class NoSelection(_Strict):
    method: Literal["none"] = "none"


class KBestSelection(_Strict):
    method: Literal["k_best"] = "k_best"
    k: int = Field(ge=1, le=1000, description="number of encoded features to keep")


class ModelBasedSelection(_Strict):
    """Keep features whose ExtraTrees importance is at least the median importance."""

    method: Literal["model_based"] = "model_based"


FeatureSelection = Annotated[
    NoSelection | KBestSelection | ModelBasedSelection, Field(discriminator="method")
]


class DummyModel(_Strict):
    """Prior (classification) or mean (regression) predictor: the baseline every plan must beat."""

    name: Literal["dummy"] = Field(
        default="dummy", description="exactly 'dummy': prior / mean baseline"
    )


class LogRegModel(_Strict):
    name: Literal["logreg"] = Field(
        default="logreg", description="exactly 'logreg': logistic regression, classification only"
    )
    C: float = Field(default=1.0, ge=1e-4, le=1e4, description="inverse L2 strength")
    max_iter: int = Field(default=1000, ge=100, le=10_000)


class RidgeModel(_Strict):
    name: Literal["ridge"] = Field(
        default="ridge", description="exactly 'ridge': ridge regression, regression only"
    )
    alpha: float = Field(default=1.0, ge=1e-4, le=1e4, description="L2 strength")


class RandomForestModel(_Strict):
    name: Literal["random_forest"] = Field(
        default="random_forest", description="exactly 'random_forest': random forest"
    )
    n_estimators: int = Field(default=300, ge=10, le=1000)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    min_samples_leaf: int = Field(default=1, ge=1, le=200)
    max_features: Literal["sqrt", "log2", "all"] = "sqrt"


class HistGBModel(_Strict):
    name: Literal["hist_gb"] = Field(
        default="hist_gb",
        description="exactly 'hist_gb': histogram gradient boosting "
        "(scikit-learn HistGradientBoosting*)",
    )
    learning_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    max_iter: int = Field(default=200, ge=10, le=2000)
    max_leaf_nodes: int = Field(default=31, ge=2, le=256)
    min_samples_leaf: int = Field(default=20, ge=1, le=500)
    l2_regularization: float = Field(default=0.0, ge=0.0, le=100.0)


class LightGBMModel(_Strict):
    name: Literal["lightgbm"] = Field(
        default="lightgbm", description="exactly 'lightgbm': LightGBM gradient boosting"
    )
    n_estimators: int = Field(default=300, ge=10, le=5000)
    learning_rate: float = Field(default=0.05, ge=0.005, le=0.5)
    num_leaves: int = Field(default=31, ge=2, le=1024)
    min_child_samples: int = Field(default=20, ge=1, le=1000)
    subsample: float = Field(default=1.0, ge=0.1, le=1.0)
    colsample_bytree: float = Field(default=1.0, ge=0.1, le=1.0)
    reg_lambda: float = Field(default=0.0, ge=0.0, le=100.0)


#: The model tags, exactly as a plan must spell them.
MODEL_NAMES = ("dummy", "logreg", "ridge", "random_forest", "hist_gb", "lightgbm")

ModelSpec = Annotated[
    DummyModel | LogRegModel | RidgeModel | RandomForestModel | HistGBModel | LightGBMModel,
    Field(
        discriminator="name",
        description="one model object; its name must be exactly one of: "
        + ", ".join(MODEL_NAMES)
        + " (no other spelling, e.g. not a scikit-learn class name)",
    ),
]

#: Models that only make sense for one task; enforced in validate_plan.
_MODEL_TASKS: dict[str, Task] = {"logreg": Task.CLASSIFICATION, "ridge": Task.REGRESSION}


class Rationale(_Strict):
    """One short line per choice, so every decision in a plan is explained and reviewable."""

    metric: OneLine
    drop_columns: OneLine
    preprocessing: OneLine
    feature_selection: OneLine
    model: OneLine
    class_weight: OneLine


#: How a plan addresses one data finding (v1.1). Code checks the action against the plan;
#: whether the problem is actually resolved is decided by code, not by this claim.
FindingAction = Literal[
    "drop_columns",
    "change_split",
    "change_metric",
    "deduplicate",
    "decline_causal_claim",
    "keep_with_reason",
    # The offline planner only: it does not act on findings; code applies the audit's fixes.
    "deferred_to_audit",
]


class FindingHandling(_Strict):
    """The planner's explicit answer to one data finding (v1.1, post-benchmark)."""

    finding_id: str = Field(min_length=1, max_length=12, description="e.g. F1, as listed")
    action: FindingAction
    reason: OneLine


class ModelPlan(_Strict):
    """A complete, bounded modelling plan."""

    task: Task
    metric: MetricName
    drop_columns: list[DropColumn] = Field(default_factory=list)
    numeric: NumericPreprocessing = NumericPreprocessing()
    categorical: CategoricalPreprocessing = CategoricalPreprocessing()
    feature_selection: FeatureSelection = NoSelection()
    model: ModelSpec
    class_weight: Literal["none", "balanced"] = "none"
    rationale: Rationale
    #: v1.1: remove duplicate development rows (equal normalised features) before
    #: cross-validation and the final fit. Code applies it; the holdout is never touched.
    deduplicate: bool = Field(
        default=False,
        description="true removes duplicate development rows before cross-validation",
    )
    #: v1.1: one entry per data finding (raw-data audit before planning). Empty for runs
    #: without findings and for every plan written by v1.0, which still load.
    findings_handled: list[FindingHandling] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _unique_drops(self) -> "ModelPlan":
        names = [d.column for d in self.drop_columns]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"drop_columns lists {duplicates} more than once")
        return self

    @property
    def dropped(self) -> set[str]:
        return {d.column for d in self.drop_columns}


def drop_rationale(drops: list[DropColumn]) -> str:
    """The rationale line for ``drop_columns`` written from the final drop list, so a plan
    whose drops code changed never says "no drops needed" above its dropped columns."""
    if not drops:
        return "no drops needed"
    detailed = "drops " + "; ".join(f"{d.column!r} ({d.reason})" for d in drops)
    if len(detailed) <= 200:
        return detailed
    names = "drops " + ", ".join(repr(d.column) for d in drops) + " (reasons per column)"
    if len(names) <= 200:
        return names
    return f"drops {len(drops)} columns (listed with their reasons in drop_columns)"


def with_drops(plan: ModelPlan, drops: list[DropColumn]) -> ModelPlan:
    """``plan`` with ``drops`` as its drop list and the drop rationale rewritten to match."""
    rationale = {**plan.rationale.model_dump(), "drop_columns": drop_rationale(drops)}
    return ModelPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "drop_columns": [d.model_dump() for d in drops],
            "rationale": rationale,
        }
    )


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class PlanIssue(_Strict):
    """One finding about a plan; errors block compilation, warnings are reported."""

    severity: Severity
    code: str
    field: str
    message: str


_UNMODELLABLE = {ColumnKind.DATETIME, ColumnKind.TEXT, ColumnKind.ID_LIKE}


def validate_plan(plan: ModelPlan, profile: DatasetProfile) -> list[PlanIssue]:
    """Check a shape-valid plan against the data profile.

    Returns every issue found (not just the first), errors before warnings, so a proposer
    can fix them all in one revision.
    """
    issues: list[PlanIssue] = []

    def error(code: str, field: str, message: str) -> None:
        issues.append(PlanIssue(severity=Severity.ERROR, code=code, field=field, message=message))

    def warn(code: str, field: str, message: str) -> None:
        issues.append(PlanIssue(severity=Severity.WARNING, code=code, field=field, message=message))

    metric = get_metric(plan.metric)
    if metric.task is not plan.task:
        error(
            "metric_task_mismatch",
            "metric",
            f"metric {plan.metric.value!r} is for {metric.task.value}, but the plan's task is "
            f"{plan.task.value}",
        )
    if plan.task is not profile.task_hint:
        warn(
            "task_differs_from_hint",
            "task",
            f"plan task is {plan.task.value}, but the target {profile.target!r} looks like "
            f"{profile.task_hint.value}",
        )
    n_classes = profile.n_classes
    if plan.task is Task.CLASSIFICATION and n_classes is not None:
        if metric.binary_only and n_classes > 2:
            error(
                "metric_needs_binary",
                "metric",
                f"{plan.metric.value!r} is defined for binary targets, but {profile.target!r} "
                f"has {n_classes} classes; use roc_auc, f1 or balanced_accuracy",
            )
        minority = profile.minority_class_fraction
        if (
            plan.metric is MetricName.ACCURACY
            and minority is not None
            and minority < config.ACCURACY_MIN_MINORITY_FRACTION
        ):
            # A warning here, not an error: the run proceeds so the audit (A7) can show
            # how misleading accuracy is next to balanced metrics, and A7 raises it to a
            # critical flag that blocks unsealing.
            warn(
                "accuracy_on_imbalanced_target",
                "metric",
                f"the minority class is {minority:.1%} of rows (< "
                f"{config.ACCURACY_MIN_MINORITY_FRACTION:.0%}); always predicting the majority "
                f"class already scores {1 - minority:.1%} accuracy. Use balanced_accuracy, "
                "f1, pr_auc or roc_auc",
            )

    required_task = _MODEL_TASKS.get(plan.model.name)
    if required_task is not None and required_task is not plan.task:
        error(
            "model_task_mismatch",
            "model",
            f"model {plan.model.name!r} supports {required_task.value} only",
        )
    if plan.class_weight == "balanced" and (
        plan.task is Task.REGRESSION or plan.model.name == "dummy"
    ):
        error(
            "class_weight_not_applicable",
            "class_weight",
            "class_weight 'balanced' needs a classification task and a weightable model",
        )

    known = {c.name for c in profile.columns}
    for drop in plan.drop_columns:
        if drop.column == profile.target:
            error("drop_target", "drop_columns", f"cannot drop the target {profile.target!r}")
        elif drop.column not in known:
            error(
                "unknown_column",
                "drop_columns",
                f"cannot drop {drop.column!r}: no such column (known: {sorted(known)})",
            )

    kept = [c for c in profile.feature_columns if c.name not in plan.dropped]
    for column in kept:
        if column.kind in _UNMODELLABLE:
            error(
                "unmodellable_column",
                "drop_columns",
                f"column {column.name!r} is {column.kind.value}; the pipeline cannot use "
                "it, so add it to drop_columns with a reason",
            )
    if not kept:
        error("no_features", "drop_columns", "no feature columns remain after drop_columns")

    if isinstance(plan.feature_selection, KBestSelection):
        width = _max_encoded_width(plan, profile, kept)
        if kept and plan.feature_selection.k >= width:
            warn(
                "k_best_keeps_everything",
                "feature_selection",
                f"k={plan.feature_selection.k} is at least the ~{width} encoded features, so "
                "k_best selects nothing out",
            )
    return sorted(issues, key=lambda i: i.severity is not Severity.ERROR)


def _max_encoded_width(plan: ModelPlan, profile: DatasetProfile, kept: list[ColumnProfile]) -> int:
    """Upper bound on the number of features after encoding (for the k_best warning)."""
    # TargetEncoder emits one column per class for multiclass targets, one otherwise.
    n_classes = profile.n_classes or 0
    target_encoded_width = n_classes if n_classes > 2 else 1
    width = 0
    for column in kept:
        if column.kind is ColumnKind.NUMERIC:
            width += 1
        elif plan.categorical.encode == "onehot":
            width += column.n_unique + (1 if column.missing_rate > 0 else 0)
        elif plan.categorical.encode == "target_cv":
            width += target_encoded_width
        else:
            width += 1
    return width
