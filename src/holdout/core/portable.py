"""Portable modelling kernel shared verbatim by Holdout and every exported ``pipeline.py``.

Why this module exists: an exported script must rebuild *exactly* the split and pipeline
that Holdout evaluated, without importing Holdout. Two hand-maintained copies would drift
silently, so the exporter embeds this file's source instead. That imposes two rules,
both enforced by tests:

* imports are limited to the standard library, pandas, numpy, scikit-learn and lightgbm;
* inputs are plain dicts and literals (``SplitConfig``, ``PipelineConfig``), never Holdout
  types. Holdout's typed layers validate first, then translate into these configs.
"""

import argparse
import hashlib
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

#: Fill value for missing categories under ``impute: constant``.
MISSING_CATEGORY = "__missing__"

#: Threads for estimators that support them (kept small: laptops and shared CI runners).
N_JOBS = 2


class SplitConfig(TypedDict):
    """Plain description of the holdout split; ``seed`` is the already-derived split seed."""

    strategy: str
    test_size: float
    seed: int
    group_col: str | None
    time_col: str | None


class PipelineConfig(TypedDict):
    """Plain description of a pipeline; produced from a validated ``ModelPlan``."""

    task: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_impute: str
    numeric_scale: str
    categorical_impute: str
    categorical_encode: str
    selection_method: str
    selection_k: int | None
    model_name: str
    model_params: dict[str, Any]
    class_weight: str
    model_seed: int
    encoder_seed: int
    selector_seed: int


# --------------------------------------------------------------------------- data loading


def file_sha256(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks so large CSVs stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path, target: str) -> tuple[pd.DataFrame, int]:
    """Read a CSV and drop rows whose target is missing.

    Rows without a label cannot be trained on or scored, and keeping them would make the
    split depend on rows that never matter. Returns the frame (re-indexed 0..n-1, so the
    index is a stable row id) and the number of rows dropped.
    """
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} not found; columns are {list(frame.columns)}")
    missing = frame[target].isna()
    return frame.loc[~missing].reset_index(drop=True), int(missing.sum())


def time_order_key(values: pd.Series) -> pd.Series:
    """Map a column to sortable time values (NaN/NaT where a value cannot be ordered).

    Numeric columns order by value (a sequence index); datetime columns are normalised to
    UTC; strings are parsed as ISO 8601 first (fast, unambiguous) and only fall back to
    mixed-format parsing when that parses strictly more values.
    """
    if pd.api.types.is_bool_dtype(values):
        raise ValueError(f"column {values.name!r} is boolean and cannot define a time order")
    if pd.api.types.is_numeric_dtype(values):
        return values.astype("float64")
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return pd.Series(values.dt.tz_convert("UTC"), index=values.index, name=values.name)
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.Series(values.dt.tz_localize("UTC"), index=values.index, name=values.name)
    parsed = pd.to_datetime(values, errors="coerce", utc=True, format="ISO8601")
    if parsed.isna().sum() > values.isna().sum():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mixed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        if mixed.notna().sum() > parsed.notna().sum():
            parsed = mixed
    return parsed


# --------------------------------------------------------------------------- splitting


def split_frame(
    frame: pd.DataFrame, target: str, config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into (development, holdout), preserving the original index.

    * ``random`` / ``stratified``: ``train_test_split`` (stratified on the target).
    * ``group``: ``GroupShuffleSplit``; ``test_size`` is a fraction of *groups*.
    * ``time``: stable sort by time; the last ``ceil(test_size * n)`` rows are the holdout.

    Development rows are returned in original order, except for ``time`` where they are
    time-sorted (``TimeSeriesSplit`` relies on that order).
    """
    n_rows = len(frame)
    positions = np.arange(n_rows)
    strategy = config["strategy"]
    if strategy == "time":
        key = time_order_key(frame[_required(config["time_col"], "time_col")])
        order = key.reset_index(drop=True).sort_values(kind="stable").index.to_numpy()
        cut = n_rows - math.ceil(config["test_size"] * n_rows)
        return frame.iloc[order[:cut]], frame.iloc[order[cut:]]
    if strategy == "group":
        groups = frame[_required(config["group_col"], "group_col")].to_numpy()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config["test_size"], random_state=config["seed"]
        )
        dev_pos, holdout_pos = next(splitter.split(positions, groups=groups))
    elif strategy in ("random", "stratified"):
        stratify = frame[target].to_numpy() if strategy == "stratified" else None
        dev_pos, holdout_pos = train_test_split(
            positions,
            test_size=config["test_size"],
            random_state=config["seed"],
            shuffle=True,
            stratify=stratify,
        )
    else:
        raise ValueError(f"unknown split strategy {strategy!r}")
    return frame.iloc[np.sort(dev_pos)], frame.iloc[np.sort(holdout_pos)]


_MISSING = "\x00<missing>"


def normalise_features(features: pd.DataFrame, significant_digits: int) -> pd.DataFrame:
    """Return a normalised copy: columns sorted by name, floats rounded, strings folded."""
    columns = sorted(features.columns, key=str)
    out: dict[str, pd.Series] = {}
    for column in columns:
        values = features[column]
        if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
            out[str(column)] = pd.Series(
                round_significant(values.astype("float64").to_numpy(), significant_digits),
                index=features.index,
            )
        else:
            text = values.astype(str).str.strip().str.lower()
            out[str(column)] = text.where(values.notna(), _MISSING)
    return pd.DataFrame(out, index=features.index)


def round_significant(values: np.ndarray, digits: int) -> np.ndarray:
    """Round to ``digits`` significant digits; NaN and infinities are kept as they are."""
    result = np.array(values, dtype="float64", copy=True)
    finite = np.isfinite(result) & (result != 0.0)
    magnitude = np.floor(np.log10(np.abs(result[finite])))
    factor = np.power(10.0, digits - 1 - magnitude)
    result[finite] = np.round(result[finite] * factor) / factor
    return result + 0.0  # turn -0.0 into 0.0 so both hash alike


def drop_duplicate_rows(
    dev: pd.DataFrame, target: str, significant_digits: int
) -> tuple[pd.DataFrame, int]:
    """Keep the first of development rows whose normalised features match (v1.1).

    The same normalisation as the duplicate audit; the label is not compared, so of two
    rows with equal features and different labels the first is kept. Never applied to the
    holdout. Returns the remaining rows (original order and index) and the number removed.
    """
    normalised = normalise_features(dev.drop(columns=[target]), significant_digits)
    duplicated = pd.util.hash_pandas_object(normalised, index=False).duplicated(keep="first")
    keep = ~duplicated.to_numpy()
    return dev.iloc[keep], int((~keep).sum())


def make_cv(strategy: str, n_splits: int, seed: int) -> Any:
    """Return the CV splitter that matches the holdout split strategy.

    Folds must respect the same structure as the holdout: a group split needs group folds
    and a time split needs forward-chaining folds, otherwise validation leaks exactly what
    the holdout was built to prevent.
    """
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "group":
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"unknown split strategy {strategy!r}")


def cv_inputs(
    dev: pd.DataFrame, target: str, config: SplitConfig
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Return (X, y, groups) for cross-validation on the development rows.

    For time splits the rows are (re-)sorted stably by time, so forward-chaining folds are
    correct even if a caller passes development rows in another order.
    """
    if config["strategy"] == "time":
        key = time_order_key(dev[_required(config["time_col"], "time_col")])
        dev = dev.iloc[key.reset_index(drop=True).sort_values(kind="stable").index.to_numpy()]
    groups = dev[config["group_col"]] if config["strategy"] == "group" else None
    return dev.drop(columns=[target]), dev[target], groups


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"split config requires {name}")
    return value


# --------------------------------------------------------------------------- pipeline


def to_string_categories(X: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical values to ``str`` (keeping missing values as ``np.nan``).

    Stateless, so it learns nothing from any fold. It exists because CSV columns can mix
    types (``1`` and ``"1"``), which breaks the sorting that encoders rely on.
    """
    frame = pd.DataFrame(X)
    missing = frame.isna().to_numpy()
    as_text = frame.astype(str).astype(object)
    return as_text.mask(missing, np.nan)


def build_pipeline(config: PipelineConfig) -> Pipeline:
    """Build an unfitted scikit-learn ``Pipeline`` from a plain config.

    Every learned step (imputers, scaler, encoder, selector, model) is inside the pipeline,
    so cross-validation fits all of them on training folds only.
    """
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if config["numeric_columns"]:
        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy=config["numeric_impute"]))
        ]
        if config["numeric_scale"] == "standard":
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), list(config["numeric_columns"])))
    if config["categorical_columns"]:
        categorical_steps: list[tuple[str, Any]] = [
            ("as_text", FunctionTransformer(to_string_categories)),
            ("impute", _categorical_imputer(config["categorical_impute"])),
            ("encode", _encoder(config)),
        ]
        transformers.append(
            ("categorical", Pipeline(categorical_steps), list(config["categorical_columns"]))
        )
    if not transformers:
        raise ValueError("pipeline has no feature columns")
    preprocess = ColumnTransformer(
        transformers, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False
    )
    steps: list[tuple[str, Any]] = [("preprocess", preprocess)]
    selector = _selector(config)
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", _model(config)))
    return Pipeline(steps)


def _categorical_imputer(strategy: str) -> SimpleImputer:
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)
    return SimpleImputer(strategy=strategy)


def _encoder(config: PipelineConfig) -> Any:
    encode = config["categorical_encode"]
    if encode == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if encode == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if encode == "target_cv":
        # TargetEncoder cross-fits internally: each training row is encoded with
        # statistics from the other internal folds, never from its own target.
        return TargetEncoder(cv=5, shuffle=True, random_state=config["encoder_seed"])
    raise ValueError(f"unknown categorical encoding {encode!r}")


def _selector(config: PipelineConfig) -> Any:
    method = config["selection_method"]
    classification = config["task"] == "classification"
    if method == "none":
        return None
    if method == "k_best":
        score_func = f_classif if classification else f_regression
        return SelectKBest(score_func=score_func, k=config["selection_k"])
    if method == "model_based":
        forest_cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
        forest = forest_cls(n_estimators=100, random_state=config["selector_seed"], n_jobs=N_JOBS)
        return SelectFromModel(forest, threshold="median")
    raise ValueError(f"unknown feature selection {method!r}")


def _model(config: PipelineConfig) -> Any:
    name = config["model_name"]
    params = config["model_params"]
    seed = config["model_seed"]
    classification = config["task"] == "classification"
    class_weight = "balanced" if config["class_weight"] == "balanced" else None
    if name == "dummy":
        return DummyClassifier(strategy="prior") if classification else DummyRegressor()
    if name == "logreg":
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=class_weight,
            random_state=seed,
        )
    if name == "ridge":
        return Ridge(alpha=params["alpha"])
    if name == "random_forest":
        forest_cls = RandomForestClassifier if classification else RandomForestRegressor
        extra = {"class_weight": class_weight} if classification else {}
        return forest_cls(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=None if params["max_features"] == "all" else params["max_features"],
            random_state=seed,
            n_jobs=N_JOBS,
            **extra,
        )
    if name == "hist_gb":
        hgb_cls = (
            HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        )
        extra = {"class_weight": class_weight} if classification else {}
        # early_stopping=False: automatic early stopping would carve an internal validation
        # split whose size depends on n_rows, making behaviour data-size dependent.
        return hgb_cls(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            early_stopping=False,
            random_state=seed,
            **extra,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        lgbm_cls = LGBMClassifier if classification else LGBMRegressor
        # deterministic + force_row_wise make results reproducible run to run.
        # class_weight is always None for regression (validate_plan rejects "balanced").
        return lgbm_cls(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1 if params["subsample"] < 1.0 else 0,
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
            class_weight=class_weight,
        )
    raise ValueError(f"unknown model {name!r}")


# --------------------------------------------------------------------------- scoring


def score(metric: str, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Score a fitted estimator; returns the metric in its natural sign (rmse is positive).

    Binary positive class is ``estimator.classes_[1]`` (scikit-learn's convention).
    Multiclass: ``roc_auc`` is one-vs-rest, ``f1`` is macro-averaged, ``pr_auc`` is refused.
    """
    with warnings.catch_warnings():
        # LightGBM records generic feature names even for array input; harmless here.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if metric in ("roc_auc", "pr_auc"):
            return _ranking_score(metric, estimator, X, y)
        predictions = estimator.predict(X)
    if metric == "f1":
        classes = list(estimator.classes_)
        if len(classes) == 2:
            return float(f1_score(y, predictions, pos_label=classes[1], zero_division=0))
        return float(f1_score(y, predictions, average="macro", zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, predictions))
    if metric == "accuracy":
        return float(accuracy_score(y, predictions))
    if metric == "rmse":
        return float(root_mean_squared_error(y, predictions))
    if metric == "mae":
        return float(mean_absolute_error(y, predictions))
    if metric == "r2":
        return float(r2_score(y, predictions))
    raise ValueError(f"unknown metric {metric!r}")


def _ranking_score(metric: str, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    probabilities = np.asarray(estimator.predict_proba(X))
    classes = list(estimator.classes_)
    if len(classes) == 2:
        positive = (y.to_numpy() == classes[1]).astype(int)
        if metric == "roc_auc":
            return float(roc_auc_score(positive, probabilities[:, 1]))
        return float(average_precision_score(positive, probabilities[:, 1]))
    if metric == "pr_auc":
        raise ValueError("pr_auc is defined for binary targets only")
    return float(roc_auc_score(y, probabilities, multi_class="ovr", labels=classes))


def cross_validate_scores(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    cv: Any,
    metric: str,
) -> tuple[list[float], float]:
    """Fit a fresh clone of ``pipeline`` per fold; return (fold scores, total fit seconds).

    Written as an explicit loop (rather than ``cross_validate``) so every fit is visibly
    on ``X.iloc[train]`` only, sequential and deterministic.
    """
    scores: list[float] = []
    started = time.perf_counter()
    for fold, (train, valid) in enumerate(cv.split(X, y, groups)):
        estimator = clone(pipeline)
        try:
            estimator.fit(X.iloc[train], y.iloc[train])
            scores.append(score(metric, estimator, X.iloc[valid], y.iloc[valid]))
        except ValueError as exc:
            raise RuntimeError(f"fold {fold + 1}: {exc}") from exc
    return scores, time.perf_counter() - started


# --------------------------------------------------------------------------- exported entry


def run_exported_cv(
    argv: list[str] | None,
    *,
    data_sha256: str,
    target: str,
    metric: str,
    n_splits: int,
    cv_seed: int,
    split_config: SplitConfig,
    pipeline_config: PipelineConfig,
    dedup_significant_digits: int | None = None,
) -> int:
    """Command-line entry point of an exported ``pipeline.py``.

    Re-applies the recorded split, discards the holdout rows unseen, removes duplicate
    development rows if the plan did, and prints the cross-validated score on the
    development rows. It never scores the holdout.
    """
    parser = argparse.ArgumentParser(description="Reproduce a Holdout cross-validation run.")
    parser.add_argument("--csv", type=Path, required=True, help="the CSV the run used")
    args = parser.parse_args(argv)
    observed = file_sha256(args.csv)
    if observed != data_sha256:
        print(
            f"warning: CSV sha256 {observed} differs from the run's {data_sha256}; "
            "scores will not match the manifest",
            file=sys.stderr,
        )
    frame, _ = load_csv(args.csv, target)
    dev, _ = split_frame(frame, target, split_config)
    if dedup_significant_digits is not None:
        dev, removed = drop_duplicate_rows(dev, target, dedup_significant_digits)
        print(f"deduplicated: {removed} development rows removed")
    X, y, groups = cv_inputs(dev, target, split_config)
    cv = make_cv(split_config["strategy"], n_splits, cv_seed)
    scores, seconds = cross_validate_scores(
        build_pipeline(pipeline_config), X, y, groups, cv, metric
    )
    print(f"metric={metric}")
    print(f"fold_scores={scores!r}")
    print(f"cv_mean={float(np.mean(scores))!r}")
    print(f"cv_std={float(np.std(scores))!r}")
    print(f"fit_seconds={seconds:.2f}")
    return 0
