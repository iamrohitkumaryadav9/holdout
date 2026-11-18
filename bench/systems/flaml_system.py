"""FLAML AutoML with a fixed 60 s budget: a strong-accuracy, zero-methodology reference.

No audits and no issue reports. Estimators are limited to LightGBM, random forest, extra
trees and (classification) L1 logistic regression; xgboost and catboost are excluded so
the reference does not depend on extra compiled libraries (FLAML's own default list
includes them). It optimises the benchmark metric with 5-fold cross-validation (FLAML's
own folds: stratified for classification), seeded with the run seed.

Column names that LightGBM refuses (JSON special characters, e.g. magic_telescope's
``fLength:``) are renamed for FLAML only, at fit and prediction alike (prereg Amendment 1:
the final run's first pass failed all 8 magic_telescope units on this, a harness bug).
"""

import re
import time
from typing import Any, Final

import numpy as np
import pandas as pd

from bench.harness import benchmark_metric
from bench.systems.base import FinalPipeline, SystemContext, SystemName, SystemOutput
from holdout.core.schemas import Task

TIME_BUDGET_SECONDS: Final = 60
CLASSIFIERS: Final = ("lgbm", "rf", "extra_tree", "lrl1")
REGRESSORS: Final = ("lgbm", "rf", "extra_tree")


#: Characters LightGBM refuses in feature names ("special JSON characters").
_UNSAFE = re.compile(r'[",:\[\]{}]')


def safe_names(columns: list[str]) -> dict[str, str]:
    """Column -> a name LightGBM accepts (prereg Amendment 1). Names without the refused
    characters are kept; the others get them replaced by ``_``, made unique with a suffix.
    The mapping is applied identically at fit and at prediction time, so the model sees the
    same data; only the labels change."""
    out: dict[str, str] = {}
    used = {str(c) for c in columns if not _UNSAFE.search(str(c))}  # never renamed
    for column in columns:
        if not _UNSAFE.search(str(column)):
            out[column] = str(column)
            continue
        name = base = _UNSAFE.sub("_", str(column))
        n = 1
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)
        out[column] = name
    return out


class RenamedColumns:
    """A fitted FLAML model that renames the raw columns before predicting."""

    def __init__(self, model: Any, names: dict[str, str]) -> None:
        self.model = model
        self.names = names

    @property
    def classes_(self) -> Any:
        return self.model.classes_

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X.rename(columns=self.names))

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return self.model.predict_proba(X.rename(columns=self.names))


class FlamlSystem:
    name: SystemName = "flaml"
    uses_llm = False

    def __init__(self, time_budget: float = TIME_BUDGET_SECONDS) -> None:
        self.time_budget = time_budget

    def run(self, ctx: SystemContext) -> SystemOutput:
        from flaml import AutoML  # optional dependency (bench extra)

        raw = ctx.dev.drop(columns=[ctx.target])
        names = safe_names(list(raw.columns))
        X = raw.rename(columns=names)
        y = ctx.dev[ctx.target]
        metric = benchmark_metric(ctx.task)
        classification = ctx.task is Task.CLASSIFICATION
        flaml_metric = metric
        if classification and y.nunique() > 2:
            flaml_metric = "roc_auc_ovr"
        automl = AutoML()
        started = time.perf_counter()
        automl.fit(
            X,
            y,
            task="classification" if classification else "regression",
            metric=flaml_metric,
            time_budget=self.time_budget,
            estimator_list=list(CLASSIFIERS if classification else REGRESSORS),
            eval_method="cv",
            n_splits=5,
            seed=ctx.seed,
            verbose=0,
            log_file_name="",
        )
        loss = float(automl.best_loss)
        # FLAML minimises a loss: 1 - AUC for roc_auc, the RMSE itself for rmse.
        validation = (1.0 - loss) if metric == "roc_auc" else loss
        return SystemOutput(
            status="completed",
            issues=[],
            pipeline=FinalPipeline(
                dropped_columns=[],
                metric=metric,
                split="stratified" if classification else "random",
                model=f"flaml:{automl.best_estimator}",
            ),
            estimator=RenamedColumns(automl, names),
            validation_metric=metric,
            validation_score=float(np.round(validation, 6)),
            extra={
                "time_budget": self.time_budget,
                "fit_seconds": round(time.perf_counter() - started, 1),
                "best_config": automl.best_config,
                "renamed_columns": sum(old != new for old, new in names.items()),
            },
        )

    def resume(self, ctx: SystemContext) -> SystemOutput:
        # No checkpoint: rerun from scratch. A time-budgeted search is seeded but not
        # deterministic (how far it gets depends on machine speed); documented as such.
        return self.run(ctx)
