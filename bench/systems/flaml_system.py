"""FLAML AutoML with a fixed 60 s budget: a strong-accuracy, zero-methodology reference.

No audits and no issue reports. Estimators are limited to LightGBM, random forest, extra
trees and (classification) L1 logistic regression; xgboost and catboost are excluded so
the reference does not depend on extra compiled libraries (FLAML's own default list
includes them). It optimises the benchmark metric with 5-fold cross-validation (FLAML's
own folds: stratified for classification), seeded with the run seed.
"""

import time
from typing import Final

import numpy as np

from bench.harness import benchmark_metric
from bench.systems.base import FinalPipeline, SystemContext, SystemName, SystemOutput
from holdout.core.schemas import Task

TIME_BUDGET_SECONDS: Final = 60
CLASSIFIERS: Final = ("lgbm", "rf", "extra_tree", "lrl1")
REGRESSORS: Final = ("lgbm", "rf", "extra_tree")


class FlamlSystem:
    name: SystemName = "flaml"
    uses_llm = False

    def __init__(self, time_budget: float = TIME_BUDGET_SECONDS) -> None:
        self.time_budget = time_budget

    def run(self, ctx: SystemContext) -> SystemOutput:
        from flaml import AutoML  # optional dependency (bench extra)

        X = ctx.dev.drop(columns=[ctx.target])
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
            estimator=automl,
            validation_metric=metric,
            validation_score=float(np.round(validation, 6)),
            extra={
                "time_budget": self.time_budget,
                "fit_seconds": round(time.perf_counter() - started, 1),
                "best_config": automl.best_config,
            },
        )

    def resume(self, ctx: SystemContext) -> SystemOutput:
        # No checkpoint: rerun from scratch. A time-budgeted search is seeded but not
        # deterministic (how far it gets depends on machine speed); documented as such.
        return self.run(ctx)
