"""Run the generated pipeline.py in a subprocess and match Holdout's CV score to 1e-9.

The subprocess runs with ``python -I`` (isolated mode: no current directory or user site
on sys.path), and the import test in test_pipeline_export.py guarantees the script never
imports holdout, so the score is produced by the exported code alone.
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from holdout.core.portable import file_sha256
from holdout.core.schemas import SplitStrategy
from holdout.data.examples import make_classification_example, make_timeseries_example
from holdout.data.load import load_dataset
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec, apply_split
from holdout.export.pipeline_export import write_pipeline_script
from holdout.plan.compiler import evaluate_cv
from holdout.plan.defaults import default_plan
from holdout.plan.schema import ModelPlan
from tests.helpers import make_grouped_df, plan_dict

CASES = {
    "stratified_default": (
        lambda: make_classification_example(n_rows=600, seed=4),
        "churned",
        SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=42),
        None,
    ),
    "time_lightgbm": (
        lambda: make_timeseries_example(n_days=80, seed=4),
        "units_sold",
        SplitSpec(strategy=SplitStrategy.TIME, time_col="date", seed=7),
        plan_dict(
            task="regression",
            metric="mae",
            model={"name": "lightgbm", "n_estimators": 50, "subsample": 0.8},
            categorical={"encode": "target_cv"},
            drop_columns=[{"column": "date", "reason": "raw timestamp"}],
        ),
    ),
    "group_logreg_kbest": (
        make_grouped_df,
        "label",
        SplitSpec(strategy=SplitStrategy.GROUP, group_col="customer_id", seed=3),
        plan_dict(
            metric="pr_auc",
            model={"name": "logreg", "C": 0.5},
            numeric={"scale": "standard"},
            feature_selection={"method": "k_best", "k": 2},
            drop_columns=[{"column": "customer_id", "reason": "entity id"}],
        ),
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        c if c != "stratified_default" else pytest.param(c, marks=pytest.mark.full_only)
        for c in sorted(CASES)
    ],
)
def test_exported_script_reproduces_the_cv_score(tmp_path: Path, case: str) -> None:
    make_frame, target, spec, plan_override = CASES[case]
    csv = tmp_path / "data.csv"
    make_frame().to_csv(csv, index=False)
    # Same loading path as the CLI, so both sides see identical rows and dtypes.
    frame = load_dataset(csv, target).frame
    profile = profile_schema(frame, target)
    plan = (
        default_plan(profile) if plan_override is None else ModelPlan.model_validate(plan_override)
    )
    dev, _ = apply_split(frame, spec, target)
    expected = evaluate_cv(plan, dev, target, spec, profile)

    script = write_pipeline_script(
        tmp_path / "pipeline.py",
        plan=plan,
        profile=profile,
        spec=spec,
        data_sha256=file_sha256(csv),
    )
    completed = subprocess.run(
        [sys.executable, "-I", str(script), "--csv", str(csv)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert completed.stderr == ""  # no sha mismatch warning
    match = re.search(r"^cv_mean=(.+)$", completed.stdout, re.MULTILINE)
    assert match is not None, completed.stdout
    assert abs(float(match.group(1)) - expected.mean) <= 1e-9
    folds = re.search(r"^fold_scores=\[(.+)\]$", completed.stdout, re.MULTILINE)
    assert folds is not None
    exported_folds = [float(v) for v in folds.group(1).split(",")]
    assert exported_folds == pytest.approx(expected.fold_scores, abs=1e-9)


def test_exported_script_warns_on_a_different_csv(tmp_path: Path) -> None:
    frame = make_classification_example(n_rows=300, seed=1)
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    profile = profile_schema(pd.read_csv(csv), "churned")
    script = write_pipeline_script(
        tmp_path / "pipeline.py",
        plan=default_plan(profile),
        profile=profile,
        spec=SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1),
        data_sha256="0" * 64,
    )
    completed = subprocess.run(
        [sys.executable, "-I", str(script), "--csv", str(csv)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "differs from the run's" in completed.stderr
