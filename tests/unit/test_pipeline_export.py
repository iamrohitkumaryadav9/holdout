"""The exported pipeline.py: deterministic, self-contained, and valid Python."""

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from holdout.core.errors import PlanValidationError
from holdout.core.schemas import SplitStrategy
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec
from holdout.export.pipeline_export import render_pipeline_script, write_pipeline_script
from holdout.plan.defaults import default_plan
from holdout.plan.schema import ModelPlan
from tests.helpers import plan_dict

ALLOWED_TOP_LEVEL = {"numpy", "pandas", "sklearn", "lightgbm"}
SPEC = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=42)


def _render(churn_df: pd.DataFrame) -> str:
    profile = profile_schema(churn_df, "churned")
    return render_pipeline_script(
        plan=default_plan(profile), profile=profile, spec=SPEC, data_sha256="a" * 64
    )


def test_same_plan_gives_byte_identical_files(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = default_plan(profile)
    first = write_pipeline_script(
        tmp_path / "a.py", plan=plan, profile=profile, spec=SPEC, data_sha256="a" * 64
    )
    second = write_pipeline_script(
        tmp_path / "b.py", plan=plan, profile=profile, spec=SPEC, data_sha256="a" * 64
    )
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()


def test_different_plans_give_different_files(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    other = ModelPlan.model_validate(plan_dict(model={"name": "logreg", "C": 0.3}))
    assert _render(churn_df) != render_pipeline_script(
        plan=other, profile=profile, spec=SPEC, data_sha256="a" * 64
    )


def test_script_imports_only_the_modelling_stack(churn_df: pd.DataFrame) -> None:
    tree = ast.parse(_render(churn_df))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    third_party = {m for m in imported if m not in sys.stdlib_module_names}
    assert third_party <= ALLOWED_TOP_LEVEL
    assert "holdout" not in imported


def test_script_documents_the_plan(churn_df: pd.DataFrame, sales_df: pd.DataFrame) -> None:
    source = _render(churn_df)
    assert source.startswith('"""Standalone reproduction of a Holdout cross-validation run.')
    assert "#   model: histogram gradient boosting" in source
    assert "CV_SEED = " in source
    profile = profile_schema(sales_df, "units_sold")
    spec = SplitSpec(strategy=SplitStrategy.TIME, time_col="date", seed=1)
    sales_source = render_pipeline_script(
        plan=default_plan(profile), profile=profile, spec=spec, data_sha256="b" * 64
    )
    assert "# Dropped columns:" in sales_source
    assert "#   'date': raw timestamp" in sales_source


def test_script_has_no_pyflakes_errors(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "pipeline.py"
    path.write_text(_render(churn_df), encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--isolated",
            "--select",
            "F,E9",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_plan_is_not_exported(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(plan_dict(metric="rmse"))
    with pytest.raises(PlanValidationError):
        render_pipeline_script(plan=plan, profile=profile, spec=SPEC, data_sha256="a" * 64)
