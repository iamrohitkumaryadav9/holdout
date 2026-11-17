"""--split parsing, plan loading, and the guarantee that rejected plans never seal."""

import json
from pathlib import Path

import pytest

from holdout.core.errors import PlanValidationError, SplitError
from holdout.core.schemas import SplitStrategy, Task
from holdout.data.split import SplitSpec
from holdout.runner import RunOptions, execute_run, load_plan, parse_split_option
from tests.helpers import plan_dict


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("auto", None),
        ("random", SplitSpec(strategy=SplitStrategy.RANDOM, seed=3)),
        ("stratified", SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=3)),
        ("group:customer", SplitSpec(strategy=SplitStrategy.GROUP, group_col="customer", seed=3)),
        ("time:date", SplitSpec(strategy=SplitStrategy.TIME, time_col="date", seed=3)),
    ],
)
def test_parse_split_option(text: str, expected: SplitSpec | None) -> None:
    assert parse_split_option(text, seed=3) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("kfold", "unknown --split"),
        ("group", "needs a column"),
        ("time:", "needs a column"),
        ("random:x", "only group and time splits take a column"),
    ],
)
def test_parse_split_option_errors(text: str, message: str) -> None:
    with pytest.raises(SplitError, match=message):
        parse_split_option(text, seed=3)


def test_load_plan_reports_every_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan_dict(metric="logloss", class_weight="auto")), encoding="utf-8")
    with pytest.raises(PlanValidationError, match="malformed") as info:
        load_plan(path)
    assert "metric" in str(info.value)
    assert "class_weight" in str(info.value)
    with pytest.raises(PlanValidationError, match="cannot read plan"):
        load_plan(tmp_path / "missing.json")


def test_rejected_plan_never_seals_a_holdout(tmp_path: Path, csv_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            plan_dict(metric="accuracy", drop_columns=[{"column": "ghost", "reason": "typo"}])
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runs"
    with pytest.raises(PlanValidationError, match="unknown_column"):
        execute_run(RunOptions(csv=csv_path, target="churned", plan_path=plan_path, out_dir=out))
    assert not out.exists()


def test_plan_task_must_match_the_run(tmp_path: Path, csv_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_dict(task="regression", metric="rmse")), encoding="utf-8")
    with pytest.raises(PlanValidationError, match="pass --task"):
        execute_run(
            RunOptions(
                csv=csv_path,
                target="churned",
                task=Task.CLASSIFICATION,
                plan_path=plan_path,
                out_dir=tmp_path / "runs",
            )
        )
