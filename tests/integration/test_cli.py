"""End-to-end CLI on the make-example CSVs: offline, all three artifacts, holdout once."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from holdout.cli import app
from holdout.data.examples import CLASSIFICATION_FILENAME, TIMESERIES_FILENAME
from holdout.data.split import TIME_ORDER_WARNING
from tests.helpers import plan_dict

runner = CliRunner()


@pytest.fixture(scope="module")
def examples(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("examples")
    result = runner.invoke(app, ["make-example", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / CLASSIFICATION_FILENAME).is_file()
    assert (out / TIMESERIES_FILENAME).is_file()
    return out


def _only_run_dir(out: Path) -> Path:
    runs = [p for p in out.iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _assert_artifacts(run_dir: Path) -> dict[str, object]:
    for name in ("pipeline.py", "run_manifest.json", "report.md", "holdout.parquet"):
        assert (run_dir / name).is_file(), name
    manifest: dict[str, object] = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    return manifest


def test_classification_example_end_to_end_with_unseal(examples: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run",
            "--csv",
            str(examples / CLASSIFICATION_FILENAME),
            "--target",
            "churned",
            "--unseal",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Split: stratified" in result.output
    assert "Optimism gap (validation - holdout)" in result.output

    run_dir = _only_run_dir(out)
    manifest = _assert_artifacts(run_dir)
    assert manifest["task"] == "classification"
    holdout = manifest["holdout"]
    assert isinstance(holdout, dict)
    assert holdout["evaluation_index"] == 1
    assert manifest["optimism_gap"] == pytest.approx(
        manifest["validation_score"] - holdout["score"]
    )  # type: ignore[operator]
    plans = manifest["plans"]
    assert isinstance(plans, list)
    assert [p["plan"]["model"]["name"] for p in plans] == ["dummy", "hist_gb"]
    assert plans[1]["cv"]["mean"] > plans[0]["cv"]["mean"]

    log = (run_dir / "holdout_access.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["outcome"] for line in log] == ["evaluated"]
    assert json.loads(log[0])["caller"] == "holdout run --unseal"
    assert "# Holdout run" in (run_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.full_only
def test_timeseries_example_warns_when_declared_tabular(examples: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run",
            "--csv",
            str(examples / TIMESERIES_FILENAME),
            "--target",
            "units_sold",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert TIME_ORDER_WARNING in result.output
    assert "alternative: --split time:date" in result.output
    manifest = _assert_artifacts(_only_run_dir(out))
    assert manifest["holdout"] == "not unsealed"
    assert manifest["optimism_gap"] is None
    # Week 2: A6 finds the time leak, so the randomly drawn holdout is marked invalid; the
    # access log records only that (nothing ever read the holdout).
    assert manifest["holdout_invalid"] is True
    assert "Start a NEW run with the corrected split" in result.output.replace("\n", " ")
    log = (_only_run_dir(out) / "holdout_access.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["outcome"] for line in log] == ["marked_invalid"]


def test_timeseries_example_with_time_split(examples: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run",
            "--csv",
            str(examples / TIMESERIES_FILENAME),
            "--target",
            "units_sold",
            "--data-type",
            "timeseries",
            "--unseal",
            # Forward-chaining folds train early folds on little data, so A12 warns that
            # the fold scores spread widely; acknowledging it is the user's call.
            "--ack-warnings",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = _assert_artifacts(_only_run_dir(out))
    split = manifest["split"]
    assert isinstance(split, dict)
    assert split["spec"]["strategy"] == "time"
    assert split["spec"]["time_col"] == "date"
    assert manifest["plans"][1]["cv"]["splitter"] == "TimeSeriesSplit"  # type: ignore[index]


@pytest.mark.full_only
def test_explicit_plan_and_split(examples: Path, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(plan_dict(metric="pr_auc", model={"name": "random_forest", "n_estimators": 50})),
        encoding="utf-8",
    )
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run",
            "--csv",
            str(examples / CLASSIFICATION_FILENAME),
            "--target",
            "churned",
            "--task",
            "classification",
            "--split",
            "random",
            "--seed",
            "7",
            "--plan",
            str(plan_path),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user override" in result.output
    manifest = _assert_artifacts(_only_run_dir(out))
    assert manifest["metric"] == "pr_auc"
    assert manifest["run_seed"] == 7


@pytest.mark.full_only
def test_ledger_warns_on_a_second_unsealed_run(examples: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    args = [
        "run",
        "--csv",
        str(examples / CLASSIFICATION_FILENAME),
        "--target",
        "churned",
        "--unseal",
        "--out",
        str(out),
    ]
    assert runner.invoke(app, args).exit_code == 0
    second = runner.invoke(app, args)
    assert second.exit_code == 0
    assert "this exact holdout was already evaluated" in second.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--target", "nope"], "target column 'nope' not found"),
        (["--target", "churned", "--split", "group:nope"], "group_col 'nope' not found"),
        (["--target", "churned", "--split", "kfold"], "unknown --split"),
    ],
)
def test_errors_exit_with_code_2_and_a_message(
    examples: Path, tmp_path: Path, extra: list[str], message: str
) -> None:
    result = runner.invoke(
        app,
        ["run", "--csv", str(examples / CLASSIFICATION_FILENAME), "--out", str(tmp_path), *extra],
    )
    assert result.exit_code == 2
    assert message in result.output
