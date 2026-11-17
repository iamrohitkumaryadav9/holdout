"""The committed examples/ stay in sync with the generator and with the plan contract."""

from pathlib import Path

from holdout.data.examples import CLASSIFICATION_FILENAME, TIMESERIES_FILENAME, write_examples
from holdout.data.load import load_dataset
from holdout.data.profile import profile_schema
from holdout.plan.schema import Severity, validate_plan
from holdout.runner import load_plan

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_committed_csvs_match_the_generator(tmp_path: Path) -> None:
    generated = write_examples(tmp_path)
    for path in generated:
        committed = EXAMPLES / path.name
        assert committed.read_bytes() == path.read_bytes(), f"regenerate {committed}"
    assert {p.name for p in generated} == {CLASSIFICATION_FILENAME, TIMESERIES_FILENAME}


def test_example_plan_is_valid_for_the_example_data() -> None:
    plan = load_plan(EXAMPLES / "plan_churn_lightgbm.json")
    frame = load_dataset(EXAMPLES / CLASSIFICATION_FILENAME, "churned").frame
    issues = validate_plan(plan, profile_schema(frame, "churned"))
    assert [i for i in issues if i.severity is Severity.ERROR] == []
