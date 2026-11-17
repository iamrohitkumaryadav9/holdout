"""Numbers-from-state: digits only inside placeholders or allow-listed identifiers."""

import pytest

from holdout.agent.report import (
    allowed_tokens,
    draft_problems,
    fill,
    fill_draft,
    placeholder_catalogue,
    placeholder_values,
    prose_numbers,
    prose_problems,
)
from holdout.agent.schemas import ReportDraft
from holdout.plan.schema import ModelPlan
from tests.helpers import plan_dict

COLUMNS = ["feature_12", "age", "q4 revenue", "refund_issued"]
VALUES = placeholder_values(
    {"validation_score": 0.71234, "dummy_score": 0.5, "n_rows": 1200, "holdout_score": None}
)
ALLOWED = allowed_tokens(COLUMNS, ModelPlan.model_validate(plan_dict(metric="f1")))


def test_placeholder_values_format_and_skip_unavailable() -> None:
    assert VALUES == {"validation_score": "0.7123", "dummy_score": "0.5000", "n_rows": "1200"}
    catalogue = placeholder_catalogue(VALUES)
    assert "{{validation_score}}" in catalogue
    assert "0.7123" not in catalogue  # meanings only, never values


@pytest.mark.parametrize(
    "text",
    [
        "The plan scores {{validation_score}} against {{dummy_score}}.",
        "A12 flagged feature_12 and A1 flagged refund_issued.",
        "The f1 metric was used; q4 revenue was dropped.",
        "Checks A1 to A12 ran.",
    ],
)
def test_allowed_prose(text: str) -> None:
    assert prose_problems(text, ALLOWED, set(VALUES)) == []


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("The plan scores 0.71.", "'0.71'"),
        ("About 85% of rows.", "'85'"),
        ("It ran 3 rounds.", "'3'"),
        ("feature_13 is strong.", "'feature_13'"),  # not a real column
        ("Score {{validation_scor}}.", "unknown placeholder"),
        ("A13 fired.", "'A13'"),  # not a check id
    ],
)
def test_rejected_prose(text: str, problem: str) -> None:
    problems = prose_problems(text, ALLOWED, set(VALUES))
    assert problems
    assert any(problem in p for p in problems)


def _draft(**overrides: object) -> ReportDraft:
    base = {
        "summary": "The plan beats the dummy: {{validation_score}} versus {{dummy_score}}.",
        "findings": ["A1 found a leak in refund_issued."],
        "caveats": [],
        "recommendations": ["Drop refund_issued."],
        "causal_claim": False,
    }
    base.update(overrides)
    return ReportDraft.model_validate(base)


def test_draft_must_mention_every_open_flag() -> None:
    assert (
        draft_problems(_draft(), allowed=ALLOWED, placeholders=set(VALUES), required_checks=["A1"])
        == []
    )
    problems = draft_problems(
        _draft(), allowed=ALLOWED, placeholders=set(VALUES), required_checks=["A1", "A12"]
    )
    assert problems == ["findings must mention flag A12"]


def test_filled_prose_contains_only_state_numbers() -> None:
    filled = fill_draft(_draft(), VALUES)
    assert filled.summary == "The plan beats the dummy: 0.7123 versus 0.5000."
    numbers = prose_numbers(filled.summary + " A12 feature_12", ALLOWED)
    assert set(numbers) <= set(VALUES.values())
    assert fill("{{unknown}}", VALUES) == "{{unknown}}"
