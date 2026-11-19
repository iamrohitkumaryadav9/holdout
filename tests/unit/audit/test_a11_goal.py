"""v1.1 goal-level A11: a causal question is a finding at intake; the report declines.

Post-benchmark fix from docs/error_analysis.md, section 1 (causal goals were answered
without any decline when the generated prose avoided causal phrasing, and offline runs never
checked the goal). Not part of the pre-registered evaluation."""

import pytest

from holdout.audit.checks.a11_causal import check_causal_claims, check_goal, decline_sentence
from holdout.audit.types import Category, CheckId, Severity

COLUMNS = ["GLCM_Pan", "Mean_Green", "class", "contacted_by_phone"]


@pytest.mark.parametrize(
    "goal",
    [
        "Does GLCM_Pan cause class?",
        "What is the effect of GLCM_Pan on class?",
        "Does being contacted by phone cause customers to convert?",
        "Find what drives class.",
    ],
)
def test_a_causal_question_is_a_goal_finding(goal: str) -> None:
    (flag,) = check_goal(goal, COLUMNS, "class")
    assert flag.check is CheckId.A11
    assert flag.category is Category.CAUSAL_CLAIM
    assert flag.severity.at_least(Severity.WARNING)
    assert flag.fix.kind == "remove_causal_claim"
    assert "goal asks a causal question" in flag.message


@pytest.mark.parametrize(
    "goal", ["predict class", "Maximise accuracy at predicting class.", "Does it not cause"]
)
def test_a_predictive_goal_is_not(goal: str) -> None:
    assert check_goal(goal, COLUMNS, "class") == []


def test_the_goal_finding_names_the_columns_asked_about() -> None:
    (flag,) = check_goal("Does GLCM_Pan cause class?", COLUMNS, "class")
    assert flag.columns == ["GLCM_Pan"]  # not the target, not unrelated columns


def test_the_decline_is_not_itself_a_causal_claim() -> None:
    for causes in (["GLCM_Pan"], [], ["a", "b"]):
        sentence = decline_sentence(causes, "class")
        assert "no causal conclusion is drawn" in sentence
        assert check_causal_claims(sentence) == []
