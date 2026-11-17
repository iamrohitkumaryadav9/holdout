"""A10 deep mode, A11 causal claims (string level), A12 unstable validation."""

import dataclasses

import pytest

from bench.traps import build_case, clean_twin
from holdout.audit.checks import a10_too_good as a10
from holdout.audit.checks import a11_causal as a11
from holdout.audit.checks import a12_stability as a12
from holdout.audit.types import CheckId, Severity
from tests.audit_harness import SEEDS, context_for, twin_context

# ------------------------------------------------------------------ A10


@pytest.mark.parametrize("seed", SEEDS)
def test_near_ceiling_score_switches_on_deep_mode_as_info(seed: int) -> None:
    # A near-ceiling planted case by construction: a leak with 1% label noise.
    ctx = context_for(build_case("direct_leak", seed, params={"noise": 0.01}), seed)
    assert a10.deep_mode(ctx)
    flag = a10.run(ctx).flags[0]
    assert flag.severity is Severity.INFO
    assert flag.evidence.deep_mode  # type: ignore[union-attr]
    assert "deep mode" in flag.message


@pytest.mark.parametrize("seed", SEEDS)
def test_ordinary_scores_stay_in_standard_mode(seed: int) -> None:
    ctx = twin_context("direct_leak", seed)
    assert not a10.deep_mode(ctx)
    assert a10.run(ctx).flags == []


# ------------------------------------------------------------------ A11


@pytest.mark.parametrize(
    "text",
    [
        "Being contacted by phone causes customers to convert.",
        "Phone contact leads to higher conversion.",
        "Tenure drives conversion.",
        "Promotions result in more sales.",
        "We estimate the effect of price on demand.",
        "Sales rose because of the promotion.",
    ],
)
def test_causal_phrasing_is_flagged(text: str) -> None:
    flags = a11.check_causal_claims(text)
    assert len(flags) == 1
    assert flags[0].check is CheckId.A11
    assert flags[0].severity is Severity.WARNING
    assert flags[0].fix.kind == "remove_causal_claim"
    assert flags[0].evidence.statements == [text]  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "text",
    [
        "Phone contact does not cause conversion; it is only associated with it.",
        "This model cannot tell whether tenure drives conversion.",
        "Tenure is strongly associated with conversion.",
        "The model predicts conversion with ROC AUC 0.72.",
        "",
    ],
)
def test_negated_or_associational_text_is_not_flagged(text: str) -> None:
    assert a11.check_causal_claims(text) == []


def test_only_causal_sentences_are_quoted() -> None:
    text = "AUC is 0.72. Visits drive conversion. Tenure is associated with it."
    flags = a11.check_causal_claims(text)
    assert flags[0].evidence.statements == ["Visits drive conversion."]  # type: ignore[union-attr]


def test_structured_claims_are_flagged() -> None:
    claim = a11.StructuredCausalClaim(cause="promo", effect="sales", statement="promo raises sales")
    flags = a11.check_causal_claims("", [claim])
    assert flags[0].evidence.source == "structured"  # type: ignore[union-attr]
    assert flags[0].evidence.phrases == ["promo -> sales"]  # type: ignore[union-attr]


def test_causal_question_trap_truth_matches_a_causal_answer() -> None:
    case = build_case("causal_question", 0)
    assert case.truth.expected_checks == [CheckId.A11]
    answer = "Yes: being contacted by phone causes customers to convert."
    assert a11.check_causal_claims(answer)[0].category == case.truth.expected_category


# ------------------------------------------------------------------ A12


def _with_folds(scores: list[float]):  # type: ignore[no-untyped-def]
    ctx = twin_context("direct_leak", 0)
    result = ctx.cv_result.model_copy(update={"fold_scores": scores})
    return dataclasses.replace(ctx, cv_result=result)


def test_widely_spread_folds_warn() -> None:
    ctx = _with_folds([0.55, 0.85, 0.6, 0.8, 0.58])  # dummy is 0.5
    flag = a12.run(ctx).flags[0]
    assert flag.severity is Severity.WARNING
    assert "spread" in flag.message
    assert flag.evidence.lift_coefficient_of_variation > 0.25  # type: ignore[union-attr,operator]


def test_a_worst_fold_near_the_dummy_warns() -> None:
    flag = a12.run(_with_folds([0.72, 0.73, 0.71, 0.72, 0.52])).flags[0]
    assert "worst fold" in flag.message


def test_plan_not_beating_the_dummy_is_not_assessed() -> None:
    outcome = a12.run(_with_folds([0.5, 0.49, 0.5, 0.51, 0.5]))
    assert outcome.flags == []
    assert "does not beat" in str(outcome.detail)


@pytest.mark.parametrize("seed", SEEDS)
def test_quiet_when_validation_is_stable(seed: int) -> None:
    # A12's clean twin: the same clean base with enough rows for stable folds. (On ~1k
    # rows with a weak model, fold scores genuinely vary by a large share of the lift, and
    # the A12 warning there is correct; A12 is never critical.)
    case = clean_twin(build_case("direct_leak", seed, n_rows=6000))
    assert a12.run(context_for(case, seed)).flags == []
