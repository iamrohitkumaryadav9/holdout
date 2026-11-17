"""A10 deep mode, A11 causal claims (string level), A12 unstable validation."""

import dataclasses

import numpy as np
import pytest

from bench.traps import build_case, clean_twin
from holdout.audit.checks import a10_too_good as a10
from holdout.audit.checks import a11_causal as a11
from holdout.audit.checks import a12_stability as a12
from holdout.audit.types import CheckId, Severity
from tests.audit_harness import context_for, twin_context
from tests.tiers import SEEDS

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


def test_nadeau_bengio_interval_matches_a_hand_computed_example() -> None:
    # d = [0.10, 0.12, 0.08, 0.11, 0.09]: mean 0.10, s^2 = 0.001 / 4 = 0.00025.
    # k = 5, n_test/n_train = 1/4: corrected variance = (1/5 + 1/4) * 0.00025 = 0.0001125,
    # se = 0.0106066; t(4, 0.975) = 2.7764451; half-width = 0.0294488.
    mean, low, high = a12.nadeau_bengio_interval(
        np.array([0.10, 0.12, 0.08, 0.11, 0.09]), test_train_ratio=0.25, confidence=0.95
    )
    assert mean == pytest.approx(0.10)
    assert low == pytest.approx(0.10 - 0.0294488, abs=1e-6)
    assert high == pytest.approx(0.10 + 0.0294488, abs=1e-6)
    # The naive interval (variance s^2 / k) would be narrower: the correction matters.
    naive_half = 2.7764451 * np.sqrt(0.00025 / 5)
    assert high - mean > naive_half


def test_interval_needs_two_folds() -> None:
    _, low, high = a12.nadeau_bengio_interval(np.array([0.1]), 0.25, 0.95)
    assert (low, high) == (float("-inf"), float("inf"))


def test_test_train_ratio_comes_from_the_run_folds() -> None:
    ctx = twin_context("direct_leak", 0)  # stratified 5-fold
    assert a12.fold_test_train_ratio(ctx) == pytest.approx(0.25, abs=0.01)


def test_spread_that_could_change_the_conclusion_warns() -> None:
    ctx = _with_folds([0.55, 0.85, 0.6, 0.8, 0.58])  # dummy folds are 0.5
    flag = a12.run(ctx).flags[0]
    assert flag.severity is Severity.WARNING
    assert "includes zero" in flag.message
    assert flag.evidence.lift_interval_low <= 0  # type: ignore[union-attr]


def test_a_fold_at_or_below_the_dummy_warns() -> None:
    flag = a12.run(_with_folds([0.72, 0.73, 0.71, 0.72, 0.49])).flags[0]
    assert flag.severity is Severity.WARNING
    assert "1 fold(s) score at or below the dummy" in flag.message


def test_spread_that_cannot_change_the_conclusion_is_info() -> None:
    # Week 2 warned here (worst fold kept little of the lift); every fold still beats the
    # dummy and the corrected interval excludes zero, so the conclusion stands.
    flag = a12.run(_with_folds([0.72, 0.73, 0.71, 0.72, 0.62])).flags[0]
    assert flag.severity is Severity.INFO
    assert flag.evidence.lift_interval_low > 0  # type: ignore[union-attr,operator]
    assert flag.evidence.fold_lifts == pytest.approx([0.22, 0.23, 0.21, 0.22, 0.12])  # type: ignore[union-attr]


def test_plan_not_beating_the_dummy_is_not_assessed() -> None:
    outcome = a12.run(_with_folds([0.5, 0.49, 0.5, 0.51, 0.5]))
    assert outcome.flags == []
    assert "does not beat" in str(outcome.detail)


@pytest.mark.parametrize("seed", SEEDS)
def test_quiet_when_validation_is_stable(seed: int) -> None:
    case = clean_twin(build_case("direct_leak", seed, n_rows=6000))
    flags = a12.run(context_for(case, seed)).flags
    assert [f.severity for f in flags] == [Severity.INFO]  # spread recorded, no warning
