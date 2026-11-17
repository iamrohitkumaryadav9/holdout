"""Trap generators: deterministic per seed, truth metadata matches the data, twins differ
only by the planted mechanism."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bench.traps import TRAPS, build_case, clean_twin
from bench.traps.synthetic import BUILDERS
from bench.traps.types import CLEAN_TRUTH_TYPE
from holdout.audit.types import Severity
from holdout.core.schemas import Task


@pytest.mark.parametrize("name", sorted(TRAPS))
def test_generators_are_deterministic_per_seed(name: str) -> None:
    first, again, other = build_case(name, 3), build_case(name, 3), build_case(name, 4)
    pd.testing.assert_frame_equal(first.df, again.df)
    assert first.truth == again.truth
    assert first.goal_text == again.goal_text
    pd.testing.assert_frame_equal(clean_twin(first).df, clean_twin(again).df)
    if name != "causal_question":  # its data is the untouched base, which also varies by seed
        assert not first.df.equals(other.df)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_bases_are_deterministic_and_neutral(name: str) -> None:
    frame, target, _ = BUILDERS[name](seed=1)
    again, _, _ = BUILDERS[name](seed=1)
    pd.testing.assert_frame_equal(frame, again)
    assert not frame.duplicated().any()
    assert target in frame.columns


@pytest.mark.parametrize("name", sorted(TRAPS))
def test_truth_metadata_is_consistent(name: str) -> None:
    case = build_case(name, 0)
    truth = case.truth
    assert truth.trap_type == name
    assert truth.expected_category is not None
    assert truth.expected_checks
    assert truth.min_expected_severity in (Severity.WARNING, Severity.CRITICAL)
    for column in truth.affected_columns:
        assert column in case.df.columns, f"{name}: affected column {column!r} missing"
    twin = clean_twin(case)
    assert twin.truth.trap_type == CLEAN_TRUTH_TYPE
    assert twin.truth.expected_category is None
    assert twin.target == case.target


def test_direct_leak_column_is_derived_from_the_target() -> None:
    case = build_case("direct_leak", 0)
    agreement = (case.df["refund_issued"] == case.df[case.target]).mean()
    assert 0.95 < agreement < 1.0
    assert "refund_issued" not in clean_twin(case).df.columns


def test_noisy_proxy_has_a_neutral_name_and_a_strong_correlation() -> None:
    case = build_case("noisy_proxy", 0)
    assert case.df["signal_strength"].corr(case.df[case.target]) > 0.7


def test_id_with_signal_is_issued_in_target_order_but_shuffled() -> None:
    case = build_case("id_with_signal", 0)
    ids = case.df["record_number"]
    assert ids.is_unique
    assert not ids.is_monotonic_increasing  # the file is shuffled
    by_id = case.df.sort_values("record_number")[case.target].to_numpy()
    assert np.all(np.diff(by_id) >= 0)  # but IDs follow the target
    twin_ids = clean_twin(case).df.sort_values("record_number")[case.target].to_numpy()
    assert not np.all(np.diff(twin_ids) >= 0)


@pytest.mark.parametrize("variant", ["exact", "near"])
def test_duplicates_share(variant: str) -> None:
    base, _, _ = BUILDERS["classification"](seed=0)
    case = build_case("cross_split_duplicates", 0, params={"variant": variant, "share": 0.15})
    assert len(case.df) == len(base) + 180
    exact_dupes = case.df.duplicated().sum()
    assert exact_dupes == (180 if variant == "exact" else 0)
    with pytest.raises(ValueError, match="unknown duplicate variant"):
        build_case("cross_split_duplicates", 0, params={"variant": "fuzzy"})


def test_group_leakage_has_entity_constant_features_and_bias() -> None:
    case = build_case("group_leakage", 0)
    per_entity = case.df.groupby("account")
    assert (per_entity["region_code"].nunique() == 1).all()
    assert (per_entity["account_vintage"].nunique() == 1).all()
    purity = per_entity[case.target].agg(lambda s: max(s.mean(), 1 - s.mean())).mean()
    twin_purity = (
        clean_twin(case)
        .df.groupby("account")[case.target]
        .agg(lambda s: max(s.mean(), 1 - s.mean()))
        .mean()
    )
    assert purity > twin_purity + 0.1  # entity-level target bias only in the trap


@pytest.mark.parametrize("mode", ["remove", "rename"])
def test_time_order_hides_the_date_and_keeps_order(mode: str) -> None:
    case = build_case("time_order", 0, params={"mode": mode})
    assert "date" not in case.df.columns
    assert ("period" in case.df.columns) == (mode == "rename")
    trend = np.corrcoef(np.arange(len(case.df)), case.df[case.target])[0, 1]
    twin_trend = np.corrcoef(np.arange(len(case.df)), clean_twin(case).df[case.target])[0, 1]
    assert trend > 0.5
    assert abs(twin_trend) < 0.1
    with pytest.raises(ValueError, match="unknown time_order mode"):
        build_case("time_order", 0, params={"mode": "shuffle"})


def test_imbalance_accuracy_subsamples_and_asks_for_accuracy() -> None:
    case = build_case("imbalance_accuracy", 0)
    assert case.df[case.target].mean() == pytest.approx(0.04, abs=0.005)
    assert case.plan_overrides == {"metric": "accuracy"}
    assert "accuracy" in case.goal_text.lower()
    twin = clean_twin(case)
    assert twin.plan_overrides == {"metric": "accuracy"}
    assert twin.df[twin.target].mean() > 0.2
    with pytest.raises(ValueError, match="classification base"):
        TRAPS["imbalance_accuracy"].apply(*BUILDERS["regression"](seed=0), 0, {})


def test_messy_formats_plant_all_three_problems() -> None:
    case = build_case("messy_formats", 0)
    income = case.df["income"].astype(str)
    assert income.str.contains(",").any()
    assert income.str.startswith("Rs ").any()
    assert (case.df["age"] == -999).mean() > 0.01
    assert case.df["segment"].str.strip().str.lower().nunique() < case.df["segment"].nunique()


def test_causal_question_only_changes_the_goal() -> None:
    case = build_case("causal_question", 0)
    twin = clean_twin(case)
    pd.testing.assert_frame_equal(case.df, twin.df)
    assert "cause" in case.goal_text
    assert "cause" not in twin.goal_text


def test_regression_variants_of_column_traps() -> None:
    frame, target, task = BUILDERS["regression"](seed=0)
    assert task is Task.REGRESSION
    leak = TRAPS["direct_leak"].apply(frame, target, task, 0, {})
    assert "final_settlement" in leak.df.columns
    proxy = TRAPS["noisy_proxy"].apply(frame, target, task, 0, {})
    assert proxy.df["signal_strength"].corr(proxy.df[target]) > 0.7
    grouped = TRAPS["group_leakage"].apply(frame, target, task, 0, {})
    assert grouped.df[target].std() > frame[target].std()


def test_clean_twin_needs_provenance() -> None:
    case = build_case("direct_leak", 0)
    with pytest.raises(ValueError, match="provenance"):
        clean_twin(replace(case, provenance=None))
