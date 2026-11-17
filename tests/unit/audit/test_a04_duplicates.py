"""A4: planted duplicates are critical and invalidate the holdout; small overlap warns."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bench.traps import build_case
from holdout.audit.checks import a04_duplicates as a4
from holdout.audit.types import Severity
from holdout.data.seal import HoldoutOverlap
from tests.audit_harness import context_for, trap_context, twin_context
from tests.tiers import SEEDS


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_exact_duplicates_are_critical_and_invalidate(seed: int) -> None:
    flag = a4.run(trap_context("cross_split_duplicates", seed)).flags[0]
    assert flag.severity is Severity.CRITICAL
    assert flag.invalidates_holdout
    evidence = flag.evidence
    assert evidence.kind == "duplicates"
    assert evidence.cross_fold_share > 0.05  # type: ignore[union-attr]
    assert evidence.holdout_overlap_share > 0.05  # type: ignore[union-attr,operator]
    assert flag.fix.kind == "deduplicate"


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_near_duplicates_below_six_significant_digits_are_caught(seed: int) -> None:
    case = build_case("cross_split_duplicates", seed, params={"variant": "near"})
    assert not case.df.duplicated().any()  # no exact duplicates: only float noise differs
    flag = a4.run(context_for(case, seed)).flags[0]
    assert flag.severity is Severity.CRITICAL


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trap", ["cross_split_duplicates", "direct_leak", "group_leakage"])
def test_quiet_on_clean_twins(trap: str, seed: int) -> None:
    assert a4.run(twin_context(trap, seed)).flags == []


class _FixedOverlap:
    def __init__(self, matching: int, rows: int) -> None:
        self.overlap = HoldoutOverlap(matching, matching, rows)

    def holdout_overlap(self, dev_features: pd.DataFrame) -> HoldoutOverlap:
        return self.overlap


def test_small_holdout_overlap_is_a_warning_with_counts() -> None:
    ctx = twin_context("cross_split_duplicates", 0)
    small = dataclasses.replace(ctx, overlap_source=_FixedOverlap(1, 300))  # 0.33% < 1%
    flag = a4.run(small).flags[0]
    assert flag.severity is Severity.WARNING
    assert not flag.invalidates_holdout
    assert "1 holdout rows (0.3%)" in flag.message
    large = dataclasses.replace(ctx, overlap_source=_FixedOverlap(3, 300))  # 1% >= 1%
    assert a4.run(large).flags[0].invalidates_holdout


def test_small_cross_fold_share_is_a_warning() -> None:
    ctx = twin_context("cross_split_duplicates", 0)
    dev = ctx.dev
    with_one_copy = pd.concat([dev, dev.iloc[[0]]]).reset_index(drop=True)
    small = dataclasses.replace(ctx, dev=with_one_copy, overlap_source=None)
    flag = a4.run(small).flags[0]
    assert flag.severity is Severity.WARNING
    assert flag.evidence.holdout_overlap_share is None  # type: ignore[union-attr]
    assert "not available" in flag.evidence.holdout_overlap_note  # type: ignore[union-attr]


def test_low_cardinality_data_reports_redundancy_without_alarm() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.choice(["x", "y"], 200), "y": rng.integers(0, 2, 200)})
    ctx = twin_context("cross_split_duplicates", 0)
    tiny = dataclasses.replace(ctx, dev=frame, target="y", overlap_source=None)
    outcome = a4.run(tiny)
    assert outcome.flags[0].evidence.dev_redundant_rows == 198  # type: ignore[union-attr]
