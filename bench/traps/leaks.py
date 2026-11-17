"""Column-level traps: a post-outcome leak, a neutral-named proxy, an ID that encodes order.

Each trap is ``(df, target, task, seed, params) -> TrapCase``; ``clean_*`` produces the
same data without the planted mechanism.
"""

from typing import Any

import numpy as np
import pandas as pd

from bench.traps.types import (
    TrapCase,
    TrapTruth,
    clean_case,
    default_goal,
    positive_indicator,
    trap_rng,
)
from holdout.audit.types import Category, CheckId, Severity
from holdout.core.schemas import Task


def direct_leak(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Add a column derived from the target, named like something recorded afterwards."""
    rng = trap_rng(seed, "direct_leak")
    noise = float(params.get("noise", 0.03))
    if task is Task.CLASSIFICATION:
        column = str(params.get("column", "refund_issued"))
        y = positive_indicator(df[target])
        flips = rng.random(len(df)) < noise
        leaked = np.where(flips, 1 - y, y)
    else:
        y = df[target].to_numpy()
        column = str(params.get("column", "final_settlement"))
        leaked = np.round(y * 1.05 + rng.normal(0, noise * np.std(y), len(df)), 2)
    out = df.copy()
    out.insert(len(out.columns) - 1, column, leaked)
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=default_goal(target),
        plan_overrides={},
        truth=TrapTruth(
            trap_type="direct_leak",
            expected_checks=[CheckId.A1, CheckId.A2],
            expected_category=Category.TARGET_LEAKAGE,
            min_expected_severity=Severity.CRITICAL,
            affected_columns=[column],
            expected_remediation="drop_columns",
        ),
    )


def noisy_proxy(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Add target-plus-noise under a neutral name: only statistics (A1) can see it."""
    rng = trap_rng(seed, "noisy_proxy")
    column = str(params.get("column", "signal_strength"))
    y = (
        positive_indicator(df[target]).astype("float64")
        if task is Task.CLASSIFICATION
        else df[target].to_numpy(dtype="float64")
    )
    scale = float(params.get("noise_sd", 0.35)) * (
        1.0 if task is Task.CLASSIFICATION else np.std(y)
    )
    out = df.copy()
    out.insert(len(out.columns) - 1, column, np.round(y + rng.normal(0, scale, len(df)), 4))
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=default_goal(target),
        plan_overrides={},
        truth=TrapTruth(
            trap_type="noisy_proxy",
            expected_checks=[CheckId.A1],
            expected_category=Category.TARGET_LEAKAGE,
            # Uncorroborated statistics cannot tell a proxy from a genuinely strong
            # feature, so A1 alone stays a warning (see docs/audits.md, A1).
            min_expected_severity=Severity.WARNING,
            affected_columns=[column],
            expected_remediation="drop_columns",
        ),
    )


def _record_numbers(order: np.ndarray, start: int = 100_001) -> np.ndarray:
    numbers = np.empty(len(order), dtype="int64")
    numbers[order] = np.arange(start, start + len(order))
    return numbers


def id_with_signal(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Issue record numbers in target order, then shuffle the file.

    The file is shuffled because a target-sorted file would make the ID monotonic, which
    Week 1 already detects and drops; the realistic trap is an ID that *looks* like an
    arbitrary number but was issued in an order correlated with the outcome.
    """
    rng = trap_rng(seed, "id_with_signal")
    column = str(params.get("column", "record_number"))
    tie_break = rng.random(len(df))
    order = np.lexsort((tie_break, df[target].to_numpy()))
    out = df.copy()
    out.insert(0, column, _record_numbers(order))
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=default_goal(target),
        plan_overrides={},
        truth=TrapTruth(
            trap_type="id_with_signal",
            expected_checks=[CheckId.A3, CheckId.A1],
            expected_category=Category.ID_SIGNAL,
            min_expected_severity=Severity.CRITICAL,
            affected_columns=[column],
            expected_remediation="drop_columns",
        ),
    )


def clean_direct_leak(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    return clean_case(df, target, task)


def clean_noisy_proxy(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    return clean_case(df, target, task)


def clean_id_with_signal(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Same record-number column, issued in random order: an ID without signal."""
    rng = trap_rng(seed, "id_with_signal")
    column = str(params.get("column", "record_number"))
    out = df.copy()
    out.insert(0, column, _record_numbers(rng.permutation(len(df))))
    return clean_case(out, target, task)
