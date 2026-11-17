"""Evaluation and data-quality traps: misleading accuracy, messy formats, causal goals."""

from typing import Any

import numpy as np
import pandas as pd

from bench.traps.types import TrapCase, TrapTruth, clean_case, trap_rng
from holdout.audit.types import Category, CheckId, Severity
from holdout.core.schemas import Task

ACCURACY_GOAL = "Maximise accuracy at predicting {target}."


def imbalance_accuracy(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Subsample the minority class to a few percent and ask for accuracy."""
    if task is not Task.CLASSIFICATION:
        raise ValueError("imbalance_accuracy needs a classification base")
    rng = trap_rng(seed, "imbalance_accuracy")
    share = float(params.get("minority_share", 0.04))
    counts = df[target].value_counts()
    minority = counts.idxmin()
    majority_rows = int(counts.max())
    keep = round(share / (1 - share) * majority_rows)
    minority_positions = np.flatnonzero(df[target].to_numpy() == minority)
    kept = rng.choice(minority_positions, size=keep, replace=False)
    mask = df[target].to_numpy() != minority
    mask[kept] = True
    out = df.loc[mask].reset_index(drop=True)
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=ACCURACY_GOAL.format(target=target),
        plan_overrides={"metric": "accuracy"},
        truth=TrapTruth(
            trap_type="imbalance_accuracy",
            expected_checks=[CheckId.A7],
            expected_category=Category.METRIC_MISMATCH,
            min_expected_severity=Severity.CRITICAL,
            affected_columns=[target],
            expected_remediation="change_metric",
        ),
    )


def clean_imbalance_accuracy(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Accuracy on the original, reasonably balanced data is a fair metric."""
    return clean_case(
        df,
        target,
        task,
        goal=ACCURACY_GOAL.format(target=target),
        plan_overrides={"metric": "accuracy"},
    )


def messy_formats(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Numbers as formatted strings, category spelling variants, and -999 sentinels."""
    rng = trap_rng(seed, "messy_formats")
    out = df.copy()
    numeric_col = str(params.get("numeric_column", "income"))
    category_col = str(params.get("category_column", "segment"))
    sentinel_col = str(params.get("sentinel_column", "age"))

    def money(value: float) -> str:
        text = f"{value:,.0f}"
        return f"Rs {text}" if rng.random() < 0.5 else text

    out[numeric_col] = [money(v) for v in out[numeric_col]]

    def variant(value: str) -> str:
        draw = rng.random()
        if draw < 0.2:
            return value.upper()
        if draw < 0.4:
            return f"  {value.title()} "
        return value

    out[category_col] = [variant(v) for v in out[category_col]]
    sentinel_rows = rng.random(len(out)) < float(params.get("sentinel_share", 0.03))
    out[sentinel_col] = np.where(sentinel_rows, -999, out[sentinel_col])
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=f"predict {target}",
        plan_overrides={},
        truth=TrapTruth(
            trap_type="messy_formats",
            expected_checks=[CheckId.A8],
            expected_category=Category.FORMAT_ISSUE,
            min_expected_severity=Severity.WARNING,
            affected_columns=[numeric_col, category_col, sentinel_col],
            expected_remediation="clean_formats",
        ),
    )


def clean_messy_formats(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    return clean_case(df, target, task)


CAUSAL_GOAL = "Does being contacted by phone cause customers to convert?"


def causal_question(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """The goal asks a causal question that a predictive model cannot answer. Detected by
    A11 on the generated report (wired in Week 3); tested on strings in Week 2."""
    return TrapCase(
        df=df.copy(),
        target=target,
        task=task,
        goal_text=str(params.get("goal", CAUSAL_GOAL)),
        plan_overrides={},
        truth=TrapTruth(
            trap_type="causal_question",
            expected_checks=[CheckId.A11],
            expected_category=Category.CAUSAL_CLAIM,
            min_expected_severity=Severity.WARNING,
            affected_columns=[],
            expected_remediation="remove_causal_claim",
        ),
    )


def clean_causal_question(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    return clean_case(df, target, task)
