"""Split-level traps: duplicated rows, entity leakage, hidden time order."""

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


def cross_split_duplicates(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Duplicate a share of rows (exact, or with float noise far below 6 significant digits)
    and shuffle, so copies land in training, validation and holdout alike."""
    rng = trap_rng(seed, "cross_split_duplicates")
    share = float(params.get("share", 0.15))
    variant = str(params.get("variant", "exact"))
    copies = df.iloc[rng.choice(len(df), size=round(share * len(df)), replace=False)].copy()
    if variant == "near":
        for column in copies.columns:
            if column != target and pd.api.types.is_float_dtype(copies[column]):
                copies[column] = copies[column] * (1 + rng.normal(0, 1e-10, len(copies)))
    elif variant != "exact":
        raise ValueError(f"unknown duplicate variant {variant!r}")
    out = pd.concat([df, copies], ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=default_goal(target),
        plan_overrides={},
        truth=TrapTruth(
            trap_type="cross_split_duplicates",
            expected_checks=[CheckId.A4],
            expected_category=Category.DUPLICATE_ROWS,
            min_expected_severity=Severity.CRITICAL,
            affected_columns=[],
            expected_remediation="deduplicate",
        ),
    )


def _entity_frame(
    df: pd.DataFrame, seed: int, params: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = trap_rng(seed, "group_leakage/entities")
    rows_per_entity = int(params.get("rows_per_entity", 6))
    column = str(params.get("column", "account"))
    n_entities = max(30, len(df) // rows_per_entity)
    entity = rng.integers(0, n_entities, len(df))
    region = rng.choice([f"r{i}" for i in range(8)], n_entities)
    vintage = rng.integers(2005, 2024, n_entities)
    band = rng.choice(["a", "b", "c", "d", "e"], n_entities)
    out = df.copy()
    position = len(out.columns) - 1
    out.insert(position, column, np.array([f"acct_{i:04d}" for i in range(n_entities)])[entity])
    out.insert(position + 1, "region_code", region[entity])
    out.insert(position + 2, "account_vintage", vintage[entity])
    out.insert(position + 3, "credit_band", band[entity])
    return out, entity


def group_leakage(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Rows belong to entities with a strong entity-level target effect, entity-constant
    features and a neutral-named entity key; a row-level split lets the model memorise
    entities."""
    out, entity = _entity_frame(df, seed, params)
    rng = trap_rng(seed, "group_leakage/bias")
    strength = float(params.get("strength", 0.8))
    n_entities = int(entity.max()) + 1
    y = out[target].to_numpy()
    if task is Task.CLASSIFICATION:
        # Works for 0/1 and for string labels: each entity draws one of the two labels with
        # the base's positive rate (identical draws to the 0/1 version).
        positive = positive_indicator(out[target])
        labels = sorted(pd.unique(out[target]), key=str)
        entity_positive = rng.random(n_entities) < positive.mean()
        entity_label = np.where(entity_positive, labels[-1], labels[0])
        take = rng.random(len(out)) < strength
        out[target] = np.where(take, entity_label[entity], y)
        if pd.api.types.is_integer_dtype(y.dtype):
            out[target] = out[target].astype(y.dtype)
    else:
        effect = rng.normal(0, 2.0 * strength * np.std(y), n_entities)
        out[target] = np.round(y + effect[entity], 2)
    column = str(params.get("column", "account"))
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=default_goal(target),
        plan_overrides={},
        truth=TrapTruth(
            trap_type="group_leakage",
            expected_checks=[CheckId.A5],
            expected_category=Category.GROUP_LEAKAGE,
            min_expected_severity=Severity.CRITICAL,
            affected_columns=[column],
            expected_remediation="change_split",
        ),
    )


def clean_group_leakage(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Same entities and entity-constant features, but no entity-level target effect."""
    out, _ = _entity_frame(df, seed, params)
    return clean_case(out, target, task)


def time_order(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Time-ordered data with drift; the date column is removed (or renamed) and the rows
    stay in time order. Apply to ``synthetic.time_ordered(drift>0)``."""
    out = _hide_time(df, params)
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=default_goal(target),
        plan_overrides={},
        truth=TrapTruth(
            trap_type="time_order",
            expected_checks=[CheckId.A6],
            expected_category=Category.TIME_LEAKAGE,
            min_expected_severity=Severity.CRITICAL,
            affected_columns=[] if params.get("mode", "remove") == "remove" else ["period"],
            expected_remediation="change_split",
        ),
    )


def clean_time_order(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    """Same transformation on a drift-free base (the catalog rebuilds it with drift=0)."""
    return clean_case(_hide_time(df, params), target, task)


def _hide_time(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    mode = str(params.get("mode", "remove"))
    if mode == "remove":
        return df.drop(columns=["date"])
    if mode == "rename":
        return df.rename(columns={"date": "period"})
    raise ValueError(f"unknown time_order mode {mode!r}")


def clean_cross_split_duplicates(
    df: pd.DataFrame, target: str, task: Task, seed: int, params: dict[str, Any]
) -> TrapCase:
    return clean_case(df, target, task)
