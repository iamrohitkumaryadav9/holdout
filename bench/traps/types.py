"""Trap cases and their ground truth."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from holdout.audit.types import Category, CheckId, Severity
from holdout.core.schemas import Task
from holdout.core.seeds import derive_seed


@dataclass(frozen=True)
class TrapTruth:
    """What a correct audit must report for a case.

    Detection = a flag of ``expected_category`` at severity >= ``min_expected_severity``.
    A clean twin has ``expected_category=None``: no flag above info is allowed.
    """

    trap_type: str
    expected_checks: list[CheckId]
    expected_category: Category | None
    min_expected_severity: Severity
    affected_columns: list[str]
    expected_remediation: str | None


@dataclass(frozen=True)
class Provenance:
    """Enough to rebuild the case (and its clean twin) deterministically."""

    trap: str
    base: str
    base_kwargs: dict[str, Any]
    seed: int
    params: dict[str, Any]


@dataclass(frozen=True)
class TrapCase:
    df: pd.DataFrame
    target: str
    task: Task
    goal_text: str
    plan_overrides: dict[str, Any]
    truth: TrapTruth
    provenance: Provenance | None = field(default=None, compare=False)


CLEAN_TRUTH_TYPE = "clean"


def clean_truth() -> TrapTruth:
    return TrapTruth(
        trap_type=CLEAN_TRUTH_TYPE,
        expected_checks=[],
        expected_category=None,
        min_expected_severity=Severity.INFO,
        affected_columns=[],
        expected_remediation=None,
    )


def clean_case(
    df: pd.DataFrame,
    target: str,
    task: Task,
    goal: str | None = None,
    plan_overrides: dict[str, Any] | None = None,
) -> TrapCase:
    """A case with no planted problem: no flag above info is expected."""
    return TrapCase(
        df=df.copy(),
        target=target,
        task=task,
        goal_text=goal if goal is not None else default_goal(target),
        plan_overrides=dict(plan_overrides or {}),
        truth=clean_truth(),
    )


def trap_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, f"bench/trap/{name}"))


def default_goal(target: str) -> str:
    return f"predict {target}"
