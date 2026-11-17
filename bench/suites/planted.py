"""Planted-trap suites on OpenML bases: the dev suite and the frozen final suite.

A task is one trap (or one clean twin) on one base. It is instantiated per seed: the seed
drives the planting randomness and is also the run seed. Bases come from the registry pool
the suite belongs to (``dev`` or ``final``); the existing contamination guard refuses any
final entry whose base or seed is registered in ``bench/dev_usage.toml``.

Trap parameters that name columns (messy formats, a causal goal) are chosen from each base
deterministically, by rules written here, so the suite file fully determines every frame.

``time_order`` on a real base is **semi-synthetic**: the (already shuffled) row order
becomes a hidden timeline, drift is injected along it (a trend in the target, two features
that move with time), the rows stay in timeline order and no time column is kept. Its clean
twin keeps the same timeline without drift. It is labelled semi-synthetic wherever results
are shown, so it is never confused with the genuinely time-ordered real-world cases.
"""

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from bench.data.openml import load_frame
from bench.data.registry import DatasetSpec, Pool, load_registry, require_pool
from bench.suite import DevUsage, SuiteEntry, check_not_contaminated, load_dev_usage
from bench.traps.catalog import TRAPS
from bench.traps.types import TrapCase, TrapTruth, clean_case, default_goal, trap_rng
from holdout.audit.types import Category, CheckId, Severity
from holdout.core.schemas import Task

SUITES_DIR = Path(__file__).resolve().parent
FINAL_V1_PATH = SUITES_DIR / "final_v1.json"
DEV_V1_PATH = SUITES_DIR / "dev_v1.json"
FINAL_SEEDS = (101, 202)
DEV_SEEDS = (11, 12)

Kind = Literal["trap", "clean"]


@dataclass(frozen=True)
class TaskSpec:
    """One suite task: a trap, or the clean twin of a trap, on a base."""

    trap: str
    base: str
    kind: Kind = "trap"
    #: Dev only: keep the file's original order (no shuffle), e.g. a class-sorted control.
    file_order: bool = False

    @property
    def id(self) -> str:
        suffix = "" if self.kind == "trap" else ":clean"
        order = ":file_order" if self.file_order else ""
        return f"{self.trap}{suffix}@{self.base}{order}"

    @property
    def semi_synthetic(self) -> bool:
        return self.trap == "time_order"


# ----------------------------------------------------------------------------- suites

#: 28 trap tasks (direct_leak on 4 bases, every other trap on 3) and 12 clean controls on
#: the final bases; no base carries the same trap twice.
FINAL_V1_TASKS: tuple[TaskSpec, ...] = (
    *(TaskSpec("direct_leak", b) for b in ("qsar_biodeg", "telco_churn", "cpu_act", "pol")),
    *(TaskSpec("noisy_proxy", b) for b in ("magic_telescope", "house_16h", "steel_plates_fault")),
    *(TaskSpec("id_with_signal", b) for b in ("credit_card_default", "wilt", "puma32h")),
    *(
        TaskSpec("cross_split_duplicates", b)
        for b in ("steel_plates_fault", "cpu_act", "magic_telescope")
    ),
    *(TaskSpec("group_leakage", b) for b in ("telco_churn", "house_16h", "qsar_biodeg")),
    *(TaskSpec("time_order", b) for b in ("credit_card_default", "puma32h", "wilt")),
    *(TaskSpec("imbalance_accuracy", b) for b in ("magic_telescope", "pol", "steel_plates_fault")),
    *(TaskSpec("messy_formats", b) for b in ("house_16h", "credit_card_default", "cpu_act")),
    *(TaskSpec("causal_question", b) for b in ("wilt", "puma32h", "telco_churn")),
    # Clean controls: one twin per base, plus two more twin types.
    TaskSpec("direct_leak", "qsar_biodeg", "clean"),
    TaskSpec("group_leakage", "telco_churn", "clean"),
    TaskSpec("cross_split_duplicates", "cpu_act", "clean"),
    TaskSpec("imbalance_accuracy", "pol", "clean"),
    TaskSpec("imbalance_accuracy", "magic_telescope", "clean"),
    TaskSpec("group_leakage", "house_16h", "clean"),
    TaskSpec("noisy_proxy", "steel_plates_fault", "clean"),
    TaskSpec("id_with_signal", "credit_card_default", "clean"),
    TaskSpec("time_order", "wilt", "clean"),
    TaskSpec("time_order", "puma32h", "clean"),
    TaskSpec("causal_question", "telco_churn", "clean"),
    TaskSpec("time_order", "credit_card_default", "clean"),
)

#: 9 traps (every type once) and 3 clean controls on the dev bases, including a
#: class-sorted file kept in its original order (for tuning A6's row-order logic).
DEV_V1_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("direct_leak", "adult"),
    TaskSpec("group_leakage", "adult"),
    TaskSpec("imbalance_accuracy", "adult"),
    TaskSpec("time_order", "adult"),
    TaskSpec("id_with_signal", "credit_approval"),
    TaskSpec("messy_formats", "credit_approval"),
    TaskSpec("causal_question", "credit_approval"),
    TaskSpec("noisy_proxy", "kin8nm"),
    TaskSpec("cross_split_duplicates", "kin8nm"),
    TaskSpec("direct_leak", "adult", "clean"),
    TaskSpec("direct_leak", "credit_approval", "clean", file_order=True),
    TaskSpec("time_order", "kin8nm", "clean"),
)


# ----------------------------------------------------------------------------- planting


def base_frame(
    spec: DatasetSpec, *, file_order: bool = False, allow_download: bool = False
) -> tuple[pd.DataFrame, str, Task]:
    """The prepared base; ``file_order`` skips the shuffle (dev controls only)."""
    if file_order:
        spec = dataclasses.replace(spec, shuffle_seed=None)
    return load_frame(spec, allow_download=allow_download)


def _positive(y: pd.Series) -> Any:
    """The label treated as positive for binary planting: the larger one in sort order."""
    labels = sorted(y.unique(), key=str)
    if len(labels) != 2:
        raise ValueError(f"binary planting needs two classes, got {len(labels)}")
    return labels[-1]


def _numeric_features(frame: pd.DataFrame, target: str) -> list[str]:
    return [
        str(c)
        for c in frame.columns
        if c != target and pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique() > 10
    ]


def trap_params(trap: str, frame: pd.DataFrame, target: str) -> dict[str, Any]:
    """Column choices for traps that need them, by fixed rules (recorded in the suite)."""
    if trap == "messy_formats":
        numeric = _numeric_features(frame, target)
        if len(numeric) < 2:
            raise ValueError("messy_formats needs two numeric features")
        by_spread = sorted(numeric, key=lambda c: (-frame[c].nunique(), c))
        objects = [
            str(c)
            for c in frame.columns
            if c != target
            and not pd.api.types.is_numeric_dtype(frame[c])
            and frame[c].nunique() <= 50
        ]
        return {
            "numeric_column": by_spread[0],
            "sentinel_column": by_spread[1],
            "category_column": sorted(objects)[0] if objects else None,
            "decimals": 0 if frame[by_spread[0]].abs().median() >= 1000 else 2,
        }
    if trap == "causal_question":
        candidates = _numeric_features(frame, target) or [
            str(c) for c in frame.columns if c != target
        ]
        feature = sorted(candidates)[0]
        return {"goal": f"Does {feature} cause {target}?"}
    return {}


def plant(task: TaskSpec, frame: pd.DataFrame, target: str, kind: Task, seed: int) -> TrapCase:
    """Build one task's case on a prepared base frame."""
    params = {**TRAPS[task.trap].params, **trap_params(task.trap, frame, target)}
    if task.trap == "time_order":
        drift = 1.0 if task.kind == "trap" else 0.0
        return time_order_real(frame, target, kind, seed, drift)
    spec = TRAPS[task.trap]
    fn = spec.apply if task.kind == "trap" else spec.clean
    return fn(frame, target, kind, seed, params)


def time_order_real(
    frame: pd.DataFrame, target: str, task: Task, seed: int, drift: float
) -> TrapCase:
    """Semi-synthetic time order on a real base (see the module docs).

    The timeline is the shuffled base's row order. With ``drift > 0``:

    * the target trends along the timeline (classification: negatives become positive
      with probability ``0.35 * drift * t``; regression: ``+ drift * std * t``);
    * the two most varied numeric features shift by ``drift * std * (t - 0.5)`` plus
      noise, so a model can tell where in time a row sits (what a random split exploits).

    Rows stay in timeline order; no time column exists. ``drift = 0`` is the clean twin.
    """
    rng = trap_rng(seed, "time_order_real")
    out = frame.copy().reset_index(drop=True)
    t = np.linspace(0.0, 1.0, len(out))
    if drift > 0:
        y = out[target]
        if task is Task.CLASSIFICATION:
            positive = _positive(y)
            negatives = (y != positive).to_numpy()
            flip = negatives & (rng.random(len(out)) < 0.35 * drift * t)
            out.loc[flip, target] = positive
        else:
            values = y.to_numpy(dtype="float64")
            out[target] = values + drift * np.std(values) * t
        for column in sorted(
            _numeric_features(frame, target), key=lambda c: (-frame[c].nunique(), c)
        )[:2]:
            values = out[column].to_numpy(dtype="float64")
            scale = float(np.std(values)) or 1.0
            out[column] = values + drift * scale * (t - 0.5) + rng.normal(0, 0.25 * scale, len(out))
    truth = TrapTruth(
        trap_type="time_order",
        expected_checks=[CheckId.A6],
        expected_category=Category.TIME_LEAKAGE,
        min_expected_severity=Severity.CRITICAL,
        affected_columns=[],
        expected_remediation="change_split",
    )
    if drift == 0:
        return clean_case(out, target, task)
    return TrapCase(
        df=out,
        target=target,
        task=task,
        goal_text=default_goal(target),
        plan_overrides={},
        truth=truth,
    )


# ----------------------------------------------------------------------------- building


def frame_sha256(frame: pd.DataFrame) -> str:
    """A stable fingerprint of a frame's contents (CSV text with full float precision)."""
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BuiltTask:
    spec: TaskSpec
    seed: int
    case: TrapCase
    params: dict[str, Any] = field(default_factory=dict)


def build_task(
    task: TaskSpec,
    seed: int,
    pool_name: Pool,
    *,
    registry: dict[str, DatasetSpec] | None = None,
    allow_download: bool = False,
) -> BuiltTask:
    registry = registry if registry is not None else load_registry()
    spec = registry[task.base]
    require_pool(spec, pool_name)
    if task.file_order and pool_name != "dev":
        raise ValueError("file-order tasks exist only in the dev suite")
    frame, target, kind = base_frame(
        spec, file_order=task.file_order, allow_download=allow_download
    )
    case = plant(task, frame, target, kind, seed)
    return BuiltTask(task, seed, case, trap_params(task.trap, frame, target))


def check_suite(
    tasks: Sequence[TaskSpec], seeds: Sequence[int], pool_name: Pool, usage: DevUsage
) -> None:
    """Final: no dev base or seed (the existing guard). Dev: every base and seed registered."""
    if pool_name == "final":
        check_not_contaminated([SuiteEntry(t.base, t.trap, s) for t in tasks for s in seeds], usage)
        return
    unregistered = sorted({t.base for t in tasks} - usage.bases) + [
        f"seed {s}" for s in seeds if s not in usage.seeds
    ]
    if unregistered:
        raise ValueError(
            "register dev bases and seeds in bench/dev_usage.toml before using them: "
            + ", ".join(map(str, unregistered))
        )


def suite_document(
    name: str,
    tasks: Sequence[TaskSpec],
    seeds: Sequence[int],
    pool_name: Pool,
    *,
    usage: DevUsage | None = None,
    registry: dict[str, DatasetSpec] | None = None,
) -> dict[str, Any]:
    """Build every task at every seed and describe it (spec, truth, frame hashes)."""
    usage = usage if usage is not None else load_dev_usage()
    check_suite(tasks, seeds, pool_name, usage)
    registry = registry if registry is not None else load_registry()
    entries = []
    for task in tasks:
        frames: dict[str, Any] = {}
        record: dict[str, Any] = {}
        for seed in seeds:
            built = build_task(task, seed, pool_name, registry=registry)
            truth = built.case.truth
            frames[str(seed)] = {
                "sha256": frame_sha256(built.case.df),
                "rows": len(built.case.df),
                "columns": len(built.case.df.columns),
            }
            record = {
                "id": task.id,
                "trap": task.trap,
                "base": task.base,
                "kind": task.kind,
                "file_order": task.file_order,
                "semi_synthetic": task.semi_synthetic,
                "base_sha256": registry[task.base].sha256,
                "target": built.case.target,
                "task": built.case.task.value,
                "goal": built.case.goal_text,
                "plan_overrides": built.case.plan_overrides,
                "params": built.params,
                "truth": {
                    "trap_type": truth.trap_type,
                    "expected_category": (
                        truth.expected_category.value if truth.expected_category else None
                    ),
                    "min_expected_severity": truth.min_expected_severity.value,
                    "affected_columns": truth.affected_columns,
                    "expected_remediation": truth.expected_remediation,
                },
            }
        entries.append({**record, "frames": frames})
    return {
        "name": name,
        "schema_version": 1,
        "pool": pool_name,
        "seeds": list(seeds),
        "tasks": entries,
    }


def write_suite(document: dict[str, Any], path: Path) -> str:
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
