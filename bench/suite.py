"""Final benchmark suite builder with a contamination guard.

The audit thresholds were tuned on synthetic trap cases (``bench/dev_usage.toml``). A
benchmark that reuses those bases or seeds would grade the thresholds on the data that
set them. :func:`build_final_suite` therefore refuses any entry whose base *or* seed is
registered as used during development, before loading anything.
"""

import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bench.traps.catalog import TRAPS
from bench.traps.types import TrapCase
from holdout.core.schemas import Task

DEV_USAGE_PATH = Path(__file__).resolve().parent / "dev_usage.toml"

BaseLoader = Callable[[str, int], tuple[pd.DataFrame, str, Task]]


class ContaminationError(ValueError):
    """A final-suite entry reuses a base dataset or seed used during development."""


@dataclass(frozen=True)
class DevUsage:
    bases: frozenset[str]
    seeds: frozenset[int]


@dataclass(frozen=True)
class SuiteEntry:
    base: str
    trap: str
    seed: int


def load_dev_usage(path: Path = DEV_USAGE_PATH) -> DevUsage:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return DevUsage(
        bases=frozenset(str(b["name"]) for b in data.get("base", [])),
        seeds=frozenset(int(s) for s in data.get("seeds", [])),
    )


def check_not_contaminated(entries: Sequence[SuiteEntry], usage: DevUsage) -> None:
    """Raise :class:`ContaminationError` naming every entry that reuses development data."""
    problems = []
    for entry in entries:
        reasons = []
        if entry.base in usage.bases:
            reasons.append(f"base {entry.base!r} was used during development")
        if entry.seed in usage.seeds:
            reasons.append(f"seed {entry.seed} was used during development")
        if entry.trap not in TRAPS:
            reasons.append(f"unknown trap {entry.trap!r}")
        if reasons:
            problems.append(f"{entry}: " + "; ".join(reasons))
    if problems:
        raise ContaminationError(
            "the final suite must use held-out bases and seeds only:\n  " + "\n  ".join(problems)
        )


def build_final_suite(
    entries: Sequence[SuiteEntry],
    load_base: BaseLoader,
    usage: DevUsage | None = None,
) -> list[TrapCase]:
    """Build the final benchmark cases, refusing contaminated entries before any loading."""
    check_not_contaminated(entries, usage if usage is not None else load_dev_usage())
    cases = []
    for entry in entries:
        frame, target, task = load_base(entry.base, entry.seed)
        spec = TRAPS[entry.trap]
        cases.append(spec.apply(frame, target, task, entry.seed, dict(spec.params)))
    return cases
