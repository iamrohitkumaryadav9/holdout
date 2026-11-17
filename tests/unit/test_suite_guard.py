"""The final-suite builder refuses development bases and seeds (contamination guard)."""

import pandas as pd
import pytest

from bench.suite import (
    ContaminationError,
    SuiteEntry,
    build_final_suite,
    load_dev_usage,
)
from bench.traps.synthetic import BUILDERS, classification
from holdout.core.schemas import Task
from tests.tiers import FULL_SEEDS

HELD_OUT_SEED = 1001


def _loader(calls: list[tuple[str, int]]):  # type: ignore[no-untyped-def]
    def load(base: str, seed: int) -> tuple[pd.DataFrame, str, Task]:
        calls.append((base, seed))
        return classification(n_rows=300, seed=seed)

    return load


def test_every_development_base_and_seed_is_registered() -> None:
    usage = load_dev_usage()
    assert set(BUILDERS) <= usage.bases
    assert set(FULL_SEEDS) <= usage.seeds  # the audit harness seeds (full tier)
    assert {0, 1, 2, 3, 4} <= usage.seeds  # bench script, perf tests, generator tests


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (SuiteEntry("classification", "direct_leak", HELD_OUT_SEED), "base 'classification'"),
        (SuiteEntry("openml:31", "direct_leak", 3), "seed 3"),
        (SuiteEntry("openml:31", "no_such_trap", HELD_OUT_SEED), "unknown trap"),
    ],
)
def test_contaminated_entries_are_refused_before_loading(entry: SuiteEntry, reason: str) -> None:
    calls: list[tuple[str, int]] = []
    with pytest.raises(ContaminationError, match=reason):
        build_final_suite([entry], _loader(calls))
    assert calls == []  # nothing was loaded


def test_one_contaminated_entry_refuses_the_whole_suite() -> None:
    calls: list[tuple[str, int]] = []
    entries = [
        SuiteEntry("openml:31", "direct_leak", HELD_OUT_SEED),
        SuiteEntry("easy_clean", "noisy_proxy", HELD_OUT_SEED + 1),
    ]
    with pytest.raises(ContaminationError, match="easy_clean"):
        build_final_suite(entries, _loader(calls))
    assert calls == []


def test_held_out_entries_are_built() -> None:
    calls: list[tuple[str, int]] = []
    cases = build_final_suite(
        [SuiteEntry("openml:31", "direct_leak", HELD_OUT_SEED)], _loader(calls)
    )
    assert calls == [("openml:31", HELD_OUT_SEED)]
    assert cases[0].truth.trap_type == "direct_leak"
