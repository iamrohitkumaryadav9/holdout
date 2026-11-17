"""Planted suites: frozen hashes, composition, the dev-seed and pool guards, and planting on
real-base shapes (string labels, the semi-synthetic time order)."""

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from bench.data.openml import cached_path
from bench.data.registry import load_registry
from bench.suite import ContaminationError, load_dev_usage
from bench.suites import frozen
from bench.suites.planted import (
    DEV_SEEDS,
    DEV_V1_PATH,
    DEV_V1_TASKS,
    FINAL_SEEDS,
    FINAL_V1_PATH,
    FINAL_V1_TASKS,
    TaskSpec,
    check_suite,
    plant,
    suite_document,
    time_order_real,
    trap_params,
)
from bench.suites.real_world import suite_sha256
from bench.traps.catalog import TRAPS
from holdout.core.schemas import Task


def test_the_frozen_suite_files_have_not_changed() -> None:
    assert suite_sha256(FINAL_V1_PATH) == frozen.FINAL_V1_SHA256
    assert suite_sha256(DEV_V1_PATH) == frozen.DEV_V1_SHA256


def test_the_final_suite_composition() -> None:
    doc = json.loads(FINAL_V1_PATH.read_text(encoding="utf-8"))
    assert (doc["name"], doc["pool"], doc["seeds"]) == ("final_v1", "final", [101, 202])
    tasks = doc["tasks"]
    assert len(tasks) == 40
    traps = Counter(t["trap"] for t in tasks if t["kind"] == "trap")
    assert sum(traps.values()) == 28
    assert set(traps) == set(TRAPS)
    assert sorted(traps.values()) == [3] * 8 + [4]
    assert sum(t["kind"] == "clean" for t in tasks) == 12
    pairs = [(t["trap"], t["base"], t["kind"]) for t in tasks]
    assert len(pairs) == len(set(pairs))
    registry = load_registry()
    for t in tasks:
        assert registry[t["base"]].pool == "final"
        assert t["semi_synthetic"] == (t["trap"] == "time_order")
        assert set(t["frames"]) == {"101", "202"}
        if t["kind"] == "clean":
            assert t["truth"]["expected_category"] is None


def test_the_dev_suite_composition() -> None:
    doc = json.loads(DEV_V1_PATH.read_text(encoding="utf-8"))
    assert doc["seeds"] == list(DEV_SEEDS)
    assert len(doc["tasks"]) == 12
    assert {t["trap"] for t in doc["tasks"] if t["kind"] == "trap"} == set(TRAPS)
    assert any(t["file_order"] for t in doc["tasks"])


def test_dev_bases_and_seeds_are_registered_and_final_ones_are_not() -> None:
    usage = load_dev_usage()
    registry = load_registry()
    dev = {s.name for s in registry.values() if s.pool == "dev"}
    final = {s.name for s in registry.values() if s.pool == "final"}
    assert dev <= usage.bases
    assert not final & usage.bases
    assert not set(FINAL_SEEDS) & usage.seeds
    assert set(DEV_SEEDS) <= usage.seeds


def test_a_dev_seed_or_base_in_the_final_suite_is_refused() -> None:
    usage = load_dev_usage()
    with pytest.raises(ContaminationError, match="seed 11 was used during development"):
        check_suite(FINAL_V1_TASKS[:1], [11], "final", usage)
    with pytest.raises(ContaminationError, match="base 'adult'"):
        check_suite([TaskSpec("direct_leak", "adult")], [101], "final", usage)
    with pytest.raises(ValueError, match="register dev bases and seeds"):
        check_suite(DEV_V1_TASKS[:1], [99], "dev", usage)


def test_the_builder_refuses_a_base_from_another_pool() -> None:
    with pytest.raises(ValueError, match="pools are disjoint"):
        suite_document("x", [TaskSpec("direct_leak", "adult")], [101], "final", usage=_no_usage())


def _no_usage():  # type: ignore[no-untyped-def]
    from bench.suite import DevUsage

    return DevUsage(bases=frozenset(), seeds=frozenset())


# ------------------------------------------------------------------ planting on real shapes


def _base(n: int = 400, labels: tuple[str, str] = ("g", "h")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n) * 1000,
            "c": rng.integers(0, 50, n),
            "y": rng.choice(list(labels), n),
        }
    )


@pytest.mark.parametrize("trap", ["direct_leak", "noisy_proxy", "group_leakage"])
def test_binary_traps_work_with_string_labels(trap: str) -> None:
    frame = _base()
    case = plant(TaskSpec(trap, "x"), frame, "y", Task.CLASSIFICATION, 101)
    assert set(case.df["y"].unique()) <= {"g", "h"}
    if trap == "direct_leak":
        leak = case.df["refund_issued"].to_numpy()
        agree = (leak == (case.df["y"] == "h").to_numpy()).mean()
        assert agree > 0.9


def test_semi_synthetic_time_order_drifts_and_its_twin_does_not() -> None:
    frame = _base(2000)
    planted = time_order_real(frame, "y", Task.CLASSIFICATION, 101, drift=1.0)
    twin = time_order_real(frame, "y", Task.CLASSIFICATION, 101, drift=0.0)
    early, late = planted.df["y"].iloc[:500], planted.df["y"].iloc[-500:]
    assert (late == "h").mean() > (early == "h").mean() + 0.1  # the target trends
    assert planted.df["b"].iloc[-500:].mean() > planted.df["b"].iloc[:500].mean()
    pd.testing.assert_frame_equal(twin.df, frame)  # same timeline, no drift
    assert planted.truth.expected_category is not None
    assert twin.truth.expected_category is None
    regression = _base(1000).assign(y=np.arange(1000.0) % 7)
    trended = time_order_real(regression, "y", Task.REGRESSION, 101, drift=1.0)
    assert trended.df["y"].iloc[-200:].mean() > trended.df["y"].iloc[:200].mean()


def test_column_choices_follow_fixed_rules() -> None:
    frame = _base()
    params = trap_params("messy_formats", frame, "y")
    assert params == {
        "numeric_column": "a",
        "sentinel_column": "b",
        "category_column": None,
        "decimals": 2,
    }
    assert trap_params("causal_question", frame, "y") == {"goal": "Does a cause y?"}


def _datasets_cached() -> bool:
    registry = load_registry()
    return all(cached_path(s).exists() for s in registry.values() if s.pool in ("dev", "final"))


@pytest.mark.full_only
@pytest.mark.skipif(not _datasets_cached(), reason="needs the OpenML files (no network in CI)")
def test_rebuilding_the_suites_reproduces_every_frame() -> None:
    for name, tasks, seeds, pool, path in (
        ("dev_v1", DEV_V1_TASKS, DEV_SEEDS, "dev", DEV_V1_PATH),
        ("final_v1", FINAL_V1_TASKS, FINAL_SEEDS, "final", FINAL_V1_PATH),
    ):
        rebuilt = suite_document(name, tasks, seeds, pool)  # type: ignore[arg-type]
        assert rebuilt == json.loads(path.read_text(encoding="utf-8"))
