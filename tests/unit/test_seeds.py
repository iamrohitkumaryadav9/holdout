"""derive_seed must be stable across processes and separate distinct streams."""

import subprocess
import sys

import pytest

from holdout.core.seeds import MAX_SEED, SEED_NAMES, derive_seed, derived_seeds


def test_derive_seed_is_deterministic_and_in_range() -> None:
    seed = derive_seed(42, "split")
    assert seed == derive_seed(42, "split")
    assert 0 <= seed <= MAX_SEED


def test_distinct_names_and_run_seeds_give_distinct_seeds() -> None:
    seeds = derived_seeds(42)
    assert set(seeds) == set(SEED_NAMES)
    assert len(set(seeds.values())) == len(SEED_NAMES)
    assert derive_seed(42, "split") != derive_seed(43, "split")


def test_derive_seed_is_stable_across_processes() -> None:
    # Python's hash() is salted per process; derive_seed must not be.
    code = "from holdout.core.seeds import derive_seed; print(derive_seed(42, 'split'))"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert outputs == {str(derive_seed(42, "split"))}


def test_known_value_pins_the_derivation_scheme() -> None:
    # Changing the scheme silently would change every past split; these pins make it loud.
    assert derive_seed(42, "split") == 2013321071
    assert derive_seed(0, "cv") == 301692499


@pytest.mark.parametrize("bad_seed", [True, 1.5, "42"])
def test_rejects_non_int_seed(bad_seed: object) -> None:
    with pytest.raises(TypeError, match="run_seed must be an int"):
        derive_seed(bad_seed, "split")  # type: ignore[arg-type]


def test_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        derive_seed(1, "")
