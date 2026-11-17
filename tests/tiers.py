"""Test tiers: ``fast`` (default) and ``full``.

The tier is fixed before collection by ``tests/conftest.py`` from ``--tier`` or the
``HOLDOUT_TEST_TIER`` environment variable, so parametrisations (seeds) can depend on it.
"""

import os

FULL = os.environ.get("HOLDOUT_TEST_TIER", "fast") == "full"

#: Seeds for per-check planted/clean tests: 5 in the full tier, 1 in the fast tier.
FULL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SEEDS: tuple[int, ...] = FULL_SEEDS if FULL else (0,)


def examples(full: int, fast: int) -> int:
    """Hypothesis example count for the current tier."""
    return full if FULL else fast
