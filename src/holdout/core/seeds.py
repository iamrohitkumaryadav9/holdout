"""Stable seed derivation.

Every ``random_state`` in Holdout is derived from the single run seed through
:func:`derive_seed`. Python's built-in ``hash()`` is salted per process for strings, so it
would silently break reproducibility across runs; sha256 is stable across processes,
platforms and Python versions.
"""

import hashlib
from typing import Final

#: Namespace prefix: changing the derivation scheme must change every derived seed.
_SCHEME: Final = "holdout/seed/v1"

#: Upper bound (inclusive) of a derived seed. 31 bits fit every consumer we use:
#: numpy ``RandomState`` (32-bit unsigned) and LightGBM's C ``int`` seed (32-bit signed).
MAX_SEED: Final = 2**31 - 1

#: The named random streams of a run, recorded in the manifest.
SEED_NAMES: Final[tuple[str, ...]] = (
    "split",
    "cv",
    "model",
    "target_encoder",
    "feature_selection",
)


def derive_seed(run_seed: int, name: str) -> int:
    """Derive the seed for the random stream ``name`` from ``run_seed``.

    Distinct names give independent-looking seeds, so the split, the CV folds and the
    model never share a random stream by accident.

    Raises:
        TypeError: if ``run_seed`` is not an ``int`` (``bool`` is rejected explicitly,
            because ``True`` silently acting as seed 1 hides a bug).
        ValueError: if ``name`` is empty.
    """
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError(f"run_seed must be an int, got {type(run_seed).__name__}")
    if not name:
        raise ValueError("seed stream name must be a non-empty string")
    digest = hashlib.sha256(f"{_SCHEME}:{run_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & MAX_SEED


def derived_seeds(run_seed: int) -> dict[str, int]:
    """Return every named seed of a run, for the manifest."""
    return {name: derive_seed(run_seed, name) for name in SEED_NAMES}
