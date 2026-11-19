"""Normalised feature-row fingerprints, shared by the seal and the duplicate audit (A4).

Two rows are "the same" when their *features* match after normalisation: floats rounded
to a fixed number of significant digits, strings stripped and lower-cased, missing values
unified. The label is never part of a fingerprint.

Fingerprints are 64-bit and stay in memory. What the seal *stores* for the holdout is an
HMAC-SHA256 of each fingerprint under a per-run random key, so the stored digests cannot be
reversed by enumerating likely rows without that key.
"""

import hashlib
import hmac

import numpy as np
import pandas as pd

# The normalisation lives in the portable kernel (v1.1) so an exported pipeline.py can
# deduplicate development rows exactly as the run did; the code is unchanged, so seal
# digests are unchanged.
from holdout.core.portable import normalise_features, round_significant

__all__ = ["keyed_digests", "normalise_features", "round_significant", "row_fingerprints"]


def row_fingerprints(features: pd.DataFrame, significant_digits: int) -> np.ndarray:
    """One unsigned 64-bit fingerprint per row of normalised features."""
    normalised = normalise_features(features, significant_digits)
    return np.asarray(pd.util.hash_pandas_object(normalised, index=False), dtype="uint64")


def keyed_digests(fingerprints: np.ndarray, key: bytes) -> list[str]:
    """HMAC-SHA256 (truncated to 128 bits, hex) of each fingerprint under ``key``."""
    return [
        hmac.new(key, int(fp).to_bytes(8, "little"), hashlib.sha256).hexdigest()[:32]
        for fp in fingerprints
    ]
