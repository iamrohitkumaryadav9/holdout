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

_MISSING = "\x00<missing>"


def normalise_features(features: pd.DataFrame, significant_digits: int) -> pd.DataFrame:
    """Return a normalised copy: columns sorted by name, floats rounded, strings folded."""
    columns = sorted(features.columns, key=str)
    out: dict[str, pd.Series] = {}
    for column in columns:
        values = features[column]
        if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
            out[str(column)] = pd.Series(
                round_significant(values.astype("float64").to_numpy(), significant_digits),
                index=features.index,
            )
        else:
            text = values.astype(str).str.strip().str.lower()
            out[str(column)] = text.where(values.notna(), _MISSING)
    return pd.DataFrame(out, index=features.index)


def round_significant(values: np.ndarray, digits: int) -> np.ndarray:
    """Round to ``digits`` significant digits; NaN and infinities are kept as they are."""
    result = np.array(values, dtype="float64", copy=True)
    finite = np.isfinite(result) & (result != 0.0)
    magnitude = np.floor(np.log10(np.abs(result[finite])))
    factor = np.power(10.0, digits - 1 - magnitude)
    result[finite] = np.round(result[finite] * factor) / factor
    return result + 0.0  # turn -0.0 into 0.0 so both hash alike


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
