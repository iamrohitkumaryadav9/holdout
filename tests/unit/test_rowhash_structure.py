"""Row fingerprints (normalisation rules) and structural entity detection."""

import numpy as np
import pandas as pd
import pytest

from holdout.data.rowhash import keyed_digests, round_significant, row_fingerprints
from holdout.data.structure import entity_candidates


def test_round_significant() -> None:
    values = np.array([123456.789, 0.000123456789, -0.0, np.nan, np.inf, 1.0000001])
    rounded = round_significant(values, 6)
    assert rounded[0] == pytest.approx(123457.0)
    assert rounded[1] == pytest.approx(0.000123457)
    assert rounded[2] == 0.0
    assert not np.signbit(rounded[2])
    assert np.isnan(rounded[3])
    assert np.isinf(rounded[4])
    assert rounded[5] == 1.0


def test_fingerprints_normalise_floats_strings_and_column_order() -> None:
    a = pd.DataFrame({"x": [1.0, 2.5], "s": ["  Mumbai", "pune"], "n": [None, "a"]})
    b = pd.DataFrame({"s": ["mumbai ", "PUNE"], "n": [None, "a"], "x": [1.0 + 1e-12, 2.5]})
    assert list(row_fingerprints(a, 6)) == list(row_fingerprints(b, 6))
    c = a.assign(x=[1.001, 2.5])
    assert row_fingerprints(c, 6)[0] != row_fingerprints(a, 6)[0]
    assert row_fingerprints(c, 6)[1] == row_fingerprints(a, 6)[1]


def test_missing_differs_from_the_string_nan() -> None:
    a = pd.DataFrame({"s": [None]})
    b = pd.DataFrame({"s": ["nan"]})
    assert row_fingerprints(a, 6)[0] != row_fingerprints(b, 6)[0]


def test_keyed_digests_depend_on_the_key() -> None:
    fps = row_fingerprints(pd.DataFrame({"x": [1.0, 2.0]}), 6)
    first = keyed_digests(fps, b"k" * 32)
    assert first == keyed_digests(fps, b"k" * 32)
    assert first != keyed_digests(fps, b"j" * 32)
    assert all(len(d) == 32 for d in first)


def _entities(n_entities: int = 40, rows: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat([f"acct_{i:03d}" for i in range(n_entities)], rows)
    return pd.DataFrame(
        {
            "account": entity,
            "plan": np.repeat(rng.choice(["a", "b", "c"], n_entities), rows),
            "x": rng.normal(size=n_entities * rows),
            "color": rng.choice(["r", "g", "b"], n_entities * rows),
            "y": rng.integers(0, 2, n_entities * rows),
        }
    )


def test_entity_candidates_ranks_entity_keys_first() -> None:
    found = entity_candidates(
        _entities(),
        exclude=["y"],
        min_mean_rows_per_value=2.0,
        min_groups=20,
        max_unique_ratio=0.5,
        max_candidates=3,
    )
    assert [c.column for c in found] == ["account"]
    assert found[0].n_groups == 40
    assert found[0].constant_feature_share == pytest.approx(1 / 3)  # plan is constant


def test_entity_candidates_skip_floats_low_cardinality_and_unique_columns() -> None:
    frame = _entities()
    frame["row"] = np.arange(len(frame))
    assert [
        c.column
        for c in entity_candidates(
            frame,
            exclude=["y", "account"],
            min_mean_rows_per_value=2.0,
            min_groups=20,
            max_unique_ratio=0.5,
            max_candidates=3,
        )
    ] == []
    assert (
        entity_candidates(
            frame.iloc[:0],
            [],
            min_mean_rows_per_value=2,
            min_groups=2,
            max_unique_ratio=1,
            max_candidates=1,
        )
        == []
    )
