"""The benchmark data layer: registry validation, disjoint pools, verified downloads, and
deterministic preparation (no network: downloads are injected)."""

import hashlib
import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.data.openml import (
    DatasetIntegrityError,
    DatasetUnavailableError,
    cached_path,
    ensure_file,
    load_frame,
    prepare,
    subsample,
)
from bench.data.registry import (
    DATASETS_PATH,
    MAX_ROWS,
    DatasetSpec,
    RegistryError,
    Subsample,
    load_registry,
    pool,
    require_pool,
)
from holdout.core.schemas import Task


@pytest.fixture(autouse=True)
def _private_dataset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loader tests write fake files: never into the real dataset cache."""
    monkeypatch.setenv("HOLDOUT_DATASETS", str(tmp_path / "datasets"))


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _spec(data: bytes, **kw: object) -> DatasetSpec:
    base = {
        "name": "toy",
        "openml_id": 999_001,
        "version": 1,
        "pool": "final",
        "target": "y",
        "task": Task.CLASSIFICATION,
        "licence": "Public",
        "rows": 10,
        "md5": "0" * 32,
        "sha256": hashlib.sha256(data).hexdigest(),
        "parquet_url": "https://example.invalid/toy.pq",
    }
    base.update(kw)
    return DatasetSpec(**base)  # type: ignore[arg-type]


TOY = pd.DataFrame({"a": range(10), "b": list("xxyyzzxxyy"), "y": [0, 1] * 5})


# ------------------------------------------------------------------ the registry


def test_the_registry_is_valid_and_its_pools_are_disjoint() -> None:
    specs = load_registry()
    by_pool = {p: pool(specs, p) for p in ("dev", "final", "real_world")}
    assert all(by_pool.values())
    ids = [{s.openml_id for s in group} for group in by_pool.values()]
    names = [{s.name for s in group} for group in by_pool.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not ids[i] & ids[j]
            assert not names[i] & names[j]
    for spec in specs.values():
        assert spec.rows <= MAX_ROWS or spec.subsample is not None
        if spec.pool == "real_world":  # used exactly as distributed
            assert not spec.deduplicate
            assert spec.shuffle_seed is None


def _write(tmp_path: Path, entries: list[str]) -> Path:
    path = tmp_path / "datasets.toml"
    path.write_text("\n\n".join(entries), encoding="utf-8")
    return path


def _entry(name: str, did: int, pool_name: str = "final", **extra: str) -> str:
    fields = {
        "name": f'"{name}"',
        "openml_id": str(did),
        "version": "1",
        "pool": f'"{pool_name}"',
        "target": '"y"',
        "task": '"classification"',
        "licence": '"Public"',
        "rows": "100",
        "md5": f'"{"0" * 32}"',
        "sha256": f'"{"a" * 64}"',
        "parquet_url": '"https://example.invalid/x.pq"',
        **extra,
    }
    return "\n".join(["[[dataset]]", *(f"{k} = {v}" for k, v in fields.items())])


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([_entry("a", 1), _entry("a", 2)], "name 'a' appears 2 times"),
        ([_entry("a", 1, "dev"), _entry("b", 1, "real_world")], "OpenML id 1 appears 2 times"),
        ([_entry("a", 1, "holdout")], "unknown pool"),
        ([_entry("a", 1, rows="60000")], "need a subsample rule"),
        ([_entry("a", 1, "real_world", deduplicate="true")], "used as distributed"),
        ([_entry("a", 1, sha256='"short"')], "sha256 must be recorded"),
    ],
)
def test_an_inconsistent_registry_is_refused(
    tmp_path: Path, entries: list[str], message: str
) -> None:
    with pytest.raises(RegistryError, match=message):
        load_registry(_write(tmp_path, entries))


def test_a_dataset_from_another_pool_is_refused() -> None:
    specs = load_registry()
    real = next(s for s in specs.values() if s.pool == "real_world")
    with pytest.raises(RegistryError, match="pools are disjoint"):
        require_pool(real, "final")
    require_pool(real, "real_world")


def test_the_registry_file_is_the_default() -> None:
    assert DATASETS_PATH.name == "datasets.toml"
    assert DATASETS_PATH.is_file()


# ------------------------------------------------------------------ verified downloads


def test_download_once_then_offline_with_the_hash_checked_every_time() -> None:
    data = _parquet(TOY)
    spec = _spec(data)
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return data

    frame, target, task = load_frame(spec, fetch=fetch)
    assert (len(frame), target, task) == (10, "y", Task.CLASSIFICATION)
    load_frame(spec, fetch=fetch, allow_download=False)  # cached: no second download
    assert calls == [spec.parquet_url]
    cached_path(spec).write_bytes(data + b"tampered")
    with pytest.raises(DatasetIntegrityError, match="cached file"):
        load_frame(spec, allow_download=False)


def test_a_download_with_the_wrong_hash_is_never_stored() -> None:
    spec = _spec(_parquet(TOY), openml_id=999_002)
    with pytest.raises(DatasetIntegrityError, match="downloaded bytes"):
        ensure_file(spec, fetch=lambda url: b"something else")
    assert not cached_path(spec).exists()


def test_offline_without_a_cached_file_fails_clearly() -> None:
    spec = _spec(_parquet(TOY), openml_id=999_003)
    with pytest.raises(DatasetUnavailableError, match="offline"):
        ensure_file(spec, allow_download=False)


# ------------------------------------------------------------------ preparation


def test_preparation_drops_ids_and_builds_the_target() -> None:
    raw = pd.DataFrame(
        {"rid": range(6), "x": [1.0, 2, 3, 4, 5, 6], "y": ["NO", "<30", ">30", "<30", "NO", "NO"]}
    )
    spec = _spec(b"", drop_columns=("rid",), positive_labels=("<30",))
    out = prepare(raw, spec)
    assert list(out.columns) == ["x", "y"]
    assert out["y"].tolist() == [0, 1, 0, 1, 0, 0]
    rings = pd.DataFrame({"x": [1, 2], "y": pd.Categorical(["9", "11"])})
    numeric = prepare(rings, _spec(b"", task=Task.REGRESSION, numeric_target=True))
    assert numeric["y"].tolist() == [9, 11]
    with pytest.raises(DatasetIntegrityError, match="not in the file"):
        prepare(raw, _spec(b"", drop_columns=("missing",)))


def test_trap_bases_are_deduplicated_and_shuffled_deterministically() -> None:
    raw = pd.DataFrame({"a": [1, 1, 2, 3, 4, 5], "b": [7, 7, 8, 9, 9, 9], "y": [0, 1, 0, 1, 0, 1]})
    spec = _spec(b"", deduplicate=True, shuffle_seed=7)
    first, second = prepare(raw, spec), prepare(raw, spec)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5  # the second (1, 7) row is a duplicate feature row
    assert sorted(first["a"]) == [1, 2, 3, 4, 5]
    assert not first.duplicated(subset=["a", "b"]).any()


def test_contiguous_subsampling_keeps_the_first_rows_in_order() -> None:
    frame = pd.DataFrame({"t": range(100), "y": [0, 1] * 50})
    out = subsample(frame, "y", Subsample("contiguous", max_rows=30))
    assert out["t"].tolist() == list(range(30))


def test_group_subsampling_keeps_whole_groups_in_file_order() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), rng.integers(1, 6, 40))
    frame = pd.DataFrame({"g": groups, "i": range(len(groups)), "y": 0})
    rule = Subsample("groups", max_rows=60, group_column="g", seed=3)
    out = subsample(frame, "y", rule)
    assert len(out) <= 60
    kept = out["g"].unique()
    full = frame[frame["g"].isin(kept)]
    assert len(full) == len(out)  # every kept group is complete
    assert out["i"].is_monotonic_increasing
    pd.testing.assert_frame_equal(out, subsample(frame, "y", rule))


def test_stratified_subsampling_keeps_class_shares() -> None:
    frame = pd.DataFrame({"x": range(1000), "y": [1] * 100 + [0] * 900})
    out = subsample(frame, "y", Subsample("stratified", max_rows=500, seed=1))
    assert len(out) == 500
    assert out["y"].mean() == pytest.approx(0.1)


def test_a_spec_change_does_not_alter_the_shuffle_of_another_base() -> None:
    raw = pd.DataFrame({"a": range(20), "y": [0, 1] * 10})
    one = prepare(raw, _spec(b"", shuffle_seed=1))
    other = prepare(raw, replace(_spec(b"", shuffle_seed=1), name="renamed"))
    pd.testing.assert_frame_equal(one, other)  # the seed, not the name, decides
