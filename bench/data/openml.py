"""Download OpenML parquet files once, verify their sha256 on every load, prepare frames.

Plain HTTPS and pyarrow only (no ``openml`` package and its compiled dependencies). Files
live in ``~/.holdout/datasets/<id>-v<version>.pq`` (``HOLDOUT_DATASETS`` overrides it).
After the first download everything works offline. A file whose sha256 differs from the
registry is an error and is never silently replaced.
"""

import hashlib
import io
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from bench.data.registry import DatasetSpec, Subsample
from holdout.core.schemas import Task

Fetch = Callable[[str], bytes]


class DatasetIntegrityError(RuntimeError):
    """A downloaded or cached dataset file does not match its recorded sha256."""


class DatasetUnavailableError(RuntimeError):
    """The dataset is not cached and downloading is not allowed."""


def datasets_dir() -> Path:
    """``HOLDOUT_DATASETS`` if set, else ``~/.holdout/datasets``.

    Deliberately independent of ``HOLDOUT_HOME`` (which tests point at a temporary
    directory to isolate the ledger and response cache): the files are read-only and
    verified by sha256 on every load, so sharing them is safe.
    """
    override = os.environ.get("HOLDOUT_DATASETS")
    return Path(override) if override else Path.home() / ".holdout" / "datasets"


def cached_path(spec: DatasetSpec) -> Path:
    return datasets_dir() / f"{spec.openml_id}-v{spec.version}.pq"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def http_fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "holdout-benchmark"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data: bytes = response.read()
    return data


def ensure_file(
    spec: DatasetSpec, *, fetch: Fetch | None = None, allow_download: bool = True
) -> Path:
    """The verified local file for ``spec``, downloading it once if needed.

    Raises:
        DatasetIntegrityError: when the cached or downloaded bytes do not match.
        DatasetUnavailableError: when the file is not cached and downloads are off.
    """
    path = cached_path(spec)
    if path.exists():
        actual = sha256_file(path)
        if actual != spec.sha256:
            raise DatasetIntegrityError(
                f"{spec.name}: cached file {path} has sha256 {actual}, registry says "
                f"{spec.sha256}; delete it to re-download, or update the registry deliberately"
            )
        return path
    if not allow_download:
        raise DatasetUnavailableError(f"{spec.name} is not cached at {path} (offline)")
    data = (fetch or http_fetch)(spec.parquet_url)
    actual = hashlib.sha256(data).hexdigest()
    if actual != spec.sha256:
        raise DatasetIntegrityError(
            f"{spec.name}: downloaded bytes have sha256 {actual}, registry says {spec.sha256}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(path)  # atomic on the same volume
    return path


def load_frame(
    spec: DatasetSpec, *, fetch: Fetch | None = None, allow_download: bool = True
) -> tuple[pd.DataFrame, str, Task]:
    """The prepared frame: row-id/ignore columns removed, target prepared, subsampled."""
    path = ensure_file(spec, fetch=fetch, allow_download=allow_download)
    raw = pd.read_parquet(io.BytesIO(path.read_bytes()))
    return prepare(raw, spec), spec.target, spec.task


def prepare(raw: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    missing = [c for c in (spec.target, *spec.drop_columns) if c not in raw.columns]
    if missing:
        raise DatasetIntegrityError(f"{spec.name}: columns {missing} not in the file")
    frame = raw.drop(columns=list(spec.drop_columns)).reset_index(drop=True)
    target = frame[spec.target]
    if spec.positive_labels:
        frame[spec.target] = target.astype(str).isin(spec.positive_labels).astype(int)
    elif spec.numeric_target:
        frame[spec.target] = pd.to_numeric(target.astype(str), errors="raise")
    elif isinstance(target.dtype, pd.CategoricalDtype):
        frame[spec.target] = target.astype(str)
    if spec.deduplicate:
        features = frame.drop(columns=[spec.target])
        frame = frame[~features.duplicated(keep="first")].reset_index(drop=True)
    if spec.subsample is not None and len(frame) > spec.subsample.max_rows:
        frame = subsample(frame, spec.target, spec.subsample)
    if spec.shuffle_seed is not None:
        order = np.random.default_rng(spec.shuffle_seed).permutation(len(frame))
        frame = frame.iloc[order].reset_index(drop=True)
    return frame


def subsample(frame: pd.DataFrame, target: str, rule: Subsample) -> pd.DataFrame:
    """Deterministic, documented reduction (see :class:`Subsample`)."""
    if rule.method == "contiguous":
        return frame.iloc[: rule.max_rows].reset_index(drop=True)
    rng = np.random.default_rng(rule.seed)
    if rule.method == "groups":
        assert rule.group_column is not None
        sizes = frame.groupby(rule.group_column, sort=True).size()
        order = rng.permutation(sizes.index.to_numpy())
        kept, total = [], 0
        for group in order:
            size = int(sizes[group])
            if total + size > rule.max_rows:
                continue
            kept.append(group)
            total += size
        mask = frame[rule.group_column].isin(set(kept))
        return frame[mask].reset_index(drop=True)  # file order kept
    fraction = rule.max_rows / len(frame)
    picked: list[np.ndarray] = []
    for _, index in sorted(
        frame.groupby(target, sort=True).groups.items(), key=lambda kv: str(kv[0])
    ):
        positions = np.asarray(index)
        n = int(np.floor(len(positions) * fraction))
        picked.append(rng.choice(positions, size=n, replace=False))
    return frame.loc[np.sort(np.concatenate(picked))].reset_index(drop=True)
