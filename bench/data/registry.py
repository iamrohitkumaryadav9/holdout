"""The dataset registry (``bench/datasets.toml``): every OpenML dataset the benchmark uses.

Each entry pins the OpenML id and version, the target and task, the licence, the row count,
OpenML's md5 and the sha256 of the downloaded parquet file, the **pool** it belongs to and
how it is prepared (columns OpenML marks as row ids or ignored, target binarisation, a
deterministic subsample for large data).

Pools are disjoint by construction and by test:

* ``dev`` - threshold and prompt tuning, the dev suite and the pilot;
* ``final`` - bases for the frozen planted-trap suite;
* ``real_world`` - the real-world suite (external ground truth); never a trap base.
"""

import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from holdout.core.schemas import Task

DATASETS_PATH = Path(__file__).resolve().parents[1] / "datasets.toml"
MAX_ROWS = 50_000

Pool = Literal["dev", "final", "real_world"]
POOLS: tuple[Pool, ...] = ("dev", "final", "real_world")


class RegistryError(ValueError):
    """The registry is inconsistent (duplicate or overlapping entries, bad fields)."""


@dataclass(frozen=True)
class Subsample:
    """Deterministic reduction to at most ``max_rows`` rows.

    * ``contiguous``: the first ``max_rows`` rows in file order (time-ordered data keeps its
      order);
    * ``groups``: whole groups of ``group_column`` drawn with ``seed`` until the next group
      would exceed ``max_rows`` (repeated entities stay complete);
    * ``stratified``: per-class sampling with ``seed`` (classification only).
    """

    method: Literal["contiguous", "groups", "stratified"]
    max_rows: int = MAX_ROWS
    group_column: str | None = None
    seed: int = 0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_id: int
    version: int
    pool: Pool
    target: str
    task: Task
    licence: str
    rows: int
    md5: str
    sha256: str
    parquet_url: str
    #: Columns OpenML marks as row ids or "ignore" attributes: removed on load.
    drop_columns: tuple[str, ...] = ()
    #: For a classification target built from a multi-class column: these labels become 1,
    #: all others 0 (documented in ``target_note``).
    positive_labels: tuple[str, ...] = ()
    #: Cast a nominal target to numbers (e.g. abalone's ring counts) for regression.
    numeric_target: bool = False
    subsample: Subsample | None = None
    #: Trap bases only (dev, final): remove exact duplicate feature rows (keep the first) and
    #: shuffle with this seed, because many OpenML files are sorted by class. The planted
    #: trap must be the only structure. Real-world datasets are used as distributed.
    deduplicate: bool = False
    shuffle_seed: int | None = None
    target_note: str = ""
    note: str = ""


def load_registry(path: Path = DATASETS_PATH) -> dict[str, DatasetSpec]:
    """Parse and validate the registry.

    Raises:
        RegistryError: on duplicate names or ids, an id or name in two pools, an unknown
            pool or task, a missing hash, or a subsample above the 50,000-row limit.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    specs: list[DatasetSpec] = []
    for entry in raw.get("dataset", []):
        try:
            sub = entry.get("subsample")
            spec = DatasetSpec(
                name=str(entry["name"]),
                openml_id=int(entry["openml_id"]),
                version=int(entry["version"]),
                pool=entry["pool"],
                target=str(entry["target"]),
                task=Task(entry["task"]),
                licence=str(entry["licence"]),
                rows=int(entry["rows"]),
                md5=str(entry["md5"]),
                sha256=str(entry["sha256"]),
                parquet_url=str(entry["parquet_url"]),
                drop_columns=tuple(entry.get("drop_columns", ())),
                positive_labels=tuple(str(v) for v in entry.get("positive_labels", ())),
                numeric_target=bool(entry.get("numeric_target", False)),
                subsample=Subsample(**sub) if sub is not None else None,
                deduplicate=bool(entry.get("deduplicate", False)),
                shuffle_seed=(
                    int(entry["shuffle_seed"]) if entry.get("shuffle_seed") is not None else None
                ),
                target_note=str(entry.get("target_note", "")),
                note=str(entry.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"bad registry entry {entry.get('name')!r}: {exc}") from exc
        specs.append(spec)
    _validate(specs)
    return {s.name: s for s in specs}


def _validate(specs: list[DatasetSpec]) -> None:
    problems: list[str] = []
    for counter, what in (
        (Counter(s.name for s in specs), "name"),
        (Counter(s.openml_id for s in specs), "OpenML id"),
    ):
        problems += [f"{what} {k!r} appears {n} times" for k, n in counter.items() if n > 1]
    for s in specs:
        if s.pool not in POOLS:
            problems.append(f"{s.name}: unknown pool {s.pool!r}")
        if len(s.sha256) != 64:
            problems.append(f"{s.name}: sha256 must be recorded (64 hex characters)")
        if s.subsample is not None:
            if s.subsample.max_rows > MAX_ROWS:
                problems.append(f"{s.name}: subsample above {MAX_ROWS} rows")
            if s.subsample.method == "groups" and not s.subsample.group_column:
                problems.append(f"{s.name}: a groups subsample needs group_column")
            if s.subsample.method == "stratified" and s.task is not Task.CLASSIFICATION:
                problems.append(f"{s.name}: stratified subsampling needs a classification task")
        elif s.rows > MAX_ROWS:
            problems.append(f"{s.name}: {s.rows} rows need a subsample rule (max {MAX_ROWS})")
        if s.positive_labels and s.task is not Task.CLASSIFICATION:
            problems.append(f"{s.name}: positive_labels only apply to classification")
        if s.pool == "real_world" and (s.deduplicate or s.shuffle_seed is not None):
            problems.append(f"{s.name}: real-world datasets are used as distributed")
    if problems:
        raise RegistryError("invalid dataset registry:\n  " + "\n  ".join(problems))


def pool(specs: dict[str, DatasetSpec], name: Pool) -> list[DatasetSpec]:
    return [s for s in specs.values() if s.pool == name]


def require_pool(spec: DatasetSpec, expected: Pool) -> None:
    """Refuse a dataset from another pool (e.g. a real-world dataset as a trap base)."""
    if spec.pool != expected:
        raise RegistryError(
            f"{spec.name} is in the {spec.pool!r} pool; only {expected!r} datasets may be "
            "used here (the pools are disjoint by design)"
        )
