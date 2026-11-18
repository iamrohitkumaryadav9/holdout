"""The benchmark harness's seal: one oracle partition per unit, scored once per system.

Every system receives only the development partition. The harness keeps the holdout in
``<results>/seals/<unit>/holdout.parquet`` (outside every system's work directory),
pins its sha256, and is the only code that reads it: :meth:`BenchSeal.evaluate` verifies
the hash, scores one fitted model with Holdout's own scorer (``portable.score``) and logs
the access. Systems never see the path.

The partition is an **oracle**: it follows the deployment framing given by the ground
truth, so the holdout measures true generalisation for every system alike:

* entity traps (planted ``group_leakage``) and Diabetes hold out unseen entities;
* time-order traps (planted, semi-synthetic) and time-ordered real cases hold out the
  latest 20% of rows;
* duplicate traps drop holdout rows whose features also occur in development (deployment
  data does not contain copies of training rows);
* everything else: a stratified (classification) or random 20% holdout.

Each system then chooses its own internal validation; the optimism gap compares that
claim with the oracle holdout.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from holdout.core import portable
from holdout.core.schemas import Task
from holdout.core.seeds import derive_seed

HOLDOUT_FRACTION = 0.2

PartitionKind = Literal["stratified", "random", "group", "time"]


@dataclass(frozen=True)
class PartitionRule:
    kind: PartitionKind
    group_column: str | None = None
    dedupe_against_dev: bool = False

    def describe(self) -> str:
        base = f"group:{self.group_column}" if self.kind == "group" else self.kind
        return base + (
            " + holdout rows duplicated in dev removed" if self.dedupe_against_dev else ""
        )


def planted_rule(trap: str, task: Task, affected_columns: list[str]) -> PartitionRule:
    """The oracle rule for a planted task (and its clean twin, so the pair is comparable)."""
    if trap == "group_leakage":
        return PartitionRule("group", group_column=(affected_columns or ["account"])[0])
    if trap == "time_order":
        return PartitionRule("time")
    default: PartitionKind = "stratified" if task is Task.CLASSIFICATION else "random"
    return PartitionRule(default, dedupe_against_dev=trap == "cross_split_duplicates")


def real_world_rule(remediations: list[str], task: Task) -> PartitionRule:
    """Group if any documented issue is fixed by a group split, else time, else default."""
    for remedy in remediations:
        if remedy.startswith("split:group:"):
            return PartitionRule("group", group_column=remedy.removeprefix("split:group:"))
    if any(r.startswith("split:time:") for r in remediations):
        return PartitionRule("time")
    return PartitionRule("stratified" if task is Task.CLASSIFICATION else "random")


def partition(
    frame: pd.DataFrame, target: str, rule: PartitionRule, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(development, holdout), deterministic for ``seed``."""
    n = len(frame)
    rng = np.random.default_rng(derive_seed(seed, "bench/partition"))
    n_holdout = round(HOLDOUT_FRACTION * n)
    if rule.kind == "time":
        mask = np.zeros(n, dtype=bool)
        mask[n - n_holdout :] = True
    elif rule.kind == "group":
        assert rule.group_column is not None
        groups = frame[rule.group_column].astype(str).to_numpy()
        unique = np.unique(groups)
        chosen, total = set(), 0
        for g in rng.permutation(unique):
            if total >= n_holdout:
                break
            chosen.add(g)
            total += int((groups == g).sum())
        mask = np.isin(groups, list(chosen))
    elif rule.kind == "stratified":
        mask = np.zeros(n, dtype=bool)
        labels = frame[target].astype(str).to_numpy()
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            take = round(HOLDOUT_FRACTION * len(positions))
            mask[rng.choice(positions, size=take, replace=False)] = True
    else:
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, size=n_holdout, replace=False)] = True
    dev, holdout = frame[~mask], frame[mask]
    if rule.dedupe_against_dev:
        features = [c for c in frame.columns if c != target]
        seen = set(map(tuple, dev[features].astype(str).to_numpy()))
        keep = [tuple(row) not in seen for row in holdout[features].astype(str).to_numpy()]
        holdout = holdout[np.asarray(keep, dtype=bool)]
    return dev.reset_index(drop=True), holdout.reset_index(drop=True)


def benchmark_metric(task: Task) -> str:
    """The common metric every system's holdout predictions are scored with."""
    return "roc_auc" if task is Task.CLASSIFICATION else "rmse"


class SealError(RuntimeError):
    """The benchmark seal was tampered with, or a model could not be scored."""


@dataclass
class BenchSeal:
    """The holdout of one unit. Only :meth:`evaluate` reads it."""

    directory: Path
    target: str
    sha256: str
    n_rows: int

    @classmethod
    def create(cls, directory: Path, holdout: pd.DataFrame, target: str) -> "BenchSeal":
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "holdout.parquet"
        holdout.to_parquet(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        seal = cls(directory, target, digest, len(holdout))
        (directory / "seal.json").write_text(
            json.dumps({"sha256": digest, "target": target, "rows": len(holdout)}), "utf-8"
        )
        return seal

    @classmethod
    def load(cls, directory: Path) -> "BenchSeal":
        meta = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
        return cls(directory, meta["target"], meta["sha256"], int(meta["rows"]))

    @property
    def path(self) -> Path:
        return self.directory / "holdout.parquet"

    def evaluate(self, estimator: Any, metric: str, *, caller: str) -> float:
        """Score a fitted model on the holdout (hash-checked, logged)."""
        data = self.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise SealError(f"the benchmark holdout in {self.directory} was modified")
        holdout = pd.read_parquet(self.path)
        X = holdout.drop(columns=[self.target])
        y = holdout[self.target]
        try:
            value = float(portable.score(metric, estimator, X, y))
        except Exception as exc:  # a system's model may fail on unseen data
            self._log(caller, metric, None, f"{type(exc).__name__}: {exc}")
            raise SealError(f"scoring {caller}'s model failed: {exc}") from exc
        self._log(caller, metric, value, None)
        return value

    def access_log(self) -> list[dict[str, Any]]:
        path = self.directory / "access.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]

    def _log(self, caller: str, metric: str, value: float | None, error: str | None) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "caller": caller,
            "metric": metric,
            "score": value,
            "error": error,
        }
        with (self.directory / "access.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
