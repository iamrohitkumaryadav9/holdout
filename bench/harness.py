"""The benchmark harness's seal: two holdouts per unit, scored by the harness only.

Every system receives only the development partition. The harness keeps two holdouts in
``<results>/seals/<suite>/<unit>/`` (outside every system's work directory), pins their
sha256 and is the only code that reads them; :meth:`BenchSeal.evaluate` verifies the hash,
scores one fitted model with Holdout's own scorer (``portable.score``) and logs the access.

* **holdout_iid**: a random (stratified for classification) sample of rows from the same
  population as the development rows, leaks intact. For entity and time traps it is drawn
  from the development entities / the earlier time block, so it shares them with training.
* **holdout_deploy**: deployment-faithful, by a rule per trap type (``DEPLOY_RULES`` and
  docs/benchmark.md):

  - entity traps (planted ``group_leakage``) and Diabetes: rows of unseen entities;
  - time traps (planted, semi-synthetic) and time-ordered real cases: the latest block;
  - ``cross_split_duplicates``: the iid rows minus rows whose features occur in development;
  - ``direct_leak``, ``noisy_proxy``, ``id_with_signal``: the iid rows with the leak column
    permuted within the holdout (the column would not exist when predicting; permuting
    keeps types and values, so no missing values crash a valid pipeline, and removes only
    the link to the target);
  - everything else, every clean twin and every clean real-world control: identical to iid.

Both holdouts are disjoint from the same development rows, so every system trains once
and is scored on both. Rules are deterministic per seed and identical for every system;
systems are not told about them.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from holdout.core import portable
from holdout.core.schemas import Task
from holdout.core.seeds import derive_seed
from holdout.llm.filelock import locked

HOLDOUT_FRACTION = 0.2

DeployKind = Literal["iid", "group", "time"]
Holdout = Literal["iid", "deploy"]

#: How the deploy holdout differs from the iid holdout, per planted trap type.
DEPLOY_RULES: Final[dict[str, str]] = {
    "direct_leak": "the leak column is permuted within the holdout",
    "noisy_proxy": "the proxy column is permuted within the holdout",
    "id_with_signal": "the identifier column is permuted within the holdout",
    "cross_split_duplicates": "holdout rows whose features also occur in development are removed",
    "group_leakage": "rows of entities that never occur in development (unseen entities)",
    "time_order": "the latest 20% of rows in timeline order (the future)",
    "imbalance_accuracy": "identical to iid",
    "messy_formats": "identical to iid",
    "causal_question": "identical to iid",
}
LEAK_TRAPS: Final = frozenset({"direct_leak", "noisy_proxy", "id_with_signal"})


@dataclass(frozen=True)
class PartitionRule:
    """How a unit's development rows and its two holdouts are drawn."""

    deploy: DeployKind = "iid"
    group_column: str | None = None
    dedupe_against_dev: bool = False
    #: Leak columns that would not exist when predicting (permuted in holdout_deploy).
    unavailable_columns: tuple[str, ...] = ()
    stratify: bool = True

    def describe(self) -> str:
        parts = ["iid: random" + (" stratified" if self.stratify else "")]
        if self.deploy == "group":
            parts.append(f"deploy: unseen {self.group_column}")
        elif self.deploy == "time":
            parts.append("deploy: latest block")
        if self.dedupe_against_dev:
            parts.append("deploy: rows duplicated in dev removed")
        if self.unavailable_columns:
            parts.append("deploy: permuted " + ",".join(self.unavailable_columns))
        if len(parts) == 1:
            parts.append("deploy = iid")
        return "; ".join(parts)

    @property
    def deploy_is_iid(self) -> bool:
        return self.deploy == "iid" and not self.dedupe_against_dev and not self.unavailable_columns


def planted_rule(
    trap: str, task: Task, affected_columns: list[str], *, clean: bool = False
) -> PartitionRule:
    """The rule for a planted task. Clean twins: deploy == iid (they have nothing to fix)."""
    stratify = task is Task.CLASSIFICATION
    if clean:
        return PartitionRule(stratify=stratify)
    if trap == "group_leakage":
        column = (affected_columns or ["account"])[0]
        return PartitionRule("group", group_column=column, stratify=stratify)
    if trap == "time_order":
        return PartitionRule("time", stratify=stratify)
    return PartitionRule(
        dedupe_against_dev=trap == "cross_split_duplicates",
        unavailable_columns=tuple(affected_columns) if trap in LEAK_TRAPS else (),
        stratify=stratify,
    )


def real_world_rule(deploy: str, task: Task) -> PartitionRule:
    """From a case's recorded ``deploy_holdout``: ``iid``, ``time`` or ``group:COLUMN``."""
    stratify = task is Task.CLASSIFICATION
    if deploy.startswith("group:"):
        return PartitionRule("group", group_column=deploy.removeprefix("group:"), stratify=stratify)
    if deploy == "time":
        return PartitionRule("time", stratify=stratify)
    if deploy == "iid":
        return PartitionRule(stratify=stratify)
    raise ValueError(f"unknown deploy holdout {deploy!r}")


def _sample(
    positions: np.ndarray, labels: np.ndarray | None, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """A (stratified) random sample of ``positions``."""
    if labels is None:
        take = round(fraction * len(positions))
        return rng.choice(positions, size=take, replace=False)
    picked = []
    for label in np.unique(labels[positions]):
        members = positions[labels[positions] == label]
        picked.append(rng.choice(members, size=round(fraction * len(members)), replace=False))
    return np.concatenate(picked) if picked else np.array([], dtype=int)


def partition(
    frame: pd.DataFrame, target: str, rule: PartitionRule, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(development, holdout_iid, holdout_deploy), deterministic for ``seed``."""
    n = len(frame)
    rng = np.random.default_rng(derive_seed(seed, "bench/partition"))
    labels = frame[target].astype(str).to_numpy() if rule.stratify else None
    deploy_mask = np.zeros(n, dtype=bool)
    if rule.deploy == "time":
        deploy_mask[n - round(HOLDOUT_FRACTION * n) :] = True
    elif rule.deploy == "group":
        assert rule.group_column is not None
        groups = frame[rule.group_column].astype(str).to_numpy()
        target_rows, total, chosen = round(HOLDOUT_FRACTION * n), 0, []
        for g in rng.permutation(np.unique(groups)):
            if total >= target_rows:
                break
            chosen.append(g)
            total += int((groups == g).sum())
        deploy_mask = np.isin(groups, chosen)
    rest = np.flatnonzero(~deploy_mask)
    iid_mask = np.zeros(n, dtype=bool)
    iid_mask[_sample(rest, labels, HOLDOUT_FRACTION, rng)] = True
    dev = frame[~(iid_mask | deploy_mask)].reset_index(drop=True)
    iid = frame[iid_mask].reset_index(drop=True)
    deploy = iid.copy() if rule.deploy == "iid" else frame[deploy_mask].reset_index(drop=True)
    if rule.dedupe_against_dev:
        features = [c for c in frame.columns if c != target]
        seen = set(map(tuple, dev[features].astype(str).to_numpy()))
        keep = [tuple(row) not in seen for row in deploy[features].astype(str).to_numpy()]
        deploy = deploy[np.asarray(keep, dtype=bool)].reset_index(drop=True)
    for column in rule.unavailable_columns:
        if column in deploy.columns:  # same values, no link to the target
            order = np.random.default_rng(derive_seed(seed, f"bench/unavailable/{column}"))
            deploy[column] = deploy[column].to_numpy()[order.permutation(len(deploy))]
    return dev, iid, deploy


def benchmark_metric(task: Task) -> str:
    """The common metric every system's holdout predictions are scored with."""
    return "roc_auc" if task is Task.CLASSIFICATION else "rmse"


class SealError(RuntimeError):
    """The benchmark seal was tampered with, or a model could not be scored."""


@dataclass
class BenchSeal:
    """The two holdouts of one unit. Only :meth:`evaluate` reads them."""

    directory: Path
    target: str
    sha256: dict[str, str]
    rows: dict[str, int]

    @classmethod
    def create(
        cls, directory: Path, iid: pd.DataFrame, deploy: pd.DataFrame, target: str
    ) -> "BenchSeal":
        directory.mkdir(parents=True, exist_ok=True)
        digests, rows = {}, {}
        for name, frame in (("iid", iid), ("deploy", deploy)):
            path = directory / f"holdout_{name}.parquet"
            frame.to_parquet(path, index=False)
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows[name] = len(frame)
        meta = {"sha256": digests, "rows": rows, "target": target}
        (directory / "seal.json").write_text(json.dumps(meta), encoding="utf-8")
        return cls(directory, target, digests, rows)

    @classmethod
    def load(cls, directory: Path) -> "BenchSeal":
        meta = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
        return cls(directory, meta["target"], dict(meta["sha256"]), dict(meta["rows"]))

    def path(self, which: Holdout) -> Path:
        return self.directory / f"holdout_{which}.parquet"

    def evaluate(self, estimator: Any, metric: str, *, caller: str, which: Holdout) -> float:
        """Score a fitted model on one holdout (hash-checked, logged)."""
        data = self.path(which).read_bytes()
        if hashlib.sha256(data).hexdigest() != self.sha256[which]:
            raise SealError(f"the benchmark {which} holdout in {self.directory} was modified")
        holdout = pd.read_parquet(self.path(which))
        X = holdout.drop(columns=[self.target])
        y = holdout[self.target]
        try:
            value = float(portable.score(metric, estimator, X, y))
        except Exception as exc:  # a system's model may fail on unseen data
            self._log(caller, which, metric, None, f"{type(exc).__name__}: {exc}")
            raise SealError(f"scoring {caller}'s model on {which} failed: {exc}") from exc
        self._log(caller, which, metric, value, None)
        return value

    def access_log(self) -> list[dict[str, Any]]:
        path = self.directory / "access.jsonl"
        if not path.exists():
            return []
        with locked(path):
            text = path.read_text("utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    def _log(
        self, caller: str, which: str, metric: str, value: float | None, error: str | None
    ) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "caller": caller,
            "holdout": which,
            "metric": metric,
            "score": value,
            "error": error,
        }
        path = self.directory / "access.jsonl"
        # Units of different systems share a seal and may score at once (--workers).
        with locked(path), path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")
