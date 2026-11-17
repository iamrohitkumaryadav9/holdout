"""The sealed holdout: written once, scored once, every access logged.

This is the mechanism behind "seal before see". Splitting happens *inside*
:meth:`SealedHoldout.seal`, which writes the holdout rows straight to disk and hands the
caller only the development rows, so no code path in the caller ever holds the holdout
DataFrame.

One documented, harness-side exception serves the duplicate audit (A4): at seal time the
seal also stores keyed digests of the holdout's normalised *feature* rows (no labels, no
values). :meth:`SealedHoldout.holdout_overlap` answers only "how many development rows
share a feature row with the holdout", as counts.
"""

import hashlib
import json
import secrets
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from holdout.core import config
from holdout.core.errors import (
    EvaluationError,
    HoldoutAlreadyUsedError,
    HoldoutInvalidError,
    HoldoutTamperedError,
    SealError,
)
from holdout.core.portable import file_sha256
from holdout.core.schemas import HoldoutResult, Scorer
from holdout.core.state import RunState
from holdout.data.rowhash import keyed_digests, row_fingerprints
from holdout.data.split import SplitSpec, apply_split

AccessOutcome = Literal[
    "evaluated",
    "forced",
    "refused_already_used",
    "refused_tampered",
    "refused_invalid",
    "scoring_failed",
    "marked_invalid",
]


class SealRecord(BaseModel):
    """Persistent seal state (``holdout_seal.json``); re-read before every evaluation.

    Fields added in Week 2 have defaults so seals written by earlier versions still load.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    target: str
    holdout_sha256: str
    n_rows: int = Field(ge=1)
    data_sha256: str
    split_spec: SplitSpec
    split_fingerprint: str
    sealed_at: datetime
    evaluations: int = Field(ge=0)
    dev_index_sha256: str | None = None
    row_digest_key: str | None = Field(default=None, description="hex HMAC key, per run")
    row_hash_significant_digits: int | None = None
    holdout_invalid: bool = False
    invalid_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class HoldoutOverlap:
    """Counts only: how much development data duplicates holdout feature rows."""

    dev_rows_matching: int
    holdout_rows_matching: int
    holdout_rows: int


def dev_index_sha256(dev: pd.DataFrame) -> str:
    """Fingerprint of which rows are development rows (sorted original row ids)."""
    ids = ",".join(str(int(i)) for i in sorted(dev.index))
    return hashlib.sha256(ids.encode()).hexdigest()


class SealedHoldout:
    """A holdout set that can be evaluated exactly once per run.

    Guarantees:
      * the holdout rows exist only in ``holdout.parquet``; this object keeps a path and
        a sha256, never the data;
      * :meth:`evaluate` verifies the sha256 before reading, and refuses a modified file;
      * the first :meth:`evaluate` succeeds, later ones raise
        :class:`HoldoutAlreadyUsedError` unless ``force=True``; the count is persisted, so
        a fresh ``SealedHoldout`` for the same run directory cannot reset it;
      * once :meth:`mark_invalid` has recorded that the holdout is contaminated,
        :meth:`evaluate` raises :class:`HoldoutInvalidError` forever, even with ``force``;
      * every call, refused or not, is appended to ``holdout_access.jsonl``; sealing,
        evaluation and invalidation are appended to the cross-run ledger.

    Honest scope: this is a *process-level* guarantee, not cryptographic security. Anyone
    with file-system access can read the parquet file, edit the state file, or start a new
    run on the same data. The seal prevents accidental and silent reuse by the agent and by
    well-behaved code, and makes any deliberate override visible in the logs.
    """

    def __init__(self, state: RunState, reuse_warnings: list[str], caveats: list[str]) -> None:
        """Use :meth:`seal` or :meth:`load`; direct construction skips sealing."""
        self._state = state
        self.reuse_warnings: list[str] = reuse_warnings
        self.caveats: list[str] = caveats

    @classmethod
    def seal(
        cls,
        frame: pd.DataFrame,
        target: str,
        spec: SplitSpec,
        state: RunState,
        data_sha256: str,
    ) -> tuple["SealedHoldout", pd.DataFrame]:
        """Split ``frame``, write and hash the holdout, return (seal, development rows).

        Raises:
            SplitError: if the spec does not fit the data.
            SealError: if this run directory already holds a seal.
        """
        if state.seal_state_path.exists() or state.holdout_path.exists():
            raise SealError(f"run {state.run_id} is already sealed; start a new run instead")
        dev, holdout = apply_split(frame, spec, target)
        holdout.to_parquet(state.holdout_path, index=True)
        key = secrets.token_bytes(32)
        digits = config.ROW_HASH_SIGNIFICANT_DIGITS
        digests = keyed_digests(row_fingerprints(holdout.drop(columns=[target]), digits), key)
        state.holdout_digests_path.write_text(
            json.dumps({"digests": dict(sorted(Counter(digests).items()))}) + "\n",
            encoding="utf-8",
        )
        record = SealRecord(
            run_id=state.run_id,
            target=target,
            holdout_sha256=file_sha256(state.holdout_path),
            n_rows=len(holdout),
            data_sha256=data_sha256,
            split_spec=spec,
            split_fingerprint=spec.fingerprint(),
            sealed_at=datetime.now(UTC),
            evaluations=0,
            dev_index_sha256=dev_index_sha256(dev),
            row_digest_key=key.hex(),
            row_hash_significant_digits=digits,
        )
        _write_record(state.seal_state_path, record)
        seal = cls(state, *_ledger_notes(state.ledger_path, record))
        _append_jsonl(state.ledger_path, _ledger_entry("sealed", record))
        return seal, dev.copy()

    @classmethod
    def load(cls, state: RunState) -> "SealedHoldout":
        """Reattach to the seal of an existing run (the evaluation count is preserved)."""
        record = _read_record(state.seal_state_path)
        return cls(state, *_ledger_notes(state.ledger_path, record))

    @property
    def record(self) -> SealRecord:
        """A fresh copy of the persisted seal state (contains no holdout data)."""
        return _read_record(self._state.seal_state_path)

    @property
    def holdout_sha256(self) -> str:
        return self.record.holdout_sha256

    @property
    def n_rows(self) -> int:
        return self.record.n_rows

    @property
    def evaluations(self) -> int:
        return self.record.evaluations

    @property
    def holdout_invalid(self) -> bool:
        return self.record.holdout_invalid

    def holdout_overlap(self, dev_features: pd.DataFrame) -> HoldoutOverlap | None:
        """Count development rows whose normalised features equal a holdout feature row.

        Returns counts only (never digests, values or labels), or None for seals written
        before row digests were recorded.
        """
        record = self.record
        path = self._state.holdout_digests_path
        if record.row_digest_key is None or record.row_hash_significant_digits is None:
            return None
        if not path.exists():
            raise SealError(f"holdout row digests missing: {path}")
        holdout_counts: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))["digests"]
        dev_digests = keyed_digests(
            row_fingerprints(dev_features, record.row_hash_significant_digits),
            bytes.fromhex(record.row_digest_key),
        )
        dev_set = set(dev_digests)
        return HoldoutOverlap(
            dev_rows_matching=sum(d in holdout_counts for d in dev_digests),
            holdout_rows_matching=sum(c for d, c in holdout_counts.items() if d in dev_set),
            holdout_rows=record.n_rows,
        )

    def mark_invalid(self, reasons: list[str], *, caller: str) -> None:
        """Record permanently that the holdout is contaminated; it can never be scored."""
        record = self.record
        record.holdout_invalid = True
        record.invalid_reasons = sorted(set(record.invalid_reasons) | set(reasons))
        _write_record(self._state.seal_state_path, record)
        self._log("marked_invalid", caller, record, None, None, None, False)
        _append_jsonl(self._state.ledger_path, _ledger_entry("invalidated", record))

    def evaluate(
        self, fitted_pipeline: Any, metric: Scorer, *, caller: str, force: bool = False
    ) -> HoldoutResult:
        """Score ``fitted_pipeline`` on the holdout. The only way to touch holdout rows.

        Args:
            fitted_pipeline: an estimator already fitted on development rows only.
            metric: the metric to compute (from :mod:`holdout.plan.metrics`).
            caller: who is asking, recorded in the access log (e.g. ``"cli:run --unseal"``).
            force: allow a repeat evaluation; it is still logged and flagged as forced.

        Raises:
            HoldoutInvalidError: if the holdout was marked contaminated (``force`` ignored).
            HoldoutAlreadyUsedError: on a repeat call without ``force``.
            HoldoutTamperedError: if the parquet file changed or vanished since sealing.
            EvaluationError: if the pipeline fails on the holdout (this still counts as a
                use, because holdout rows reached the pipeline).
        """
        if not caller.strip():
            raise ValueError("caller must name who is evaluating the holdout")
        record = self.record
        observed = (
            file_sha256(self._state.holdout_path) if self._state.holdout_path.exists() else None
        )
        if record.holdout_invalid:
            self._log("refused_invalid", caller, record, observed, metric, None, force)
            raise HoldoutInvalidError(
                f"the holdout of run {record.run_id} is contaminated "
                f"({'; '.join(record.invalid_reasons)}); it can never be evaluated. "
                "Start a new run with the corrected split."
            )
        if record.evaluations >= 1 and not force:
            self._log("refused_already_used", caller, record, observed, metric, None, force)
            raise HoldoutAlreadyUsedError(
                f"the holdout of run {record.run_id} was already evaluated "
                f"{record.evaluations} time(s); looking again turns it into a validation set. "
                "Pass force=True to override; the override is logged."
            )
        if observed != record.holdout_sha256:
            self._log("refused_tampered", caller, record, observed, metric, None, force)
            raise HoldoutTamperedError(
                f"{self._state.holdout_path} sha256 is {observed or 'missing'}, expected "
                f"{record.holdout_sha256}; the holdout was modified after sealing"
            )

        record.evaluations += 1
        _write_record(self._state.seal_state_path, record)
        holdout = pd.read_parquet(self._state.holdout_path)
        X = holdout.drop(columns=[record.target])
        y = holdout[record.target]
        try:
            value = metric.score(fitted_pipeline, X, y)
        except Exception as exc:
            self._log("scoring_failed", caller, record, observed, metric, None, force)
            raise EvaluationError(f"scoring the holdout failed: {exc}") from exc

        outcome: AccessOutcome = "forced" if record.evaluations > 1 else "evaluated"
        evaluated_at = self._log(outcome, caller, record, observed, metric, value, force)
        entry = _ledger_entry("evaluated", record)
        entry.update({"timestamp": evaluated_at.isoformat(), "metric": metric.name.value})
        entry["score"] = value
        _append_jsonl(self._state.ledger_path, entry)
        return HoldoutResult(
            metric=metric.name,
            score=value,
            n_rows=len(holdout),
            holdout_sha256=record.holdout_sha256,
            evaluation_index=record.evaluations,
            forced=record.evaluations > 1,
            caller=caller,
            evaluated_at=evaluated_at,
        )

    def _log(
        self,
        outcome: AccessOutcome,
        caller: str,
        record: SealRecord,
        observed: str | None,
        metric: Scorer | None,
        value: float | None,
        force: bool,
    ) -> datetime:
        timestamp = datetime.now(UTC)
        _append_jsonl(
            self._state.access_log_path,
            {
                "timestamp": timestamp.isoformat(),
                "run_id": record.run_id,
                "caller": caller,
                "outcome": outcome,
                "force": force,
                "metric": None if metric is None else metric.name.value,
                "score": value,
                "expected_sha256": record.holdout_sha256,
                "observed_sha256": observed,
                "evaluations_after": record.evaluations,
            },
        )
        return timestamp


def _ledger_entry(event: str, record: SealRecord) -> dict[str, Any]:
    return {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": record.run_id,
        "data_sha256": record.data_sha256,
        "split_fingerprint": record.split_fingerprint,
        "holdout_sha256": record.holdout_sha256,
    }


def _ledger_notes(ledger_path: Path, record: SealRecord) -> tuple[list[str], list[str]]:
    """(reuse warnings, caveats) from earlier runs on the same data.

    Reuse warnings: a holdout from this data was already *evaluated*. Caveats: this data was
    already *inspected* (sealed and cross-validated) in an earlier run, so this holdout was
    drawn after inspection; strongest when an earlier holdout was found contaminated.
    """
    if not ledger_path.exists():
        return [], []
    evaluated_same: set[str] = set()
    evaluated_other: set[str] = set()
    inspected_other: set[str] = set()
    invalidated: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("data_sha256") != record.data_sha256 or entry["run_id"] == record.run_id:
            continue
        same_split = entry["split_fingerprint"] == record.split_fingerprint
        event = entry.get("event", "evaluated")
        if event == "evaluated":
            (evaluated_same if same_split else evaluated_other).add(entry["run_id"])
        elif event == "invalidated":
            invalidated.add(entry["run_id"])
        elif event == "sealed" and not same_split:
            inspected_other.add(entry["run_id"])
    warnings: list[str] = []
    if evaluated_same:
        warnings.append(
            f"this exact holdout was already evaluated in run(s) {sorted(evaluated_same)}; "
            "another evaluation would make it a validation set across runs"
        )
    if evaluated_other:
        warnings.append(
            f"a holdout from the same data (different split) was evaluated in run(s) "
            f"{sorted(evaluated_other)}; the holdout rows likely overlap"
        )
    caveats: list[str] = []
    if invalidated:
        caveats.append(
            "this holdout was re-drawn after the data had been inspected (run(s) "
            f"{sorted(invalidated)} had a contaminated holdout); its score may be slightly "
            "optimistic"
        )
    elif inspected_other:
        caveats.append(
            f"this data was already inspected under a different split in run(s) "
            f"{sorted(inspected_other)}; the holdout was drawn after that inspection, so its "
            "score may be slightly optimistic"
        )
    return warnings, caveats


def _read_record(path: Path) -> SealRecord:
    if not path.exists():
        raise SealError(f"no seal found at {path}")
    return SealRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _write_record(path: Path, record: SealRecord) -> None:
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
