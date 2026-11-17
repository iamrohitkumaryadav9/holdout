"""The sealed holdout: written once, scored once, every access logged.

This is the mechanism behind "seal before see". Splitting happens *inside*
:meth:`SealedHoldout.seal`, which writes the holdout rows straight to disk and hands the
caller only the development rows, so no code path in the caller ever holds the holdout
DataFrame.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from holdout.core.errors import (
    EvaluationError,
    HoldoutAlreadyUsedError,
    HoldoutTamperedError,
    SealError,
)
from holdout.core.portable import file_sha256
from holdout.core.schemas import HoldoutResult, Scorer
from holdout.core.state import RunState
from holdout.data.split import SplitSpec, apply_split

AccessOutcome = Literal[
    "evaluated", "forced", "refused_already_used", "refused_tampered", "scoring_failed"
]


class SealRecord(BaseModel):
    """Persistent seal state (``holdout_seal.json``); re-read before every evaluation."""

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


class SealedHoldout:
    """A holdout set that can be evaluated exactly once per run.

    Guarantees:
      * the holdout rows exist only in ``holdout.parquet``; this object keeps a path and
        a sha256, never the data;
      * :meth:`evaluate` verifies the sha256 before reading, and refuses a modified file;
      * the first :meth:`evaluate` succeeds, later ones raise
        :class:`HoldoutAlreadyUsedError` unless ``force=True``; the count is persisted, so
        a fresh ``SealedHoldout`` for the same run directory cannot reset it;
      * every call, refused or not, is appended to ``holdout_access.jsonl``, and every
        evaluation to the cross-run ledger, which warns when the same data's holdout was
        already scored in an earlier run.

    Honest scope: this is a *process-level* guarantee, not cryptographic security. Anyone
    with file-system access can read the parquet file, edit the state file, or start a new
    run on the same data. The seal prevents accidental and silent reuse by the agent and by
    well-behaved code, and makes any deliberate override visible in the logs.
    """

    def __init__(self, state: RunState, reuse_warnings: list[str]) -> None:
        """Use :meth:`seal` or :meth:`load`; direct construction skips sealing."""
        self._state = state
        self.reuse_warnings: list[str] = reuse_warnings

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
        )
        _write_record(state.seal_state_path, record)
        seal = cls(state, _ledger_warnings(state.ledger_path, record))
        return seal, dev.copy()

    @classmethod
    def load(cls, state: RunState) -> "SealedHoldout":
        """Reattach to the seal of an existing run (the evaluation count is preserved)."""
        record = _read_record(state.seal_state_path)
        return cls(state, _ledger_warnings(state.ledger_path, record))

    @property
    def holdout_sha256(self) -> str:
        return _read_record(self._state.seal_state_path).holdout_sha256

    @property
    def n_rows(self) -> int:
        return _read_record(self._state.seal_state_path).n_rows

    @property
    def evaluations(self) -> int:
        return _read_record(self._state.seal_state_path).evaluations

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
            HoldoutAlreadyUsedError: on a repeat call without ``force``.
            HoldoutTamperedError: if the parquet file changed or vanished since sealing.
            EvaluationError: if the pipeline fails on the holdout (this still counts as a
                use, because holdout rows reached the pipeline).
        """
        if not caller.strip():
            raise ValueError("caller must name who is evaluating the holdout")
        record = _read_record(self._state.seal_state_path)
        observed = (
            file_sha256(self._state.holdout_path) if self._state.holdout_path.exists() else None
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
        _append_jsonl(
            self._state.ledger_path,
            {
                "event": "evaluated",
                "timestamp": evaluated_at.isoformat(),
                "run_id": record.run_id,
                "data_sha256": record.data_sha256,
                "split_fingerprint": record.split_fingerprint,
                "holdout_sha256": record.holdout_sha256,
                "metric": metric.name.value,
                "score": value,
            },
        )
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
        metric: Scorer,
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
                "metric": metric.name.value,
                "score": value,
                "expected_sha256": record.holdout_sha256,
                "observed_sha256": observed,
                "evaluations_after": record.evaluations,
            },
        )
        return timestamp


def _ledger_warnings(ledger_path: Path, record: SealRecord) -> list[str]:
    """Warn when an earlier run already scored a holdout carved from the same data."""
    if not ledger_path.exists():
        return []
    same_split: set[str] = set()
    other_split: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("data_sha256") != record.data_sha256 or entry["run_id"] == record.run_id:
            continue
        target = (
            same_split if entry["split_fingerprint"] == record.split_fingerprint else other_split
        )
        target.add(entry["run_id"])
    warnings: list[str] = []
    if same_split:
        warnings.append(
            f"this exact holdout was already evaluated in run(s) {sorted(same_split)}; "
            "another evaluation would make it a validation set across runs"
        )
    if other_split:
        warnings.append(
            f"a holdout from the same data (different split) was evaluated in run(s) "
            f"{sorted(other_split)}; the holdout rows likely overlap"
        )
    return warnings


def _read_record(path: Path) -> SealRecord:
    if not path.exists():
        raise SealError(f"no seal found at {path}")
    return SealRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _write_record(path: Path, record: SealRecord) -> None:
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
