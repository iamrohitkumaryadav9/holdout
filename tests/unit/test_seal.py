"""The seal: one evaluation per run, tamper detection, complete access log, ledger."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from holdout.core.errors import (
    EvaluationError,
    HoldoutAlreadyUsedError,
    HoldoutTamperedError,
    SealError,
)
from holdout.core.schemas import SplitStrategy
from holdout.core.state import RunState
from holdout.data.seal import SealedHoldout
from holdout.data.split import SplitSpec
from tests.helpers import AccuracyScorer

TARGET = "churned"
SPEC = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=11)


def _seal(
    df: pd.DataFrame, out: Path, data_sha: str = "data-sha"
) -> tuple[SealedHoldout, pd.DataFrame, RunState]:
    state = RunState.create(out, run_seed=11)
    seal, dev = SealedHoldout.seal(df, TARGET, SPEC, state, data_sha256=data_sha)
    return seal, dev, state


def _fitted(dev: pd.DataFrame) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent").fit(dev.drop(columns=[TARGET]), dev[TARGET])


def _log(state: RunState) -> list[dict[str, object]]:
    lines = state.access_log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_seal_returns_only_dev_rows_and_writes_hashed_holdout(
    churn_df: pd.DataFrame, tmp_path: Path
) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    holdout = pd.read_parquet(state.holdout_path)
    assert set(dev.index).isdisjoint(holdout.index)
    assert len(dev) + len(holdout) == len(churn_df)
    assert seal.n_rows == len(holdout)
    assert len(seal.holdout_sha256) == 64
    assert seal.evaluations == 0
    assert not hasattr(seal, "holdout")  # no attribute exposes the rows


def test_first_evaluation_succeeds_and_second_raises(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, _ = _seal(churn_df, tmp_path)
    result = seal.evaluate(_fitted(dev), accuracy_scorer, caller="test")
    assert result.evaluation_index == 1
    assert not result.forced
    assert 0.5 < result.score < 1.0
    with pytest.raises(HoldoutAlreadyUsedError, match="already evaluated 1 time"):
        seal.evaluate(_fitted(dev), accuracy_scorer, caller="test-again")


def test_force_allows_repeat_and_flags_it(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    seal.evaluate(_fitted(dev), accuracy_scorer, caller="first")
    forced = seal.evaluate(_fitted(dev), accuracy_scorer, caller="second", force=True)
    assert forced.forced
    assert forced.evaluation_index == 2
    assert [e["outcome"] for e in _log(state)] == ["evaluated", "forced"]


def test_reloaded_seal_keeps_the_evaluation_count(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    seal.evaluate(_fitted(dev), accuracy_scorer, caller="first")
    reattached = SealedHoldout.load(state)
    with pytest.raises(HoldoutAlreadyUsedError):
        reattached.evaluate(_fitted(dev), accuracy_scorer, caller="new-process")


def test_tampered_parquet_raises(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    holdout = pd.read_parquet(state.holdout_path)
    holdout.iloc[0, 0] = -999
    holdout.to_parquet(state.holdout_path, index=True)
    with pytest.raises(HoldoutTamperedError, match="modified after sealing"):
        seal.evaluate(_fitted(dev), accuracy_scorer, caller="test")
    assert seal.evaluations == 0  # a refused call does not consume the evaluation


def test_deleted_parquet_raises(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    state.holdout_path.unlink()
    with pytest.raises(HoldoutTamperedError, match="missing"):
        seal.evaluate(_fitted(dev), accuracy_scorer, caller="test")


def test_access_log_records_every_attempt(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    seal.evaluate(_fitted(dev), accuracy_scorer, caller="cli")
    with pytest.raises(HoldoutAlreadyUsedError):
        seal.evaluate(_fitted(dev), accuracy_scorer, caller="sneaky")
    state.holdout_path.write_bytes(b"tampered")
    with pytest.raises(HoldoutTamperedError):
        seal.evaluate(_fitted(dev), accuracy_scorer, caller="forced", force=True)

    entries = _log(state)
    assert [e["outcome"] for e in entries] == [
        "evaluated",
        "refused_already_used",
        "refused_tampered",
    ]
    assert [e["caller"] for e in entries] == ["cli", "sneaky", "forced"]
    for entry in entries:
        assert entry["expected_sha256"] == seal.holdout_sha256
        assert entry["timestamp"]
    assert entries[0]["score"] is not None
    assert entries[1]["score"] is None
    assert entries[2]["observed_sha256"] != seal.holdout_sha256


def test_failed_scoring_counts_as_a_use(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, _, state = _seal(churn_df, tmp_path)
    with pytest.raises(EvaluationError, match="scoring the holdout failed"):
        seal.evaluate(DummyClassifier(), accuracy_scorer, caller="unfitted")  # never fitted
    assert seal.evaluations == 1
    assert _log(state)[-1]["outcome"] == "scoring_failed"


def test_caller_is_required(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, _ = _seal(churn_df, tmp_path)
    with pytest.raises(ValueError, match="caller"):
        seal.evaluate(_fitted(dev), accuracy_scorer, caller="  ")


def test_run_cannot_be_sealed_twice(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    _, _, state = _seal(churn_df, tmp_path)
    with pytest.raises(SealError, match="already sealed"):
        SealedHoldout.seal(churn_df, TARGET, SPEC, state, data_sha256="x")


def test_load_without_seal_raises(tmp_path: Path) -> None:
    with pytest.raises(SealError, match="no seal found"):
        SealedHoldout.load(RunState.create(tmp_path, 1))


def test_ledger_warns_about_reuse_across_runs(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, _ = _seal(churn_df, tmp_path)
    assert seal.reuse_warnings == []
    first = seal.evaluate(_fitted(dev), accuracy_scorer, caller="run-1")
    assert first.evaluation_index == 1

    same, _, _ = _seal(churn_df, tmp_path)
    assert any("this exact holdout was already evaluated" in w for w in same.reuse_warnings)

    state = RunState.create(tmp_path, run_seed=99)
    other_spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=99)
    other, _ = SealedHoldout.seal(churn_df, TARGET, other_spec, state, data_sha256="data-sha")
    assert any("rows likely overlap" in w for w in other.reuse_warnings)

    unrelated, _, _ = _seal(churn_df, tmp_path, data_sha="other-data")
    assert unrelated.reuse_warnings == []
