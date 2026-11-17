"""Week 2 seal additions: the A4 hash exception, invalidation, caveats, dev fingerprint."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from holdout.core.errors import HoldoutInvalidError, SealError
from holdout.core.schemas import SplitStrategy
from holdout.core.state import RunState
from holdout.data.seal import SealedHoldout, dev_index_sha256
from holdout.data.split import SplitSpec
from tests.helpers import AccuracyScorer

TARGET = "churned"
SPEC = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=11)


def _seal(
    df: pd.DataFrame, out: Path, spec: SplitSpec = SPEC
) -> tuple[SealedHoldout, pd.DataFrame, RunState]:
    state = RunState.create(out, run_seed=11)
    seal, dev = SealedHoldout.seal(df, TARGET, spec, state, data_sha256="data")
    return seal, dev, state


def test_stored_digests_never_expose_holdout_values_or_labels(
    churn_df: pd.DataFrame, tmp_path: Path
) -> None:
    seal, _, state = _seal(churn_df, tmp_path)
    stored = json.loads(state.holdout_digests_path.read_text(encoding="utf-8"))
    assert set(stored) == {"digests"}
    digests = stored["digests"]
    assert sum(digests.values()) == seal.n_rows
    for digest, count in digests.items():
        assert len(digest) == 32
        assert int(digest, 16) >= 0
        assert isinstance(count, int)
    holdout = pd.read_parquet(state.holdout_path)
    raw = state.holdout_digests_path.read_text(encoding="utf-8")
    for column in holdout.columns:
        for value in holdout[column].dropna().astype(str).unique()[:20]:
            if len(value) >= 4:  # short tokens could appear inside hex by chance
                assert value not in raw, f"holdout value {value!r} leaked into the digests"
    record = json.loads(state.seal_state_path.read_text(encoding="utf-8"))
    assert TARGET not in json.dumps(record["split_spec"])  # record holds no labels
    assert len(bytes.fromhex(record["row_digest_key"])) == 32


def test_overlap_returns_counts_only(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    clean = seal.holdout_overlap(dev.drop(columns=[TARGET]))
    assert clean is not None
    assert (clean.dev_rows_matching, clean.holdout_rows_matching) == (0, 0)
    holdout = pd.read_parquet(state.holdout_path)  # the test peeks to plant duplicates
    leaked = pd.concat([dev, holdout.iloc[:5]]).drop(columns=[TARGET])
    overlap = seal.holdout_overlap(leaked)
    assert overlap is not None
    assert overlap.dev_rows_matching == 5
    assert overlap.holdout_rows_matching == 5
    assert overlap.holdout_rows == seal.n_rows


def test_invalid_holdout_refuses_evaluation_even_with_force(
    churn_df: pd.DataFrame, tmp_path: Path, accuracy_scorer: AccuracyScorer
) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    seal.mark_invalid(["A5: shares groups"], caller="audit")
    fitted = DummyClassifier().fit(dev.drop(columns=[TARGET]), dev[TARGET])
    for force in (False, True):
        with pytest.raises(HoldoutInvalidError, match="contaminated"):
            seal.evaluate(fitted, accuracy_scorer, caller="test", force=force)
    assert SealedHoldout.load(state).holdout_invalid
    outcomes = [
        json.loads(line)["outcome"] for line in state.access_log_path.read_text().splitlines()
    ]
    assert outcomes == ["marked_invalid", "refused_invalid", "refused_invalid"]
    events = [json.loads(line)["event"] for line in state.ledger_path.read_text().splitlines()]
    assert events == ["sealed", "invalidated"]


def test_caveats_after_inspection_and_invalidation(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    first, _, _ = _seal(churn_df, tmp_path)
    assert first.caveats == []
    same, _, _ = _seal(churn_df, tmp_path)  # same split: no inspection caveat
    assert same.caveats == []
    other_spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=3)
    other, _, _ = _seal(churn_df, tmp_path, other_spec)
    assert any("already inspected under a different split" in c for c in other.caveats)
    other.mark_invalid(["A4"], caller="audit")
    redrawn, _, _ = _seal(churn_df, tmp_path, SplitSpec(strategy=SplitStrategy.RANDOM, seed=4))
    assert any("re-drawn after the data had been inspected" in c for c in redrawn.caveats)


def test_dev_fingerprint_recorded(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    seal, dev, _ = _seal(churn_df, tmp_path)
    assert seal.record.dev_index_sha256 == dev_index_sha256(dev)


def test_week1_seal_without_digests_still_loads(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    _, dev, state = _seal(churn_df, tmp_path)
    record = json.loads(state.seal_state_path.read_text(encoding="utf-8"))
    for key in (
        "dev_index_sha256",
        "row_digest_key",
        "row_hash_significant_digits",
        "holdout_invalid",
        "invalid_reasons",
    ):
        record.pop(key)
    state.seal_state_path.write_text(json.dumps(record), encoding="utf-8")
    old = SealedHoldout.load(state)
    assert old.holdout_overlap(dev.drop(columns=[TARGET])) is None
    assert not old.holdout_invalid


def test_missing_digest_file_is_an_error(churn_df: pd.DataFrame, tmp_path: Path) -> None:
    seal, dev, state = _seal(churn_df, tmp_path)
    state.holdout_digests_path.unlink()
    with pytest.raises(SealError, match="digests missing"):
        seal.holdout_overlap(dev.drop(columns=[TARGET]))
