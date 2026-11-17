"""RunState allocates unique run directories and owns the artifact layout."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from holdout.core.seeds import derived_seeds
from holdout.core.state import LEDGER_FILENAME, RunState


def test_create_allocates_a_fresh_directory(tmp_path: Path) -> None:
    now = datetime(2026, 11, 17, 9, 30, tzinfo=UTC)
    state = RunState.create(tmp_path, run_seed=7, now=now)
    assert state.run_dir.is_dir()
    assert state.run_id.startswith("20261117T093000Z-")
    assert state.seeds == derived_seeds(7)


def test_two_runs_in_the_same_second_get_distinct_directories(tmp_path: Path) -> None:
    now = datetime(2026, 11, 17, tzinfo=UTC)
    first = RunState.create(tmp_path, 1, now=now)
    second = RunState.create(tmp_path, 1, now=now)
    assert first.run_dir != second.run_dir


def test_artifact_paths_live_in_the_run_directory(tmp_path: Path) -> None:
    state = RunState.create(tmp_path, 1)
    for path in (
        state.holdout_path,
        state.seal_state_path,
        state.access_log_path,
        state.manifest_path,
        state.pipeline_path,
        state.report_path,
    ):
        assert path.parent == state.run_dir
    assert state.ledger_path == tmp_path / LEDGER_FILENAME


def test_existing_run_directory_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force an id collision: a second run must fail rather than share the first run's seal.
    monkeypatch.setattr("holdout.core.state.secrets.token_hex", lambda _n: "abcdef")
    now = datetime(2026, 11, 17, tzinfo=UTC)
    RunState.create(tmp_path, 1, now=now)
    with pytest.raises(FileExistsError):
        RunState.create(tmp_path, 1, now=now)
