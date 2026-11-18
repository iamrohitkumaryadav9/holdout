"""The final benchmark is budget-checked as a whole: final_v1 and real_world_v1 together,
every configured system, with room for the units in flight. A suite that fits on its own
must not start when the two together do not fit."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bench import run as bench_run
from bench.budget import preflight
from bench.config import load_bench_config
from bench.run import FINAL_SUITES, benchmark_units, units_for
from bench.systems.base import SystemName
from holdout.llm import config
from holdout.llm.budget import SpendLedger

WORKERS = 4
SYSTEMS: tuple[SystemName, ...] = ("holdout", "audits_only", "llm_judge", "react", "flaml")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(config.PROJECT_CAP_ENV, "1000")
    monkeypatch.delenv(config.SPEND_PURPOSE_ENV, raising=False)
    return tmp_path / "home"


def _required(units: list[bench_run.Unit], root: Path) -> float:
    report = preflight(units, root, load_bench_config(), purpose="benchmark", workers=WORKERS)
    return report.total_inr + report.headroom_inr


def test_the_whole_benchmark_must_fit_not_each_suite(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_bench_config()
    root = tmp_path / "results"
    alone = {s: _required(units_for(s, list(SYSTEMS)), root) for s in FINAL_SUITES}
    together = _required(benchmark_units(cfg.systems), root)
    assert together > max(alone.values())
    # Leave room for either suite alone, but not for both: the case the pilot exposed.
    available = (max(alone.values()) + together) / 2
    SpendLedger().append({"kind": "call", "timestamp": datetime.now(UTC).isoformat(),
                          "inr": 1000.0 - available})  # fmt: skip
    SpendLedger().record_balance(2000.0, datetime.now().astimezone().date())
    for suite in FINAL_SUITES:  # what the old per-suite check saw: each fits
        alone_report = preflight(units_for(suite, list(SYSTEMS)), root, cfg,
                                 purpose="benchmark", workers=WORKERS)  # fmt: skip
        assert alone_report.fits, alone_report.render()
    for suite in FINAL_SUITES:
        args = ["--suite", suite, "--systems", "flaml", "--results", str(root),
                "--purpose", "benchmark", "--workers", str(WORKERS)]  # fmt: skip
        assert bench_run.main([*args, "--plan"]) == 6  # even a free subset is checked whole
        out = capsys.readouterr().out
        assert "scope: the whole final benchmark (final_v1 + real_world_v1" in out
        assert "fits: NO" in out
        assert "unit-start guard would stop the run" in out
        assert bench_run.main(args) == 6  # and the run refuses to start
        assert "refusing to start" in capsys.readouterr().out
    assert not (root / "final_v1").exists()  # nothing ran


def test_dev_runs_are_checked_on_their_own(home: Path, tmp_path: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    SpendLedger().record_balance(2000.0, datetime.now().astimezone().date())
    args = ["--suite", "dev_v1", "--tasks", "direct_leak@adult", "--seeds", "11",
            "--systems", "holdout", "--results", str(tmp_path / "r"), "--plan"]  # fmt: skip
    assert bench_run.main(args) == 0
    assert "scope: this invocation (dev_v1)" in capsys.readouterr().out
