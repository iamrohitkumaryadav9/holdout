"""Full CLI runs with audits on 50k rows stay within the 90-second budget.

Marked slow; CI runs it on Ubuntu only. Cases: deep mode (a planted leak drives A10 into
deep mode, so A1 runs without row sampling) and the A5 worst case (entity data, where the
group comparison refits the plan several times).
"""

import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench.traps import build_case
from holdout.cli import EXIT_UNSEAL_REFUSED, app

BUDGET_SECONDS = 90.0
ROWS = 50_000

runner = CliRunner()


@pytest.mark.slow
@pytest.mark.parametrize(("trap", "expect_deep"), [("direct_leak", True), ("group_leakage", False)])
def test_full_run_with_audits_on_50k_rows_within_budget(
    trap: str, expect_deep: bool, tmp_path: Path
) -> None:
    case = build_case(trap, 1, n_rows=ROWS)
    csv = tmp_path / "data.csv"
    case.df.to_csv(csv, index=False)
    started = time.perf_counter()
    result = runner.invoke(
        app,
        [
            "run",
            "--csv",
            str(csv),
            "--target",
            case.target,
            "--unseal",
            "--out",
            str(tmp_path / "runs"),
        ],
    )
    seconds = time.perf_counter() - started
    assert result.exit_code in (0, EXIT_UNSEAL_REFUSED), result.output
    assert seconds < BUDGET_SECONDS, f"{trap}: {seconds:.1f}s exceeds {BUDGET_SECONDS}s"
    assert ("deep mode" in result.output) == expect_deep
