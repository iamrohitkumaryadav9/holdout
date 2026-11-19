"""The exported cost ledger: what it keeps, and that docs/results.md regenerates without the
local spend ledger."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bench import analyze
from bench.costs import EXPORT_NAME, cost_calls, export_lines, load_export, write_export
from bench.run import append_result
from bench.systems.base import BenchResult
from holdout.llm import config
from holdout.llm.budget import SpendLedger

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)


def _result(
    task: str, system: str = "react", extra: dict[str, object] | None = None
) -> BenchResult:
    now = datetime.now(UTC)
    return BenchResult(suite="final_v1", task_id=task, system=system, seed=101,  # type: ignore[arg-type]
                       variant="planted", status="completed", benchmark_metric="roc_auc",
                       cost_inr=99.0, extra=extra or {}, started_at=now,
                       finished_at=now)  # fmt: skip


def _line(inr: float, **where: object) -> dict[str, object]:
    return {"kind": "call", "timestamp": datetime.now(UTC).isoformat(), "inr": inr,
            "tier_billed": "standard", "model": "gemini-3.8-flash", "prompt_tokens": 10,
            "path_like": "C:/secret/path", **where}  # fmt: skip


def test_the_export_keeps_billing_fields_only_and_tags_each_line(tmp_path: Path) -> None:
    ledger = SpendLedger()
    ledger.append(_line(2.0, run_id="final_v1/t0__planted__s101/react"))  # legacy: run id
    time.sleep(0.001)
    ledger.append(_line(3.0, unit="final_v1/t1__planted__s101/react"))  # tagged
    ledger.append(_line(1.5, run_id="20261118T000001Z-ab"))  # a Holdout agent run
    ledger.append(_line(50.0, run_id="unrelated"))
    ledger.record_balance(100.0)
    results = [_result("t0"), _result("t1"),
               _result("t2", "holdout", {"run_dirs": ["C:/x/20261118T000001Z-ab"]})]  # fmt: skip
    rows = export_lines(results, ledger.calls())
    assert sorted(r["inr"] for r in rows) == [1.5, 2.0, 3.0]  # nothing unrelated
    assert {r["unit"] for r in rows} == {"final_v1/t0__planted__s101/react",
                                         "final_v1/t1__planted__s101/react",
                                         "final_v1/t2__planted__s101/holdout"}  # fmt: skip
    assert {r["system"] for r in rows} == {"react", "holdout"}
    assert all("path_like" not in r and "amount_inr" not in r for r in rows)
    # Attribution from the export alone matches the ledger's.
    assert analyze.attributed_cost(results[0], rows) == pytest.approx(2.0)
    assert analyze.attributed_cost(results[2], rows) == pytest.approx(1.5)


def test_cost_calls_merge_the_local_ledger_and_the_export_without_double_counting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    shared = _line(2.0, run_id="final_v1/t0__planted__s101/react")
    SpendLedger().append(shared)
    exported = {**shared, "unit": "final_v1/t0__planted__s101/react", "system": "react"}
    only_exported = _line(4.0, unit="final_v1/t9__planted__s101/react")
    (root / EXPORT_NAME).write_text(
        "".join(json.dumps(x) + "\n" for x in (exported, only_exported)), encoding="utf-8"
    )
    calls = cost_calls(root)
    assert sorted(float(c["inr"]) for c in calls) == [2.0, 4.0]  # the shared line once
    assert analyze.attributed_cost(_result("t0"), calls) == pytest.approx(2.0)
    assert analyze.attributed_cost(_result("t9"), calls) == pytest.approx(4.0)


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "results"
    append_result(root, _result("t0"))
    SpendLedger().append(_line(2.5, run_id="final_v1/t0__planted__s101/react"))
    path = write_export(root)
    assert path.name == EXPORT_NAME
    (row,) = load_export(root)
    assert row["inr"] == 2.5
    assert row["unit"] == "final_v1/t0__planted__s101/react"


def test_results_md_regenerates_identically_without_the_local_ledger(tmp_path: Path) -> None:
    """The committed docs/results.md is reproduced from committed files alone."""
    assert not SpendLedger().path.exists()  # this test's HOLDOUT_HOME has no ledger
    out = tmp_path / "results.md"
    assert analyze.main(["--results", str(ROOT / "results"), "--out", str(out)]) == 0
    expected = (ROOT / "docs" / "results.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert out.read_text(encoding="utf-8") == expected
