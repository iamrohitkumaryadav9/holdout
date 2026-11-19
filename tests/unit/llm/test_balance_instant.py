"""A prepaid balance recorded mid-day subtracts only the spend after the instant it was
recorded; date-only records keep counting from the start of their day."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from holdout.cli import app
from holdout.llm import config
from holdout.llm.budget import SpendLedger
from holdout.llm.spend import build_report, render


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)


def _spend(inr: float, at: datetime) -> None:
    SpendLedger().append({"kind": "call", "timestamp": at.astimezone(UTC).isoformat(),
                          "inr": inr, "run_id": "r"})  # fmt: skip


def test_a_mid_day_balance_does_not_double_count_the_days_earlier_spend() -> None:
    """The reported case: Rs 1,255.89 spent earlier today, Rs 198.89 recorded at 18:13."""
    now = datetime.now().astimezone()
    _spend(1255.89, now - timedelta(minutes=30))  # earlier today, already in Google's figure
    ledger = SpendLedger()
    record = ledger.record_balance(198.89)  # now
    assert record.at is not None
    assert ledger.balance_left() == pytest.approx(198.89)  # not 198.89 - 1255.89 = -1057
    _spend(10.0, datetime.now().astimezone() + timedelta(seconds=1))  # after the record
    assert ledger.balance_left() == pytest.approx(188.89)


def test_an_explicit_instant_subtracts_only_the_spend_after_it() -> None:
    base = datetime.now().astimezone().replace(microsecond=0) - timedelta(hours=3)
    _spend(5.0, base - timedelta(minutes=1))
    _spend(7.0, base)  # at the instant: already shown in the balance
    _spend(11.0, base + timedelta(minutes=1))
    ledger = SpendLedger()
    ledger.record_balance(100.0, base)
    assert ledger.balance_left() == pytest.approx(100.0 - 11.0)  # strictly after
    with pytest.raises(ValueError, match="future"):
        ledger.record_balance(1.0, datetime.now().astimezone() + timedelta(hours=1))


def test_date_only_records_still_count_from_the_start_of_the_day(tmp_path: Path) -> None:
    today = datetime.now().astimezone()
    start = today.replace(hour=0, minute=0, second=0, microsecond=0)
    _spend(3.0, start - timedelta(minutes=5))  # yesterday: excluded
    _spend(4.0, start + timedelta(minutes=5))  # today: subtracted
    # A record written before exact instants existed (no "as_of_at").
    SpendLedger().append({"kind": "balance", "amount_inr": 50.0, "as_of": today.date().isoformat(),
                          "timestamp": datetime.now(UTC).isoformat()})  # fmt: skip
    assert SpendLedger().balance_left() == pytest.approx(46.0)
    SpendLedger().record_balance(60.0, today.date())  # a bare date keeps that meaning
    assert SpendLedger().balance_left() == pytest.approx(56.0)
    text = "\n".join(render(build_report()))
    assert "(date only)" in text
    assert "from the start of" in text


def test_the_cli_records_now_or_an_iso_datetime() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["spend", "--set-balance", "198.89"])
    assert result.exit_code == 0, result.output
    assert "spend after that is subtracted" in result.output
    (line,) = [e for e in SpendLedger().entries() if e["kind"] == "balance"]
    assert datetime.fromisoformat(line["as_of_at"]).tzinfo is not None
    earlier = (datetime.now() - timedelta(hours=1)).replace(microsecond=0).isoformat()
    result = runner.invoke(app, ["spend", "--set-balance", "50", "--as-of", earlier, "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["balance"]["as_of_at"] is not None
    assert report["balance"]["recorded_inr"] == 50.0
    future = (datetime.now() + timedelta(days=1)).isoformat(timespec="minutes")
    bad = runner.invoke(app, ["spend", "--set-balance", "1", "--as-of", future])
    assert bad.exit_code == 2
    assert "future" in bad.output
    assert runner.invoke(app, ["spend", "--set-balance", "1", "--as-of",
                               date.today().isoformat()]).exit_code == 0  # fmt: skip
