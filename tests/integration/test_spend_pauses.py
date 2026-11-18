"""Spend pauses: a call refused by a cap (exit 6) and a billing refusal (exit 7) stop the run
cleanly, checkpointed, and the run resumes once the cap or the credit allows."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench.traps import build_case
from holdout.agent import driver
from holdout.agent.driver import PAUSED_BILLING, PAUSED_BUDGET, resume_agent
from holdout.cli import app
from holdout.llm import config
from holdout.llm.budget import SpendLedger
from holdout.llm.client import FakeLLM
from holdout.llm.retry import BillingLLMError
from tests.agent_harness import fake_for, run_case, write_case

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private ledger: these tests fill it close to the cap. They test the project cap
    itself, so their spend is benchmark spend (general spend stops at the reserve first;
    the reserve has its own tests)."""
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)
    monkeypatch.setenv(config.SPEND_PURPOSE_ENV, "benchmark")
    return tmp_path / "home"


def _near_the_cap(inr: float = 749.95) -> None:
    SpendLedger().append(
        {"kind": "call", "timestamp": datetime.now(UTC).isoformat(), "run_id": "earlier",
         "model": config.DEFAULT_MODEL, "inr": inr}
    )  # fmt: skip


def test_the_project_cap_pauses_before_the_call_and_the_env_override_resumes(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _near_the_cap()
    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))
    outcome = run_case(case, tmp_path, llm=working)
    assert outcome.status == PAUSED_BUDGET
    assert working.calls == []  # refused before sending
    paused = outcome.paused
    assert paused is not None
    assert (paused.outage.kind, paused.outage.node) == ("budget_cap", "plan")
    for text in (
        "Budget cap reached",
        "project cap of Rs 750",
        "Spend so far: this run Rs 0.00, project Rs 749.95",
        config.PROJECT_CAP_ENV,
        f"holdout agent --resume {outcome.run_id}",
    ):
        assert text in paused.message
    before = SpendLedger().total_inr()

    still_capped = resume_agent(outcome.run_dir, client_factory=lambda: working)
    assert still_capped.status == PAUSED_BUDGET  # nothing changed: refused again, not sent
    assert working.calls == []

    monkeypatch.setenv(config.PROJECT_CAP_ENV, "800")
    resumed = resume_agent(outcome.run_dir, client_factory=lambda: working)
    assert resumed.status == "completed"
    assert resumed.manifest is not None
    assert resumed.manifest.agent is not None
    assert [o.kind for o in resumed.manifest.agent.outages] == ["budget_cap", "budget_cap"]
    assert resumed.manifest.agent.project_cap_inr == 800.0
    assert SpendLedger().total_inr() == pytest.approx(before + resumed.manifest.agent.total_inr)


def test_a_billing_refusal_pauses_without_retrying_and_resumes_after_a_top_up(
    home: Path, tmp_path: Path
) -> None:
    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))
    broke = FakeLLM(
        default=working.default,
        scripts={"plan": [BillingLLMError(429, "Your prepayment credits are depleted")]},
    )
    outcome = run_case(case, tmp_path, llm=broke)
    assert outcome.status == PAUSED_BILLING
    assert len(broke.calls_for("plan")) == 1  # never retried
    paused = outcome.paused
    assert paused is not None
    assert (paused.outage.kind, paused.outage.status, paused.outage.retries) == ("billing", 429, 0)
    assert "holdout spend --set-balance" in paused.message
    assert "Top up" in paused.message
    assert f"holdout agent --resume {outcome.run_id}" in paused.message
    failed = [e for e in SpendLedger().entries() if e["kind"] == "failed"]
    assert len(failed) == 1

    resumed = resume_agent(outcome.run_dir, client_factory=lambda: working)
    assert resumed.status == "completed"
    assert resumed.manifest is not None
    assert resumed.manifest.agent is not None
    assert [o.kind for o in resumed.manifest.agent.outages] == ["billing"]
    report = (resumed.run_dir / "report.md").read_text(encoding="utf-8")
    assert "billing / quota refusal" in report


@pytest.mark.parametrize(
    ("setup", "code", "title"),
    [("cap", 6, "Budget cap reached - run paused."), ("billing", 7, "billing / quota problem")],
)
def test_the_cli_exits_6_or_7_with_a_message_and_no_traceback(
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: str,
    code: int,
    title: str,
) -> None:
    case = build_case("direct_leak", 0)
    csv = write_case(case, tmp_path)
    working = fake_for(case, csv)
    if setup == "cap":
        _near_the_cap()
        client = working
    else:
        client = FakeLLM(default=working.default, scripts={"plan": [BillingLLMError(402)]})

    class _Client:  # stands in for GeminiClient(model=...) inside the driver
        def __new__(cls, model: str, **_: object) -> FakeLLM:  # type: ignore[misc]
            return client

    monkeypatch.setattr(driver, "GeminiClient", _Client)
    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["agent", "--csv", str(csv), "--target", case.target, "--autopilot", "--no-cache",
         "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == code, result.output
    assert title in result.output
    assert "Traceback" not in result.output
    assert "holdout agent --resume" in " ".join(result.output.split())


@pytest.mark.full_only
def test_verbose_progress_shows_running_totals_and_threshold_warnings(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _near_the_cap(370.0)  # just under the 50% mark (Rs 375)
    # Recorded today after Rs 370 spent today: the estimate is already below Rs 50.
    SpendLedger().record_balance(160.0, datetime.now().astimezone().date())
    case = build_case("direct_leak", 0)
    csv = write_case(case, tmp_path)
    working = fake_for(case, csv)

    class _Client:
        def __new__(cls, model: str, **_: object) -> FakeLLM:  # type: ignore[misc]
            return working

    monkeypatch.setattr(driver, "GeminiClient", _Client)
    # A mark at Rs 370.05, crossed by the first call (the real marks are tested in unit tests).
    monkeypatch.setattr(config, "PROJECT_WARN_FRACTIONS", (0.4934,))
    result = runner.invoke(
        app,
        ["agent", "--csv", str(csv), "--target", case.target, "--autopilot", "--no-cache",
         "--verbose", "--out", str(tmp_path / "runs")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "| run Rs" in text
    assert "of 15 | project Rs 370." in text
    assert "of 750 | balance left about Rs" in text
    assert text.count("PROJECT SPEND PASSED 49% OF THE CAP") == 1
    assert "top up now; calls may start failing" in text  # standing warning at the start
