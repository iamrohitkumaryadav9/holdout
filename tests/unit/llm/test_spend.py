"""Spend control: caps and their override, live totals, threshold and balance warnings,
billing classification, and the ``holdout spend`` command."""

import json
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from google.genai import errors as genai_errors
from typer.testing import CliRunner

from holdout.cli import app
from holdout.core.errors import BudgetExceededError, LLMBillingError, LLMConfigError
from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger, cost_inr
from holdout.llm.client import LLMRequest, Usage
from holdout.llm.progress import LineProgress
from holdout.llm.retry import RetryPolicy, call_with_retries, classify
from holdout.llm.spend import build_report

runner = CliRunner()
MODEL = config.DEFAULT_MODEL
REQUEST = LLMRequest("t", "1", "s", "u", {}, seed=1)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private HOLDOUT_HOME, so a test's ledger never leaks into other tests."""
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)
    return tmp_path / "home"


def _usage_costing(inr: float) -> Usage:
    """Usage whose cost is ``inr`` (output tokens only), give or take a token."""
    per_token = cost_inr(MODEL, Usage(0, 1_000_000, 0), datetime.now(UTC).date()) / 1_000_000
    return Usage(prompt_tokens=0, output_tokens=round(inr / per_token), thinking_tokens=0)


def _spent(ledger: SpendLedger, inr: float, *, when: datetime | None = None) -> None:
    stamp = (when or datetime.now(UTC)).isoformat()
    ledger.append(
        {"kind": "call", "timestamp": stamp, "run_id": "seed", "model": MODEL, "inr": inr}
    )


# ------------------------------------------------------------------ caps


def test_the_project_cap_defaults_to_750_and_only_the_env_var_overrides_it(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert config.PROJECT_CAP_INR == 750.0
    assert config.RUN_CAP_INR == 15.0
    assert config.project_cap_inr() == 750.0
    assert Budget(run_id="r", model=MODEL).project_cap_inr == 750.0
    monkeypatch.setenv(config.PROJECT_CAP_ENV, "900")
    assert Budget(run_id="r", model=MODEL).project_cap_inr == 900.0
    for bad in ("abc", "-5", "0", "inf", "nan"):
        monkeypatch.setenv(config.PROJECT_CAP_ENV, bad)
        with pytest.raises(LLMConfigError, match=config.PROJECT_CAP_ENV):
            config.project_cap_inr()


def test_a_call_that_could_cross_the_project_cap_is_refused(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.SPEND_PURPOSE_ENV, "benchmark")  # past the reserve: cap only
    ledger = SpendLedger()
    _spent(ledger, 749.9)
    budget = Budget(run_id="r", model=MODEL)
    with pytest.raises(BudgetExceededError) as refused:
        budget.check(REQUEST, "plan")
    error = refused.value
    assert (error.scope, error.cap_inr) == ("project", 750.0)
    assert error.project_spent_inr == pytest.approx(749.9)
    assert error.estimate_inr > 0.1  # the worst case: every output token used
    monkeypatch.setenv(config.PROJECT_CAP_ENV, "800")
    assert Budget(run_id="r", model=MODEL).check(REQUEST, "plan") > 0  # override lets it pass


def test_a_call_that_could_cross_the_run_cap_is_refused(home: Path) -> None:
    budget = Budget(run_id="r", model=MODEL, spent_run_inr=14.99)
    with pytest.raises(BudgetExceededError, match="run cap") as refused:
        budget.check(REQUEST, "report")
    assert (refused.value.scope, refused.value.cap_inr) == ("run", 15.0)


def test_the_real_cost_is_charged_including_thinking_tokens(home: Path) -> None:
    budget = Budget(run_id="r", model=MODEL)
    usage = Usage(prompt_tokens=1200, output_tokens=300, thinking_tokens=500)
    inr, _ = budget.charge(usage, node="plan", prompt_name="plan", retries=2)
    assert inr == pytest.approx(cost_inr(MODEL, usage, budget.today))
    (line,) = SpendLedger().calls()
    assert (line["thinking_tokens"], line["retries"], line["inr"]) == (500, 2, inr)
    assert budget.spent_run_inr == pytest.approx(inr)


# ------------------------------------------------------------------ warnings


def test_project_warnings_fire_once_at_50_75_and_90_percent(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.PROJECT_CAP_ENV, "100")
    budget = Budget(run_id="r", model=MODEL)
    _, first = budget.charge(_usage_costing(40.0), node="n", prompt_name="p")
    assert first == []
    _, crossed_50 = budget.charge(_usage_costing(15.0), node="n", prompt_name="p")
    assert len(crossed_50) == 1
    assert "PASSED 50% OF THE CAP" in crossed_50[0]
    _, again = budget.charge(_usage_costing(1.0), node="n", prompt_name="p")
    assert again == []  # once each
    _, both = budget.charge(_usage_costing(36.0), node="n", prompt_name="p")
    assert ["75%" in both[0], "90%" in both[1]] == [True, True]


def test_the_default_marks_are_375_562_50_and_675(home: Path) -> None:
    marks = [f * config.PROJECT_CAP_INR for f in config.PROJECT_WARN_FRACTIONS]
    assert marks == [375.0, 562.5, 675.0]


def test_balance_warnings_below_150_and_below_50(home: Path) -> None:
    ledger = SpendLedger()
    ledger.record_balance(200.0, datetime.now().astimezone().date())
    _spent(ledger, 45.0)  # left 155
    budget = Budget(run_id="r", model=MODEL)
    _, soon = budget.charge(_usage_costing(10.0), node="n", prompt_name="p")
    assert len(soon) == 1
    assert "top up soon" in soon[0]
    _, quiet = budget.charge(_usage_costing(1.0), node="n", prompt_name="p")
    assert quiet == []
    _spent(ledger, 90.0)  # left about 54
    _, now = budget.charge(_usage_costing(10.0), node="n", prompt_name="p")
    assert len(now) == 1
    assert "top up now; calls may start failing" in now[0]


def test_balance_left_subtracts_spend_from_the_start_of_the_recorded_day(home: Path) -> None:
    ledger = SpendLedger()
    _spent(ledger, 30.0, when=datetime.now(UTC) - timedelta(days=3))  # before: excluded
    _spent(ledger, 2.0)
    ledger.record_balance(747.0, datetime.now().astimezone().date())
    _spent(ledger, 5.0)
    assert ledger.balance_left() == pytest.approx(740.0)  # today's 2 + 5, not the old 30
    with pytest.raises(ValueError, match="zero or more"):
        ledger.record_balance(-1.0, date.today())


def test_progress_lines_show_running_totals(home: Path) -> None:
    SpendLedger().record_balance(747.0, datetime.now().astimezone().date())
    budget = Budget(run_id="r", model=MODEL)
    inr, _ = budget.charge(_usage_costing(0.25), node="plan", prompt_name="plan")
    lines: list[str] = []
    LineProgress(lines.append).call_finished(
        "plan", "plan", 2.5, cache_hit=False, tokens=900, inr=inr, retries=0,
        spend=budget.snapshot(inr),
    )  # fmt: skip
    assert lines[0].endswith(
        f"this call Rs {inr:.3f} | run Rs {inr:.2f} of 15 | project Rs {inr:.2f} of 750 | "
        f"balance left about Rs {747.0 - inr:.2f}"
    )


# ------------------------------------------------------------------ billing errors

RATE_LIMIT_429 = (
    "You exceeded your current quota, please check your plan and billing details. "
    "Quota exceeded for metric: generate_content_requests_per_minute. Please retry in 18s. "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '18s'}"
)


@pytest.mark.parametrize(
    ("code", "message", "kind"),
    [
        (429, RATE_LIMIT_429, "http"),  # an ordinary rate limit: retried
        (429, "Your prepayment credits are depleted.", "billing"),
        (429, "Daily quota exhausted for this project.", "billing"),
        (402, "Payment required", "billing"),
        (403, "The billing account for this project is disabled.", "billing"),
        (400, "Invalid JSON schema.", "http"),
        (503, "The model is overloaded.", "http"),
    ],
)
def test_billing_and_quota_errors_are_told_apart_from_rate_limits(
    code: int, message: str, kind: str
) -> None:
    error = genai_errors.APIError(code, {"error": {"code": code, "message": message}})
    failure = classify(error)
    assert failure.kind == kind
    if message == RATE_LIMIT_429:
        assert (failure.transient, failure.retry_after) == (True, 18.0)


def test_a_billing_error_is_never_retried() -> None:
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        raise genai_errors.APIError(429, {"error": {"code": 429, "message": "credits depleted"}})

    with pytest.raises(LLMBillingError, match="billing or quota") as raised:
        call_with_retries(fn, policy=RetryPolicy(), sleep=lambda s: None, rng=random.Random(0))
    assert (len(calls), raised.value.retries, raised.value.status) == (1, 0, 429)


# ------------------------------------------------------------------ holdout spend


def _seed_ledger() -> None:
    ledger = SpendLedger()
    now = datetime.now(UTC)
    for days, run_id, model, inr in [
        (30, "run-old", "gemini-3.6-flash", 10.0),
        (3, "run-mid", MODEL, 4.0),
        (0, "run-new", MODEL, 1.5),
    ]:
        when = (now - timedelta(days=days)).isoformat()
        ledger.append(
            {"kind": "call", "timestamp": when, "run_id": run_id, "model": model, "inr": inr,
             "retries": 1}
        )  # fmt: skip
    ledger.append({"kind": "cache_hit", "timestamp": now.isoformat(), "run_id": "run-new",
                   "model": MODEL, "inr": 0.0, "retries": 0})  # fmt: skip
    ledger.append({"kind": "failed", "timestamp": now.isoformat(), "run_id": "run-new",
                   "model": MODEL, "inr": 0.0, "retries": 5})  # fmt: skip


def test_spend_set_balance_and_the_text_report(home: Path) -> None:
    _seed_ledger()
    recorded = runner.invoke(app, ["spend", "--set-balance", "747"])
    assert recorded.exit_code == 0, recorded.output
    assert "Recorded a prepaid balance of Rs 747.00" in recorded.output
    out = runner.invoke(app, ["spend"]).output
    assert "today" in out
    assert "Rs      1.50" in out  # today
    assert "Rs      5.50" in out  # last 7 days
    assert "Rs     15.50" in out  # all time
    assert out.index("run-new") < out.index("run-mid") < out.index("run-old")  # latest first
    assert "gemini-3.6-flash" in out
    assert "calls 3, retries 8, cache hits 1, failed calls 1" in out
    assert "project cap         Rs    750.00 (default)" in out
    assert "remaining Rs 734.50" in out
    assert "balance left        Rs    745.50" in out  # 747 minus today's 1.50
    assert "94.1 INR per USD" in out
    assert "reconcile" in out.lower()


def test_spend_json_and_the_env_cap(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ledger()
    monkeypatch.setenv(config.PROJECT_CAP_ENV, "1000")
    report = json.loads(runner.invoke(app, ["spend", "--json"]).output)
    assert report["all_time_inr"] == pytest.approx(15.5)
    assert report["last_7_days_inr"] == pytest.approx(5.5)
    assert (report["project_cap_inr"], report["project_cap_source"]) == (1000.0, "env")
    assert report["balance"] is None
    assert report["models"]["gemini-3.6-flash"] == {"calls": 1, "inr": 10.0}
    assert [r["run_id"] for r in report["runs"]] == ["run-new", "run-mid", "run-old"]
    assert {p["model"] for p in report["prices"]} == set(config.PRICE_TABLE)


def test_spend_warns_about_a_low_balance_and_rejects_bad_input(home: Path) -> None:
    low = runner.invoke(app, ["spend", "--set-balance", "40"])
    assert "top up now" in low.output
    assert build_report().balance is not None
    for args in (
        ["spend", "--set-balance", "-3"],
        ["spend", "--set-balance", "100", "--as-of", "2999-01-01"],
        ["spend", "--as-of", "2026-01-01"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, args
        assert "Traceback" not in result.output
