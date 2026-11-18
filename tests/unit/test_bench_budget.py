"""Benchmark budget safety: the pre-flight refusal and the reserve."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from bench.budget import Measured, estimate
from bench.config import load_bench_config
from holdout.core.errors import BudgetExceededError
from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger
from holdout.llm.client import LLMRequest

REQUEST = LLMRequest("t", "1", "s", "u", {}, seed=1)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)
    monkeypatch.delenv(config.SPEND_PURPOSE_ENV, raising=False)
    return tmp_path / "home"


def _spent(inr: float) -> None:
    SpendLedger().append(
        {"kind": "call", "timestamp": datetime.now(UTC).isoformat(), "run_id": "x", "inr": inr}
    )


MEASURED = {
    "holdout": Measured(1.0, 3, date(2026, 11, 20)),
    "llm_judge": Measured(0.8, 3, date(2026, 11, 20)),
    "react": Measured(6.0, 3, date(2026, 11, 20)),
}


def test_the_benchmark_config_is_pinned_and_consistent() -> None:
    cfg = load_bench_config()
    assert cfg.model == "gemini-3.8-flash"
    assert cfg.run_cap_inr == config.RUN_CAP_INR
    assert cfg.cost_margin == 0.20
    assert cfg.systems == ("holdout", "audits_only", "llm_judge", "react", "flaml")
    assert len(cfg.sha256) == 64


def test_a_run_that_fits_is_allowed(home: Path) -> None:
    SpendLedger().record_balance(700.0, datetime.now().astimezone().date())
    report = estimate(
        {"holdout": 10, "react": 10, "flaml": 10},
        load_bench_config(),
        purpose="benchmark",
        measured=MEASURED,
        today=date(2026, 11, 20),
    )
    assert report.fits, report.render()
    assert report.total_inr == pytest.approx((10 * 1.0 + 10 * 6.0) * 1.2)
    assert "fits: yes" in report.render()


def test_a_run_that_does_not_fit_is_refused_with_what_is_short(home: Path) -> None:
    _spent(700.0)
    SpendLedger().record_balance(60.0, datetime.now().astimezone().date())
    report = estimate(
        {"react": 20},
        load_bench_config(),
        purpose="benchmark",
        measured=MEASURED,
        today=date(2026, 11, 20),
    )
    assert not report.fits
    text = report.render()
    assert "fits: NO" in text
    assert "exceeds the Rs 50.00 left under the project cap (short by Rs 94.00)" in text
    assert "top up first" in text  # the balance (60 - 700 spent today) is short too


def test_general_spend_is_measured_against_the_cap_minus_the_reserve(home: Path) -> None:
    SpendLedger().record_balance(700.0, datetime.now().astimezone().date())
    _spent(140.0)
    report = estimate(
        {"holdout": 10}, load_bench_config(), purpose="general", measured=MEASURED,
        today=date(2026, 11, 20),
    )  # fmt: skip
    assert report.available_inr == pytest.approx(750 - 600 - 140)
    assert not report.fits
    assert "the cap minus the reserve" in report.render()


def test_unmeasured_systems_are_bounded_by_the_cap_and_a_missing_balance_refuses(
    home: Path,
) -> None:
    report = estimate(
        {"holdout": 5, "flaml": 5}, load_bench_config(), purpose="benchmark", measured={},
        today=date(2026, 11, 20),
    )  # fmt: skip
    assert report.total_inr == 5 * 15.0  # worst case: the per-run cap for every unit
    assert "worst case = per-run cap" in report.render()
    measured_only = estimate(
        {"holdout": 5}, load_bench_config(), purpose="benchmark", measured=MEASURED,
        today=date(2026, 11, 20),
    )  # fmt: skip
    assert "no prepaid balance recorded" in measured_only.render()


def test_completed_units_cost_nothing(home: Path) -> None:
    report = estimate(
        {"holdout": 0, "react": 0}, load_bench_config(), purpose="benchmark",
        measured=MEASURED, today=date(2026, 11, 20),
    )  # fmt: skip
    assert report.total_inr == 0
    assert report.fits


def test_the_price_doubling_is_priced_in(home: Path) -> None:
    SpendLedger().record_balance(700.0, datetime.now().astimezone().date())
    before = estimate(
        {"holdout": 10}, load_bench_config(), purpose="benchmark", measured=MEASURED,
        today=date(2026, 12, 31),
    )  # fmt: skip
    after = estimate(
        {"holdout": 10}, load_bench_config(), purpose="benchmark", measured=MEASURED,
        today=date(2027, 1, 1),
    )  # fmt: skip
    assert after.price_factor == pytest.approx(2.0)
    assert after.total_inr == pytest.approx(2 * before.total_inr)


# ------------------------------------------------------------------ the reserve


def test_general_spend_may_not_enter_the_reserve(home: Path) -> None:
    assert config.BENCHMARK_RESERVE_INR == 600.0
    _spent(149.95)
    general = Budget(run_id="r", model=config.DEFAULT_MODEL)
    assert general.purpose == "general"
    with pytest.raises(BudgetExceededError, match="benchmark reserve") as refused:
        general.check(REQUEST, "plan")
    assert (refused.value.scope, refused.value.cap_inr) == ("reserve", 150.0)
    benchmark = Budget(run_id="r", model=config.DEFAULT_MODEL, purpose="benchmark")
    assert benchmark.check(REQUEST, "plan") > 0  # benchmark spend may use the reserve


def test_the_purpose_comes_from_the_environment_and_is_recorded(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.SPEND_PURPOSE_ENV, "benchmark")
    assert Budget(run_id="r", model=config.DEFAULT_MODEL).purpose == "benchmark"
    monkeypatch.setenv(config.SPEND_PURPOSE_ENV, "tuning")
    with pytest.raises(Exception, match="must be one of"):
        config.spend_purpose()


def test_a_reserve_refusal_pauses_the_agent_with_exit_6(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from bench.traps import build_case
    from holdout.agent import driver
    from holdout.cli import app
    from tests.agent_harness import fake_for, write_case

    _spent(149.99)
    case = build_case("direct_leak", 0)
    csv = write_case(case, tmp_path)
    fake = fake_for(case, csv)

    class _Client:
        def __new__(cls, model: str) -> object:  # type: ignore[misc]
            return fake

    monkeypatch.setattr(driver, "GeminiClient", _Client)
    result = CliRunner().invoke(
        app,
        ["agent", "--csv", str(csv), "--target", case.target, "--autopilot", "--no-cache",
         "--out", str(tmp_path / "runs")],
    )  # fmt: skip
    assert result.exit_code == 6, result.output
    assert "benchmark reserve" in " ".join(result.output.split())
    assert fake.calls == []  # refused before sending
