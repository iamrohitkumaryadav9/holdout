"""Cached-input and Flex pricing, the trust setting, and how both are recorded."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger, billing_tier, cost_inr
from holdout.llm.client import GeminiClient, LLMRequest, Usage, served_tier
from holdout.llm.spend import build_report, render

MODEL = "gemini-3.8-flash"
DAY = date(2026, 11, 18)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.TRUST_CACHE_DISCOUNT_ENV, raising=False)
    return tmp_path / "home"


def test_the_verified_prices_are_in_the_tables() -> None:
    standard = config.price_for(MODEL, DAY)
    flex = config.price_for(MODEL, DAY, "flex")
    assert (standard.input_usd_per_million, standard.output_usd_per_million) == (0.75, 3.75)
    assert standard.cached_input_usd_per_million == 0.075
    assert (flex.input_usd_per_million, flex.output_usd_per_million) == (0.375, 1.875)
    assert flex.cached_input_usd_per_million == 0.0375
    later = config.price_for(MODEL, date(2027, 1, 1), "flex")
    assert later.input_usd_per_million == 0.75  # Flex doubles too
    with pytest.raises(KeyError):
        config.price_for("gemini-3.6-flash", DAY, "flex")  # no verified Flex price


def test_flex_costs_half_and_unknown_tiers_are_priced_as_standard() -> None:
    usage = Usage(prompt_tokens=1_000_000, output_tokens=1_000_000, thinking_tokens=0)
    standard = cost_inr(MODEL, usage, DAY)
    flex = cost_inr(MODEL, Usage(1_000_000, 1_000_000, 0, tier="flex"), DAY)
    assert flex == pytest.approx(standard / 2)
    assert standard == pytest.approx((0.75 + 3.75) * config.USD_TO_INR)
    for tier in (None, "priority", "provisioned_throughput"):
        assert billing_tier(Usage(1, 1, 0, tier=tier)) == "standard"


def test_cached_tokens_are_list_priced_unless_the_discount_is_trusted(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = Usage(prompt_tokens=1_000_000, output_tokens=0, thinking_tokens=0,
                  cached_tokens=800_000, tier="flex")  # fmt: skip
    listed = cost_inr(MODEL, usage, DAY)
    assert listed == pytest.approx(0.375 * config.USD_TO_INR)  # all input at full rate
    discounted = cost_inr(MODEL, usage, DAY, cache_discount=True)
    expected = (200_000 * 0.375 + 800_000 * 0.0375) / 1e6 * config.USD_TO_INR
    assert discounted == pytest.approx(expected)
    monkeypatch.setenv(config.TRUST_CACHE_DISCOUNT_ENV, "true")
    assert cost_inr(MODEL, usage, DAY) == pytest.approx(expected)
    # A model without a recorded cached price never gets a discount.
    other = cost_inr("gemini-3.6-flash", Usage(1_000_000, 0, 0, cached_tokens=900_000), DAY,
                     cache_discount=True)  # fmt: skip
    assert other == pytest.approx(0.75 * config.USD_TO_INR)


def test_the_ledger_records_cached_tokens_tier_and_both_valuations(home: Path) -> None:
    budget = Budget(run_id="r", model=MODEL)
    usage = Usage(prompt_tokens=10_000, output_tokens=500, thinking_tokens=100,
                  cached_tokens=8_000, tier="flex")  # fmt: skip
    inr, _ = budget.charge(usage, node="react", prompt_name="react")
    (line,) = SpendLedger().calls()
    assert (line["cached_tokens"], line["tier_served"], line["tier_billed"]) == (
        8000,
        "flex",
        "flex",
    )
    assert line["inr"] == pytest.approx(line["inr_list"]) == pytest.approx(inr)  # caps: list
    assert line["inr_cache_adjusted"] < line["inr_list"]
    ledger = SpendLedger()
    assert ledger.total_list_inr() == pytest.approx(line["inr_list"])
    assert ledger.total_cache_adjusted_inr() == pytest.approx(line["inr_cache_adjusted"])


def test_holdout_spend_shows_list_and_cache_adjusted_totals(home: Path) -> None:
    budget = Budget(run_id="r", model=MODEL)
    budget.charge(Usage(1_000_000, 0, 0, cached_tokens=900_000, tier="flex"),
                  node="n", prompt_name="p")  # fmt: skip
    SpendLedger().append({"kind": "call", "timestamp": "2026-11-18T00:00:00+00:00", "inr": 1.0})
    report = build_report()
    assert report.all_time_list_inr == pytest.approx(0.375 * config.USD_TO_INR + 1.0)
    assert report.all_time_cache_adjusted_inr < report.all_time_list_inr
    assert (report.cached_tokens, report.prompt_tokens) == (900_000, 1_000_000)
    text = "\n".join(render(report))
    assert "at list price" in text
    assert "cache-adjusted" in text
    assert "caps count: list price" in text
    assert json.loads(report.model_dump_json())["trust_cache_discount"] is False


def test_the_served_tier_and_cached_tokens_are_read_from_the_response() -> None:
    assert served_tier("ON_DEMAND_FLEX") == "flex"
    assert served_tier(SimpleNamespace(value="ON_DEMAND")) == "standard"
    assert served_tier(None) is None
    configs: list[object] = []

    def generate_content(*, model: str, contents: str, config: object) -> object:
        configs.append(config)
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=5,
                                thoughts_token_count=0, cached_content_token_count=60,
                                traffic_type="ON_DEMAND_FLEX")  # fmt: skip
        return SimpleNamespace(text="{}", usage_metadata=usage)

    client = GeminiClient(sdk_client=SimpleNamespace(models=SimpleNamespace(
        generate_content=generate_content)))  # fmt: skip
    response = client.generate(LLMRequest("t", "1", "s", "u", {}, seed=1, service_tier="flex"))
    assert (response.usage.cached_tokens, response.usage.tier) == (60, "flex")
    assert configs[0].service_tier.value == "flex"
    plain = client.generate(LLMRequest("t", "1", "s", "u", {}, seed=1))
    assert plain is not None
    assert configs[1].service_tier is None  # the API default (standard)
