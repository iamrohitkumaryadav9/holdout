"""Prereg Amendment 1: Flex patience (10 attempts, base 5 s, cap 300 s, Retry-After capped),
one standard-tier fallback per call on the same model, per-call tier recording and pricing
at the tier that served it, and dropped connections treated as transient."""

import json
import random
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from bench.analyze import fallback_table
from bench.systems.base import BenchResult
from holdout.core.errors import LLMBillingError, LLMUnavailableError
from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger, cost_inr
from holdout.llm.cache import ResponseCache
from holdout.llm.client import FakeLLM, GeminiClient, LLMRequest
from holdout.llm.retry import (
    FLEX_RETRY_POLICY,
    BillingLLMError,
    RetryPolicy,
    TransientLLMError,
    call_with_flex_fallback,
    classify,
)
from holdout.llm.structured import LLMSession, RenderedPrompt

MODEL = "gemini-3.8-flash"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.TRUST_CACHE_DISCOUNT_ENV, raising=False)


def _flaky(failures: dict[str | None, int], tiers: list[str | None]) -> Any:
    def fn(tier: str | None) -> str:
        tiers.append(tier)
        if failures.get(tier, 0) > 0:
            failures[tier] -= 1
            raise TransientLLMError(503)
        return f"ok on {tier}"

    return fn


def test_the_flex_policy_matches_the_amendment() -> None:
    assert RetryPolicy(base_seconds=5.0, cap_seconds=300.0, max_attempts=10) == FLEX_RETRY_POLICY
    assert RetryPolicy() == RetryPolicy(base_seconds=2.0, cap_seconds=60.0, max_attempts=6)


def test_flex_waits_back_off_with_full_jitter_and_honour_retry_after() -> None:
    waits: list[float] = []
    tiers: list[str | None] = []
    result, retries, fell_back = call_with_flex_fallback(
        _flaky({"flex": 9}, tiers), tier="flex", policy=RetryPolicy(), sleep=waits.append,
        rng=random.Random(0),
    )  # fmt: skip
    assert (result, retries, fell_back) == ("ok on flex", 9, False)
    assert tiers == ["flex"] * 10
    for n, wait in enumerate(waits, start=1):
        assert 0 <= wait <= min(300.0, 5.0 * 2 ** (n - 1))
    asked: list[float] = []
    count = iter(range(100))

    def retry_after(tier: str | None) -> str:
        if next(count) < 2:
            raise TransientLLMError(503, retry_after=900.0)  # capped at 300 s
        return "ok"

    call_with_flex_fallback(retry_after, tier="flex", policy=RetryPolicy(), sleep=asked.append,
                            rng=random.Random(0))  # fmt: skip
    assert asked == [300.0, 300.0]


def test_an_exhausted_flex_call_falls_back_once_to_standard() -> None:
    tiers: list[str | None] = []
    waits: list[float] = []
    result, retries, fell_back = call_with_flex_fallback(
        _flaky({"flex": 10, "standard": 2}, tiers), tier="flex", policy=RetryPolicy(),
        sleep=waits.append, rng=random.Random(0),
    )  # fmt: skip
    assert (result, fell_back) == ("ok on standard", True)
    assert tiers == ["flex"] * 10 + ["standard"] * 3  # the standard policy for the fallback
    assert retries == 10 + 2
    with pytest.raises(LLMUnavailableError, match="after 10 failed Flex attempts") as caught:
        call_with_flex_fallback(_flaky({"flex": 10, "standard": 6}, []), tier="flex",
                                policy=RetryPolicy(), sleep=lambda s: None,
                                rng=random.Random(0))  # fmt: skip
    assert caught.value.retries == 10 + 6 - 1  # every failed attempt on both tiers counted


def test_standard_and_default_tiers_never_fall_back() -> None:
    for tier in ("standard", None):
        tiers: list[str | None] = []
        with pytest.raises(LLMUnavailableError, match="all 6 attempts"):
            call_with_flex_fallback(_flaky({tier: 99}, tiers), tier=tier, policy=RetryPolicy(),
                                    sleep=lambda s: None, rng=random.Random(0))  # fmt: skip
        assert tiers == [tier] * 6


def test_billing_refusals_neither_retry_nor_fall_back() -> None:
    tiers: list[str | None] = []

    def billing(tier: str | None) -> str:
        tiers.append(tier)
        raise BillingLLMError()

    with pytest.raises(LLMBillingError):
        call_with_flex_fallback(billing, tier="flex", policy=RetryPolicy(),
                                sleep=lambda s: None, rng=random.Random(0))  # fmt: skip
    assert tiers == ["flex"]


def test_dropped_connections_are_transient() -> None:
    dropped = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    failure = classify(dropped)
    assert (failure.transient, failure.kind, failure.status) == (True, "connection", None)
    assert classify(ConnectionResetError()).transient
    assert classify(ValueError("bad request")).transient is False


def _raw(tier: str) -> Any:
    body = {"candidates": [{"content": {"parts": [{"text": '{"word": "ok"}'}]}}],
            "usageMetadata": {"promptTokenCount": 1_000_000, "candidatesTokenCount": 0,
                              "serviceTier": tier}}  # fmt: skip
    return SimpleNamespace(sdk_http_response=SimpleNamespace(body=json.dumps(body), headers={}))


class _APIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"{code} UNAVAILABLE")
        self.code = code


def test_gemini_records_the_fallback_and_prices_it_at_standard() -> None:
    sent: list[Any] = []

    def generate_content(*, model: str, contents: str, config: Any) -> Any:
        sent.append((model, config.service_tier.value if config.service_tier else None))
        if config.service_tier is not None and config.service_tier.value == "flex":
            raise _APIError(503)
        return _raw("standard")

    client = GeminiClient(
        MODEL, sdk_client=SimpleNamespace(models=SimpleNamespace(
            generate_content=generate_content)), sleep=lambda s: None,
    )  # fmt: skip
    response = client.generate(LLMRequest("t", "1", "s", "u", {}, seed=1, service_tier="flex"))
    assert [t for _, t in sent] == ["flex"] * 10 + ["standard"]
    assert {m for m, _ in sent} == {MODEL}  # never another model
    usage = response.usage
    assert (usage.tier_requested, usage.tier, usage.fell_back) == ("flex", "standard", True)
    assert response.retries == 10
    day = date(2026, 11, 18)
    assert cost_inr(MODEL, usage, day) == pytest.approx(0.75 * config.USD_TO_INR)  # standard
    Budget(run_id="r", model=MODEL).charge(usage, node="n", prompt_name="t")
    (line,) = SpendLedger().calls()
    assert (line["tier_requested"], line["tier_served"], line["tier_billed"],
            line["fell_back"]) == ("flex", "standard", "standard", True)  # fmt: skip


def test_the_session_records_fallbacks_per_call() -> None:
    script: list[Any] = [TransientLLMError(503)] * 10 + [{"word": "ok"}]
    fake = FakeLLM(scripts={"t": script})
    session = LLMSession(client=fake, budget=Budget(run_id="r", model=MODEL), seed=1,
                         cache=ResponseCache(enabled=False), service_tier="flex")  # fmt: skip
    from pydantic import BaseModel

    class Word(BaseModel):
        word: str

    session.call(node="n", prompt=RenderedPrompt("t", "1", "x", "s", "u"), schema=Word)
    (record,) = session.records
    assert (record.tier_requested, record.tier, record.fell_back) == ("flex", "standard", True)
    assert [c.service_tier for c in fake.calls] == ["flex"] * 10 + ["standard"]


def test_the_analysis_reports_fallbacks_per_system_and_unit() -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC)

    def result(system: str, task: str, calls: int, fallbacks: int) -> BenchResult:
        return BenchResult(suite="final_v1", task_id=task, system=system, seed=101,  # type: ignore[arg-type]
                           variant="planted", status="completed", benchmark_metric="roc_auc",
                           calls=calls, fallbacks=fallbacks, started_at=now,
                           finished_at=now)  # fmt: skip

    text = "\n".join(fallback_table([result("react", "a", 10, 3), result("react", "b", 10, 0),
                                     result("holdout", "a", 5, 0)]))  # fmt: skip
    assert "| react | 3 | 15.0% | 1 of 2 |" in text
    assert "| holdout | 0 | 0.0% | 0 of 1 |" in text
    assert "react a s101: 3" in text
