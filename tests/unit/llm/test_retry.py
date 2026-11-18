"""The retry policy: full-jitter backoff, Retry-After, transient vs permanent errors,
the SDK's own retries switched off, and progress lines."""

import logging
import random
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from google.genai import errors as genai_errors
from hypothesis import given, settings
from hypothesis import strategies as st

from holdout.core.errors import LLMRequestError, LLMUnavailableError
from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger
from holdout.llm.cache import ResponseCache
from holdout.llm.client import FakeLLM, GeminiClient
from holdout.llm.progress import LineProgress
from holdout.llm.retry import (
    RetryEvent,
    RetryPolicy,
    TransientLLMError,
    call_with_retries,
    classify,
)
from holdout.llm.structured import LLMSession, RenderedPrompt
from tests.tiers import examples

POLICY = RetryPolicy()


def _failing(errors: list[BaseException], result: str = "ok") -> tuple[list[int], object]:
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if errors:
            raise errors.pop(0)
        return result

    return calls, fn


def _run(
    errors: list[BaseException], policy: RetryPolicy = POLICY, seed: int = 0
) -> tuple[str, int, list[float], list[RetryEvent]]:
    sleeps: list[float] = []
    events: list[RetryEvent] = []
    _, fn = _failing(errors)
    result, retries = call_with_retries(
        fn,  # type: ignore[arg-type]
        policy=policy,
        sleep=sleeps.append,
        rng=random.Random(seed),
        on_retry=events.append,
    )
    return result, retries, sleeps, events


def test_policy_defaults_match_the_documented_contract() -> None:
    assert (POLICY.base_seconds, POLICY.cap_seconds, POLICY.max_attempts) == (2.0, 60.0, 6)


@settings(max_examples=examples(60, 20), deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_full_jitter_waits_stay_inside_the_exponential_ceiling(seed: int) -> None:
    errors: list[BaseException] = [TransientLLMError(503) for _ in range(5)]
    result, retries, sleeps, events = _run(errors, seed=seed)
    assert (result, retries) == ("ok", 5)
    for attempt, wait in enumerate(sleeps, start=1):
        assert 0.0 <= wait <= min(60.0, 2.0 * 2 ** (attempt - 1))
    assert [e.attempt for e in events] == [1, 2, 3, 4, 5]
    assert [e.wait_seconds for e in events] == sleeps


def test_waits_are_capped() -> None:
    policy = RetryPolicy(max_attempts=12)
    _, _, sleeps, _ = _run([TransientLLMError(429) for _ in range(11)], policy=policy)
    assert max(sleeps) <= 60.0
    # A server asking for longer than the cap is capped too.
    _, _, sleeps, _ = _run([TransientLLMError(429, retry_after=600.0)])
    assert sleeps == [60.0]


def test_retry_after_is_honoured_in_seconds_and_as_a_date() -> None:
    _, _, sleeps, events = _run([TransientLLMError(503, retry_after=7.0)])
    assert sleeps == [7.0]
    assert events[0].retry_after == 7.0
    seconds = genai_errors.APIError(
        503,
        {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}},
        httpx.Response(503, headers={"Retry-After": "3"}),
    )
    failure = classify(seconds)
    assert (failure.transient, failure.status, failure.retry_after) == (True, 503, 3.0)
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    dated = genai_errors.APIError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "slow down"}},
        httpx.Response(429, headers={"Retry-After": when}),
    )
    wait = classify(dated).retry_after
    assert wait is not None
    assert 25.0 <= wait <= 30.0


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_errors_are_never_retried(status: int) -> None:
    error = genai_errors.APIError(status, {"error": {"code": status, "message": "no"}})
    calls, fn = _failing([error])
    sleeps: list[float] = []
    with pytest.raises(LLMRequestError, match=f"HTTP {status}") as raised:
        call_with_retries(
            fn,  # type: ignore[arg-type]
            policy=POLICY,
            sleep=sleeps.append,
            rng=random.Random(0),
        )
    assert raised.value.status == status
    assert (len(calls), sleeps) == (1, [])


def test_a_rejected_request_carries_the_api_reason() -> None:
    error = genai_errors.APIError(
        400, {"error": {"code": 400, "message": "The specified schema produces a constraint "
                                                "that has too many states for serving."}}
    )  # fmt: skip
    _, fn = _failing([error])
    with pytest.raises(LLMRequestError, match="too many states for serving"):
        call_with_retries(fn, policy=POLICY, sleep=lambda s: None, rng=random.Random(0))  # type: ignore[arg-type]


def test_schema_and_programming_errors_are_not_transient() -> None:
    assert not classify(ValueError("bad schema")).transient


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("slow"), TimeoutError("slow")], ids=["httpx", "builtin"]
)
def test_timeouts_are_transient(error: BaseException) -> None:
    result, retries, _, events = _run([error])
    assert (result, retries) == ("ok", 1)
    assert (events[0].kind, events[0].status) == ("timeout", None)


@pytest.mark.parametrize("status", [429, 500, 503, 504])
def test_exhaustion_after_six_attempts_is_typed(status: int) -> None:
    errors: list[BaseException] = [TransientLLMError(status) for _ in range(6)]
    calls, fn = _failing(errors)
    sleeps: list[float] = []
    with pytest.raises(LLMUnavailableError, match="on all 6 attempts") as raised:
        call_with_retries(
            fn,  # type: ignore[arg-type]
            policy=POLICY,
            sleep=sleeps.append,
            rng=random.Random(0),
        )
    assert (raised.value.status, raised.value.retries) == (status, 5)
    assert (len(calls), len(sleeps)) == (6, 5)


def test_every_retry_is_logged_with_status_attempt_and_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="holdout.llm"):
        _run([TransientLLMError(503, retry_after=4.0)])
    assert "HTTP 503 on attempt 1 of 6; retrying in 4.0 s" in caplog.text


def test_gemini_client_disables_the_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    client = GeminiClient()
    options = client._client._api_client._http_options
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1  # Holdout's policy is the only retry loop
    assert config.REQUEST_TIMEOUT_SECONDS == 90
    assert options.timeout == 90_000  # milliseconds; a timeout is retried (see above)


# ------------------------------------------------------------------ session and progress

PROMPT = RenderedPrompt(name="t", version="1", sha256="abc", system="sys", user="data")


def _session(tmp_path: Path, llm: FakeLLM, lines: list[str]) -> LLMSession:
    budget = Budget(run_id="r", model=llm.model, ledger=SpendLedger(tmp_path / "ledger.jsonl"))
    return LLMSession(
        client=llm,
        budget=budget,
        seed=1,
        cache=ResponseCache(tmp_path / "cache"),
        progress=LineProgress(lines.append),
    )


def test_retries_are_counted_and_shown_but_cost_no_tokens(tmp_path: Path) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        word: str

    lines: list[str] = []
    llm = FakeLLM(scripts={"t": [TransientLLMError(503), TransientLLMError(429), {"word": "a"}]})
    session = _session(tmp_path, llm, lines)
    assert session.call(node="plan", prompt=PROMPT, schema=Answer).word == "a"
    (record,) = session.records
    assert record.retries == 2
    assert record.prompt_tokens == len("sys" + "data") // 4  # one successful call's usage
    assert any("HTTP 503 on attempt 1; retrying in" in line for line in lines)
    assert any("HTTP 429 on attempt 2; retrying in" in line for line in lines)
    assert lines[0].endswith("gemini t [plan]: calling")
    assert "2 retries" in lines[-1]
    assert all(line.startswith("[") and line[9] == "]" for line in lines)  # [HH:MM:SS]


def test_an_outage_is_recorded_with_its_node_and_raised(tmp_path: Path) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        word: str

    llm = FakeLLM(scripts={"t": [TransientLLMError(503) for _ in range(6)]})
    session = _session(tmp_path, llm, [])
    with pytest.raises(LLMUnavailableError) as raised:
        session.call(node="plan", prompt=PROMPT, schema=Answer)
    assert (raised.value.node, raised.value.status, raised.value.retries) == ("plan", 503, 5)
    (record,) = session.records
    assert (record.valid, record.retries, record.inr) == (False, 5, 0.0)
    assert record.errors[0].startswith("LLM unavailable")
