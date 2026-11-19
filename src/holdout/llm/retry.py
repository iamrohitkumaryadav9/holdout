"""Retries for transient LLM failures: exponential backoff with full jitter.

Transient: HTTP 429, 500, 502, 503, 504, request timeouts and dropped connections.
Everything else (400, 401, 403, schema problems) fails at once. The wait before retry ``n``
(1-based) is drawn uniformly from ``[0, min(cap, base * 2**(n-1))]`` ("full jitter"),
unless the server sent a
``Retry-After`` header, which is honoured (capped). With base 2 s, cap 60 s and 6 attempts,
jittered waits total at most 2 + 4 + 8 + 16 + 32 = 62 s per call; if the server asks for
long waits with Retry-After, at most five waits of 60 s (5 minutes). Every retry is logged
(logger ``holdout.llm``, WARNING) with the status, attempt and wait, and reported to an
optional ``on_retry`` callback for progress output.

This module owns retries. The google-genai SDK's own (tenacity) retries are disabled
explicitly in :class:`holdout.llm.client.GeminiClient` (``HttpRetryOptions(attempts=1)``),
so the two can never multiply. Retries cost no tokens but are counted in the manifest.
"""

import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final, Literal

from holdout.core.errors import LLMBillingError, LLMRequestError, LLMUnavailableError

#: HTTP 502 added by prereg Amendment 4 (a Bad Gateway failed a ReAct unit unretried).
TRANSIENT_STATUS: Final = frozenset({429, 500, 502, 503, 504})

_log = logging.getLogger("holdout.llm")


@dataclass(frozen=True)
class RetryPolicy:
    base_seconds: float = 2.0
    cap_seconds: float = 60.0
    max_attempts: int = 6


#: Flex requests are served from spare capacity and may be refused with 503/429 while it is
#: scarce; Google's Flex guidance is to retry with exponential backoff and to allow long
#: waits. Up to 10 attempts, full jitter from base 5 s, each wait (and Retry-After) capped at
#: 300 s (prereg Amendment 1). Standard-tier calls keep ``RetryPolicy()``.
FLEX_RETRY_POLICY: Final = RetryPolicy(base_seconds=5.0, cap_seconds=300.0, max_attempts=10)


@dataclass(frozen=True)
class RetryEvent:
    """One failed attempt that will be retried."""

    status: int | None
    kind: Literal["http", "timeout", "connection"]
    attempt: int  # the attempt that failed (1-based)
    wait_seconds: float
    retry_after: float | None


class TransientLLMError(Exception):
    """A retryable failure (used by test doubles; real ones come from the SDK)."""

    def __init__(self, status: int | None, retry_after: float | None = None) -> None:
        super().__init__(f"transient LLM failure (status {status})")
        self.status = status
        self.retry_after = retry_after


class BillingLLMError(Exception):
    """A billing / quota-exhausted refusal (used by test doubles)."""

    def __init__(self, status: int | None = 429, message: str = "prepaid credits depleted"):
        super().__init__(f"{status} {message}")
        self.status = status
        self.code = status


#: Words in an API error that mean money or an exhausted quota, not a passing overload.
_BILLING = re.compile(
    r"billing|prepa(?:y|id)|credits?\b|payment|insufficient (?:funds|balance)|"
    r"account balance|quota",
    re.IGNORECASE,
)
#: Markers of a short-window rate limit: Gemini's ordinary 429 also says "exceeded your
#: current quota ... check your plan and billing details", so these win over _BILLING.
_RATE_LIMIT = re.compile(r"per[ _-]?minute|retryDelay|retry in \d", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


@dataclass(frozen=True)
class Failure:
    transient: bool
    status: int | None
    kind: Literal["http", "timeout", "connection", "billing"]
    retry_after: float | None


def classify(exc: BaseException) -> Failure:
    """Decide whether ``exc`` is worth retrying, and read any Retry-After header.

    Billing failures are recognised from the status and the error text, because the API
    has no dedicated code for them: HTTP 402, or a 400 / 403 / 429 whose text names
    billing, prepaid credit, payment, a balance or a quota. A 429 that carries rate-limit
    markers (a per-minute quota, a ``retryDelay``, "retry in N s") is an ordinary rate limit
    and is retried, even though Gemini's rate-limit text also mentions quota and billing.
    Billing failures are never retried.
    """
    if isinstance(exc, BillingLLMError):
        return Failure(False, exc.status, "billing", None)
    if isinstance(exc, TransientLLMError):
        return Failure(True, exc.status, "http", exc.retry_after)
    if _is_timeout(exc):
        return Failure(True, None, "timeout", None)
    if _is_dropped_connection(exc):
        return Failure(True, None, "connection", None)
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status = status if isinstance(status, int) else None
    text = str(exc)
    rate_limited = status == 429 and _RATE_LIMIT.search(text) is not None
    if status == 402 or (
        status in {400, 403, 429} and not rate_limited and _BILLING.search(text) is not None
    ):
        return Failure(False, status, "billing", None)
    return Failure(status in TRANSIENT_STATUS, status, "http", _retry_after(exc))


def _reason(exc: BaseException) -> str:
    """The API's own explanation of a rejected request (it never contains the API key),
    truncated: without it a 400 cannot be diagnosed from the results."""
    message = getattr(exc, "message", None)
    if not message:
        return ""
    return f": {' '.join(str(message).split())[:300]}"


def _is_timeout(exc: BaseException) -> bool:
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        return isinstance(exc, TimeoutError)
    return isinstance(exc, TimeoutError | httpx.TimeoutException)


#: Transport failures with no HTTP response (the server closed or reset the connection);
#: by class name too, because httpcore's own classes are not httpx's.
_DROPPED: Final = frozenset({
    "RemoteProtocolError", "ConnectError", "ReadError", "WriteError", "LocalProtocolError",
    "ProtocolError", "NetworkError",
})  # fmt: skip


def _is_dropped_connection(exc: BaseException) -> bool:
    """A connection dropped before any HTTP response (prereg Amendment 1: transient, like a
    503; seen during the Flex outage as httpx.RemoteProtocolError)."""
    if isinstance(exc, ConnectionError) or type(exc).__name__ in _DROPPED:
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        return False
    return isinstance(exc, httpx.TransportError)


def _retry_after(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait: the Retry-After header, else the RetryInfo
    ``retryDelay`` in a Google API error body."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = None if headers is None else headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        delay = _RETRY_DELAY.search(str(exc))
        return float(delay.group(1)) if delay else None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(str(value))
        except (TypeError, ValueError):
            return None
        return max(0.0, (when - datetime.now(UTC)).total_seconds())


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None],
    rng: random.Random,
    on_retry: Callable[[RetryEvent], None] | None = None,
    what: str = "LLM call",
) -> tuple[T, int]:
    """Run ``fn`` with retries; return (result, number of retries).

    Raises:
        LLMRequestError: on a non-transient failure (never retried).
        LLMUnavailableError: when transient failures exhaust every attempt.
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(), attempt - 1
        except Exception as exc:
            failure = classify(exc)
            label = {"timeout": "a timeout", "connection": "a dropped connection"}.get(
                failure.kind, f"HTTP {failure.status}"
            )
            if failure.kind == "billing":
                raise LLMBillingError(
                    f"{what} was refused for billing or quota reasons ({label}); not retried",
                    status=failure.status,
                    retries=attempt - 1,
                ) from exc
            if not failure.transient:
                raise LLMRequestError(
                    f"{what} failed with {label} ({type(exc).__name__}); not retried"
                    + _reason(exc),
                    status=failure.status,
                ) from exc
            if attempt == policy.max_attempts:
                raise LLMUnavailableError(
                    f"{what} failed with {label} on all {policy.max_attempts} attempts",
                    status=failure.status,
                    retries=attempt - 1,
                ) from exc
            ceiling = min(policy.cap_seconds, policy.base_seconds * 2 ** (attempt - 1))
            wait = (
                min(policy.cap_seconds, failure.retry_after)
                if failure.retry_after is not None
                else rng.uniform(0.0, ceiling)
            )
            _log.warning(
                "%s: %s on attempt %d of %d; retrying in %.1f s",
                what,
                label,
                attempt,
                policy.max_attempts,
                wait,
            )
            if on_retry is not None:
                on_retry(
                    RetryEvent(failure.status, failure.kind, attempt, wait, failure.retry_after)
                )
            sleep(wait)
    raise AssertionError("unreachable")


def call_with_flex_fallback[T](
    fn: Callable[[str | None], T],
    *,
    tier: str | None,
    policy: RetryPolicy,
    flex_policy: RetryPolicy = FLEX_RETRY_POLICY,
    sleep: Callable[[float], None],
    rng: random.Random,
    on_retry: Callable[[RetryEvent], None] | None = None,
    what: str = "LLM call",
) -> tuple[T, int, bool]:
    """Run ``fn(tier)`` with retries; return (result, retries, fell back).

    A Flex request uses ``flex_policy``. If it is still unavailable after every attempt,
    the SAME request is sent once more on the standard tier with the standard ``policy``
    (prereg Amendment 1): same model, same prompt, same parameters. Any other tier uses
    ``policy`` and never falls back. Billing and non-transient failures are never retried
    and never fall back.

    Raises:
        LLMUnavailableError: when the standard fallback is unavailable too (``retries``
            counts every failed attempt on both tiers).
    """
    if tier != "flex":
        result, retries = call_with_retries(
            lambda: fn(tier), policy=policy, sleep=sleep, rng=rng, on_retry=on_retry, what=what
        )
        return result, retries, False
    try:
        result, retries = call_with_retries(
            lambda: fn("flex"), policy=flex_policy, sleep=sleep, rng=rng, on_retry=on_retry,
            what=f"{what} [flex]",
        )  # fmt: skip
        return result, retries, False
    except LLMUnavailableError as exc:
        flex_failures = exc.retries + 1
        _log.warning("%s: Flex unavailable after %d attempts; one fallback to the standard "
                     "tier", what, flex_failures)  # fmt: skip
    try:
        result, retries = call_with_retries(
            lambda: fn("standard"), policy=policy, sleep=sleep, rng=rng, on_retry=on_retry,
            what=f"{what} [standard fallback]",
        )  # fmt: skip
    except LLMUnavailableError as exc:
        raise LLMUnavailableError(
            f"{exc} (after {flex_failures} failed Flex attempts)",
            status=exc.status,
            retries=exc.retries + flex_failures,
        ) from exc
    return result, retries + flex_failures, True
