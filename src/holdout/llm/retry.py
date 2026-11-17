"""Retries for transient LLM failures: exponential backoff with full jitter.

Transient: HTTP 429, 500, 503, 504 and request timeouts. Everything else (400, 401, 403,
schema problems) fails at once. The wait before retry ``n`` (1-based) is drawn uniformly
from ``[0, min(cap, base * 2**(n-1))]`` ("full jitter"), unless the server sent a
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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final, Literal

from holdout.core.errors import LLMRequestError, LLMUnavailableError

TRANSIENT_STATUS: Final = frozenset({429, 500, 503, 504})

_log = logging.getLogger("holdout.llm")


@dataclass(frozen=True)
class RetryPolicy:
    base_seconds: float = 2.0
    cap_seconds: float = 60.0
    max_attempts: int = 6


@dataclass(frozen=True)
class RetryEvent:
    """One failed attempt that will be retried."""

    status: int | None
    kind: Literal["http", "timeout"]
    attempt: int  # the attempt that failed (1-based)
    wait_seconds: float
    retry_after: float | None


class TransientLLMError(Exception):
    """A retryable failure (used by test doubles; real ones come from the SDK)."""

    def __init__(self, status: int | None, retry_after: float | None = None) -> None:
        super().__init__(f"transient LLM failure (status {status})")
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class Failure:
    transient: bool
    status: int | None
    kind: Literal["http", "timeout"]
    retry_after: float | None


def classify(exc: BaseException) -> Failure:
    """Decide whether ``exc`` is worth retrying, and read any Retry-After header."""
    if isinstance(exc, TransientLLMError):
        return Failure(True, exc.status, "http", exc.retry_after)
    if _is_timeout(exc):
        return Failure(True, None, "timeout", None)
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status = status if isinstance(status, int) else None
    return Failure(status in TRANSIENT_STATUS, status, "http", _retry_after(exc))


def _is_timeout(exc: BaseException) -> bool:
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        return isinstance(exc, TimeoutError)
    return isinstance(exc, TimeoutError | httpx.TimeoutException)


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
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
            label = "a timeout" if failure.kind == "timeout" else f"HTTP {failure.status}"
            if not failure.transient:
                raise LLMRequestError(
                    f"{what} failed with {label} ({type(exc).__name__}); not retried",
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
