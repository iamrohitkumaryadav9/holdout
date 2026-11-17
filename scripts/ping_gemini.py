"""Ping Gemini: one minimal request per model (manual; never run in CI).

Sends "Reply with the single word OK" once to each model (default: every priced model),
with a 30 s timeout, one attempt (no retries), no cache, and prints the HTTP status,
latency, tokens and cost. The calls obey the same per-run and project caps as agent runs
and are charged to the same ledger; project spend and the prepaid balance left are printed
before and after. Requires GEMINI_API_KEY in the environment.

    python -m scripts.ping_gemini
    python -m scripts.ping_gemini --model gemini-3.8-flash

Exit codes: 0 every model answered; 1 a request was rejected (e.g. HTTP 400 / 404);
5 a model was unavailable or timed out; 6 a spend cap refused the call; 7 billing / quota.
"""

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from typing import Final

from holdout.core.errors import (
    BudgetExceededError,
    LLMBillingError,
    LLMConfigError,
    LLMRequestError,
    LLMUnavailableError,
)
from holdout.llm import config
from holdout.llm.budget import Budget
from holdout.llm.client import GeminiClient, LLMRequest
from holdout.llm.retry import RetryPolicy
from holdout.llm.spend import build_report, headline

PROMPT: Final = "Reply with the single word OK"
TIMEOUT_SECONDS: Final = 30.0
#: Room for low-level thinking plus the one-word answer (thinking counts toward this).
MAX_OUTPUT_TOKENS: Final = 256

EXIT_REJECTED, EXIT_UNAVAILABLE, EXIT_BUDGET, EXIT_BILLING = 1, 5, 6, 7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        action="append",
        help="model to ping (repeatable; default: every priced model)",
    )
    args = parser.parse_args()
    if not os.environ.get(config.API_KEY_ENV):
        print(f"{config.API_KEY_ENV} is not set.", file=sys.stderr)
        return 2
    models: list[str] = args.model or sorted(config.PRICE_TABLE)
    try:
        for model in models:
            config.require_priced(model, datetime.now(UTC).date())
        before = build_report()
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"before: {headline(before)}")
    _balance_warning(before.balance.warning if before.balance else None)

    run_id = f"ping-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    codes: list[int] = []
    for model in models:
        code = _ping(model, run_id)
        codes.append(code)
        if code == EXIT_BUDGET:
            break  # a cap refused the call: do not try the next model

    after = build_report()
    print(f"after:  {headline(after)}")
    _balance_warning(after.balance.warning if after.balance else None)
    for code in (EXIT_BUDGET, EXIT_BILLING, EXIT_UNAVAILABLE, EXIT_REJECTED):
        if code in codes:
            return code
    return 0


def _ping(model: str, run_id: str) -> int:
    request = LLMRequest(
        prompt_name="ping",
        prompt_version="1",
        system="",
        user=PROMPT,
        response_schema={},  # plain text
        seed=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    budget = Budget(run_id=run_id, model=model)
    label = f"{model:24}"
    try:
        budget.check(request, "ping")
    except BudgetExceededError as exc:
        print(f"{label} NOT SENT: {exc}")
        return EXIT_BUDGET
    client = GeminiClient(
        model, policy=RetryPolicy(max_attempts=1), timeout_seconds=TIMEOUT_SECONDS
    )
    started = time.perf_counter()
    try:
        response = client.generate(request)
    except (LLMBillingError, LLMUnavailableError, LLMRequestError) as exc:
        seconds = time.perf_counter() - started
        status = getattr(exc, "status", None)
        shown = "timeout" if status is None else f"HTTP {status}"
        retries = getattr(exc, "retries", 0)
        budget.record_free("failed", node="ping", prompt_name="ping", retries=retries)
        if isinstance(exc, LLMBillingError):
            print(f"{label} BILLING {shown} after {seconds:.2f} s: {exc} (top up, then retry)")
            return EXIT_BILLING
        if isinstance(exc, LLMUnavailableError):
            print(f"{label} UNAVAILABLE {shown} after {seconds:.2f} s: {exc}")
            return EXIT_UNAVAILABLE
        print(f"{label} REJECTED {shown} after {seconds:.2f} s: {exc}")
        return EXIT_REJECTED
    seconds = time.perf_counter() - started
    inr, warnings = budget.charge(response.usage, node="ping", prompt_name="ping")
    usage = response.usage
    print(
        f"{label} OK HTTP 200  {seconds:6.2f} s  tokens in {usage.prompt_tokens} out "
        f"{usage.output_tokens} thinking {usage.thinking_tokens}  reply {response.text.strip()!r}"
    )
    print(f"{'':24} {budget.snapshot(inr).line()}")
    for text in warnings:
        print(f"!!! WARNING: {text}")
    return 0


def _balance_warning(text: str | None) -> None:
    if text is not None:
        print(f"!!! WARNING: {text}")


if __name__ == "__main__":
    raise SystemExit(main())
