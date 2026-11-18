"""LLM configuration: the only place a model name, a price or a cap is written down.

Sources, checked on 2026-11-17 (see docs/design.md, "Agent"):

* models: https://aistudio.google.com/docs/models - ``gemini-3.8-flash`` is the current
  stable Flash model.
* thinking: https://aistudio.google.com/docs/thinking - Gemini 3.x is controlled with
  ``thinking_level`` (``gemini-3.8-flash`` supports low / medium / high; thinking cannot be
  switched off). Holdout uses ``low`` and bounds thinking plus answer with
  ``max_output_tokens``.
* prices: https://aistudio.google.com/docs/pricing - paid tier, USD per 1M tokens, output
  price includes thinking tokens; the promotional price doubles from 2027-01-01.
* exchange rate: about 94.1 INR per USD on 2026-11-13 (mid-market quotes from news and
  rate-aggregator sources); update ``USD_TO_INR`` and ``EXCHANGE_RATE_AS_OF`` when it moves.
"""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

#: The model every agent call uses. Never written anywhere else.
DEFAULT_MODEL: Final = "gemini-3.8-flash"

#: Thinking cap for every call (Gemini 3.x ``thinking_level``).
THINKING_LEVEL: Final = "low"

#: Upper bound on answer + thinking tokens per call; also the pre-call cost estimate.
MAX_OUTPUT_TOKENS: Final = 4096

TEMPERATURE: Final = 0.0
#: Per-request timeout; a timeout is transient and retried by holdout.llm.retry.
REQUEST_TIMEOUT_SECONDS: Final = 90
# Retries for 429 / 5xx / timeouts: see holdout.llm.retry.RetryPolicy (the single owner).

#: Spend caps in rupees. The project cap covers every Gemini call Holdout ever makes (agent
#: runs, the smoke and ping scripts); only ``HOLDOUT_PROJECT_CAP_INR`` overrides it.
RUN_CAP_INR: Final = 15.0
PROJECT_CAP_INR: Final = 750.0
PROJECT_CAP_ENV: Final = "HOLDOUT_PROJECT_CAP_INR"
#: Rupees of the project cap kept for the final benchmark run. Spend of any other purpose
#: (dev, tuning, smoke, ping, ordinary agent runs) may not go into it.
BENCHMARK_RESERVE_INR: Final = 600.0
#: The purpose of a process's spend: "general" unless this is "benchmark" (set only by
#: ``python -m bench.run --purpose benchmark``).
SPEND_PURPOSE_ENV: Final = "HOLDOUT_SPEND_PURPOSE"
SPEND_PURPOSES: Final = ("general", "benchmark")
#: Fractions of the project cap at which a one-time warning is printed.
PROJECT_WARN_FRACTIONS: Final = (0.50, 0.75, 0.90)
#: Remaining prepaid balance (rupees) below which Holdout warns.
BALANCE_TOP_UP_SOON_INR: Final = 150.0
BALANCE_TOP_UP_NOW_INR: Final = 50.0


def spend_purpose() -> str:
    """``HOLDOUT_SPEND_PURPOSE`` (default "general").

    Raises:
        LLMConfigError: for an unknown purpose.
    """
    value = os.environ.get(SPEND_PURPOSE_ENV, "general").strip() or "general"
    if value not in SPEND_PURPOSES:
        from holdout.core.errors import LLMConfigError

        raise LLMConfigError(f"{SPEND_PURPOSE_ENV} must be one of {SPEND_PURPOSES}, got {value!r}")
    return value


def project_cap_inr() -> float:
    """The project cap: ``HOLDOUT_PROJECT_CAP_INR`` if set, else ``PROJECT_CAP_INR``.

    Raises:
        LLMConfigError: if the variable is set but is not a positive number.
    """
    raw = os.environ.get(PROJECT_CAP_ENV)
    if raw is None or not raw.strip():
        return PROJECT_CAP_INR
    from holdout.core.errors import LLMConfigError

    try:
        value = float(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{PROJECT_CAP_ENV}={raw!r} is not a number of rupees") from exc
    if not value > 0 or value != value or value == float("inf"):
        raise LLMConfigError(f"{PROJECT_CAP_ENV} must be a positive finite amount, got {raw!r}")
    return value


USD_TO_INR: Final = 94.1
EXCHANGE_RATE_AS_OF: Final = date(2026, 11, 13)
PRICES_AS_OF: Final = date(2026, 11, 17)
PRICE_NOTE: Final = (
    "Gemini 3.8 Flash paid-tier prices double from 2027-01-01 "
    "(input 0.75 -> 1.50, output 3.75 -> 7.50 USD per 1M tokens)."
)


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens from ``effective_from`` (output includes thinking tokens).

    ``cached_input_usd_per_million`` is the price of prompt tokens served from a context
    cache (implicit caching); None when not recorded (cached tokens are then priced as
    ordinary input, never cheaper than known).
    """

    effective_from: date
    input_usd_per_million: float
    output_usd_per_million: float
    cached_input_usd_per_million: float | None = None


#: Standard tier. Source: https://aistudio.google.com/docs/pricing, read 2026-11-18 (cached
#: input "Context caching: $0.075 through December 31, 2026 / $0.15 starting January 1, 2027").
PRICE_TABLE: Final[dict[str, tuple[Price, ...]]] = {
    "gemini-3.8-flash": (
        Price(date(2026, 1, 1), 0.75, 3.75, 0.075),
        Price(date(2027, 1, 1), 1.50, 7.50, 0.15),
    ),
    # Alternative for ``--model`` (same pricing page, same day). No price change is
    # announced; add a dated entry here when one is. (gemini-3-flash-preview was removed: it
    # returns 404 for this project's key, so it cannot be pinged or used.) Its cached-input
    # price was not checked, so cached tokens are priced as input.
    "gemini-3.6-flash": (Price(date(2026, 1, 1), 0.75, 3.75),),
}

#: Flex tier (50% of standard). Sources, read 2026-11-18: the pricing page lists Flex for
#: gemini-3.8-flash at input $0.375 / output $1.875 / context caching $0.0375 through
#: 2026-12-31 and $0.75 / $3.75 / $0.075 from 2027-01-01; the Flex inference page
#: (https://aistudio.google.com/docs/flex-inference) says Flex is "priced at 50% of the
#: standard API". A model without an entry here cannot be run on Flex.
FLEX_PRICE_TABLE: Final[dict[str, tuple[Price, ...]]] = {
    "gemini-3.8-flash": (
        Price(date(2026, 1, 1), 0.375, 1.875, 0.0375),
        Price(date(2027, 1, 1), 0.75, 3.75, 0.075),
    ),
}

#: Service tiers a call may request (``GenerateContentConfig.service_tier``).
SERVICE_TIERS: Final = ("standard", "flex")
#: Client timeout under Flex: requests may queue for minutes ("1-15 min target"; the Flex
#: page recommends "10 minutes or more").
FLEX_REQUEST_TIMEOUT_SECONDS: Final = 900
#: Use the cached-input price for caps and charges. Off until the discount is confirmed on
#: Google's bill; until then cached tokens are charged at the full input price (list).
TRUST_CACHE_DISCOUNT_ENV: Final = "HOLDOUT_TRUST_CACHE_DISCOUNT"


def trust_cache_discount() -> bool:
    return os.environ.get(TRUST_CACHE_DISCOUNT_ENV, "").strip().lower() in {"1", "true", "yes"}


def require_priced(model: str, on: date, tier: str = "standard") -> None:
    """Refuse a model (or a tier) Holdout cannot price: spend must never be computed as zero.

    Raises:
        LLMConfigError: with the priced models listed.
    """
    from holdout.core.errors import LLMConfigError

    try:
        price_for(model, on, tier)
    except KeyError as exc:
        table = FLEX_PRICE_TABLE if tier == "flex" else PRICE_TABLE
        raise LLMConfigError(
            f"model {model!r} has no price for the {tier} tier in holdout.llm.config; "
            f"priced models for {tier}: {', '.join(sorted(table))}"
        ) from exc


def price_for(model: str, on: date, tier: str = "standard") -> Price:
    """The price in force for ``model`` on day ``on`` in service ``tier``.

    Raises:
        KeyError: for a model (or tier) without a price, so spend can never be computed as
            zero.
    """
    table = FLEX_PRICE_TABLE if tier == "flex" else PRICE_TABLE
    if tier not in SERVICE_TIERS:
        raise KeyError(f"unknown service tier {tier!r}")
    entries = [p for p in table[model] if p.effective_from <= on]
    if not entries:
        raise KeyError(f"no price for {model!r} on {on}")
    return max(entries, key=lambda p: p.effective_from)


def holdout_home() -> Path:
    """Per-user state outside the repository (spend ledger, LLM cache).

    ``HOLDOUT_HOME`` overrides ``~/.holdout`` (tests point it at a temporary directory).
    """
    override = os.environ.get("HOLDOUT_HOME")
    return Path(override) if override else Path.home() / ".holdout"


#: Environment variable holding the API key. The key is read only from here, and never
#: logged, cached, checkpointed or written to a manifest.
API_KEY_ENV: Final = "GEMINI_API_KEY"
