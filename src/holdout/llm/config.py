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
REQUEST_TIMEOUT_SECONDS: Final = 120
#: Retries for rate limits (429) and server errors (5xx), with exponential backoff.
MAX_TRANSIENT_RETRIES: Final = 3

#: Spend caps in rupees.
RUN_CAP_INR: Final = 15.0
PROJECT_CAP_INR: Final = 1200.0

USD_TO_INR: Final = 94.1
EXCHANGE_RATE_AS_OF: Final = date(2026, 11, 13)
PRICES_AS_OF: Final = date(2026, 11, 17)
PRICE_NOTE: Final = (
    "Gemini 3.8 Flash paid-tier prices double from 2027-01-01 "
    "(input 0.75 -> 1.50, output 3.75 -> 7.50 USD per 1M tokens)."
)


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens from ``effective_from`` (output includes thinking tokens)."""

    effective_from: date
    input_usd_per_million: float
    output_usd_per_million: float


PRICE_TABLE: Final[dict[str, tuple[Price, ...]]] = {
    "gemini-3.8-flash": (
        Price(date(2026, 1, 1), 0.75, 3.75),
        Price(date(2027, 1, 1), 1.50, 7.50),
    ),
}


def price_for(model: str, on: date) -> Price:
    """The price in force for ``model`` on day ``on``.

    Raises:
        KeyError: for a model without a price, so spend can never be computed as zero.
    """
    entries = [p for p in PRICE_TABLE[model] if p.effective_from <= on]
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
