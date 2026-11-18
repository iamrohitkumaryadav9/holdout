"""Spend accounting in rupees, from real token counts, with run and project caps.

Each call is checked *before* it is sent, against a worst-case estimate (the prompt's
characters / 4 plus the full ``max_output_tokens``, priced as output), so a single call can
never push spend past a cap. After the call it is charged with the real counts from the
response's usage metadata (prompt, output and thinking tokens). The project total lives in
an append-only ledger outside the repository (``~/.holdout/spend_ledger.jsonl``) so it
persists across runs, processes and scripts.

The ledger holds four kinds of line: ``call`` (a paid call, with its cost and retries),
``cache_hit`` and ``failed`` (free; kept for the call statistics) and ``balance`` (the
prepaid balance Google showed, recorded with ``holdout spend --set-balance``). Lines written
before kinds existed are paid calls.
"""

import json
import logging
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from holdout.core.errors import BudgetExceededError
from holdout.llm import config
from holdout.llm.client import LLMRequest, Usage
from holdout.llm.filelock import locked

_log = logging.getLogger("holdout.llm")


def billing_tier(usage: Usage) -> str:
    """The tier a call is priced at: the tier that served it when reported as standard or
    flex; anything else (not reported, priority, ...) is priced as standard - the
    conservative choice for a spend guard."""
    return usage.tier if usage.tier in ("standard", "flex") else "standard"


def cost_inr(model: str, usage: Usage, on: date, *, cache_discount: bool | None = None) -> float:
    """Rupees for a call at the tier that served it; thinking is billed at the output rate.

    ``cache_discount`` (default: ``HOLDOUT_TRUST_CACHE_DISCOUNT``) prices the prompt
    tokens served from a context cache at the cached-input rate; otherwise every prompt
    token is priced at the full input rate (the list price, used for caps until the
    discount is confirmed on Google's bill).
    """
    if cache_discount is None:
        cache_discount = config.trust_cache_discount()
    price = config.price_for(model, on, billing_tier(usage))
    cached = min(max(usage.cached_tokens, 0), usage.prompt_tokens)
    cached_rate = price.cached_input_usd_per_million
    if not cache_discount or cached_rate is None:
        cached_rate = price.input_usd_per_million
    usd = (
        (usage.prompt_tokens - cached) * price.input_usd_per_million
        + cached * cached_rate
        + (usage.output_tokens + usage.thinking_tokens) * price.output_usd_per_million
    ) / 1_000_000
    return usd * config.USD_TO_INR


def estimate_inr(model: str, request: LLMRequest, on: date) -> float:
    """Worst-case cost before sending: every allowed output token is used, no cache hit,
    at the standard rate (a Flex request is never cheaper than this)."""
    worst = Usage(
        prompt_tokens=(len(request.system) + len(request.user)) // 4 + 1,
        output_tokens=request.max_output_tokens,
        thinking_tokens=0,
    )
    return cost_inr(model, worst, on)


def default_ledger_path() -> Path:
    return config.holdout_home() / "spend_ledger.jsonl"


@dataclass(frozen=True)
class BalanceRecord:
    amount_inr: float
    as_of: date  # local calendar date the amount was shown by Google
    recorded_at: str


@dataclass
class SpendLedger:
    """Append-only JSON-lines ledger of every Gemini call, across runs and processes."""

    path: Path = field(default_factory=default_ledger_path)

    def entries(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with locked(self.path):  # another process may be appending (bench.run --workers)
            text = self.path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                entry: dict[str, Any] = json.loads(line)
                entry.setdefault("kind", "call")
                yield entry

    def calls(self) -> list[dict[str, Any]]:
        return [e for e in self.entries() if e["kind"] == "call"]

    def total_inr(self) -> float:
        return sum(float(e["inr"]) for e in self.calls())

    def total_list_inr(self) -> float:
        """Every call at the list price (cached tokens at the full input rate). Entries from
        before cached tokens were recorded count at what they were charged (list)."""
        return sum(float(e.get("inr_list", e["inr"])) for e in self.calls())

    def total_cache_adjusted_inr(self) -> float:
        """Every call with cached tokens at the cached-input rate (what Google should bill
        if the implicit-cache discount applies)."""
        return sum(float(e.get("inr_cache_adjusted", e["inr"])) for e in self.calls())

    def append(self, entry: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, sort_keys=True) + "\n"
        with locked(self.path), self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------ prepaid balance

    def record_balance(self, amount_inr: float, as_of: date) -> BalanceRecord:
        if not (amount_inr >= 0 and math.isfinite(amount_inr)):
            raise ValueError(f"a balance must be zero or more rupees, got {amount_inr}")
        record = BalanceRecord(amount_inr, as_of, datetime.now(UTC).isoformat())
        self.append(
            {
                "kind": "balance",
                "amount_inr": amount_inr,
                "as_of": as_of.isoformat(),
                "timestamp": record.recorded_at,
            }
        )
        return record

    def latest_balance(self) -> BalanceRecord | None:
        latest: BalanceRecord | None = None
        for e in self.entries():
            if e["kind"] == "balance":
                latest = BalanceRecord(
                    float(e["amount_inr"]), date.fromisoformat(e["as_of"]), e["timestamp"]
                )
        return latest

    def spent_since(self, day: date) -> float:
        """Spend on calls made on or after ``day`` (local calendar date)."""
        return sum(float(e["inr"]) for e in self.calls() if local_date(e["timestamp"]) >= day)

    def balance_left(self) -> float | None:
        """Recorded balance minus ledger spend since its date (start of that day, which
        errs low: Google's figure may already include part of that day's spend)."""
        balance = self.latest_balance()
        if balance is None:
            return None
        return balance.amount_inr - self.spent_since(balance.as_of)


def local_date(timestamp: str) -> date:
    return datetime.fromisoformat(timestamp).astimezone().date()


def balance_warning(left: float | None) -> str | None:
    """The standing warning for a remaining balance, if any."""
    if left is None:
        return None
    if left < config.BALANCE_TOP_UP_NOW_INR:
        return (
            f"Prepaid balance left is about Rs {left:.2f} (below Rs "
            f"{config.BALANCE_TOP_UP_NOW_INR:.0f}): top up now; calls may start failing."
        )
    if left < config.BALANCE_TOP_UP_SOON_INR:
        return (
            f"Prepaid balance left is about Rs {left:.2f} (below Rs "
            f"{config.BALANCE_TOP_UP_SOON_INR:.0f}): top up soon."
        )
    return None


@dataclass(frozen=True)
class SpendSnapshot:
    """Running totals after a call, for progress lines."""

    call_inr: float
    run_inr: float
    run_cap_inr: float
    project_inr: float
    project_cap_inr: float
    balance_left_inr: float | None

    def line(self) -> str:
        balance = (
            "balance left unknown (holdout spend --set-balance)"
            if self.balance_left_inr is None
            else f"balance left about Rs {self.balance_left_inr:.2f}"
        )
        return (
            f"this call Rs {self.call_inr:.3f} | run Rs {self.run_inr:.2f} of "
            f"{self.run_cap_inr:g} | project Rs {self.project_inr:.2f} of "
            f"{self.project_cap_inr:g} | {balance}"
        )


@dataclass
class Budget:
    run_id: str
    model: str
    # Read at construction (not import) so configuration changes apply to new budgets.
    run_cap_inr: float = field(default_factory=lambda: config.RUN_CAP_INR)
    project_cap_inr: float = field(default_factory=lambda: config.project_cap_inr())
    ledger: SpendLedger = field(default_factory=SpendLedger)
    spent_run_inr: float = 0.0
    today: date = field(default_factory=lambda: datetime.now(UTC).date())
    #: "general" spend may not go into the benchmark reserve; "benchmark" spend may.
    purpose: str = field(default_factory=lambda: config.spend_purpose())
    reserve_inr: float = field(default_factory=lambda: config.BENCHMARK_RESERVE_INR)

    def check(self, request: LLMRequest, node: str) -> float:
        """Raise before a call that could exceed a cap; return the worst-case estimate."""
        estimate = estimate_inr(self.model, request, self.today)
        project = self.ledger.total_inr()
        if self.spent_run_inr + estimate > self.run_cap_inr:
            raise BudgetExceededError(
                f"run cap: Rs {self.spent_run_inr:.2f} spent in this run, the next call "
                f"({node}) could cost up to Rs {estimate:.2f}, cap Rs {self.run_cap_inr:.2f}",
                scope="run",
                cap_inr=self.run_cap_inr,
                run_spent_inr=self.spent_run_inr,
                project_spent_inr=project,
                estimate_inr=estimate,
            )
        if project + estimate > self.project_cap_inr:
            raise BudgetExceededError(
                f"project cap: Rs {project:.2f} spent across runs, the next call ({node}) "
                f"could cost up to Rs {estimate:.2f}, cap Rs {self.project_cap_inr:.2f}",
                scope="project",
                cap_inr=self.project_cap_inr,
                run_spent_inr=self.spent_run_inr,
                project_spent_inr=project,
                estimate_inr=estimate,
            )
        general_cap = self.project_cap_inr - self.reserve_inr
        if self.purpose != "benchmark" and project + estimate > general_cap:
            raise BudgetExceededError(
                f"benchmark reserve: Rs {project:.2f} spent across runs, the next call "
                f"({node}) could cost up to Rs {estimate:.2f}; spend that is not the "
                f"benchmark may not exceed Rs {general_cap:.2f} (cap Rs "
                f"{self.project_cap_inr:g} minus the Rs {self.reserve_inr:g} benchmark reserve)",
                scope="reserve",
                cap_inr=general_cap,
                run_spent_inr=self.spent_run_inr,
                project_spent_inr=project,
                estimate_inr=estimate,
            )
        return estimate

    def charge(
        self, usage: Usage, *, node: str, prompt_name: str, retries: int = 0
    ) -> tuple[float, list[str]]:
        """Record the real cost of a call (in the run and the project ledger).

        Returns the cost and the warnings this call triggered: each project-cap fraction
        (50 / 75 / 90 %) and each balance level (below Rs 150, below Rs 50) is announced
        by the call that crosses it, so each is printed once.
        """
        inr = cost_inr(self.model, usage, self.today)
        inr_list = cost_inr(self.model, usage, self.today, cache_discount=False)
        inr_cached = cost_inr(self.model, usage, self.today, cache_discount=True)
        project_before = self.ledger.total_inr()
        left_before = self.ledger.balance_left()
        self.spent_run_inr += inr
        self.ledger.append(
            {
                "kind": "call",
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": self.run_id,
                "node": node,
                "prompt": prompt_name,
                "model": self.model,
                "prompt_tokens": usage.prompt_tokens,
                "output_tokens": usage.output_tokens,
                "thinking_tokens": usage.thinking_tokens,
                "cached_tokens": usage.cached_tokens,
                "tier_requested": usage.tier_requested,
                "tier_served": usage.tier,
                "tier_billed": billing_tier(usage),
                "fell_back": usage.fell_back,
                # Both valuations; "inr" (what the caps count) is one of them, chosen by
                # HOLDOUT_TRUST_CACHE_DISCOUNT.
                "inr_list": inr_list,
                "inr_cache_adjusted": inr_cached,
                "retries": retries,
                "purpose": self.purpose,
                "inr": inr,
            }
        )
        warnings = self._crossings(project_before, project_before + inr, left_before, inr)
        for text in warnings:
            _log.warning("%s", text)
        return inr, warnings

    def record_free(self, kind: str, *, node: str, prompt_name: str, retries: int = 0) -> None:
        """A cache hit or a failed call: no cost, kept for the call statistics."""
        self.ledger.append(
            {
                "kind": kind,
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": self.run_id,
                "node": node,
                "prompt": prompt_name,
                "model": self.model,
                "retries": retries,
                "purpose": self.purpose,
                "inr": 0.0,
            }
        )

    def snapshot(self, call_inr: float) -> SpendSnapshot:
        return SpendSnapshot(
            call_inr=call_inr,
            run_inr=self.spent_run_inr,
            run_cap_inr=self.run_cap_inr,
            project_inr=self.ledger.total_inr(),
            project_cap_inr=self.project_cap_inr,
            balance_left_inr=self.ledger.balance_left(),
        )

    def _crossings(
        self, before: float, after: float, left_before: float | None, inr: float
    ) -> list[str]:
        warnings = []
        for fraction in config.PROJECT_WARN_FRACTIONS:
            mark = fraction * self.project_cap_inr
            if before < mark <= after:
                warnings.append(
                    f"PROJECT SPEND PASSED {fraction:.0%} OF THE CAP: Rs {after:.2f} of "
                    f"Rs {self.project_cap_inr:g} spent (mark Rs {mark:.2f})."
                )
        if left_before is not None:
            left_after = left_before - inr
            for level in (config.BALANCE_TOP_UP_SOON_INR, config.BALANCE_TOP_UP_NOW_INR):
                if left_before >= level > left_after:
                    message = balance_warning(left_after)
                    if message is not None and message not in warnings:
                        warnings.append(message)
        return warnings
