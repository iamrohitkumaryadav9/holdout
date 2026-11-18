"""Spend reports from the project ledger (``holdout spend``, the ping and smoke scripts).

Everything is computed from the ledger's token-priced entries, live. Google's billing page
updates with a delay, so the two are reconciled by hand after paid runs and top-ups
(``holdout spend --set-balance``).
"""

import os
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from holdout.llm import config
from holdout.llm.budget import SpendLedger, balance_warning, local_date


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunSpend(_Record):
    run_id: str
    last_call_at: str
    calls: int
    inr: float


class ModelSpend(_Record):
    calls: int
    inr: float


class BalanceStatus(_Record):
    recorded_inr: float
    as_of: date
    spent_since_inr: float
    left_inr: float
    warning: str | None


class PriceRow(_Record):
    model: str
    effective_from: date
    input_usd_per_million: float
    output_usd_per_million: float


class SpendReport(_Record):
    ledger_path: str
    today: date
    today_inr: float
    last_7_days_inr: float
    all_time_inr: float
    #: All-time spend valued at list price (cached tokens at the full input rate) and with
    #: the implicit-cache discount; ``all_time_inr`` is what the caps count.
    all_time_list_inr: float = 0.0
    all_time_cache_adjusted_inr: float = 0.0
    cached_tokens: int = 0
    prompt_tokens: int = 0
    trust_cache_discount: bool = False
    runs: list[RunSpend]
    models: dict[str, ModelSpend]
    calls: int
    retries: int
    cache_hits: int
    failed_calls: int
    run_cap_inr: float
    project_cap_inr: float
    project_cap_source: Literal["default", "env"]
    project_remaining_inr: float
    project_used_fraction: float
    balance: BalanceStatus | None
    prices: list[PriceRow]
    usd_to_inr: float
    exchange_rate_as_of: date
    prices_as_of: date
    price_note: str


def build_report(ledger: SpendLedger | None = None, today: date | None = None) -> SpendReport:
    ledger = ledger if ledger is not None else SpendLedger()
    today = today if today is not None else datetime.now().astimezone().date()
    entries = list(ledger.entries())
    calls = [e for e in entries if e["kind"] == "call"]

    def spent(since: date) -> float:
        return sum(float(e["inr"]) for e in calls if local_date(e["timestamp"]) >= since)

    by_run: dict[str, list[dict[str, object]]] = {}
    for e in calls:
        by_run.setdefault(str(e.get("run_id", "?")), []).append(e)
    runs = sorted(
        (
            RunSpend(
                run_id=run_id,
                last_call_at=max(str(e["timestamp"]) for e in items),
                calls=len(items),
                inr=sum(float(str(e["inr"])) for e in items),
            )
            for run_id, items in by_run.items()
        ),
        key=lambda r: r.last_call_at,
        reverse=True,
    )[:10]
    models: dict[str, ModelSpend] = {}
    for e in calls:
        name = str(e.get("model", "?"))
        before = models.get(name, ModelSpend(calls=0, inr=0.0))
        models[name] = ModelSpend(calls=before.calls + 1, inr=before.inr + float(e["inr"]))
    total = sum(float(e["inr"]) for e in calls)
    cap = config.project_cap_inr()
    from_env = bool(os.environ.get(config.PROJECT_CAP_ENV, "").strip())
    recorded = ledger.latest_balance()
    balance = None
    if recorded is not None:
        since = ledger.spent_since(recorded.as_of)
        left = recorded.amount_inr - since
        balance = BalanceStatus(
            recorded_inr=recorded.amount_inr,
            as_of=recorded.as_of,
            spent_since_inr=since,
            left_inr=left,
            warning=balance_warning(left),
        )
    return SpendReport(
        ledger_path=str(ledger.path),
        today=today,
        today_inr=spent(today),
        last_7_days_inr=spent(today - timedelta(days=6)),
        all_time_inr=total,
        all_time_list_inr=ledger.total_list_inr(),
        all_time_cache_adjusted_inr=ledger.total_cache_adjusted_inr(),
        cached_tokens=sum(int(e.get("cached_tokens", 0)) for e in calls),
        prompt_tokens=sum(int(e.get("prompt_tokens", 0)) for e in calls),
        trust_cache_discount=config.trust_cache_discount(),
        runs=runs,
        models=dict(sorted(models.items())),
        calls=len(calls),
        retries=sum(int(e.get("retries", 0)) for e in entries if e["kind"] in {"call", "failed"}),
        cache_hits=sum(e["kind"] == "cache_hit" for e in entries),
        failed_calls=sum(e["kind"] == "failed" for e in entries),
        run_cap_inr=config.RUN_CAP_INR,
        project_cap_inr=cap,
        project_cap_source="env" if from_env else "default",
        project_remaining_inr=cap - total,
        project_used_fraction=total / cap,
        balance=balance,
        prices=[
            PriceRow(
                model=model,
                effective_from=p.effective_from,
                input_usd_per_million=p.input_usd_per_million,
                output_usd_per_million=p.output_usd_per_million,
            )
            for model, rows in sorted(config.PRICE_TABLE.items())
            for p in rows
        ],
        usd_to_inr=config.USD_TO_INR,
        exchange_rate_as_of=config.EXCHANGE_RATE_AS_OF,
        prices_as_of=config.PRICES_AS_OF,
        price_note=config.PRICE_NOTE,
    )


def headline(report: SpendReport) -> str:
    """One line for scripts: project spend, cap and balance left."""
    balance = (
        "balance left unknown (record it with holdout spend --set-balance)"
        if report.balance is None
        else f"balance left about Rs {report.balance.left_inr:.2f}"
    )
    return (
        f"project spend Rs {report.all_time_inr:.2f} of Rs {report.project_cap_inr:g} "
        f"(Rs {report.project_remaining_inr:.2f} left under the cap); {balance}"
    )


def render(report: SpendReport) -> list[str]:
    """The plain-text ``holdout spend`` report."""
    cap_from = (
        f" (from {config.PROJECT_CAP_ENV})" if report.project_cap_source == "env" else " (default)"
    )
    lines = [
        f"Gemini spend (ledger {report.ledger_path})",
        "",
        f"  today ({report.today})   Rs {report.today_inr:9.2f}",
        f"  last 7 days         Rs {report.last_7_days_inr:9.2f}",
        f"  all time            Rs {report.all_time_inr:9.2f}",
        f"    at list price     Rs {report.all_time_list_inr:9.2f} (cached tokens at the "
        "full input rate)",
        f"    cache-adjusted    Rs {report.all_time_cache_adjusted_inr:9.2f} (cached tokens at "
        f"the cached rate; {report.cached_tokens} of {report.prompt_tokens} prompt tokens "
        "cached)",
        "    caps count: "
        + (
            "cache-adjusted (HOLDOUT_TRUST_CACHE_DISCOUNT is set)"
            if report.trust_cache_discount
            else "list price (set HOLDOUT_TRUST_CACHE_DISCOUNT once the discount is on the bill)"
        ),
        "",
        f"  project cap         Rs {report.project_cap_inr:9.2f}{cap_from}; "
        f"used {report.project_used_fraction:.1%}, remaining Rs {report.project_remaining_inr:.2f}",
        f"  per-run cap         Rs {report.run_cap_inr:9.2f}",
    ]
    if report.balance is None:
        lines.append("  prepaid balance     not recorded (holdout spend --set-balance <rupees>)")
    else:
        b = report.balance
        lines += [
            f"  prepaid balance     Rs {b.recorded_inr:9.2f} recorded for {b.as_of}",
            f"  spent since then    Rs {b.spent_since_inr:9.2f} (from the start of {b.as_of})",
            f"  balance left        Rs {b.left_inr:9.2f} (estimate)",
        ]
    lines += [
        "",
        f"  calls {report.calls}, retries {report.retries}, cache hits {report.cache_hits}, "
        f"failed calls {report.failed_calls}",
        "",
        "  latest runs (top 10)",
    ]
    if not report.runs:
        lines.append("    none yet")
    for r in report.runs:
        lines.append(f"    {r.run_id:28} {r.last_call_at[:19]}  calls {r.calls:3}  Rs {r.inr:.3f}")
    lines += ["", "  per model"]
    if not report.models:
        lines.append("    none yet")
    for name, m in report.models.items():
        lines.append(f"    {name:28} calls {m.calls:4}  Rs {m.inr:.3f}")
    lines += [
        "",
        f"  prices (USD per 1M tokens, as of {report.prices_as_of}; output includes thinking)",
    ]
    for p in report.prices:
        lines.append(
            f"    {p.model:28} from {p.effective_from}  input {p.input_usd_per_million:.2f}"
            f"  output {p.output_usd_per_million:.2f}"
        )
    lines += [
        f"  exchange rate {report.usd_to_inr} INR per USD (as of {report.exchange_rate_as_of})",
        f"  {report.price_note}",
        "",
        "  Holdout's ledger is computed live from token counts; Google's billing page updates "
        "with a delay. Reconcile the two after paid runs and after each top-up.",
    ]
    return lines
