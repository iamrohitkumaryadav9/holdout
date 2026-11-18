"""Benchmark budget pre-flight: ``python -m bench.run --suite ... --plan``.

Before a run starts, the cost of every *pending* unit (completed units are excluded) is
estimated per system from the pilot's measured cost per run, plus the configured margin
(20%), rescaled if Gemini's price changes between the pilot date and today (the price
table is dated; the promotional price doubles on 2027-01-01). The total is compared with:

* the remaining project cap for this purpose: ``cap - spent`` for benchmark spend, ``cap -
  reserve - spent`` for anything else (the benchmark reserve);
* the prepaid balance left (recorded with ``holdout spend --set-balance``).

A run that does not fit is refused with every shortfall spelled out; so is any LLM spend
without a recorded balance. An LLM system without a pilot measurement (the pilot itself) is
estimated at its hard worst case, the per-run cap per unit.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from bench.config import BenchConfig
from holdout.llm import config as llm_config
from holdout.llm.budget import SpendLedger

LLM_SYSTEMS = frozenset({"holdout", "llm_judge", "react"})


@dataclass(frozen=True)
class Measured:
    per_run_inr: float
    runs: int
    measured_on: date


@dataclass(frozen=True)
class SystemEstimate:
    system: str
    units: int
    per_run_inr: float | None
    source: str
    total_inr: float


@dataclass
class PreflightReport:
    purpose: str
    estimates: list[SystemEstimate]
    margin: float
    price_factor: float
    total_inr: float
    available_inr: float
    balance_left_inr: float | None
    shortfalls: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return not self.shortfalls

    def render(self) -> str:
        lines = [f"Budget pre-flight ({self.purpose} spend)"]
        for e in self.estimates:
            per = "n/a" if e.per_run_inr is None else f"Rs {e.per_run_inr:.3f}"
            lines.append(
                f"  {e.system:12} {e.units:4} pending units x {per} per run ({e.source})"
                f" = Rs {e.total_inr:.2f}"
            )
        lines.append(
            f"  margin {self.margin:.0%}, price factor {self.price_factor:g}: estimated total "
            f"Rs {self.total_inr:.2f}"
        )
        lines.append(f"  available under the cap for this purpose: Rs {self.available_inr:.2f}")
        balance = (
            "not recorded" if self.balance_left_inr is None else f"Rs {self.balance_left_inr:.2f}"
        )
        lines.append(f"  prepaid balance left: {balance}")
        if self.fits:
            lines.append("  fits: yes")
        else:
            lines.append("  fits: NO")
            lines += [f"  short: {s}" for s in self.shortfalls]
        return "\n".join(lines)


def measured_costs(pilot_root: Path) -> dict[str, Measured]:
    """Mean cost per completed run, per system, from the pilot's results files."""
    costs: dict[str, list[float]] = {}
    dates: dict[str, date] = {}
    for path in sorted(pilot_root.glob("*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            if result["status"] != "completed":
                continue
            costs.setdefault(result["system"], []).append(float(result["cost_inr"]))
            day = datetime.fromisoformat(result["started_at"]).date()
            dates[result["system"]] = min(dates.get(result["system"], day), day)
    return {
        system: Measured(sum(values) / len(values), len(values), dates[system])
        for system, values in costs.items()
    }


def _price_factor(model: str, measured_on: date, today: date) -> float:
    then = llm_config.price_for(model, measured_on)
    now = llm_config.price_for(model, today)
    # Input and output prices change together in the table; use the larger ratio (worst case).
    return max(
        now.input_usd_per_million / then.input_usd_per_million,
        now.output_usd_per_million / then.output_usd_per_million,
    )


def estimate(
    pending_by_system: dict[str, int],
    config: BenchConfig,
    *,
    purpose: str,
    ledger: SpendLedger | None = None,
    measured: dict[str, Measured] | None = None,
    today: date | None = None,
) -> PreflightReport:
    ledger = ledger if ledger is not None else SpendLedger()
    measured = measured if measured is not None else measured_costs(config.pilot_results)
    today = today if today is not None else datetime.now(UTC).date()
    shortfalls: list[str] = []
    estimates: list[SystemEstimate] = []
    factor = 1.0
    for system, units in pending_by_system.items():
        if system not in LLM_SYSTEMS:
            estimates.append(SystemEstimate(system, units, 0.0, "no LLM calls", 0.0))
            continue
        m = measured.get(system)
        if m is None:
            # No measurement (the pilot itself): bound it by the hard per-run cap. A call
            # past that cap is never sent, so this is a true worst case, not a guess.
            estimates.append(
                SystemEstimate(
                    system,
                    units,
                    config.run_cap_inr,
                    "not measured: worst case = per-run cap",
                    units * config.run_cap_inr,
                )
            )
            continue
        system_factor = _price_factor(config.model, m.measured_on, today)
        factor = max(factor, system_factor)
        total = units * m.per_run_inr * (1 + config.cost_margin) * system_factor
        estimates.append(
            SystemEstimate(
                system, units, m.per_run_inr, f"pilot mean of {m.runs} runs, {m.measured_on}", total
            )
        )
    total = sum(e.total_inr for e in estimates)
    spent = ledger.total_inr()
    cap = llm_config.project_cap_inr()
    reserve = 0.0 if purpose == "benchmark" else llm_config.BENCHMARK_RESERVE_INR
    available = cap - reserve - spent
    if total > available:
        where = "the project cap" if purpose == "benchmark" else "the cap minus the reserve"
        shortfalls.append(
            f"estimated Rs {total:.2f} exceeds the Rs {available:.2f} left under {where} "
            f"(short by Rs {total - available:.2f})"
        )
    balance = ledger.balance_left()
    if total > 0:
        if balance is None:
            shortfalls.append(
                "no prepaid balance recorded: record it with 'holdout spend --set-balance'"
            )
        elif total > balance:
            shortfalls.append(
                f"estimated Rs {total:.2f} exceeds the prepaid balance left, about Rs "
                f"{balance:.2f} (short by Rs {total - balance:.2f}): top up first"
            )
    return PreflightReport(
        purpose=purpose,
        estimates=estimates,
        margin=config.cost_margin,
        price_factor=factor,
        total_inr=total,
        available_inr=available,
        balance_left_inr=balance,
        shortfalls=shortfalls,
    )


def preflight(
    units: Sequence[Any], results_root: Path, config: BenchConfig, *, purpose: str
) -> PreflightReport:
    """Estimate the pending units of ``units`` (see :func:`estimate`)."""
    from bench.run import pending

    # No isinstance check: under ``python -m bench.run`` the units are instances of
    # ``__main__.Unit``, not ``bench.run.Unit``; a type filter silently dropped them all
    # and estimated Rs 0 (found by the pilot's pre-flight). Units are duck-typed.
    all_units = list(units)
    counts: dict[str, int] = {}
    for unit, _ in pending(all_units, results_root):
        counts[unit.system] = counts.get(unit.system, 0) + 1
    for unit in all_units:
        counts.setdefault(unit.system, 0)
    if not all_units:
        raise ValueError("no units to estimate")
    return estimate(counts, config, purpose=purpose)
