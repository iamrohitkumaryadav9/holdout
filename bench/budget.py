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
#: Holdout-family systems may start exactly one follow-up agent run (corrected split) with
#: its own per-run cap, so one unit can spend up to two caps.
FOLLOWUP_SYSTEMS = frozenset({"holdout", "llm_judge"})


def unit_worst_case(system: str, run_cap_inr: float) -> float:
    """The most one unit of ``system`` can spend: no call past a per-run cap is sent."""
    if system not in LLM_SYSTEMS:
        return 0.0
    return run_cap_inr * (2 if system in FOLLOWUP_SYSTEMS else 1)


def limits_left(
    ledger: SpendLedger,
    *,
    purpose: str,
    invocation_start_inr: float | None = None,
    max_spend_inr: float | None = None,
) -> dict[str, float]:
    """Rupees left under every limit that applies to spend of this ``purpose``.

    * project cap: cap - spent (every purpose);
    * benchmark reserve: cap - reserve - spent (general spend may not enter it);
    * this invocation's cap (``--max-spend``, e.g. the pilot cap), when set;
    * the prepaid balance left, when recorded.
    """
    spent = ledger.total_inr()
    cap = llm_config.project_cap_inr()
    left = {"project cap": cap - spent}
    if purpose != "benchmark":
        left["benchmark reserve"] = cap - llm_config.BENCHMARK_RESERVE_INR - spent
    if max_spend_inr is not None:
        start = invocation_start_inr if invocation_start_inr is not None else spent
        left["this run's spend cap"] = max_spend_inr - (spent - start)
    balance = ledger.balance_left()
    if balance is not None:
        left["prepaid balance"] = balance
    return left


def start_refusal(
    worst_case_inr: float,
    in_flight_worst_inr: float,
    left: dict[str, float],
) -> str | None:
    """Why a unit may not start, or None: every limit must still cover this unit's worst
    case on top of the worst case of every unit already running (their spend is not in
    the ledger yet). A unit that cannot spend (no LLM) always starts."""
    if worst_case_inr <= 0:
        return None
    needed = worst_case_inr + in_flight_worst_inr
    short = [f"{name}: Rs {value:.2f} left" for name, value in left.items() if value < needed]
    if not short:
        return None
    return (
        f"needs Rs {needed:.2f} (Rs {worst_case_inr:.2f} worst case for this unit + Rs "
        f"{in_flight_worst_inr:.2f} for units still running), but " + "; ".join(short)
    )


@dataclass(frozen=True)
class Measured:
    per_run_inr: float
    runs: int
    measured_on: date
    #: "flex" only if every paid call of every measured run was served by Flex.
    tier: str = "standard"


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
    headroom_inr: float = 0.0
    workers: int = 1
    scope: str = ""

    @property
    def fits(self) -> bool:
        return not self.shortfalls

    def render(self) -> str:
        lines = [f"Budget pre-flight ({self.purpose} spend)"]
        if self.scope:
            lines.append(f"  scope: {self.scope}")
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
        lines.append(
            f"  headroom for units in flight ({self.workers} worker(s), worst case each): "
            f"Rs {self.headroom_inr:.2f}"
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
    tiers: dict[str, set[str]] = {}
    for path in sorted(pilot_root.glob("*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            if result["status"] != "completed":
                continue
            system = result["system"]
            costs.setdefault(system, []).append(float(result["cost_inr"]))
            day = datetime.fromisoformat(result["started_at"]).date()
            dates[system] = min(dates.get(system, day), day)
            served = result.get("tiers_served") or {"standard": 1}  # older results: standard
            tiers.setdefault(system, set()).update(served)
    return {
        system: Measured(
            sum(values) / len(values),
            len(values),
            dates[system],
            "flex" if tiers.get(system) == {"flex"} else "standard",
        )
        for system, values in costs.items()
    }


def _price_factor(
    model: str, measured_on: date, today: date, measured_tier: str = "standard",
    tier: str = "standard",
) -> float:  # fmt: skip
    """Today's price at the configured tier over the price the measurement was made at."""
    then = llm_config.price_for(model, measured_on, measured_tier)
    now = llm_config.price_for(model, today, tier)
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
    workers: int = 1,
) -> PreflightReport:
    """Estimate pending spend and check it, plus the unit-start guard's headroom, against
    the cap for ``purpose`` and the prepaid balance.

    The guard starts a unit only while every limit covers the worst case of the units in
    flight, so the last ``workers`` LLM units need up to ``workers`` worst cases of room
    beyond what has been spent: that headroom is added to the requirement.
    """
    ledger = ledger if ledger is not None else SpendLedger()
    measured = measured if measured is not None else measured_costs(config.pilot_results)
    today = today if today is not None else datetime.now(UTC).date()
    shortfalls: list[str] = []
    estimates: list[SystemEstimate] = []
    factors: list[float] = []
    for system, units in pending_by_system.items():
        if system not in LLM_SYSTEMS:
            estimates.append(SystemEstimate(system, units, 0.0, "no LLM calls", 0.0))
            continue
        m = measured.get(system)
        if m is None:
            # No measurement (the pilot itself): bound it by the hard per-run cap(s). A call
            # past a cap is never sent, so this is a true worst case, not a guess.
            worst = unit_worst_case(system, config.run_cap_inr)
            estimates.append(
                SystemEstimate(
                    system,
                    units,
                    worst,
                    "not measured: worst case = per-run cap"
                    + (" x 2 (follow-up run)" if system in FOLLOWUP_SYSTEMS else ""),
                    units * worst,
                )
            )
            continue
        system_factor = _price_factor(
            config.model, m.measured_on, today, m.tier, config.service_tier
        )
        factors.append(system_factor)
        total = units * m.per_run_inr * (1 + config.cost_margin) * system_factor
        source = f"pilot mean of {m.runs} runs, {m.measured_on}, {m.tier} tier"
        estimates.append(SystemEstimate(system, units, m.per_run_inr, source, total))
    total = sum(e.total_inr for e in estimates)
    llm_units = sorted(
        (unit_worst_case(s, config.run_cap_inr) for s, n in pending_by_system.items()
         for _ in range(n) if s in LLM_SYSTEMS),
        reverse=True,
    )  # fmt: skip
    headroom = sum(llm_units[: max(1, workers)])
    spent = ledger.total_inr()
    cap = llm_config.project_cap_inr()
    reserve = 0.0 if purpose == "benchmark" else llm_config.BENCHMARK_RESERVE_INR
    available = cap - reserve - spent
    where = "the project cap" if purpose == "benchmark" else "the cap minus the reserve"
    if total > available:
        shortfalls.append(
            f"estimated Rs {total:.2f} exceeds the Rs {available:.2f} left under {where} "
            f"(short by Rs {total - available:.2f})"
        )
    elif total + headroom > available:
        shortfalls.append(
            f"estimated Rs {total:.2f} plus Rs {headroom:.2f} of headroom for the units in "
            f"flight ({max(1, workers)} worker(s) x worst case) exceeds the Rs "
            f"{available:.2f} left under {where} (short by Rs "
            f"{total + headroom - available:.2f}): the unit-start guard would stop the run "
            "before the end"
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
        elif total + headroom > balance:
            shortfalls.append(
                f"estimated Rs {total:.2f} plus Rs {headroom:.2f} of headroom for the units "
                f"in flight exceeds the prepaid balance left, about Rs {balance:.2f}: top up "
                "first"
            )
    return PreflightReport(
        purpose=purpose,
        estimates=estimates,
        margin=config.cost_margin,
        price_factor=max(factors) if factors else 1.0,
        total_inr=total,
        headroom_inr=headroom if total > 0 else 0.0,
        workers=max(1, workers),
        available_inr=available,
        balance_left_inr=balance,
        shortfalls=shortfalls,
    )


def preflight(
    units: Sequence[Any],
    results_root: Path,
    config: BenchConfig,
    *,
    purpose: str,
    retry_failed: bool = False,
    retry_failed_matching: str | None = None,
    workers: int = 1,
    scope: str = "",
) -> PreflightReport:
    """Estimate the pending units of ``units`` together (see :func:`estimate`); units may
    span suites. ``scope`` names what was estimated, for the report."""
    from bench.run import pending

    # No isinstance check: under ``python -m bench.run`` the units are instances of
    # ``__main__.Unit``, not ``bench.run.Unit``; a type filter silently dropped them all
    # and estimated Rs 0 (found by the pilot's pre-flight). Units are duck-typed.
    all_units = list(units)
    counts: dict[str, int] = {}
    for unit, _ in pending(all_units, results_root, retry_failed=retry_failed,
                           retry_failed_matching=retry_failed_matching):  # fmt: skip
        counts[unit.system] = counts.get(unit.system, 0) + 1
    for unit in all_units:
        counts.setdefault(unit.system, 0)
    if not all_units:
        raise ValueError("no units to estimate")
    report = estimate(counts, config, purpose=purpose, workers=workers)
    report.scope = scope
    return report
