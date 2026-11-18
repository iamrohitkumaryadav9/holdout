"""The pilot report: measured cost and time per run, what consumed the money, and the
projected cost of the full benchmark (``python -m bench.run ... --report <path>``).

Everything comes from the results files and the spend ledger. The projection multiplies
each system's mean measured cost per run by its number of units in the full benchmark
(planted suite: 40 tasks x 2 seeds; real-world suite: 9 cases x 2 variants x 2 seeds) and
adds the configured margin. ReAct is also projected *without* observation truncation, from
the observation tokens it would have re-sent with an untruncated history.
"""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bench.config import BenchConfig
from bench.score import load_results
from holdout.llm import config as llm_config
from holdout.llm.budget import SpendLedger

PLANTED_UNITS = 40 * 2
REAL_WORLD_UNITS = 9 * 2 * 2
FULL_UNITS = PLANTED_UNITS + REAL_WORLD_UNITS


def write_pilot_report(
    results_root: Path,
    suite: str,
    path: Path,
    config: BenchConfig,
    *,
    ledger: SpendLedger | None = None,
) -> str:
    text = pilot_report(results_root, suite, config, ledger=ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


def pilot_report(
    results_root: Path, suite: str, config: BenchConfig, *, ledger: SpendLedger | None = None
) -> str:
    ledger = ledger if ledger is not None else SpendLedger()
    results = sorted(load_results(results_root, suite), key=lambda r: r.started_at)
    if not results:
        return "# Pilot report\n\nNo results.\n"
    lines = ["# Pilot report", "", f"Suite `{suite}`, results in `{results_root}`.", ""]
    lines += [
        "## Units",
        "",
        "| system | task | seed | status | Rs | compute s | API wait s | calls | retries "
        "| first-attempt valid | error |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        firsts = sum(q.first_attempts for q in r.first_attempt.values())
        valid = sum(q.valid for q in r.first_attempt.values())
        error = (r.error or "").replace("|", "/")[:120]
        lines.append(
            f"| {r.system} | {r.task_id} | {r.seed} | {r.status} | {r.cost_inr:.3f} | "
            f"{r.time_per_run:.0f} | {r.api_seconds:.0f} | {r.calls} | {r.retries} | "
            f"{f'{valid}/{firsts}' if firsts else '-'} | {error} |"
        )
    # ------------------------------------------------------------------ what consumed it
    lines += ["", "## What consumed the money", ""]
    by_node: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in results:
        for node, inr in r.extra.get("inr_by_node", {}).items():
            by_node[r.system][node] += float(inr)
    lines += ["| system | node | Rs |", "|---|---|---|"]
    for system in sorted(by_node):
        for node, inr in sorted(by_node[system].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {system} | {node} | {inr:.3f} |")
    react = [r for r in results if r.system == "react" and r.extra.get("inr_by_step")]
    for r in react:
        steps = ", ".join(f"{v:.3f}" for v in r.extra["inr_by_step"])
        lines.append("")
        lines.append(
            f"ReAct on `{r.task_id}`: {r.extra.get('steps')} steps; Rs per LLM call: {steps}; "
            f"prompt tokens {r.extra.get('prompt_tokens')}, observation tokens re-sent with the "
            f"history (estimate) {r.extra.get('observation_tokens_sent_estimate')} truncated vs "
            f"{r.extra.get('observation_tokens_sent_untruncated_estimate')} untruncated."
        )
    started = min(r.started_at for r in results)
    ledger_spend = sum(
        float(e["inr"]) for e in ledger.calls() if datetime.fromisoformat(e["timestamp"]) >= started
    )
    attributed = sum(r.cost_inr for r in results)
    lines += [
        "",
        f"Ledger spend since the pilot started: Rs {ledger_spend:.3f}; attributed to results: "
        f"Rs {attributed:.3f}; unattributed (e.g. a unit that failed or paused after "
        f"spending, or other processes): Rs {ledger_spend - attributed:.3f}.",
    ]
    # ------------------------------------------------------------------ per system
    lines += [
        "",
        "## Measured per run",
        "",
        "| system | runs | Rs/run list (mean) | Rs/run list (max) | Rs/run cache-adjusted | "
        "cached share of prompt | tiers served | compute s/run | API wait s/run |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    per_system: dict[str, list] = defaultdict(list)  # type: ignore[type-arg]
    for r in results:
        if r.status == "completed":
            per_system[r.system].append(r)
    for system, runs in sorted(per_system.items()):
        costs = [r.cost_inr for r in runs]
        prompt = sum(r.prompt_tokens for r in runs)
        cached = sum(r.cached_tokens for r in runs)
        tiers: dict[str, int] = defaultdict(int)
        for r in runs:
            for tier, n in r.tiers_served.items():
                tiers[tier] += n
        lines.append(
            f"| {system} | {len(runs)} | {np.mean(costs):.3f} | {max(costs):.3f} | "
            f"{np.mean([r.cost_adjusted for r in runs]):.3f} | "
            f"{f'{cached / prompt:.1%}' if prompt else '-'} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(tiers.items())) or '-'} | "
            f"{np.mean([r.time_per_run for r in runs]):.0f} | "
            f"{np.mean([r.api_seconds for r in runs]):.0f} |"
        )
    # ------------------------------------------------------------------ projection
    today = datetime.now(UTC).date()
    price = llm_config.price_for(config.model, today)
    input_inr_per_token = price.input_usd_per_million * llm_config.USD_TO_INR / 1_000_000
    margin = 1 + config.cost_margin
    lines += [
        "",
        f"## Projected full benchmark ({FULL_UNITS} units per system: planted "
        f"{PLANTED_UNITS}, real world {REAL_WORLD_UNITS}; margin {config.cost_margin:.0%})",
        "",
        "| system | Rs list (with ReAct truncation) | Rs list (without truncation) | "
        "Rs cache-adjusted | compute hours |",
        "|---|---|---|---|---|",
    ]
    total_with = total_without = total_adjusted = 0.0
    for system, runs in sorted(per_system.items()):
        mean = float(np.mean([r.cost_inr for r in runs]))
        with_trunc = mean * FULL_UNITS * margin
        adjusted = float(np.mean([r.cost_adjusted for r in runs])) * FULL_UNITS * margin
        if system == "react":
            extra_tokens = np.mean(
                [
                    r.extra.get("observation_tokens_sent_untruncated_estimate", 0)
                    - r.extra.get("observation_tokens_sent_estimate", 0)
                    for r in runs
                ]
            )
            without = (mean + extra_tokens * input_inr_per_token) * FULL_UNITS * margin
        else:
            without = with_trunc
        hours = float(np.mean([r.time_per_run for r in runs])) * FULL_UNITS / 3600
        total_with += with_trunc
        total_without += without
        total_adjusted += adjusted
        lines.append(
            f"| {system} | {with_trunc:.2f} | {without:.2f} | {adjusted:.2f} | {hours:.1f} |"
        )
    spent = ledger.total_inr()
    cap = llm_config.project_cap_inr()
    balance = ledger.balance_left()
    lines += [
        f"| **total** | **{total_with:.2f}** | **{total_without:.2f}** | "
        f"**{total_adjusted:.2f}** | |",
        "",
        f"- Project spend so far: Rs {spent:.2f}; remaining under the Rs {cap:g} cap "
        f"(benchmark spend may use the reserve): Rs {cap - spent:.2f}.",
        f"- Benchmark reserve: Rs {llm_config.BENCHMARK_RESERVE_INR:g}.",
        "- Prepaid balance left (estimate): "
        + ("not recorded" if balance is None else f"Rs {balance:.2f}")
        + ".",
        f"- Prices as of {llm_config.PRICES_AS_OF} ({price.input_usd_per_million} / "
        f"{price.output_usd_per_million} USD per 1M input / output tokens today); "
        f"{llm_config.PRICE_NOTE}",
        "- The pilot ran on three dev tasks (adult 48k rows, credit-approval 690, kin8nm 8k); "
        "the final bases range from 1k to 30k rows, so per-run cost may differ.",
    ]
    return "\n".join(lines) + "\n"


def measured_summary(results_root: Path, suite: str) -> dict[str, float]:
    """Mean cost per completed run, per system (for scripts)."""
    out: dict[str, list[float]] = defaultdict(list)
    for r in load_results(results_root, suite):
        if r.status == "completed":
            out[r.system].append(r.cost_inr)
    return {k: float(np.mean(v)) for k, v in out.items()}


if __name__ == "__main__":  # pragma: no cover - convenience
    import sys

    from bench.config import load_bench_config

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/pilot")
    print(pilot_report(root, "dev_v1", load_bench_config()))
    print(json.dumps(measured_summary(root, "dev_v1")))
