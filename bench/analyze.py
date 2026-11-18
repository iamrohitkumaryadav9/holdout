"""Deterministic tables (markdown) and charts (PNG) from the results files only.

``python -m bench.analyze --results results --out docs/results.md --charts docs/img``

Every number here is computed from ``results/<suite>/<system>.jsonl`` and the frozen suites;
nothing is typed by hand. The task is the unit of every CI and test (seeds averaged within
a task; McNemar pairs per task); anonymised variants are shown next to the originals, never
pooled with them. The semi-synthetic time-order trap is labelled as such.
"""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bench.score import (
    TaskScore,
    TaskTruth,
    bootstrap_mean,
    clean_alarms,
    clopper_pearson_upper,
    load_results,
    mcnemar,
    optimism_gap,
    paired_bootstrap,
    planted_truths,
    real_world_truths,
    relative_to_dummy,
    task_scores,
)
from bench.systems.base import BenchResult
from holdout.audit.checks.a11_causal import check_causal_claims

SYSTEM_ORDER = ("holdout", "audits_only", "llm_judge", "react", "flaml")


def _systems(results: Sequence[BenchResult]) -> list[str]:
    present = {r.system for r in results}
    return [s for s in SYSTEM_ORDER if s in present]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(mean: float, low: float, high: float) -> str:
    return f"{_pct(mean)} [{_pct(low)}, {_pct(high)}]"


def trap_label(trap: str, semi_synthetic: bool) -> str:
    return f"{trap} (semi-synthetic)" if semi_synthetic else trap


def detection_tables(
    results: list[BenchResult], truths: dict[tuple[str, str], TaskTruth]
) -> list[str]:
    systems = _systems(results)
    scores = [s for s in task_scores(results, truths) if not s.secondary]
    lines = ["### Detection by trap type (share of task-seed runs)", ""]
    lines += ["| trap | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    by_trap: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in scores:
        truth = truths[(s.task_id, s.variant)]
        by_trap[trap_label(truth.trap, truth.semi_synthetic)][s.system].extend(
            float(v) for v in s.detected_by_seed.values()
        )
    for trap in sorted(by_trap):
        cells = [
            _pct(float(np.mean(by_trap[trap][sy]))) if by_trap[trap][sy] else "-" for sy in systems
        ]
        lines.append(f"| {trap} | " + " | ".join(cells) + " |")
    lines += ["", "### Detection recall (task = unit, seeds averaged; 95% bootstrap CI)", ""]
    lines += ["| system | recall | both seeds | either seed | per seed |", "|---|---|---|---|---|"]
    for sy in systems:
        mine = [s for s in scores if s.system == sy]
        if not mine:
            continue
        seeds = sorted({k for s in mine for k in s.detected_by_seed})

        def seed_rate(k: int, runs: list[TaskScore] = mine) -> float:
            return float(np.mean([s.detected_by_seed[k] for s in runs if k in s.detected_by_seed]))

        per_seed = ", ".join(f"{k}: {_pct(seed_rate(k))}" for k in seeds)
        lines.append(
            f"| {sy} | {_ci(*bootstrap_mean([s.rate for s in mine]))} | "
            f"{_pct(float(np.mean([s.both for s in mine])))} | "
            f"{_pct(float(np.mean([s.either for s in mine])))} | {per_seed} |"
        )
    return lines


def remediation_table(
    results: list[BenchResult], truths: dict[tuple[str, str], TaskTruth]
) -> list[str]:
    systems = _systems(results)
    lines = ["### Remediation (the final pipeline fixed it; scorable traps only)", ""]
    lines += ["| system | remediated | scorable runs |", "|---|---|---|"]
    for sy in systems:
        values = [
            float(v)
            for s in task_scores(results, truths)
            if s.system == sy and not s.secondary
            for v in s.remediated_by_seed.values()
            if v is not None
        ]
        if values:
            lines.append(f"| {sy} | {_pct(float(np.mean(values)))} | {len(values)} |")
    return lines


def false_alarm_table(
    results: list[BenchResult], truths: dict[tuple[str, str], TaskTruth]
) -> list[str]:
    systems = _systems(results)
    lines = [
        "### False alarms on clean controls (task = unit: any seed or variant alarming counts)",
        "",
        "| system | critical | one-sided 95% upper bound | warning or above "
        "| any issue (info too) |",
        "|---|---|---|---|---|",
    ]
    levels = {
        level: clean_alarms(results, truths, level) for level in ("critical", "warning", "any")
    }
    for sy in systems:
        crit = levels["critical"].get(sy, {})
        if not crit:
            continue
        k, n = sum(crit.values()), len(crit)
        warn = levels["warning"].get(sy, {})
        anyl = levels["any"].get(sy, {})
        lines.append(
            f"| {sy} | {k}/{n} = {_pct(k / n)} | {_pct(clopper_pearson_upper(k, n))} | "
            f"{_pct(sum(warn.values()) / max(len(warn), 1))} | "
            f"{_pct(sum(anyl.values()) / max(len(anyl), 1))} |"
        )
    return lines


def _per_task_mean(results: list[BenchResult], system: str, fn) -> dict[str, float]:  # type: ignore[no-untyped-def]
    values: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.system == system and r.status == "completed":
            v = fn(r)
            if v is not None:
                values[f"{r.task_id}/{r.variant}"].append(v)
    return {k: float(np.mean(v)) for k, v in values.items()}


def outcome_table(results: list[BenchResult]) -> list[str]:
    systems = _systems(results)
    lines = [
        "### Scores, optimism, cost and answer quality (completed runs; task = unit)",
        "",
        "| system | optimism gap (95% CI) | holdout vs dummy | holdout - FLAML | cost/run | "
        "time/run | calls | retries | first-attempt valid | repairs/run | substitutions/run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    flaml = _per_task_mean(results, "flaml", lambda r: r.holdout_score)
    for sy in systems:
        runs = [r for r in results if r.system == sy and r.status == "completed"]
        if not runs:
            continue
        gaps = list(_per_task_mean(results, sy, optimism_gap).values())
        gap = _ci(*bootstrap_mean(gaps)).replace("%", "") if gaps else "-"
        rel = list(_per_task_mean(results, sy, relative_to_dummy).values())
        mine = _per_task_mean(results, sy, lambda r: r.holdout_score)
        shared = sorted(set(mine) & set(flaml))
        vs_flaml = (
            f"{np.mean([mine[k] - flaml[k] for k in shared]):+.4f}"
            if shared and sy != "flaml"
            else "-"
        )
        firsts = sum(q.first_attempts for r in runs for q in r.first_attempt.values())
        valid = sum(q.valid for r in runs for q in r.first_attempt.values())
        lines.append(
            f"| {sy} | {gap} | {np.mean(rel):.3f} | {vs_flaml} | "
            f"Rs {np.mean([r.cost_inr for r in runs]):.2f} | "
            f"{np.mean([r.seconds for r in runs]):.0f} s | "
            f"{np.mean([r.calls for r in runs]):.1f} | {sum(r.retries for r in runs)} | "
            f"{_pct(valid / firsts) if firsts else '-'} | "
            f"{np.mean([r.repairs for r in runs]):.2f} | "
            f"{np.mean([r.substitutions for r in runs]):.2f} |"
        )
    lines += ["", "Optimism gap: the system's own validation claim minus its holdout score in "
              "the same metric (positive = optimistic)."]  # fmt: skip
    return lines


def mcnemar_table(
    results: list[BenchResult], truths: dict[tuple[str, str], TaskTruth]
) -> list[str]:
    systems = _systems(results)
    scores = [s for s in task_scores(results, truths) if not s.secondary]
    by = {(s.system, s.task_id, s.variant, s.issue_id): s for s in scores}
    keys = sorted({(s.task_id, s.variant, s.issue_id) for s in scores})
    lines = [
        "### Pairwise detection: exact McNemar per task (detected in both seeds; either seed)",
        "",
        "| A vs B | A only | B only | p (both) | p (either) |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(systems):
        for b in systems[i + 1 :]:
            shared = [k for k in keys if (a, *k) in by and (b, *k) in by]
            if not shared:
                continue
            both = mcnemar([by[(a, *k)].both for k in shared], [by[(b, *k)].both for k in shared])
            either = mcnemar(
                [by[(a, *k)].either for k in shared], [by[(b, *k)].either for k in shared]
            )
            lines.append(
                f"| {a} vs {b} | {both.only_a} | {both.only_b} | {both.p_value:.4f} | "
                f"{either.p_value:.4f} |"
            )
    return lines


def real_world_table(results: list[BenchResult]) -> list[str]:
    truths = real_world_truths()
    systems = _systems(results)
    scores = task_scores(results, truths)
    lines = [
        "### Real-world issues: original vs anonymised (detection rate over seeds)",
        "",
        "| case / issue | " + " | ".join(f"{s} orig / anon" for s in systems) + " |",
        "|---|" + "---|" * len(systems),
    ]
    rows = sorted({(s.task_id, s.issue_id, s.secondary) for s in scores})
    for task_id, issue_id, secondary in rows:
        cells = []
        for sy in systems:
            pair = []
            for variant in ("original", "anonymised"):
                match = [s for s in scores if (s.task_id, s.issue_id, s.system, s.variant)
                         == (task_id, issue_id, sy, variant)]  # fmt: skip
                pair.append(_pct(match[0].rate) if match else "-")
            cells.append(" / ".join(pair))
        label = f"{task_id} / {issue_id}" + (" (secondary)" if secondary else "")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "#### H5 (exploratory): paired original - anonymised detection, per system", ""]
    lines += ["| system | mean difference [95% CI] |", "|---|---|"]
    for sy in systems:
        orig = {
            (s.task_id, s.issue_id): s.rate
            for s in scores
            if s.system == sy and s.variant == "original" and not s.secondary
        }
        anon = {
            (s.task_id, s.issue_id): s.rate
            for s in scores
            if s.system == sy and s.variant == "anonymised" and not s.secondary
        }
        shared = sorted(set(orig) & set(anon))
        if shared:
            mean, low, high = paired_bootstrap([orig[k] for k in shared], [anon[k] for k in shared])
            lines.append(f"| {sy} | {mean:+.3f} [{low:+.3f}, {high:+.3f}] |")
    return lines


def a11_table(results: list[BenchResult], truths: dict[tuple[str, str], TaskTruth]) -> list[str]:
    lines = [
        "### A11 flags on generated reports of tasks whose goal is not causal",
        "",
        "An upper bound on A11's false-positive rate: whether a flagged sentence really is a "
        "causal claim needs a human label (sentences listed in the appendix).",
        "",
        "| system | reports | flagged | rate |",
        "|---|---|---|---|",
    ]
    for sy in _systems(results):
        reports = [
            r for r in results
            if r.system == sy and r.report_text
            and truths.get((r.task_id, r.variant)) is not None
            and truths[(r.task_id, r.variant)].trap != "causal_question"
        ]  # fmt: skip
        if reports:
            flagged = sum(bool(check_causal_claims(r.report_text or "")) for r in reports)
            lines.append(f"| {sy} | {len(reports)} | {flagged} | {_pct(flagged / len(reports))} |")
    return lines


def charts(
    results: list[BenchResult], truths: dict[tuple[str, str], TaskTruth], out: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    systems = _systems(results)
    scores = [s for s in task_scores(results, truths) if not s.secondary]
    paths = []
    rates = [
        float(np.mean([s.rate for s in scores if s.system == sy]))
        if any(s.system == sy for s in scores)
        else 0.0
        for sy in systems
    ]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(systems, rates, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("detection recall")
    ax.set_title("Planted traps: detection by system")
    fig.tight_layout()
    path = out / "detection.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)
    gaps = [list(_per_task_mean(results, sy, optimism_gap).values()) for sy in systems]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.boxplot([g or [0.0] for g in gaps], tick_labels=systems)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("validation - holdout")
    ax.set_title("Optimism gap per task")
    fig.tight_layout()
    path = out / "optimism_gap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(path)
    return paths


def report(root: Path, suite: str) -> list[str]:
    results = load_results(root, suite)
    if not results:
        return [f"## {suite}", "", "No results yet."]
    if suite == "real_world_v1":
        truths = real_world_truths()
        body = [*real_world_table(results), "", *false_alarm_table(results, truths)]
    else:
        truths = planted_truths(suite)
        body = [
            *detection_tables(results, truths),
            "",
            *remediation_table(results, truths),
            "",
            *false_alarm_table(results, truths),
            "",
            *mcnemar_table(results, truths),
        ]
    statuses: dict[str, int] = defaultdict(int)
    for r in results:
        statuses[r.status] += 1
    head = [
        f"## {suite}",
        "",
        "Runs: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())),
        "",
    ]
    return head + body + [""] + outcome_table(results) + [""] + a11_table(results, truths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--suites", nargs="+", default=["final_v1", "real_world_v1"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--charts", type=Path, default=None)
    args = parser.parse_args(argv)
    lines = ["# Benchmark results", ""]
    for suite in args.suites:
        lines += [*report(args.results, suite), ""]
        if args.charts is not None and suite != "real_world_v1":
            results = load_results(args.results, suite)
            if results:
                charts(results, planted_truths(suite), args.charts / suite)
    text = "\n".join(lines) + "\n"
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
