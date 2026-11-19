"""Post-hoc comparison of one system across two code versions (v1.1 vs v1.0).

``python -m bench.compare --suite dev_v1 --before results_v11/v1_0 --after results_v11/v1_1``

Scored with the pre-registered rules of :mod:`bench.score` (detection, remediation, clean
false alarms; the task is the unit, seeds kept apart). Every change is listed task by task:
each detection lost and gained, each false alarm (on a clean control) new and gone, per
seed. This is exploratory - not part of the pre-registered evaluation.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.score import (
    TaskTruth,
    false_alarm,
    load_results,
    planted_truths,
    real_world_truths,
    task_scores,
)
from bench.systems.base import BenchResult
from holdout.audit.types import Severity


@dataclass(frozen=True)
class Summary:
    runs: int
    failed: int
    recall: float | None  # mean over tasks of the per-task detection rate
    recall_critical: float | None
    remediated: float | None  # share of scorable trap runs
    clean_tasks: int
    clean_critical: int  # clean tasks with a critical alarm (or a failure) in any seed
    clean_warning: int


@dataclass(frozen=True)
class Change:
    kind: str  # "detection lost" | "detection gained" | "false alarm new" | "false alarm gone"
    task: str  # task_id/variant (issue)
    seed: int
    detail: str


def truths_for(suite: str) -> dict[tuple[str, str], TaskTruth]:
    return real_world_truths() if suite == "real_world_v1" else planted_truths(suite)


def _mine(results: list[BenchResult], system: str) -> list[BenchResult]:
    return [r for r in results if r.system == system and r.status != "paused"]


def summarise(
    results: list[BenchResult], truths: dict[tuple[str, str], TaskTruth], system: str
) -> Summary:
    mine = _mine(results, system)
    scores = [s for s in task_scores(mine, truths) if not s.secondary]
    remedies = [v for s in scores for v in s.remediated_by_seed.values() if v is not None]
    clean: dict[str, tuple[bool, bool]] = {}
    for r in mine:
        truth = truths.get((r.task_id, r.variant))
        if truth is None or truth.kind != "clean":
            continue
        crit, warn = clean.get(r.task_id, (False, False))
        clean[r.task_id] = (crit or false_alarm(r, "critical"), warn or false_alarm(r, "warning"))

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return Summary(
        runs=len(mine),
        failed=sum(r.status == "failed" for r in mine),
        recall=mean([s.rate for s in scores]),
        recall_critical=mean(
            [float(np.mean(list(s.detected_critical_by_seed.values()))) for s in scores]
        ),
        remediated=mean([float(v) for v in remedies]),
        clean_tasks=len(clean),
        clean_critical=sum(c for c, _ in clean.values()),
        clean_warning=sum(w for _, w in clean.values()),
    )


def _issues_text(result: BenchResult | None, minimum: str) -> str:
    if result is None:
        return "no run"
    if result.status == "failed":
        return f"failed: {(result.error or '')[:80]}"
    keep = [i for i in result.issues if i.severity.at_least(Severity(minimum))]
    return (
        "; ".join(
            f"{i.source} {i.severity.value} {i.category.value}"
            + (f" [{', '.join(i.columns)}]" if i.columns else "")
            for i in keep
        )
        or "none"
    )


def changes(
    before: list[BenchResult],
    after: list[BenchResult],
    truths: dict[tuple[str, str], TaskTruth],
    system: str,
) -> list[Change]:
    """Every per-seed difference in detection (traps) and in warning-or-critical false
    alarms (clean controls) between the two versions."""
    b, a = _mine(before, system), _mine(after, system)
    out: list[Change] = []
    b_scores = {(s.task_id, s.variant, s.issue_id): s for s in task_scores(b, truths)}
    for s in task_scores(a, truths):
        old = b_scores.get((s.task_id, s.variant, s.issue_id))
        if old is None:
            continue
        label = f"{s.task_id}/{s.variant} ({s.issue_id})" + (" [secondary]" if s.secondary else "")
        for seed, now in sorted(s.detected_by_seed.items()):
            was = old.detected_by_seed.get(seed)
            if was is None or was == now:
                continue
            kind = "detection gained" if now else "detection lost"
            out.append(Change(kind, label, seed, ""))
    by_key = {(r.task_id, r.variant, r.seed): r for r in b}
    for r in a:
        truth = truths.get((r.task_id, r.variant))
        if truth is None or truth.kind != "clean":
            continue
        old_r = by_key.get((r.task_id, r.variant, r.seed))
        if old_r is None:
            continue
        for level in ("critical", "warning"):
            was, now = false_alarm(old_r, level), false_alarm(r, level)
            if was == now:
                continue
            kind = f"false alarm {'new' if now else 'gone'} ({level})"
            source = r if now else old_r
            out.append(Change(kind, f"{r.task_id}/{r.variant}", r.seed,
                              _issues_text(source, level)))  # fmt: skip
    return sorted(out, key=lambda c: (c.kind, c.task, c.seed))


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render(
    suite: str, before: Path, after: Path, system: str = "audits_only", *, title: str = ""
) -> list[str]:
    truths = truths_for(suite)
    b = load_results(before, suite)
    a = load_results(after, suite)
    sb, sa = summarise(b, truths, system), summarise(a, truths, system)
    lines = [f"### {title or suite}", ""]
    lines += [
        "| | v1.0 | v1.1 |",
        "|---|---|---|",
        f"| runs (failed) | {sb.runs} ({sb.failed}) | {sa.runs} ({sa.failed}) |",
        f"| detection recall (task = unit) | {_pct(sb.recall)} | {_pct(sa.recall)} |",
        f"| detected at critical | {_pct(sb.recall_critical)} | {_pct(sa.recall_critical)} |",
        f"| remediated (scorable runs) | {_pct(sb.remediated)} | {_pct(sa.remediated)} |",
        f"| clean controls with a critical false alarm | {sb.clean_critical}/{sb.clean_tasks} | "
        f"{sa.clean_critical}/{sa.clean_tasks} |",
        f"| clean controls with a warning-or-critical false alarm | "
        f"{sb.clean_warning}/{sb.clean_tasks} | {sa.clean_warning}/{sa.clean_tasks} |",
        "",
    ]
    diffs = changes(b, a, truths, system)
    if not diffs:
        lines += ["No per-seed change in detection or in false alarms.", ""]
        return lines
    lines += ["| change | task | seed | issues behind it |", "|---|---|---|---|"]
    lines += [f"| {c.kind} | {c.task} | {c.seed} | {c.detail or '-'} |" for c in diffs]
    return [*lines, ""]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--system", default="audits_only")
    args = parser.parse_args(argv)
    print("\n".join(render(args.suite, args.before, args.after, args.system)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
