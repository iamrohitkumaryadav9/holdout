"""Scoring: detection, remediation, false alarms, optimism, cost, and the statistics.

Definitions (docs/benchmark.md):

* **Detection** (planted): the expected category reported at severity >= the trap's minimum
  (the Week 2 rule; no column match). Real world: one of the issue's categories at
  severity >= its minimum, naming one of its columns when it has any.
* **Remediation**: the final pipeline actually fixed it - the column is dropped, the split is
  the required group or time split, duplicates were removed, the metric is not accuracy, the
  report has no A11 flag (checked with the same lexicon for every system's report).
  Format issues are not scored for remediation (the pipeline description cannot show them).
* **False alarm** on a clean control: a critical issue (primary); warning or above
  (secondary); any reported issue, info included (tertiary).
* **Optimism gap**: the system's own validation claim minus its holdout score in the same
  metric (sign-adjusted so positive means optimistic), on both holdouts:
  ``optimism_gap_deploy`` (deployment-faithful; H3's primary) and ``optimism_gap_iid``
  (leaks intact; secondary). Real cases with ``holdout_scored = false`` have no gap.
* **Statistics** use the task as the unit: seeds are averaged within a task before the
  paired bootstrap; McNemar pairs are per task ("detected in both seeds", with "either seed"
  as a secondary); anonymised variants are never pooled with originals.
"""

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from bench.suites.planted import DEV_V1_PATH, FINAL_V1_PATH
from bench.suites.real_world import REAL_WORLD_PATH, load_anonymisations, load_real_world
from bench.systems.base import BenchResult
from holdout.audit.checks.a11_causal import check_causal_claims
from holdout.audit.types import Category, Severity

LOWER_IS_BETTER = frozenset({"rmse", "mae", "log_loss"})
SUITE_PATHS = {"final_v1": FINAL_V1_PATH, "dev_v1": DEV_V1_PATH}


@dataclass(frozen=True)
class Expectation:
    issue_id: str
    categories: tuple[Category, ...]
    min_severity: Severity
    columns: tuple[str, ...]
    match_columns: bool
    remediation: str | None
    secondary: bool = False


@dataclass(frozen=True)
class TaskTruth:
    suite: str
    task_id: str
    variant: str
    kind: str  # trap | clean
    trap: str
    semi_synthetic: bool
    expectations: tuple[Expectation, ...] = field(default=())


# ----------------------------------------------------------------------------- truth


def planted_truths(suite: str) -> dict[tuple[str, str], TaskTruth]:
    doc = json.loads(SUITE_PATHS[suite].read_text(encoding="utf-8"))
    truths = {}
    for task in doc["tasks"]:
        truth = task["truth"]
        expectations: tuple[Expectation, ...] = ()
        if task["kind"] == "trap":
            expectations = (
                Expectation(
                    issue_id=task["trap"],
                    categories=(Category(truth["expected_category"]),),
                    min_severity=Severity(truth["min_expected_severity"]),
                    columns=tuple(truth["affected_columns"]),
                    match_columns=False,
                    remediation=_planted_remedy(task, truth),
                ),
            )
        truths[(task["id"], "planted")] = TaskTruth(
            suite, task["id"], "planted", task["kind"], task["trap"], task["semi_synthetic"],
            expectations,
        )  # fmt: skip
    return truths


def _planted_remedy(task: dict[str, object], truth: dict[str, object]) -> str | None:
    columns = list(truth["affected_columns"])  # type: ignore[call-overload]
    remedy = truth["expected_remediation"]
    if remedy == "drop_columns":
        return "drop:" + ",".join(columns)
    if remedy == "change_split":
        return f"split:group:{columns[0]}" if task["trap"] == "group_leakage" else "split:time:"
    if remedy == "deduplicate":
        return "deduplicate"
    if remedy == "change_metric":
        return "metric:not_accuracy"
    if remedy == "remove_causal_claim":
        return "report:no_causal_claim"
    return None  # clean_formats: not visible in a pipeline description


def real_world_truths(path: Path = REAL_WORLD_PATH) -> dict[tuple[str, str], TaskTruth]:
    mappings = load_anonymisations()
    truths = {}
    for case in load_real_world(path):
        for variant in ("original", "anonymised"):
            rename: Callable[[str], str] = (
                (lambda c: c) if variant == "original" else mappings[case.id].rename
            )
            expectations = tuple(
                Expectation(
                    issue_id=issue.id,
                    categories=issue.categories,
                    min_severity=issue.min_severity,
                    columns=tuple(rename(c) for c in issue.columns),
                    match_columns=bool(issue.columns),
                    remediation=_rename_remedy(issue.remediation, rename),
                    secondary=issue.secondary,
                )
                for issue in case.scored_issues
            )
            truths[(case.id, variant)] = TaskTruth(
                "real_world_v1",
                case.id,
                variant,
                "trap" if case.kind == "issues" else "clean",
                case.id,
                False,
                expectations,
            )
    return truths


def _rename_remedy(remedy: str, rename: Callable[[str], str]) -> str:
    for prefix in ("drop:", "split:group:"):
        if remedy.startswith(prefix):
            return prefix + rename(remedy.removeprefix(prefix))
    if remedy.startswith("split:time:"):
        return "split:time:"
    return remedy


# ----------------------------------------------------------------------------- per result


def detected(result: BenchResult, expectation: Expectation) -> bool:
    for issue in result.issues:
        if issue.category not in expectation.categories:
            continue
        if not issue.severity.at_least(expectation.min_severity):
            continue
        if expectation.match_columns and not set(issue.columns) & set(expectation.columns):
            continue
        return True
    return False


def detected_critical(result: BenchResult, expectation: Expectation) -> bool:
    critical = Expectation(
        expectation.issue_id,
        expectation.categories,
        Severity.CRITICAL,
        expectation.columns,
        expectation.match_columns,
        expectation.remediation,
    )
    return detected(result, critical)


def remediated(result: BenchResult, expectation: Expectation) -> bool | None:
    """True / False, or None when not scorable (format issues, or no final pipeline)."""
    remedy = expectation.remediation
    pipeline = result.pipeline
    if remedy is None:
        return None
    if remedy == "report:no_causal_claim":
        return bool(result.report_text) and not check_causal_claims(result.report_text or "")
    if pipeline is None:
        return False
    if remedy.startswith("drop:"):
        return set(remedy.removeprefix("drop:").split(",")) <= set(pipeline.dropped_columns)
    if remedy.startswith("split:group:"):
        return pipeline.split == "group:" + remedy.removeprefix("split:group:")
    if remedy.startswith("split:time:"):
        return pipeline.split.startswith("time:")
    if remedy == "deduplicate":
        return pipeline.deduplicated
    if remedy == "metric:not_accuracy":
        return pipeline.metric is not None and pipeline.metric != "accuracy"
    return None


def false_alarm(result: BenchResult, level: str) -> bool:
    """On a clean control: ``critical`` | ``warning`` (or above) | ``any`` issue."""
    if level == "any":
        return bool(result.issues)
    minimum = Severity.CRITICAL if level == "critical" else Severity.WARNING
    return any(i.severity.at_least(minimum) for i in result.issues)


def holdout_score(result: BenchResult, which: str = "deploy") -> float | None:
    """The benchmark-metric holdout score; None for real cases scored on detection and
    remediation only (``holdout_scored = false``)."""
    if not result.holdout_scored:
        return None
    return result.holdout_deploy if which == "deploy" else result.holdout_iid


def optimism_gap(result: BenchResult, which: str = "deploy") -> float | None:
    """The system's own validation claim minus its score on holdout ``which`` (deploy or
    iid) in the same metric, sign-adjusted so positive means optimistic."""
    own = result.holdout_deploy_own_metric if which == "deploy" else result.holdout_iid_own_metric
    if result.validation_score is None or own is None or not result.holdout_scored:
        return None
    gap = result.validation_score - own
    return -gap if (result.validation_metric or "") in LOWER_IS_BETTER else gap


def relative_to_dummy(result: BenchResult, which: str = "deploy") -> float | None:
    """Share of the way from the dummy to a perfect score (roc_auc), or the RMSE reduction
    relative to the dummy (rmse). 0 = dummy, 1 = perfect."""
    s = holdout_score(result, which)
    d = result.dummy_holdout_deploy if which == "deploy" else result.dummy_holdout_iid
    if s is None or d is None:
        return None
    if result.benchmark_metric in LOWER_IS_BETTER:
        return (d - s) / d if d else None
    return (s - d) / (1 - d) if d < 1 else None


# ----------------------------------------------------------------------------- per task


@dataclass(frozen=True)
class TaskScore:
    """One system on one task (one variant), seeds kept apart and averaged."""

    task_id: str
    variant: str
    system: str
    issue_id: str
    secondary: bool
    detected_by_seed: dict[int, bool]
    detected_critical_by_seed: dict[int, bool]
    remediated_by_seed: dict[int, bool | None]

    @property
    def rate(self) -> float:
        return float(np.mean(list(self.detected_by_seed.values())))

    @property
    def both(self) -> bool:
        return all(self.detected_by_seed.values())

    @property
    def either(self) -> bool:
        return any(self.detected_by_seed.values())


def task_scores(
    results: Iterable[BenchResult], truths: dict[tuple[str, str], TaskTruth]
) -> list[TaskScore]:
    grouped: dict[tuple[str, str, str], list[BenchResult]] = defaultdict(list)
    for r in results:
        if r.status == "completed" or r.status == "stopped":
            grouped[(r.task_id, r.variant, r.system)].append(r)
    scores = []
    for (task_id, variant, system), runs in sorted(grouped.items()):
        truth = truths.get((task_id, variant))
        if truth is None or truth.kind != "trap":
            continue
        for exp in truth.expectations:
            scores.append(
                TaskScore(
                    task_id,
                    variant,
                    system,
                    exp.issue_id,
                    exp.secondary,
                    {r.seed: detected(r, exp) for r in runs},
                    {r.seed: detected_critical(r, exp) for r in runs},
                    {r.seed: remediated(r, exp) for r in runs},
                )
            )
    return scores


def clean_alarms(
    results: Iterable[BenchResult], truths: dict[tuple[str, str], TaskTruth], level: str
) -> dict[str, dict[str, bool]]:
    """system -> clean task (id/variant) -> any seed or variant raised an alarm (H2's unit:
    the clean TASK; a task counts as a false alarm if any seed or variant alarmed)."""
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in results:
        truth = truths.get((r.task_id, r.variant))
        if truth is None or truth.kind != "clean" or r.status not in ("completed", "stopped"):
            continue
        key = r.task_id
        out[r.system][key] = out[r.system].get(key, False) or false_alarm(r, level)
    return dict(out)


# ----------------------------------------------------------------------------- statistics


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a - b over tasks with a percentile 95% CI (tasks resampled in pairs)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("paired samples of equal, non-zero length are needed")
    diff = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_mean(
    values: Sequence[float], *, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@dataclass(frozen=True)
class McNemar:
    only_a: int  # a detected, b did not
    only_b: int
    both: int
    neither: int
    p_value: float  # exact, two-sided


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> McNemar:
    """Exact McNemar test (binomial on the discordant pairs), two-sided."""
    if len(a) != len(b):
        raise ValueError("paired outcomes of equal length are needed")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    both = sum(x and y for x, y in zip(a, b, strict=True))
    neither = len(a) - only_a - only_b - both
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return McNemar(only_a, only_b, both, neither, min(1.0, p))


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided upper bound of a binomial rate."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(stats.beta.ppf(confidence, k + 1, n - k))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def load_results(root: Path, suite: str) -> list[BenchResult]:
    """The latest result per unit across every system's file of ``suite``."""
    latest: dict[tuple[object, ...], BenchResult] = {}
    for path in sorted((root / suite).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = BenchResult.model_validate_json(line)
                latest[r.unit_key] = r
    return list(latest.values())
