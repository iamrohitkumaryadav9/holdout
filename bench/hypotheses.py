"""The confirmatory analysis fixed by the pre-registration (docs/preregistration.md).

Each hypothesis has one estimate, one 95% CI, one p-value and one decision rule, all
computed from the results files only. The unit is the task: seeds are averaged within a
task (or, for detection, a task counts when detected in both seeds), and anonymised
variants are never pooled with originals.

* **H1** Holdout's detection recall on planted traps exceeds ReAct's. Exact McNemar on the
  planted final_v1 trap tasks (detected in both seeds), two-sided; supported when Holdout
  detects more tasks and the Holm-adjusted p < 0.05. Estimate: the mean per-task difference
  in detection rate (seeds averaged) with a paired bootstrap CI.
* **H2** Holdout's critical false-alarm rate on clean controls is at most 10%. Unit: the
  clean task (final_v1 clean twins and real-world clean controls; a task alarms if any seed
  or variant raised a critical issue). Pass: the observed rate <= 10%. Reported with the
  one-sided 95% Clopper-Pearson upper bound and the exact one-sided binomial p for
  H0: rate >= 10%. With 16 tasks that p cannot fall below 0.9^16 = 0.185, so H2's decision
  is the observed-rate rule; its p still enters the Holm family (which makes H1, H3 and H4
  more conservative, never less).
* **H3** Holdout's optimism gap is smaller than ReAct's. Primary: the deploy gap; secondary:
  the iid gap. Per planted final_v1 task with a gap for both systems (seeds averaged), the
  difference Holdout - ReAct; two-sided paired sign-flip permutation test (exact up to 20
  tasks, otherwise 100,000 random sign flips, seed 0); supported when the mean difference
  is negative and the Holm-adjusted p < 0.05.
* **H4** Deterministic audits beat the LLM judge on detection: Holdout (deterministic audits)
  vs llm_judge (the same agent with the audit layer replaced by the Gemini verifier), as H1.
* **H5** (exploratory, outside Holm): real-world detection does not depend on column names;
  the paired original - anonymised difference per issue with a bootstrap CI; "no material
  dependence" when the CI lies within +/-0.20.

Holm-Bonferroni over H1-H4 at alpha = 0.05.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from bench.score import (
    TaskTruth,
    clean_alarms,
    clopper_pearson_upper,
    holm,
    mcnemar,
    optimism_gap,
    paired_bootstrap,
    task_scores,
)
from bench.systems.base import BenchResult

ALPHA = 0.05
H2_BOUND = 0.10
H5_MARGIN = 0.20
EXACT_SIGN_FLIP_MAX = 20
SIGN_FLIP_RESAMPLES = 100_000


@dataclass(frozen=True)
class Outcome:
    name: str
    statement: str
    n: int
    estimate: float | None
    ci: tuple[float, float] | None
    p_value: float | None
    p_holm: float | None
    decision: str
    detail: str


def sign_flip_p(diffs: Sequence[float], *, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(diffs) = 0. Exact enumeration
    up to ``EXACT_SIGN_FLIP_MAX`` pairs, else ``SIGN_FLIP_RESAMPLES`` random flips (the
    observed assignment counted, so p > 0)."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return 1.0
    observed = abs(d.mean())
    if n <= EXACT_SIGN_FLIP_MAX:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        means = np.abs((signs * d).mean(axis=1))
        return float(np.mean(means >= observed - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(SIGN_FLIP_RESAMPLES, n))
    means = np.abs((signs * d).mean(axis=1))
    return float((np.sum(means >= observed - 1e-12) + 1) / (SIGN_FLIP_RESAMPLES + 1))


def _detection(
    name: str, statement: str, a: str, b: str,
    planted: list[BenchResult], truths: dict[tuple[str, str], TaskTruth],
) -> tuple[Outcome, float | None]:  # fmt: skip
    scores = [s for s in task_scores(planted, truths) if not s.secondary]
    by = {(s.system, s.task_id, s.variant, s.issue_id): s for s in scores}
    keys = sorted({(s.task_id, s.variant, s.issue_id) for s in scores})
    shared = [k for k in keys if (a, *k) in by and (b, *k) in by]
    if not shared:
        return Outcome(name, statement, 0, None, None, None, None, "not evaluable",
                       f"no planted trap task has results for both {a} and {b}"), None  # fmt: skip
    test = mcnemar([by[(a, *k)].both for k in shared], [by[(b, *k)].both for k in shared])
    mean, low, high = paired_bootstrap([by[(a, *k)].rate for k in shared],
                                       [by[(b, *k)].rate for k in shared])  # fmt: skip
    detail = (f"{a} only {test.only_a}, {b} only {test.only_b}, both {test.both}, neither "
              f"{test.neither} (detected in both seeds)")  # fmt: skip
    return Outcome(name, statement, len(shared), mean, (low, high), test.p_value, None, "",
                   detail), (1.0 if test.only_a > test.only_b else -1.0)  # fmt: skip


def _per_task(results: list[BenchResult], system: str, which: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for r in results:
        if r.system == system and r.status == "completed" and r.holdout_scored:
            gap = optimism_gap(r, which)
            if gap is not None:
                values.setdefault(f"{r.task_id}/{r.variant}", []).append(gap)
    return {k: float(np.mean(v)) for k, v in values.items()}


def _gap(planted: list[BenchResult], which: str) -> tuple[int, float, tuple[float, float], float]:
    mine, theirs = _per_task(planted, "holdout", which), _per_task(planted, "react", which)
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        return 0, float("nan"), (float("nan"), float("nan")), 1.0
    diffs = [mine[k] - theirs[k] for k in shared]
    mean, low, high = paired_bootstrap([mine[k] for k in shared], [theirs[k] for k in shared])
    return len(shared), mean, (low, high), sign_flip_p(diffs)


def confirmatory(
    planted: list[BenchResult],
    planted_truth: dict[tuple[str, str], TaskTruth],
    real: list[BenchResult],
    real_truth: dict[tuple[str, str], TaskTruth],
) -> list[Outcome]:
    """H1-H4 with Holm-adjusted p-values and decisions, then H5 (exploratory)."""
    h1, h1_sign = _detection(
        "H1", "Holdout's detection recall on planted traps exceeds ReAct's",
        "holdout", "react", planted, planted_truth,
    )  # fmt: skip
    h4, h4_sign = _detection(
        "H4", "deterministic audits (holdout) beat the LLM judge (llm_judge) on detection",
        "holdout", "llm_judge", planted, planted_truth,
    )  # fmt: skip
    # H2: clean tasks of both suites, critical level, any seed or variant.
    real_alarms = clean_alarms(real, real_truth, "critical").get("holdout", {})
    alarms = {**clean_alarms(planted, planted_truth, "critical").get("holdout", {}),
              **{f"real/{k}": v for k, v in real_alarms.items()}}  # fmt: skip
    k, n = sum(alarms.values()), len(alarms)
    h2 = Outcome(
        "H2", f"Holdout's critical false-alarm rate on clean tasks is at most {H2_BOUND:.0%}",
        n, (k / n) if n else None, (0.0, clopper_pearson_upper(k, n)) if n else None,
        float(stats.binom.cdf(k, n, H2_BOUND)) if n else None, None, "",
        f"{k} of {n} clean tasks raised a critical issue; one-sided 95% Clopper-Pearson upper "
        f"bound {clopper_pearson_upper(k, n):.1%}" if n else "no clean task results",
    )  # fmt: skip
    gap_n, gap_mean, gap_ci, gap_p = _gap(planted, "deploy")
    iid_n, iid_mean, iid_ci, iid_p = _gap(planted, "iid")
    h3 = Outcome(
        "H3", "Holdout's optimism gap (deploy holdout) is smaller than ReAct's",
        gap_n, gap_mean if gap_n else None, gap_ci if gap_n else None,
        gap_p if gap_n else None, None, "",
        f"mean per-task difference holdout - react (negative favours Holdout); secondary, iid "
        f"holdout: {iid_mean:+.4f} [{iid_ci[0]:+.4f}, {iid_ci[1]:+.4f}], sign-flip p "
        f"{iid_p:.4f}, n {iid_n}" if iid_n else "mean per-task difference holdout - react",
    )  # fmt: skip
    family = {o.name: o.p_value for o in (h1, h2, h3, h4) if o.p_value is not None}
    adjusted = holm(family)
    out = []
    for o in (h1, h2, h3, h4):
        p_holm = adjusted.get(o.name)
        if o.p_value is None or o.estimate is None:
            decision = "not evaluable"
        elif o.name == "H2":
            decision = "pass" if o.estimate <= H2_BOUND else "fail"
        elif o.name == "H3":
            ok = o.estimate < 0 and p_holm is not None and p_holm < ALPHA
            decision = "supported" if ok else "not supported"
        else:
            sign = h1_sign if o.name == "H1" else h4_sign
            ok = sign is not None and sign > 0 and p_holm is not None and p_holm < ALPHA
            decision = "supported" if ok else "not supported"
        out.append(Outcome(o.name, o.statement, o.n, o.estimate, o.ci, o.p_value, p_holm,
                           decision, o.detail))  # fmt: skip
    out.append(_h5(real, real_truth))
    return out


def _h5(real: list[BenchResult], truths: dict[tuple[str, str], TaskTruth]) -> Outcome:
    scores = [s for s in task_scores(real, truths) if s.system == "holdout" and not s.secondary]
    orig = {(s.task_id, s.issue_id): s.rate for s in scores if s.variant == "original"}
    anon = {(s.task_id, s.issue_id): s.rate for s in scores if s.variant == "anonymised"}
    shared = sorted(set(orig) & set(anon))
    statement = "(exploratory) Holdout's real-world detection does not depend on column names"
    if not shared:
        return Outcome("H5", statement, 0, None, None, None, None, "not evaluable",
                       "no real-world issue has both variants")  # fmt: skip
    mean, low, high = paired_bootstrap([orig[k] for k in shared], [anon[k] for k in shared])
    within = low >= -H5_MARGIN and high <= H5_MARGIN
    return Outcome("H5", statement, len(shared), mean, (low, high), None, None,
                   "no material dependence" if within else "inconclusive or dependent",
                   f"original - anonymised detection rate; margin +/-{H5_MARGIN:.2f}; "
                   "outside the Holm family")  # fmt: skip


def render(outcomes: list[Outcome]) -> list[str]:
    lines = [
        "## Pre-registered hypotheses",
        "",
        "| | hypothesis | n (tasks) | estimate [95% CI] | p | p (Holm, H1-H4) | decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        est = "-" if o.estimate is None else f"{o.estimate:+.4f}"
        if o.ci is not None and o.estimate is not None:
            est += f" [{o.ci[0]:+.4f}, {o.ci[1]:+.4f}]"
        p = "-" if o.p_value is None else f"{o.p_value:.4f}"
        ph = "-" if o.p_holm is None else f"{o.p_holm:.4f}"
        lines.append(f"| {o.name} | {o.statement} | {o.n} | {est} | {p} | {ph} | "
                     f"**{o.decision}** |")  # fmt: skip
    lines += ["", *(f"- {o.name}: {o.detail}" for o in outcomes)]
    return lines
