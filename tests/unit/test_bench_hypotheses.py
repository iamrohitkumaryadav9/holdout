"""The pre-registered confirmatory analysis on hand-built results (bench/hypotheses.py)."""

import itertools
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from bench.hypotheses import H2_BOUND, confirmatory, render, sign_flip_p
from bench.score import Expectation, TaskTruth, clean_alarms, failure_rates
from bench.systems.base import BenchResult, ReportedIssue
from holdout.audit.types import Category, Severity

N_TRAPS = 12
POST = Category.POST_OUTCOME_COLUMN
N_CLEAN = 6
SEEDS = (101, 202)


def _truths() -> dict[tuple[str, str], TaskTruth]:
    exp = Expectation("direct_leak", (Category.TARGET_LEAKAGE,), Severity.CRITICAL, ("x",),
                      False, "drop:x")  # fmt: skip
    truths = {(f"t{i}", "planted"): TaskTruth("final_v1", f"t{i}", "planted", "trap",
                                               "direct_leak", False, (exp,))
              for i in range(N_TRAPS)}  # fmt: skip
    for i in range(N_CLEAN):
        truths[(f"c{i}", "planted")] = TaskTruth("final_v1", f"c{i}", "planted", "clean",
                                                  "direct_leak", False)  # fmt: skip
    return truths


def _real_truths() -> dict[tuple[str, str], TaskTruth]:
    exp = Expectation("leak", (Category.POST_OUTCOME_COLUMN,), Severity.WARNING, (), False,
                      None)  # fmt: skip
    out = {}
    for v in ("original", "anonymised"):
        out[("bank", v)] = TaskTruth("real_world_v1", "bank", v, "trap", "bank", False, (exp,))
        out[("mush", v)] = TaskTruth("real_world_v1", "mush", v, "clean", "mush", False)
    return out


def _r(system: str, task: str, seed: int, *, detected: bool = False, critical: bool = False,
       gap: float | None = None, variant: str = "planted", suite: str = "final_v1",
       category: Category = Category.TARGET_LEAKAGE) -> BenchResult:  # fmt: skip
    issues = []
    if detected or critical:
        issues.append(ReportedIssue(category=category, severity=Severity.CRITICAL, columns=["x"],
                                    source="A1", message=""))  # fmt: skip
    now = datetime.now(UTC)
    extra: dict[str, Any] = {}
    if gap is not None:
        extra = {
            "validation_metric": "roc_auc",
            "validation_score": 0.8 + gap,
            "holdout_deploy_own_metric": 0.8,
            "holdout_iid_own_metric": 0.8 + gap / 2,
        }
    return BenchResult(suite=suite, task_id=task, system=system, seed=seed,  # type: ignore[arg-type]
                       variant=variant, status="completed", benchmark_metric="roc_auc",  # type: ignore[arg-type]
                       issues=issues, started_at=now, finished_at=now, **extra)  # fmt: skip


def _planted(holdout_hits: int, react_hits: int, judge_hits: int,
             critical_clean: int = 0) -> list[BenchResult]:  # fmt: skip
    out = []
    for i, seed in itertools.product(range(N_TRAPS), SEEDS):
        out.append(_r("holdout", f"t{i}", seed, detected=i < holdout_hits, gap=0.01))
        out.append(_r("react", f"t{i}", seed, detected=i < react_hits, gap=0.06 + 0.01 * i))
        out.append(_r("llm_judge", f"t{i}", seed, detected=i < judge_hits))
    for i, seed in itertools.product(range(N_CLEAN), SEEDS):
        out.append(_r("holdout", f"c{i}", seed, critical=i < critical_clean and seed == 101))
    return out


def _real(orig: bool = True, anon: bool = True) -> list[BenchResult]:
    out = []
    for seed in SEEDS:
        for variant, hit in (("original", orig), ("anonymised", anon)):
            out.append(_r("holdout", "bank", seed, detected=hit, variant=variant,
                          suite="real_world_v1", category=POST))  # fmt: skip
            out.append(_r("holdout", "mush", seed, variant=variant, suite="real_world_v1"))
    return out


def _by_name(outcomes: list[Any]) -> dict[str, Any]:
    return {o.name: o for o in outcomes}


def test_a_clear_win_supports_h1_h3_h4_and_h2_passes() -> None:
    out = _by_name(confirmatory(_planted(12, 2, 3), _truths(), _real(), _real_truths()))
    assert out["H1"].decision == "supported"
    assert out["H1"].estimate == pytest.approx(10 / 12)
    assert "holdout only 10, react only 0" in out["H1"].detail
    assert out["H1"].p_value == pytest.approx(2 * 0.5**10)
    assert out["H4"].decision == "supported"
    assert out["H3"].decision == "supported"
    assert out["H3"].estimate < 0  # Holdout's gap smaller
    assert out["H3"].n == N_TRAPS
    assert "secondary, iid holdout" in out["H3"].detail
    # H2: 0 of 6 clean planted + 1 clean real task; decided by the rate rule alone.
    assert out["H2"].n == N_CLEAN + 1
    assert out["H2"].decision == "pass"
    assert out["H2"].p_value is None  # not in the Holm family
    assert out["H2"].p_holm is None
    assert f"exact binomial p (H0: rate >= 10%) {0.9 ** (N_CLEAN + 1):.4f}" in out["H2"].detail
    assert "descriptive only" in out["H2"].detail
    # Holm over H1, H3 and H4 only: never below the raw p, and m = 3.
    for name in ("H1", "H3", "H4"):
        assert out[name].p_holm >= out[name].p_value
    smallest = min(out[n].p_value for n in ("H1", "H3", "H4"))
    assert min(out[n].p_holm for n in ("H1", "H3", "H4")) == pytest.approx(3 * smallest)
    assert out["H5"].decision == "no material dependence"
    assert out["H5"].p_holm is None  # exploratory: outside the Holm family


def test_direction_and_the_h2_rule_decide() -> None:
    out = _by_name(confirmatory(_planted(2, 12, 2, critical_clean=2), _truths(), _real(),
                                _real_truths()))  # fmt: skip
    assert out["H1"].p_value < 0.05  # significant, but in ReAct's favour
    assert out["H1"].decision == "not supported"
    assert out["H4"].decision == "not supported"  # no discordant pairs: p = 1
    assert out["H2"].estimate == pytest.approx(2 / (N_CLEAN + 1))
    assert out["H2"].estimate > H2_BOUND
    assert out["H2"].decision == "fail"
    assert "2 of 7 clean tasks did not pass" in out["H2"].detail
    h5 = _by_name(confirmatory(_planted(2, 2, 2), _truths(), _real(True, False),
                               _real_truths()))["H5"]  # fmt: skip
    assert h5.estimate == pytest.approx(1.0)
    assert h5.decision == "inconclusive or dependent"


def test_missing_systems_are_not_evaluable_and_render() -> None:
    only_holdout = [r for r in _planted(5, 5, 5) if r.system == "holdout"]
    out = _by_name(confirmatory(only_holdout, _truths(), [], _real_truths()))
    assert out["H1"].decision == out["H3"].decision == out["H4"].decision == "not evaluable"
    assert out["H5"].decision == "not evaluable"
    text = "\n".join(render(list(out.values())))
    assert "| H1 |" in text
    assert "**not evaluable**" in text
    assert "p (Holm: H1, H3, H4)" in text


def test_the_analysis_shows_unit_failure_rates_prominently() -> None:
    from bench.analyze import failure_table

    results = [r.model_copy(update={"status": "failed"}) if r.task_id == "t0" else r
               for r in _planted(5, 5, 5)]  # fmt: skip
    text = "\n".join(failure_table(results))
    assert "| react | 2 of 24 | **8.3%** |" in text
    assert "| llm_judge | 2 of 24 | **8.3%** |" in text


def test_failed_units_count_as_misses_and_paused_ones_are_not_results() -> None:
    results = _planted(12, 12, 12)
    failed = [
        r.model_copy(update={"status": "failed"})
        if r.system == "react" and r.task_id in ("t0", "t1") else r
        for r in results
    ]  # fmt: skip
    out = _by_name(confirmatory(failed, _truths(), _real(), _real_truths()))
    assert "holdout only 2, react only 0" in out["H1"].detail  # the crashes are misses
    assert out["H1"].n == N_TRAPS
    paused = [r for r in results if not (r.system == "react" and r.task_id == "t0")] + [
        r.model_copy(update={"status": "paused"})
        for r in results if r.system == "react" and r.task_id == "t0"
    ]  # fmt: skip
    assert _by_name(confirmatory(paused, _truths(), _real(), _real_truths()))["H1"].n == (
        N_TRAPS - 1
    )
    # A failed clean unit counts AGAINST the system: it stays in the denominator as a non-pass.
    clean_fail = [
        r.model_copy(update={"status": "failed"}) if (r.task_id, r.seed) == ("c1", 202) else r
        for r in _planted(12, 2, 2)
    ]  # fmt: skip
    h2 = _by_name(confirmatory(clean_fail, _truths(), _real(), _real_truths()))["H2"]
    assert h2.n == N_CLEAN + 1
    assert h2.estimate == pytest.approx(1 / (N_CLEAN + 1))
    assert "1 with a failed unit" in h2.detail
    failing_more = [r.model_copy(update={"status": "failed"}) if r.task_id in ("c1", "c2") else r
                    for r in _planted(12, 2, 2)]  # fmt: skip
    assert _by_name(confirmatory(failing_more, _truths(), _real(),
                                 _real_truths()))["H2"].decision == "fail"  # fmt: skip
    for level in ("critical", "warning", "any"):  # every false-alarm metric
        assert clean_alarms(clean_fail, _truths(), level)["holdout"]["c1"] is True
    assert failure_rates(clean_fail)["holdout"] == (1, 2 * (N_TRAPS + N_CLEAN))
    paused_only = [r.model_copy(update={"status": "paused"}) for r in _planted(1, 1, 1)]
    assert failure_rates(paused_only) == {}  # paused units are not results yet


def test_the_sign_flip_test_is_exact_for_small_samples() -> None:
    assert sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(2 / 8)  # all-plus or all-minus
    assert sign_flip_p([0.0, 0.0]) == 1.0
    assert sign_flip_p([]) == 1.0
    rng = np.random.default_rng(1)
    big = rng.normal(0.5, 1.0, 40)
    p = sign_flip_p(big.tolist())
    assert 0 < p < 0.05
    assert sign_flip_p(big.tolist()) == p  # fixed seed: deterministic
