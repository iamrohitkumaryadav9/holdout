"""Scorer correctness on hand-built results, and the statistics."""

from datetime import UTC, datetime
from typing import Any

import pytest

from bench.score import (
    Expectation,
    TaskTruth,
    clean_alarms,
    clopper_pearson_upper,
    detected,
    false_alarm,
    holm,
    mcnemar,
    optimism_gap,
    paired_bootstrap,
    planted_truths,
    real_world_truths,
    relative_to_dummy,
    remediated,
    task_scores,
)
from bench.systems.base import BenchResult, FinalPipeline, ReportedIssue
from holdout.audit.types import Category, Severity

NOW = datetime(2026, 11, 20, tzinfo=UTC)


def _result(**kw: Any) -> BenchResult:
    base: dict[str, Any] = {
        "suite": "final_v1",
        "task_id": "t",
        "system": "holdout",
        "seed": 101,
        "variant": "planted",
        "status": "completed",
        "benchmark_metric": "roc_auc",
        "started_at": NOW,
        "finished_at": NOW,
    }
    base.update(kw)
    return BenchResult(**base)


def _issue(category: str, severity: str, *columns: str) -> ReportedIssue:
    return ReportedIssue(
        category=Category(category), severity=Severity(severity), columns=list(columns), source="x"
    )


LEAK = Expectation(
    "direct_leak",
    (Category.TARGET_LEAKAGE,),
    Severity.CRITICAL,
    ("refund_issued",),
    match_columns=False,
    remediation="drop:refund_issued",
)
REAL = Expectation(
    "duration",
    (Category.POST_OUTCOME_COLUMN, Category.TARGET_LEAKAGE),
    Severity.WARNING,
    ("duration",),
    match_columns=True,
    remediation="drop:duration",
)


def test_detection_needs_the_category_at_the_minimum_severity() -> None:
    assert detected(_result(issues=[_issue("target_leakage", "critical")]), LEAK)
    assert not detected(_result(issues=[_issue("target_leakage", "warning")]), LEAK)
    assert not detected(_result(issues=[_issue("post_outcome_column", "critical")]), LEAK)


def test_real_world_detection_also_needs_one_of_the_columns() -> None:
    assert detected(_result(issues=[_issue("post_outcome_column", "warning", "duration")]), REAL)
    assert not detected(_result(issues=[_issue("target_leakage", "critical", "age")]), REAL)
    assert not detected(_result(issues=[_issue("target_leakage", "info", "duration")]), REAL)


def test_remediation_checks_the_final_pipeline() -> None:
    dropped = _result(pipeline=FinalPipeline(dropped_columns=["refund_issued"]))
    assert remediated(dropped, LEAK) is True
    assert remediated(_result(pipeline=FinalPipeline()), LEAK) is False
    assert remediated(_result(pipeline=None), LEAK) is False
    group = Expectation("g", (Category.GROUP_LEAKAGE,), Severity.CRITICAL, ("account",), False,
                        "split:group:account")  # fmt: skip
    assert remediated(_result(pipeline=FinalPipeline(split="group:account")), group) is True
    assert remediated(_result(pipeline=FinalPipeline(split="stratified")), group) is False
    time = Expectation("t", (Category.TIME_LEAKAGE,), Severity.CRITICAL, (), False, "split:time:")
    assert remediated(_result(pipeline=FinalPipeline(split="time:<row>")), time) is True
    metric = Expectation("m", (Category.METRIC_MISMATCH,), Severity.CRITICAL, (), False,
                         "metric:not_accuracy")  # fmt: skip
    assert remediated(_result(pipeline=FinalPipeline(metric="balanced_accuracy")), metric)
    assert not remediated(_result(pipeline=FinalPipeline(metric="accuracy")), metric)
    causal = Expectation("c", (Category.CAUSAL_CLAIM,), Severity.WARNING, (), False,
                         "report:no_causal_claim")  # fmt: skip
    assert remediated(_result(report_text="Tenure is associated with churn."), causal)
    assert not remediated(_result(report_text="Phone contact causes conversion."), causal)
    formats = Expectation("f", (Category.FORMAT_ISSUE,), Severity.WARNING, (), False, None)
    assert remediated(_result(), formats) is None


def test_false_alarm_levels() -> None:
    quiet = _result(issues=[_issue("unstable_validation", "info")])
    warned = _result(issues=[_issue("unstable_validation", "warning")])
    loud = _result(issues=[_issue("time_leakage", "critical")])
    assert [false_alarm(r, "critical") for r in (quiet, warned, loud)] == [False, False, True]
    assert [false_alarm(r, "warning") for r in (quiet, warned, loud)] == [False, True, True]
    assert false_alarm(quiet, "any")


def test_clean_alarms_count_a_task_once_if_any_seed_or_variant_alarmed() -> None:
    truths = {
        ("c1", "planted"): TaskTruth("final_v1", "c1", "planted", "clean", "direct_leak", False),
        ("c2", "planted"): TaskTruth("final_v1", "c2", "planted", "clean", "direct_leak", False),
    }
    results = [
        _result(task_id="c1", seed=101),
        _result(task_id="c1", seed=202, issues=[_issue("time_leakage", "critical")]),
        _result(task_id="c2", seed=101),
        _result(task_id="c2", seed=202),
    ]
    assert clean_alarms(results, truths, "critical") == {"holdout": {"c1": True, "c2": False}}


def test_optimism_gap_and_relative_scores_are_sign_aware() -> None:
    auc = _result(validation_metric="roc_auc", validation_score=0.9,
                  holdout_deploy_own_metric=0.6, holdout_iid_own_metric=0.85)  # fmt: skip
    assert optimism_gap(auc) == pytest.approx(0.3)  # deploy is the default
    assert optimism_gap(auc, "deploy") == pytest.approx(0.3)
    assert optimism_gap(auc, "iid") == pytest.approx(0.05)
    rmse = _result(validation_metric="rmse", validation_score=1.0, holdout_deploy_own_metric=1.5)
    assert optimism_gap(rmse) == pytest.approx(0.5)  # holdout error larger: optimistic
    both = _result(holdout_deploy=0.75, dummy_holdout_deploy=0.5, holdout_iid=0.9,
                   dummy_holdout_iid=0.5)  # fmt: skip
    assert relative_to_dummy(both) == 0.5
    assert relative_to_dummy(both, "iid") == pytest.approx(0.8)
    reg = _result(benchmark_metric="rmse", holdout_deploy=2.0, dummy_holdout_deploy=4.0)
    assert relative_to_dummy(reg) == pytest.approx(0.5)
    unscored = _result(validation_score=0.9, holdout_deploy_own_metric=0.6, holdout_scored=False)
    assert optimism_gap(unscored) is None  # detection and remediation only


def test_task_scores_keep_seeds_apart() -> None:
    truths = {("t", "planted"): TaskTruth("final_v1", "t", "planted", "trap", "direct_leak",
                                           False, (LEAK,))}  # fmt: skip
    results = [
        _result(seed=101, issues=[_issue("target_leakage", "critical")]),
        _result(seed=202),
    ]
    (score,) = task_scores(results, truths)
    assert score.detected_by_seed == {101: True, 202: False}
    assert (score.rate, score.both, score.either) == (0.5, False, True)


# ------------------------------------------------------------------ statistics


def test_mcnemar_is_exact_and_two_sided() -> None:
    a = [True] * 8 + [False] * 2
    b = [False] * 8 + [True] * 2
    result = mcnemar(a, b)
    assert (result.only_a, result.only_b) == (8, 2)
    assert result.p_value == pytest.approx(0.109375)  # 2 * P(X <= 2), X ~ Bin(10, 0.5)
    assert mcnemar([True, False], [True, False]).p_value == 1.0


def test_clopper_pearson_upper_bound() -> None:
    assert clopper_pearson_upper(0, 16) == pytest.approx(1 - 0.05 ** (1 / 16))
    assert clopper_pearson_upper(0, 16) > 0.17  # why H2's bound is reported, not the criterion
    assert clopper_pearson_upper(3, 3) == 1.0


def test_holm_adjustment() -> None:
    adjusted = holm({"H1": 0.01, "H2": 0.04, "H3": 0.03, "H4": 0.20})
    assert adjusted == pytest.approx({"H1": 0.04, "H3": 0.09, "H2": 0.09, "H4": 0.20})


def test_paired_bootstrap_brackets_the_mean_difference() -> None:
    mean, low, high = paired_bootstrap([1, 1, 1, 0, 1], [0, 1, 0, 0, 0])
    assert mean == pytest.approx(0.6)
    assert low <= mean <= high
    with pytest.raises(ValueError, match="paired samples"):
        paired_bootstrap([1], [1, 0])


def test_the_analysis_renders_tables_and_charts_from_results_files(tmp_path: Any) -> None:
    from bench.analyze import charts, report
    from bench.run import append_result

    trap, clean = "direct_leak@qsar_biodeg", "direct_leak:clean@qsar_biodeg"
    for system, caught in (("holdout", True), ("react", False)):
        for seed in (101, 202):
            issues = [_issue("target_leakage", "critical", "refund_issued")] if caught else []
            append_result(
                tmp_path,
                _result(
                    task_id=trap, system=system, seed=seed, issues=issues,
                    pipeline=FinalPipeline(dropped_columns=["refund_issued"] if caught else []),
                    validation_metric="roc_auc", validation_score=0.8,
                    holdout_deploy_own_metric=0.75, holdout_deploy=0.75, dummy_holdout_deploy=0.5,
                    holdout_iid_own_metric=0.79, holdout_iid=0.79, dummy_holdout_iid=0.5,
                    cost_inr=1.0, report_text="Tenure is associated with the outcome.",
                ),
            )  # fmt: skip
            append_result(tmp_path, _result(task_id=clean, system=system, seed=seed))
    text = "\n".join(report(tmp_path, "final_v1"))
    assert "| direct_leak | 100.0% | 0.0% |" in text
    assert "| holdout | 0/1 = 0.0% |" in text  # one clean task, no critical alarm
    assert "holdout vs react" in text
    assert "optimism gap" in text.lower()
    paths = charts(_load(tmp_path), planted_truths("final_v1"), tmp_path / "img")
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def _load(root: Any) -> list[BenchResult]:
    from bench.score import load_results

    return load_results(root, "final_v1")


def test_truths_are_built_from_the_frozen_suites() -> None:
    planted = planted_truths("final_v1")
    assert sum(t.kind == "trap" for t in planted.values()) == 28
    assert sum(t.semi_synthetic for t in planted.values()) == 6  # 3 time traps + 3 twins
    real = real_world_truths()
    anon = real[("bank_marketing", "anonymised")]
    duration = next(e for e in anon.expectations if e.issue_id == "duration_post_outcome")
    assert duration.columns[0].startswith("col_")
    assert duration.remediation is not None
    assert duration.remediation.startswith("drop:col_")
