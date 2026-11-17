"""Metric registry: completeness, directions and lookup errors."""

import pytest

from holdout.core.schemas import Direction, MetricName, Task
from holdout.plan.metrics import METRICS, get_metric


def test_registry_covers_every_metric_name() -> None:
    assert set(METRICS) == set(MetricName)


@pytest.mark.parametrize(
    ("name", "task", "direction"),
    [
        ("roc_auc", Task.CLASSIFICATION, Direction.HIGHER),
        ("pr_auc", Task.CLASSIFICATION, Direction.HIGHER),
        ("rmse", Task.REGRESSION, Direction.LOWER),
        ("mae", Task.REGRESSION, Direction.LOWER),
        ("r2", Task.REGRESSION, Direction.HIGHER),
    ],
)
def test_metric_metadata(name: str, task: Task, direction: Direction) -> None:
    metric = get_metric(name)
    assert metric.task is task
    assert metric.direction is direction


def test_pr_auc_is_binary_only() -> None:
    assert get_metric("pr_auc").binary_only
    assert not get_metric("roc_auc").binary_only


def test_improvement_respects_direction() -> None:
    assert get_metric("roc_auc").improvement(0.8, 0.5) == pytest.approx(0.3)
    assert get_metric("rmse").improvement(2.0, 5.0) == pytest.approx(3.0)
    assert get_metric("rmse").improvement(5.0, 2.0) < 0


def test_unknown_metric_names_the_options() -> None:
    with pytest.raises(KeyError, match="choose from"):
        get_metric("logloss")
