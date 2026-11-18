"""Mapping a system's free-text validation metric onto benchmark metrics."""

import math

import pytest

from bench.systems.metric_names import normalise_claim


@pytest.mark.parametrize(
    ("metric", "score", "expected_metric", "expected_score"),
    [
        ("roc_auc", 0.8, "roc_auc", 0.8),
        ("AUC", 0.8, "roc_auc", 0.8),
        ("average_precision", 0.4, "pr_auc", 0.4),
        ("neg_root_mean_squared_error", -0.3, "rmse", 0.3),
        ("neg_mean_squared_error", -0.09, "rmse", 0.3),  # the pilot's kin8nm claim
        ("mean_squared_error", 0.04, "rmse", 0.2),
        ("neg_mean_absolute_error", -0.5, "mae", 0.5),
        ("r2", 0.6, "r2", 0.6),
    ],
)
def test_claims_are_converted(
    metric: str, score: float, expected_metric: str, expected_score: float
) -> None:
    mapped, value, _ = normalise_claim(metric, score)
    assert mapped == expected_metric
    assert value is not None
    assert math.isclose(value, expected_score, rel_tol=1e-9)


def test_unknown_or_impossible_claims_are_left_alone_with_a_reason() -> None:
    assert normalise_claim("log_loss_weighted", 0.3) == (
        "log_loss_weighted",
        0.3,
        "unknown metric 'log_loss_weighted': no own-metric optimism gap",
    )
    assert "cannot convert" in normalise_claim("mse", -1.0)[2]
    assert normalise_claim(None, None) == (None, None, "")
