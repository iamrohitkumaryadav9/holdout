"""Map a system's free-text validation metric onto a benchmark metric.

ReAct reports its validation metric in its own words, often as a scikit-learn scorer name
(``neg_mean_squared_error``, ``neg_root_mean_squared_error``, ``average_precision``). To
compute its optimism gap fairly, the claim is converted to a metric the harness can score,
with the score converted accordingly:

* ``neg_`` scorers are negated (scikit-learn reports errors as negative numbers);
* mean squared error becomes RMSE (square root);
* aliases (``auc`` -> roc_auc, ``average_precision`` -> pr_auc, ...) are renamed.

A metric that cannot be mapped is left as reported; its own-metric gap is then missing,
and the result says why.
"""

import math
from typing import Final

ALIASES: Final[dict[str, str]] = {
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "roc_auc_ovr": "roc_auc",
    "rocauc": "roc_auc",
    "pr_auc": "pr_auc",
    "average_precision": "pr_auc",
    "f1": "f1",
    "f1_score": "f1",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "rmse": "rmse",
    "root_mean_squared_error": "rmse",
    "mae": "mae",
    "mean_absolute_error": "mae",
    "r2": "r2",
    "r2_score": "r2",
}
_MSE: Final = frozenset({"mse", "mean_squared_error"})


def normalise_claim(
    metric: str | None, score: float | None
) -> tuple[str | None, float | None, str]:
    """(benchmark metric or the original, converted score, note)."""
    if metric is None or score is None:
        return metric, score, ""
    name = metric.strip().lower().replace(" ", "_").replace("-", "_")
    negated = name.startswith("neg_")
    name = name.removeprefix("neg_")
    value = -score if negated else score
    if name in _MSE:
        if value < 0:
            return metric, score, f"cannot convert a negative MSE ({score}) to RMSE"
        return "rmse", math.sqrt(value), f"{metric} {score} converted to RMSE"
    if name in ALIASES:
        mapped = ALIASES[name]
        note = "" if (mapped == metric and not negated) else f"{metric} {score} read as {mapped}"
        return mapped, value, note
    return metric, score, f"unknown metric {metric!r}: no own-metric optimism gap"
