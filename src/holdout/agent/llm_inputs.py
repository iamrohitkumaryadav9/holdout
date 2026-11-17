"""What the LLM may see about the data: schema, summary statistics, at most 5 rows.

The full dataset is never sent. Sample rows come from development rows only (never the
holdout), are capped at :data:`MAX_SAMPLE_ROWS`, and long values are truncated. The profile
summary is target-blind (computed before sealing).
"""

from typing import Any, Final

import pandas as pd

from holdout.data.profile import DatasetProfile

#: Hard cap on rows sent to the LLM in any prompt.
MAX_SAMPLE_ROWS: Final = 5
#: Characters kept per value in sample rows and column samples.
MAX_VALUE_CHARS: Final = 60


def profile_summary(profile: DatasetProfile) -> dict[str, Any]:
    """Per-column schema and statistics; no rows."""
    return {
        "n_rows": profile.n_rows,
        "n_columns": profile.n_cols,
        "target": profile.target,
        "task_hint": profile.task_hint.value,
        "target_class_frequencies": profile.target_class_frequencies,
        "columns": [
            {
                "name": c.name,
                "kind": c.kind.value,
                "dtype": c.dtype,
                "missing_rate": round(c.missing_rate, 4),
                "n_unique": c.n_unique,
                "examples": [_clip(e) for e in c.examples],
            }
            for c in profile.columns
        ],
        "datetime_candidates": profile.datetime_candidates,
        "entity_candidates": profile.entity_candidates
        + [e.column for e in profile.structural_entities],
    }


def sample_rows(dev: pd.DataFrame, seed: int, n: int = MAX_SAMPLE_ROWS) -> list[dict[str, str]]:
    """At most :data:`MAX_SAMPLE_ROWS` development rows, values as clipped strings."""
    count = min(n, MAX_SAMPLE_ROWS, len(dev))
    rows = dev.sample(n=count, random_state=seed) if count else dev.iloc[:0]
    return [{str(k): _clip(v) for k, v in row.items()} for row in rows.to_dict(orient="records")]


def column_samples(dev: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    """Column names with up to 5 distinct sample values each (for A2 hints)."""
    out = []
    for column in dev.columns:
        if column == target:
            continue
        values = dev[column].dropna().astype(str).unique()[:MAX_SAMPLE_ROWS]
        out.append({"column": str(column), "samples": [_clip(v) for v in values]})
    return out


def _clip(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 3] + "..."
