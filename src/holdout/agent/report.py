"""The numbers-from-state rule for agent reports.

The LLM writes prose; code writes every number. Prose may contain ``{{placeholder}}``
names from :func:`placeholder_values`, which code fills from run state. A token with a digit
anywhere in the prose is rejected unless it is an allow-listed identifier: check ids
(A1-A12), column names (``feature_12``), metric and model names (``f1``, ``r2``) and other
plan field values. Rejected drafts get one repair through the structured-call contract.
"""

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from holdout.agent.schemas import ReportDraft
from holdout.audit.types import CheckId
from holdout.core.schemas import MetricName
from holdout.plan.schema import ModelPlan

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
#: Words: letters, digits, underscores and inner hyphens; a decimal part stays attached.
_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[-.][A-Za-z0-9_]+)*")
_HAS_DIGIT = re.compile(r"\d")
_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?%?")

#: Placeholder catalogue: name -> meaning (shown to the LLM, without values).
PLACEHOLDER_MEANINGS: dict[str, str] = {
    "validation_score": "cross-validated score of the final plan on development rows",
    "validation_std": "standard deviation of that score across folds",
    "dummy_score": "cross-validated score of the dummy baseline",
    "holdout_score": "score on the sealed holdout (only if it was evaluated)",
    "optimism_gap": "validation score minus holdout score (only if evaluated)",
    "n_rows": "rows in the dataset",
    "n_dev_rows": "development rows",
    "n_holdout_rows": "sealed holdout rows",
    "n_fix_rounds": "audit-and-fix rounds the agent ran",
    "n_critical": "critical audit flags on the final plan",
    "n_warning": "warning audit flags on the final plan",
}


def placeholder_values(facts: Mapping[str, float | int | None]) -> dict[str, str]:
    """Formatted values for the placeholders that exist in this run (None = unavailable)."""
    values: dict[str, str] = {}
    for name in PLACEHOLDER_MEANINGS:
        value = facts.get(name)
        if value is None:
            continue
        values[name] = str(value) if isinstance(value, int) else f"{value:.4f}"
    return values


def allowed_tokens(columns: Iterable[str], plan: ModelPlan | None) -> set[str]:
    """Identifiers that may contain digits without being numbers."""
    allowed = {c.value.lower() for c in CheckId} | {m.value.lower() for m in MetricName}
    for column in columns:
        allowed.add(column.lower())
        allowed.update(t.lower() for t in _TOKEN.findall(column))
    if plan is not None:
        allowed.update(_plan_tokens(plan.model_dump(mode="json")))
    return allowed


def _plan_tokens(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {t for k, v in value.items() for t in {str(k).lower(), *_plan_tokens(v)}}
    if isinstance(value, list):
        return {t for v in value for t in _plan_tokens(v)}
    if isinstance(value, str):
        return {value.lower(), *(t.lower() for t in _TOKEN.findall(value))}
    return set()


def prose_problems(text: str, allowed: set[str], placeholders: set[str]) -> list[str]:
    """Numeric literals outside placeholders, and unknown placeholders."""
    problems = [
        f"unknown placeholder {{{{{name}}}}}"
        for name in PLACEHOLDER.findall(text)
        if name not in placeholders
    ]
    bare = PLACEHOLDER.sub(" ", text)
    for token in _TOKEN.findall(bare):
        if _HAS_DIGIT.search(token) and token.lower() not in allowed:
            problems.append(f"numeric literal {token!r}: use a placeholder instead")
    return problems


def draft_problems(
    draft: ReportDraft,
    *,
    allowed: set[str],
    placeholders: set[str],
    required_checks: Iterable[str],
) -> list[str]:
    """Everything wrong with a draft (empty when it may be used)."""
    problems: list[str] = []
    for field, text in _texts(draft):
        problems.extend(f"{field}: {p}" for p in prose_problems(text, allowed, placeholders))
    findings = " ".join(draft.findings)
    for check in required_checks:
        if not re.search(rf"\b{re.escape(check)}\b", findings):
            problems.append(f"findings must mention flag {check}")
    return problems


def fill(text: str, values: Mapping[str, str]) -> str:
    return PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def fill_draft(draft: ReportDraft, values: Mapping[str, str]) -> ReportDraft:
    return draft.model_copy(
        update={
            "summary": fill(draft.summary, values),
            "findings": [fill(t, values) for t in draft.findings],
            "caveats": [fill(t, values) for t in draft.caveats],
            "recommendations": [fill(t, values) for t in draft.recommendations],
        }
    )


def prose_numbers(text: str, allowed: set[str]) -> list[str]:
    """Numbers in filled prose that are not part of allow-listed identifiers (for tests)."""
    numbers = []
    for token in _TOKEN.findall(text):
        if token.lower() in allowed:
            continue
        numbers.extend(_NUMBER.findall(token))
    return numbers


def full_text(draft: ReportDraft) -> str:
    return "\n".join(text for _, text in _texts(draft))


def _texts(draft: ReportDraft) -> list[tuple[str, str]]:
    return (
        [("summary", draft.summary)]
        + [(f"findings[{i}]", t) for i, t in enumerate(draft.findings)]
        + [(f"caveats[{i}]", t) for i, t in enumerate(draft.caveats)]
        + [(f"recommendations[{i}]", t) for i, t in enumerate(draft.recommendations)]
    )


def placeholder_catalogue(values: Mapping[str, str]) -> str:
    """The catalogue shown to the LLM: names and meanings only, never values."""
    return json.dumps({f"{{{{{name}}}}}": PLACEHOLDER_MEANINGS[name] for name in values}, indent=1)
