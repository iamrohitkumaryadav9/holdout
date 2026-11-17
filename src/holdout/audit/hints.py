"""Post-outcome column hints: a provider interface and a deterministic offline provider.

A post-outcome column records something known only after the target was decided (a
refund after churn, a settlement after default). Names and values often give it away.
``HintProvider`` is the seam where an LLM provider plugs in (Week 3); the shipped
``LexiconHintProvider`` matches outcome-timing words and tokens shared with the target.

Hints are suggestions, not verdicts: the A2 check raises one above info only when the
column also carries signal (A1), and it upgrades A1 to critical when both agree.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

#: Words that suggest the value was recorded after the outcome.
OUTCOME_TIMING_TERMS: frozenset[str] = frozenset(
    {
        "after", "approved", "approval", "cancel", "cancelled", "canceled", "cancellation",
        "chargeback", "churn", "churned", "closed", "collected", "deceased", "default",
        "defaulted", "died", "discharge", "discharged", "final", "outcome", "paid", "payout",
        "post", "recovered", "refund", "refunded", "resolution", "resolved", "result",
        "returned", "settled", "settlement", "terminated", "termination", "writeoff",
    }
)  # fmt: skip

#: Outcome-timing words that, as *values* of a categorical column, suggest a status
#: recorded after the outcome (e.g. status = "refunded").
OUTCOME_VALUE_TERMS: frozenset[str] = frozenset(
    {
        "approved", "cancelled", "canceled", "chargeback", "churned", "closed", "collected",
        "defaulted", "discharged", "paid", "recovered", "refunded", "rejected", "resolved",
        "returned", "settled", "terminated", "won", "writeoff",
    }
)  # fmt: skip

#: Tokens too generic to count as "shared with the target name".
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "amount", "an", "and", "at", "by", "count", "customer", "date", "flag", "for",
        "has", "id", "if", "in", "is", "num", "number", "of", "on", "or", "predict", "rate",
        "the", "to", "total", "type", "value", "was", "whether", "will", "with", "y",
    }
)  # fmt: skip

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lower-case word tokens, splitting camelCase, snake_case, kebab-case and spaces."""
    return [t for t in _SPLIT.split(_CAMEL.sub(" ", text).lower()) if t]


@dataclass(frozen=True)
class ColumnSample:
    name: str
    values: list[str]


@dataclass(frozen=True)
class HintRequest:
    columns: list[ColumnSample]
    target: str
    goal_text: str


@dataclass(frozen=True)
class ColumnHint:
    column: str
    reasons: list[str]
    terms: list[str]


class HintProvider(Protocol):
    """Suggests columns that may have been recorded after the outcome."""

    @property
    def name(self) -> str: ...

    def suggest(self, request: HintRequest) -> Sequence[ColumnHint]: ...


class LexiconHintProvider:
    """Deterministic offline hints from outcome-timing words and target-name tokens."""

    @property
    def name(self) -> str:
        return "lexicon"

    def suggest(self, request: HintRequest) -> list[ColumnHint]:
        target_terms = {t for t in tokens(request.target) if t not in STOPWORDS}
        goal = request.goal_text.lower()
        goal_object = goal.split("predict", 1)[1] if "predict" in goal else ""
        target_terms |= {t for t in tokens(goal_object) if t not in STOPWORDS}
        hints: list[ColumnHint] = []
        for column in request.columns:
            if column.name == request.target:
                continue
            name_terms = set(tokens(column.name))
            reasons: list[str] = []
            matched: set[str] = set()
            timing = sorted(name_terms & OUTCOME_TIMING_TERMS)
            if timing:
                reasons.append(f"name contains outcome-timing word(s) {timing}")
                matched.update(timing)
            shared = sorted(name_terms & target_terms)
            if shared:
                reasons.append(f"name shares {shared} with the target {request.target!r}")
                matched.update(shared)
            value_terms = sorted(
                {t for v in column.values for t in tokens(v)} & OUTCOME_VALUE_TERMS
            )
            if value_terms:
                reasons.append(f"values look like post-outcome statuses {value_terms}")
                matched.update(value_terms)
            if reasons:
                hints.append(ColumnHint(column=column.name, reasons=reasons, terms=sorted(matched)))
        return hints
