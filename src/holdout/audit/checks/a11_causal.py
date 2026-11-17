"""A11 causal claims: a predictive model cannot say that X causes Y.

``check_causal_claims`` scans report text for causal phrasing ("causes", "leads to",
"drives", "results in", "the effect of X on Y", "because of") with simple negation
handling ("does not cause" is not a claim) and an exemption for verbs whose object is one of
the run's own statistics ("resulting in an optimism gap"), and takes structured claims
directly. In
Week 2 it is tested on strings; Week 3 wires it to the generated report. Until then the
audit runner records A11 as skipped (not a failure).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from holdout.audit.types import (
    AuditFlag,
    Category,
    CausalEvidence,
    CheckId,
    RemoveCausalClaim,
    Severity,
)

CAUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("causes", re.compile(r"\bcaus(?:e|es|ed|ing)\b")),
    ("leads to", re.compile(r"\blead(?:s|ing)? to\b|\bled to\b")),
    ("drives", re.compile(r"\bdriv(?:e|es|ing|en)\b|\bdrove\b")),
    ("results in", re.compile(r"\bresult(?:s|ed|ing)? in\b")),
    ("effect of X on Y", re.compile(r"\beffects? of\b[^.]*?\bon\b")),
    ("because of", re.compile(r"\bbecause of\b")),
)

_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "cannot",
        "can't",
        "doesn't",
        "don't",
        "didn't",
        "isn't",
        "won't",
        "without",
    }
)
_NEGATION_WINDOW = 4
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z']+")

#: Patterns whose object can be one of the run's own statistics.
_VERB_PATTERNS = frozenset({"causes", "leads to", "drives", "results in"})
#: Nouns naming the run's evaluation statistics. "X resulting in an optimism gap of ..." or
#: "dropping the column led to a lower validation score" describe arithmetic or the modelling
#: process, not cause and effect in the data, so they are not A11 claims. Deliberately
#: narrow: outcome words of the domain ("churn", "conversion rate", "sales") are not here.
_EVALUATION_OBJECTS = frozenset(
    {
        "score",
        "scores",
        "gap",
        "auc",
        "roc",
        "f1",
        "accuracy",
        "rmse",
        "mae",
        "r2",
        "metric",
        "baseline",
        "fold",
        "folds",
    }
)
_OBJECT_WINDOW = 5
#: The object of the verb ends at the clause: "leads to higher conversion, improving the
#: score" is still a claim about conversion.
_CLAUSE_END = re.compile(r"[,;:.!?()]|\b(?:and|but|while|which|so|whereas)\b")


@dataclass(frozen=True)
class StructuredCausalClaim:
    cause: str
    effect: str
    statement: str


def check_causal_claims(
    report_text: str, structured_claims: Sequence[StructuredCausalClaim] = ()
) -> list[AuditFlag]:
    """Flags for every causal statement in ``report_text`` and every structured claim."""
    flags: list[AuditFlag] = []
    statements: list[str] = []
    phrases: set[str] = set()
    for sentence in _SENTENCE.split(report_text.strip()):
        found = _causal_phrases(sentence)
        if found:
            statements.append(sentence.strip())
            phrases.update(found)
    if statements:
        flags.append(_flag("text", statements, sorted(phrases)))
    if structured_claims:
        flags.append(
            _flag(
                "structured",
                [c.statement for c in structured_claims],
                [f"{c.cause} -> {c.effect}" for c in structured_claims],
            )
        )
    return flags


def _causal_phrases(sentence: str) -> list[str]:
    lowered = sentence.lower()
    found: list[str] = []
    for name, pattern in CAUSAL_PATTERNS:
        for match in pattern.finditer(lowered):
            preceding = _WORD.findall(lowered[: match.start()])[-_NEGATION_WINDOW:]
            if _NEGATIONS.intersection(preceding):
                continue
            obj = _CLAUSE_END.split(lowered[match.end() :], maxsplit=1)[0]
            following = _WORD.findall(obj)[:_OBJECT_WINDOW]
            if name in _VERB_PATTERNS and _EVALUATION_OBJECTS.intersection(following):
                continue  # about the run's own statistics, not the data
            found.append(name)
            break
    return found


def _flag(
    source: Literal["text", "structured"], statements: list[str], phrases: list[str]
) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A11,
        category=Category.CAUSAL_CLAIM,
        severity=Severity.WARNING,
        columns=[],
        evidence=CausalEvidence(source=source, statements=statements, phrases=phrases),
        message=(
            "The report makes causal claims; a predictive model on observational data shows "
            "association, not cause and effect."
        ),
        fix=RemoveCausalClaim(statements=statements),
    )
