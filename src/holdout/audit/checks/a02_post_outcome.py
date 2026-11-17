"""A2 post-outcome column hints: names and values that suggest "recorded afterwards".

The provider (offline lexicon now, an LLM from Week 3) suggests columns. A hint alone is
never more than a warning, and is only a warning when the column also carries signal in
A1; a suggestive name on a column with no signal stays info. When A2 and A1 agree on a
column the plan uses, A1 is upgraded to critical by the audit runner.
"""

from holdout.audit.checks import CheckOutcome
from holdout.audit.checks.a01_single_feature import SingleFeatureResult
from holdout.audit.context import AuditContext
from holdout.audit.hints import ColumnSample, HintRequest
from holdout.audit.types import AuditFlag, Category, CheckId, DropColumns, HintEvidence, Severity

#: Distinct values per column shown to the provider (most frequent first).
_SAMPLE_VALUES = 20


def run(ctx: AuditContext, single: SingleFeatureResult | None) -> tuple[CheckOutcome, set[str]]:
    """Return the outcome and the set of hinted columns (for A1 corroboration)."""
    request = HintRequest(
        columns=[ColumnSample(name=c, values=_top_values(ctx, c)) for c in ctx.features],
        target=ctx.target,
        goal_text=ctx.goal_text,
    )
    hints = list(ctx.hint_provider.suggest(request))
    t = ctx.thresholds.a02_hints
    flags: list[AuditFlag] = []
    for hint in sorted(hints, key=lambda h: h.column):
        score = single.scores.get(hint.column) if single is not None else None
        used = hint.column in ctx.used_features
        has_signal = score is not None and (
            (
                score.relative_lift is not None
                and score.relative_lift >= t.min_relative_lift_for_warning
            )
            or score.normalized_score >= t.min_normalized_score_for_warning
        )
        severity = Severity.WARNING if (has_signal and used) else Severity.INFO
        reason_text = "; ".join(hint.reasons)
        if severity is Severity.WARNING:
            message = (
                f"{hint.column!r} may be recorded after the outcome ({reason_text}) and it "
                "carries signal; check when its value becomes known."
            )
        else:
            message = f"{hint.column!r} has a suggestive name or values ({reason_text}), " + (
                "but carries little signal." if used else "but the plan drops it."
            )
        flags.append(
            AuditFlag(
                check=CheckId.A2,
                category=Category.POST_OUTCOME_COLUMN,
                severity=severity,
                columns=[hint.column],
                evidence=HintEvidence(
                    provider=ctx.hint_provider.name,
                    reasons=hint.reasons,
                    matched_terms=hint.terms,
                    single_feature_relative_lift=None if score is None else score.relative_lift,
                    used_by_plan=used,
                ),
                message=message,
                fix=DropColumns(
                    columns=[hint.column],
                    reason="likely recorded after the outcome; not available at prediction time",
                ),
            )
        )
    detail = f"provider {ctx.hint_provider.name}: {len(hints)} hinted column(s)"
    return CheckOutcome(flags=flags, detail=detail), {h.column for h in hints}


def _top_values(ctx: AuditContext, column: str) -> list[str]:
    counts = ctx.dev[column].dropna().astype(str).value_counts()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [str(value) for value, _ in ranked[:_SAMPLE_VALUES]]
