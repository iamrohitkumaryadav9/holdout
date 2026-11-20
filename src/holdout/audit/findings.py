"""Data findings (v1.1): problems found on the raw data before the planner chooses a plan.

Post-benchmark v1.1, not part of the pre-registered evaluation. The error analysis
(docs/error_analysis.md, section 1) found that Holdout's LLM planner silently avoided 16 of
20 problems its own audit layer would have reported: it dropped an identifier or a leak, or
chose a different metric, so the audit of the *plan* never saw the problem at the severity
that matters. A data finding is recorded from an audit of a probe plan (every usable column,
the default model) on the raw development rows, before the planner, and it persists for the
whole run whatever the plan later drops:

* the planner must answer every finding explicitly (``ModelPlan.findings_handled``);
* code decides whether each one was resolved (:func:`resolution`), and the report lists
  them all ("Problems found and how they were handled");
* a split-level finding (an entity or time leak) is never cleared by dropping a column
  (:func:`persisting_split_flags`, fix 2).
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from holdout.audit.types import (
    AuditFlag,
    AuditReport,
    ChangeMetric,
    CheckId,
    Deduplicate,
    DropColumns,
    Severity,
)
from holdout.core.schemas import EntityScope, SplitStrategy
from holdout.data.split import SplitSpec
from holdout.plan.schema import FindingAction, FindingHandling, ModelPlan

#: The structural checks whose raw-data flags become findings.
FINDING_CHECKS = frozenset(
    {CheckId.A1, CheckId.A2, CheckId.A3, CheckId.A4, CheckId.A5, CheckId.A6, CheckId.A8}
)
FindingLevel = Literal["column", "split", "rows", "goal"]


class DataFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    level: FindingLevel
    source: Literal["raw_data", "goal"]
    flag: AuditFlag


class FindingStatus(BaseModel):
    """How one finding ended up, decided by code from the final run state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str
    check: CheckId
    severity: Severity
    columns: list[str]
    problem: str
    planner_action: FindingAction | None
    planner_reason: str | None
    resolved: bool
    outcome: str


def _level(flag: AuditFlag) -> FindingLevel:
    if flag.check in (CheckId.A5, CheckId.A6):
        return "split"
    if flag.check is CheckId.A4:
        return "rows"
    if flag.check is CheckId.A11:
        return "goal"
    return "column"


def collect_findings(
    report: AuditReport, goal_flags: Sequence[AuditFlag] = ()
) -> list[DataFinding]:
    """Warning-or-critical flags of the structural checks on the raw data, plus goal-level
    flags, numbered F1, F2, ... in a stable order. An entity key is a finding at any
    severity (A5 info: the split already groups by it, or entities were declared known), so
    the report always says how it was handled."""
    raw = [
        f
        for f in report.flags
        if f.check in FINDING_CHECKS
        and (f.severity.at_least(Severity.WARNING) or f.check is CheckId.A5)
    ]
    out: list[DataFinding] = []
    for flag in raw:
        out.append(
            DataFinding(id=f"F{len(out) + 1}", level=_level(flag), source="raw_data", flag=flag)
        )
    for flag in goal_flags:
        out.append(DataFinding(id=f"F{len(out) + 1}", level="goal", source="goal", flag=flag))
    return out


# ----------------------------------------------------------------------------- handling


def _resolves(plan: ModelPlan, finding: DataFinding) -> FindingAction | None:
    """The action the plan itself already takes on ``finding``, if any."""
    fix = finding.flag.fix
    if isinstance(fix, DropColumns) and fix.columns and set(fix.columns) <= plan.dropped:
        return "drop_columns"
    if isinstance(fix, ChangeMetric) and plan.metric is fix.metric:
        return "change_metric"
    if isinstance(fix, Deduplicate) and fix.scope == "dev" and plan.deduplicate:
        return "deduplicate"  # a holdout overlap (before_split) is not resolved by the plan
    return None


def handling_problems(
    plan: ModelPlan, findings: Sequence[DataFinding], *, offline: bool = False
) -> list[str]:
    """Why ``plan.findings_handled`` does not answer ``findings`` (empty when it does).

    Every finding needs exactly one entry; an action must be one the plan can actually
    take for that finding (dropping its columns, a split change only for a split-level
    finding, declining a causal claim only for a goal finding). ``deferred_to_audit`` is
    the offline planner's answer only: a model must say what it does.
    """
    if not findings:
        return []
    ids = {f.id: f for f in findings}
    problems: list[str] = []
    seen: dict[str, int] = {}
    for entry in plan.findings_handled:
        seen[entry.finding_id] = seen.get(entry.finding_id, 0) + 1
        finding = ids.get(entry.finding_id)
        if finding is None:
            problems.append(f"findings_handled names {entry.finding_id!r}, which is not a finding")
            continue
        columns = set(finding.flag.columns)
        if entry.action == "drop_columns" and not (columns and columns <= plan.dropped):
            problems.append(
                f"{entry.finding_id}: action drop_columns, but the plan does not drop "
                f"{sorted(columns - plan.dropped)}"
            )
        if entry.action == "change_split" and finding.level != "split":
            problems.append(f"{entry.finding_id}: change_split applies to split-level findings")
        if entry.action == "decline_causal_claim" and finding.level != "goal":
            problems.append(f"{entry.finding_id}: decline_causal_claim applies to the goal")
        if entry.action == "change_metric" and not isinstance(finding.flag.fix, ChangeMetric):
            problems.append(f"{entry.finding_id}: change_metric does not address this finding")
        if entry.action == "deferred_to_audit" and not offline:
            problems.append(
                f"{entry.finding_id}: deferred_to_audit is reserved for the offline planner; "
                "choose an action and give a reason"
            )
        if entry.action == "deduplicate" and not plan.deduplicate:
            problems.append(
                f"{entry.finding_id}: action deduplicate, but plan.deduplicate is false"
            )
    for fid in ids:
        if seen.get(fid, 0) == 0:
            problems.append(f"findings_handled has no entry for {fid}; answer every finding")
        elif seen[fid] > 1:
            problems.append(f"findings_handled lists {fid} more than once")
    return problems


#: What code does about each kind of finding when the offline planner defers it.
_DEFERRED: dict[FindingLevel, str] = {
    "column": "the offline planner does not act on findings; code applies the audit's fixes",
    "rows": "the offline planner does not act on findings; code applies the audit's fixes",
    "split": (
        "the offline planner does not act on findings; code ends a leaking run and starts "
        "one with the corrected split"
    ),
    "goal": "the offline planner does not act on findings; code adds the decline to the report",
}


def offline_handling(plan: ModelPlan, findings: Sequence[DataFinding]) -> list[FindingHandling]:
    """The offline planner's answers: it cannot reason about the data, so it defers every
    finding to code; the outcome column then says what code did."""
    return [
        FindingHandling(
            finding_id=f.id,
            action="deferred_to_audit",
            reason=_HANDLED_BY_SPLIT
            if f.level == "split" and f.flag.severity is Severity.INFO
            else _DEFERRED[f.level],
        )
        for f in findings
    ]


_HANDLED_BY_SPLIT = (
    "the offline planner does not act on findings; the split already keeps each entity on one side"
)


def with_handling(plan: ModelPlan, handled: Sequence[FindingHandling]) -> ModelPlan:
    return ModelPlan.model_validate(
        {**plan.model_dump(mode="json"), "findings_handled": [h.model_dump() for h in handled]}
    )


# ----------------------------------------------------------------------------- no masking


def split_resolved(finding: DataFinding, spec: SplitSpec) -> bool:
    """A split-level finding is resolved only by the split itself (or, for entities, by a
    declaration that predictions are for known entities) - never by dropping a column."""
    if finding.flag.check is CheckId.A5:
        return spec.strategy is SplitStrategy.GROUP or spec.entity_scope is EntityScope.KNOWN
    if finding.flag.check is CheckId.A6:
        return spec.strategy is SplitStrategy.TIME
    return True


def persisting_split_flags(
    findings: Sequence[DataFinding], report: AuditReport, spec: SplitSpec
) -> list[AuditFlag]:
    """Critical split-level findings not resolved by the split and not already raised at
    critical by this audit: they are re-asserted (fix 2), so dropping the entity key with a
    row-level split kept cannot hide them, and the holdout-integrity rule still applies."""
    raised = {
        (f.check, tuple(f.columns))
        for f in report.flags
        if f.severity is Severity.CRITICAL and f.check in (CheckId.A5, CheckId.A6)
    }
    out: list[AuditFlag] = []
    for finding in findings:
        flag = finding.flag
        if finding.level != "split" or flag.severity is not Severity.CRITICAL:
            continue
        if split_resolved(finding, spec) or (flag.check, tuple(flag.columns)) in raised:
            continue
        out.append(
            flag.model_copy(
                update={
                    "message": (
                        f"Data finding {finding.id} persists: {flag.message} Dropping the "
                        "column does not remove it - rows of the same entity or period are "
                        "still on both sides of a row-level split."
                    ),
                    "invalidates_holdout": True,
                }
            )
        )
    return out


# ----------------------------------------------------------------------------- outcome


def _resolved_text(taken: FindingAction, finding: DataFinding, plan: ModelPlan) -> str:
    """What the final plan did about a column or rows finding, naming it."""
    fix = finding.flag.fix
    if taken == "drop_columns" and isinstance(fix, DropColumns):
        return "resolved: " + ", ".join(repr(c) for c in fix.columns) + " dropped"
    if taken == "change_metric":
        return f"resolved: metric changed to {plan.metric.value}"
    if taken == "deduplicate":
        return "resolved: duplicate development rows removed"
    return f"resolved: {taken.replace('_', ' ')}"


def split_outcome(spec: SplitSpec, plan: ModelPlan) -> str:
    """How the split resolves an entity or time finding, in words."""
    if spec.strategy is SplitStrategy.GROUP and spec.group_col:
        text = f"resolved: group split on {spec.group_col!r}"
        if spec.group_col in plan.dropped:
            text += "; the key is not a feature (predictions are for new entities)"
        return text
    if spec.strategy is SplitStrategy.TIME and spec.time_col:
        return f"resolved: time split on {spec.time_col!r}"
    if spec.entity_scope is EntityScope.KNOWN:
        return "resolved: entities declared known, so a row-level split is legitimate"
    return f"resolved: the split is {spec.strategy.value}"


def resolution(
    findings: Sequence[DataFinding], plan: ModelPlan, spec: SplitSpec, final: AuditReport
) -> list[FindingStatus]:
    """Each finding's outcome, decided by code from the final plan, split and audit."""
    answers = {h.finding_id: h for h in plan.findings_handled}
    out: list[FindingStatus] = []
    for finding in findings:
        flag = finding.flag
        answer = answers.get(finding.id)
        if finding.level == "split":
            resolved = split_resolved(finding, spec)
            outcome = (
                split_outcome(spec, plan)
                if resolved
                else "open: the split still leaks; the holdout is invalid and a new run with "
                "the corrected split is needed"
            )
        elif finding.level == "goal":
            resolved = True
            outcome = "handled: the report declines a causal conclusion"
        else:
            taken = _resolves(plan, finding)
            still = [
                f for f in final.flags
                if f.check is flag.check and f.columns == flag.columns
                and f.severity.at_least(Severity.WARNING)
            ]  # fmt: skip
            if taken is not None:
                resolved, outcome = True, _resolved_text(taken, finding, plan)
            elif still:
                resolved = False
                outcome = f"open: the final audit still raises it ({still[0].severity.value})"
            else:
                resolved = True
                outcome = "not raised by the final audit (the plan as fitted no longer shows it)"
        out.append(
            FindingStatus(
                finding_id=finding.id,
                check=flag.check,
                severity=flag.severity,
                columns=list(flag.columns),
                problem=flag.message,
                planner_action=answer.action if answer is not None else None,
                planner_reason=answer.reason if answer is not None else None,
                resolved=resolved,
                outcome=outcome,
            )
        )
    return out
