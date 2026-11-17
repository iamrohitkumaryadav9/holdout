"""The planner seam: who proposes plans, explanations, report prose, hints and EDA.

``LLMPlanner`` asks the model through :class:`LLMSession` (validated output, one repair,
typed failure). ``OfflinePlanner`` makes no API call: the default plan, flag messages as
explanations, template prose, lexicon hints. Both feed the same graph, so offline runs are
comparable with agent runs.

Deterministic code stays the source of truth: fixes are applied by :func:`apply_fixes`;
the LLM's integrated plan is accepted only if it contains every fix and validates. When it
does not, the code-applied plan is used and a :class:`Substitution` is recorded, because a
remediation achieved by the code fallback must not be credited to the LLM.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from holdout.agent.llm_inputs import profile_summary, sample_rows
from holdout.agent.prompts import load_prompt
from holdout.agent.report import (
    allowed_tokens,
    draft_problems,
    fill_draft,
    full_text,
    placeholder_catalogue,
    prose_problems,
)
from holdout.agent.schemas import (
    EDAProposal,
    EDASnippet,
    ExplainFixResponse,
    HintResponse,
    ReportDraft,
    Substitution,
)
from holdout.audit.checks.a11_causal import StructuredCausalClaim, check_causal_claims
from holdout.audit.hints import ColumnHint, HintProvider, HintRequest, LexiconHintProvider
from holdout.audit.types import AuditFlag, CausalEvidence, ChangeMetric, DropColumns, Severity
from holdout.core.errors import AgentError, LLMOutputError
from holdout.core.schemas import Task
from holdout.data.profile import DatasetProfile
from holdout.llm.structured import LLMSession
from holdout.plan.defaults import default_plan
from holdout.plan.schema import DropColumn, ModelPlan, validate_plan
from holdout.plan.schema import Severity as IssueSeverity

# ----------------------------------------------------------------------------- fixes


def plan_fixes(flags: Sequence[AuditFlag]) -> list[AuditFlag]:
    """Critical flags whose typed fix changes the plan (drop columns, change metric)."""
    return [
        f
        for f in flags
        if f.severity is Severity.CRITICAL
        and not f.invalidates_holdout
        and isinstance(f.fix, DropColumns | ChangeMetric)
    ]


def fix_labels(flags: Sequence[AuditFlag]) -> list[str]:
    labels: list[str] = []
    for flag in flags:
        if isinstance(flag.fix, DropColumns):
            labels.extend(f"drop:{c}" for c in flag.fix.columns)
        elif isinstance(flag.fix, ChangeMetric):
            labels.append(f"metric:{flag.fix.metric.value}")
    return sorted(set(labels))


def apply_fixes(plan: ModelPlan, flags: Sequence[AuditFlag]) -> ModelPlan:
    """The deterministic application of every plan fix (the source of truth)."""
    drops = list(plan.drop_columns)
    dropped = {d.column for d in drops}
    metric = plan.metric
    rationale = plan.rationale.model_dump()
    for flag in flags:
        if isinstance(flag.fix, DropColumns):
            for column in flag.fix.columns:
                if column not in dropped:
                    drops.append(
                        DropColumn(
                            column=column,
                            reason=f"audit {flag.check.value}: {flag.fix.reason}"[:200],
                        )
                    )
                    dropped.add(column)
            rationale["drop_columns"] = "includes columns the audit found leaking"
        elif isinstance(flag.fix, ChangeMetric):
            metric = flag.fix.metric
            rationale["metric"] = f"{metric.value}: chosen by audit {flag.check.value}"
    return ModelPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "drop_columns": [d.model_dump() for d in drops],
            "metric": metric.value,
            "rationale": rationale,
        }
    )


def missed_fixes(revised: ModelPlan, flags: Sequence[AuditFlag]) -> list[str]:
    missed = []
    for label in fix_labels(flags):
        kind, value = label.split(":", 1)
        if kind == "drop" and value not in revised.dropped:
            missed.append(label)
        if kind == "metric" and revised.metric.value != value:
            missed.append(label)
    return missed


def plan_errors(plan: ModelPlan, profile: DatasetProfile, task: Task) -> list[str]:
    errors = [
        f"[{i.code}] {i.message}"
        for i in validate_plan(plan, profile)
        if i.severity is IssueSeverity.ERROR
    ]
    if plan.task is not task:
        errors.append(f"[task] the plan's task is {plan.task.value}; this run is {task.value}")
    return errors


# ----------------------------------------------------------------------------- contexts


@dataclass(frozen=True)
class PlanContext:
    goal: str
    target: str
    task: Task
    profile: DatasetProfile
    dev: pd.DataFrame
    seed: int
    history: list[dict[str, Any]]
    eda: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FixOutcome:
    plan: ModelPlan
    explanations: dict[str, str]
    substitution: Substitution | None


@dataclass(frozen=True)
class ReportContext:
    goal: str
    facts: dict[str, Any]
    values: dict[str, str]
    flags: list[AuditFlag]
    explanations: dict[str, str]
    caveats: list[str]
    columns: list[str]
    plan: ModelPlan
    round: int


@dataclass(frozen=True)
class ReportOutcome:
    draft: ReportDraft
    causal_flags: list[AuditFlag]
    substitution: Substitution | None


def flag_key(flag: AuditFlag) -> str:
    return f"{flag.check.value}:{','.join(flag.columns)}"


def _open_flags(flags: Sequence[AuditFlag]) -> list[AuditFlag]:
    return [f for f in flags if f.severity.at_least(Severity.WARNING)]


class Planner(Protocol):
    @property
    def offline(self) -> bool: ...

    def propose_plan(self, ctx: PlanContext) -> ModelPlan: ...

    def explain_and_fix(
        self, plan: ModelPlan, flags: list[AuditFlag], profile: DatasetProfile, round_: int
    ) -> FixOutcome: ...

    def draft_report(self, ctx: ReportContext) -> ReportOutcome: ...

    def hint_provider(self, goal: str) -> HintProvider: ...

    def propose_eda(self, profile: DatasetProfile, goal: str) -> list[EDASnippet]: ...


# ----------------------------------------------------------------------------- offline


class OfflinePlanner:
    """No API calls: the default plan, code explanations, template prose, lexicon hints."""

    @property
    def offline(self) -> bool:
        return True

    def propose_plan(self, ctx: PlanContext) -> ModelPlan:
        return default_plan(ctx.profile, ctx.task)

    def explain_and_fix(
        self, plan: ModelPlan, flags: list[AuditFlag], profile: DatasetProfile, round_: int
    ) -> FixOutcome:
        return FixOutcome(
            plan=apply_fixes(plan, plan_fixes(flags)),
            explanations={flag_key(f): f.message for f in _open_flags(flags)},
            substitution=None,
        )

    def draft_report(self, ctx: ReportContext) -> ReportOutcome:
        holdout = (
            "The sealed holdout scored {{holdout_score}}."
            if "holdout_score" in ctx.values
            else "The sealed holdout was not evaluated."
        )
        draft = ReportDraft(
            summary=(
                "The final plan scores {{validation_score}} in cross-validation against "
                "{{dummy_score}} for the dummy baseline. " + holdout
            ),
            findings=[
                f"{f.check.value} ({f.severity.value}): {f.message}" for f in _open_flags(ctx.flags)
            ],
            caveats=list(ctx.caveats),
            recommendations=["Review the audit table before relying on the score."],
            causal_claim=False,
        )
        return ReportOutcome(
            draft=fill_draft(draft, ctx.values), causal_flags=[], substitution=None
        )

    def hint_provider(self, goal: str) -> HintProvider:
        return LexiconHintProvider()

    def propose_eda(self, profile: DatasetProfile, goal: str) -> list[EDASnippet]:
        return []


# ----------------------------------------------------------------------------- LLM


@dataclass
class LLMPlanner:
    session: LLMSession

    @property
    def offline(self) -> bool:
        return False

    def propose_plan(self, ctx: PlanContext) -> ModelPlan:
        """Ask for a plan; validation errors go back once, then the run fails."""
        prompt = load_prompt("plan")
        feedback: list[str] = []
        for attempt in (1, 2):
            rendered = prompt.render(
                goal=ctx.goal,
                target={"target": ctx.target, "task": ctx.task.value},
                profile=profile_summary(ctx.profile),
                sample_rows=sample_rows(ctx.dev, ctx.seed),
                history=ctx.history,
                eda=ctx.eda,
                feedback=feedback or "none",
            )
            plan = self.session.call(node="plan", prompt=rendered, schema=ModelPlan)
            feedback = plan_errors(plan, ctx.profile, ctx.task)
            if not feedback:
                return plan
            if attempt == 2:
                raise AgentError(
                    "the proposed plan was rejected twice by validation: " + "; ".join(feedback)
                )
        raise AssertionError("unreachable")

    def explain_and_fix(
        self, plan: ModelPlan, flags: list[AuditFlag], profile: DatasetProfile, round_: int
    ) -> FixOutcome:
        open_flags = _open_flags(flags)
        fixes = plan_fixes(flags)
        allowed = allowed_tokens([c.name for c in profile.columns], plan)
        task = plan.task
        last_missed: list[str] = fix_labels(fixes)

        def check(answer: ExplainFixResponse) -> list[str]:
            nonlocal last_missed
            problems = []
            explained = {f"{e.check.value}:{','.join(e.columns)}" for e in answer.explanations}
            for flag in open_flags:
                if flag_key(flag) not in explained:
                    problems.append(f"explain flag {flag_key(flag)}")
            for e in answer.explanations:
                problems.extend(
                    f"explanation {e.check.value}: {p}"
                    for p in prose_problems(e.explanation, allowed, set())
                )
            last_missed = missed_fixes(answer.revised_plan, fixes)
            problems.extend(f"apply fix {m}" for m in last_missed)
            problems.extend(plan_errors(answer.revised_plan, profile, task))
            return problems

        rendered = load_prompt("explain_fix").render(
            plan=plan.model_dump(mode="json"),
            flags=[_flag_brief(f) for f in open_flags],
            fixes=fix_labels(fixes),
            profile=profile_summary(profile),
        )
        try:
            answer = self.session.call(
                node="explain_and_fix", prompt=rendered, schema=ExplainFixResponse, check=check
            )
        except LLMOutputError:
            return FixOutcome(
                plan=apply_fixes(plan, fixes),
                explanations={flag_key(f): f.message for f in open_flags},
                substitution=Substitution(
                    node="explain_and_fix", round=round_, missed_fixes=sorted(last_missed)
                ),
            )
        return FixOutcome(
            plan=answer.revised_plan,
            explanations={
                f"{e.check.value}:{','.join(e.columns)}": e.explanation for e in answer.explanations
            },
            substitution=None,
        )

    def draft_report(self, ctx: ReportContext) -> ReportOutcome:
        allowed = allowed_tokens(ctx.columns, ctx.plan)
        placeholders = set(ctx.values)
        required = sorted({f.check.value for f in _open_flags(ctx.flags)})

        def check(draft: ReportDraft) -> list[str]:
            return draft_problems(
                draft, allowed=allowed, placeholders=placeholders, required_checks=required
            )

        def render(feedback: list[str]) -> Any:
            return load_prompt("report").render(
                goal=ctx.goal,
                facts=ctx.facts,
                placeholders=placeholder_catalogue(ctx.values),
                flags=[
                    {**_flag_brief(f), "explanation": ctx.explanations.get(flag_key(f), f.message)}
                    for f in _open_flags(ctx.flags)
                ],
                caveats=ctx.caveats,
                feedback=feedback or "none",
            )

        draft = self.session.call(node="report", prompt=render([]), schema=ReportDraft, check=check)
        filled = fill_draft(draft, ctx.values)
        causal = _causal_flags(filled)
        if not causal:
            return ReportOutcome(draft=filled, causal_flags=[], substitution=None)
        feedback = [
            "remove every causal claim: a predictive model shows association, not cause and "
            "effect; set causal_claim to false"
        ] + [
            f"causal statement: {s}"
            for f in causal
            if isinstance(f.evidence, CausalEvidence)
            for s in f.evidence.statements
        ]

        def check_again(answer: ReportDraft) -> list[str]:
            return check(answer) + (
                ["remove the causal claims"]
                if _causal_flags(fill_draft(answer, ctx.values))
                else []
            )

        try:
            revised = self.session.call(
                node="report", prompt=render(feedback), schema=ReportDraft, check=check_again
            )
            return ReportOutcome(
                draft=fill_draft(revised, ctx.values), causal_flags=causal, substitution=None
            )
        except LLMOutputError:
            return ReportOutcome(
                draft=_strip_causal(filled),
                causal_flags=causal,
                substitution=Substitution(
                    node="report", round=ctx.round, missed_fixes=["remove_causal_claim"]
                ),
            )

    def hint_provider(self, goal: str) -> HintProvider:
        return MergedHintProvider([LexiconHintProvider(), GeminiHintProvider(self.session)])

    def propose_eda(self, profile: DatasetProfile, goal: str) -> list[EDASnippet]:
        rendered = load_prompt("eda").render(
            goal=goal, target=profile.target, profile=profile_summary(profile)
        )
        return self.session.call(node="eda", prompt=rendered, schema=EDAProposal).snippets


def _flag_brief(flag: AuditFlag) -> dict[str, Any]:
    return {
        "check": flag.check.value,
        "severity": flag.severity.value,
        "columns": flag.columns,
        "category": flag.category.value,
        "message": flag.message,
    }


def _causal_flags(draft: ReportDraft) -> list[AuditFlag]:
    structured = (
        [StructuredCausalClaim(cause="(model)", effect="(target)", statement="causal_claim=true")]
        if draft.causal_claim
        else []
    )
    return check_causal_claims(full_text(draft), structured)


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _strip_causal(draft: ReportDraft) -> ReportDraft:
    """Deterministic fallback: drop causal sentences and state the limitation."""

    def clean(text: str) -> str:
        kept = [s for s in _SENTENCE.split(text) if not check_causal_claims(s)]
        return " ".join(kept)

    return draft.model_copy(
        update={
            "summary": clean(draft.summary) or "The run's results are summarised below.",
            "findings": [t for t in (clean(f) for f in draft.findings) if t],
            "caveats": [
                *[t for t in (clean(c) for c in draft.caveats) if t],
                "This model shows association, not cause and effect.",
            ],
            "recommendations": [t for t in (clean(r) for r in draft.recommendations) if t],
            "causal_claim": False,
        }
    )


# ----------------------------------------------------------------------------- hints


@dataclass
class GeminiHintProvider:
    """A2 hints from the LLM: column names plus up to 5 sample values, and the goal."""

    session: LLMSession

    @property
    def name(self) -> str:
        return "gemini"

    def suggest(self, request: HintRequest) -> list[ColumnHint]:
        rendered = load_prompt("hints").render(
            target=request.target,
            goal=request.goal_text,
            columns=[
                {"column": c.name, "samples": c.values[:5]}
                for c in request.columns
                if c.name != request.target
            ],
        )
        answer = self.session.call(node="audit", prompt=rendered, schema=HintResponse)
        known = {c.name for c in request.columns}
        return [
            ColumnHint(column=s.column, reasons=[f"LLM: {s.reason}"], terms=[])
            for s in answer.suspects
            if s.column in known and s.column != request.target
        ]


@dataclass
class MergedHintProvider:
    """Union of several providers' hints per column (A2's severity rules are unchanged)."""

    providers: list[HintProvider]

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.providers)

    def suggest(self, request: HintRequest) -> list[ColumnHint]:
        merged: dict[str, ColumnHint] = {}
        for provider in self.providers:
            for hint in provider.suggest(request):
                previous = merged.get(hint.column)
                merged[hint.column] = (
                    hint
                    if previous is None
                    else ColumnHint(
                        column=hint.column,
                        reasons=previous.reasons + hint.reasons,
                        terms=sorted(set(previous.terms) | set(hint.terms)),
                    )
                )
        return [merged[c] for c in sorted(merged)]
