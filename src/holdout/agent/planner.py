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
from collections.abc import Callable, Sequence
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
    state_identifiers,
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
from holdout.audit.findings import DataFinding, handling_problems
from holdout.audit.hints import ColumnHint, HintProvider, HintRequest, LexiconHintProvider
from holdout.audit.types import (
    AuditFlag,
    CausalEvidence,
    ChangeMetric,
    Deduplicate,
    DropColumns,
    Severity,
)
from holdout.core.errors import AgentError, LLMOutputError
from holdout.core.schemas import Task
from holdout.data.profile import DatasetProfile
from holdout.llm.structured import LLMSession
from holdout.plan.defaults import default_plan
from holdout.plan.schema import MODEL_NAMES, DropColumn, ModelPlan, drop_rationale, validate_plan
from holdout.plan.schema import Severity as IssueSeverity

# ----------------------------------------------------------------------------- fixes


def _is_plan_fix(fix: object) -> bool:
    # v1.1: deduplicating the development rows is a plan action (ModelPlan.deduplicate);
    # deduplicating before the split is not - that needs a new run.
    return isinstance(fix, DropColumns | ChangeMetric) or (
        isinstance(fix, Deduplicate) and fix.scope == "dev"
    )


def plan_fixes(flags: Sequence[AuditFlag]) -> list[AuditFlag]:
    """Critical flags whose typed fix changes the plan (drop columns, change metric,
    deduplicate the development rows)."""
    return [
        f
        for f in flags
        if f.severity is Severity.CRITICAL and not f.invalidates_holdout and _is_plan_fix(f.fix)
    ]


def fix_labels(flags: Sequence[AuditFlag]) -> list[str]:
    labels: list[str] = []
    for flag in flags:
        if isinstance(flag.fix, DropColumns):
            labels.extend(f"drop:{c}" for c in flag.fix.columns)
        elif isinstance(flag.fix, ChangeMetric):
            labels.append(f"metric:{flag.fix.metric.value}")
        elif isinstance(flag.fix, Deduplicate) and flag.fix.scope == "dev":
            labels.append("deduplicate:dev")
    return sorted(set(labels))


def fix_constraints(plan: ModelPlan, flags: Sequence[AuditFlag]) -> list[str]:
    """Every plan fix as an explicit, checkable constraint on the revised plan."""
    constraints = []
    for label in fix_labels(flags):
        kind, value = label.split(":", 1)
        if kind == "drop":
            constraints.append(
                f'revised_plan.drop_columns MUST contain an entry {{"column": "{value}", '
                '"reason": ...}; keep every column the current plan already drops'
            )
        elif kind == "deduplicate":
            constraints.append("revised_plan.deduplicate MUST be true")
        else:
            constraints.append(f'revised_plan.metric MUST be "{value}"')
    constraints.append(
        f'revised_plan.task MUST stay "{plan.task.value}" and revised_plan.model.name MUST be '
        "exactly one of: " + ", ".join(MODEL_NAMES)
    )
    return constraints


def apply_fixes(plan: ModelPlan, flags: Sequence[AuditFlag]) -> ModelPlan:
    """The deterministic application of every plan fix (the source of truth)."""
    drops = list(plan.drop_columns)
    dropped = {d.column for d in drops}
    metric = plan.metric
    deduplicate = plan.deduplicate
    rationale = plan.rationale.model_dump()
    for flag in flags:
        if isinstance(flag.fix, Deduplicate) and flag.fix.scope == "dev":
            deduplicate = True
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
            rationale["drop_columns"] = drop_rationale(drops)  # the final list, named
        elif isinstance(flag.fix, ChangeMetric):
            metric = flag.fix.metric
            rationale["metric"] = f"{metric.value}: chosen by audit {flag.check.value}"
    return ModelPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "drop_columns": [d.model_dump() for d in drops],
            "metric": metric.value,
            "rationale": rationale,
            "deduplicate": deduplicate,
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
        if kind == "deduplicate" and not revised.deduplicate:
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
    #: Plan fields the user stated (e.g. a required metric). Only the offline planner
    #: applies them, since it cannot read the goal text; an LLM planner reads the goal.
    overrides: dict[str, Any] = field(default_factory=dict)
    #: v1.1: problems found on the raw development rows; the plan must answer each one.
    findings: list[DataFinding] = field(default_factory=list)


def findings_summary(findings: Sequence[DataFinding]) -> list[dict[str, Any]]:
    """What the planner is told about each data finding (code-computed, data only)."""
    return [
        {
            "id": f.id,
            "level": f.level,
            "check": f.flag.check.value,
            "severity": f.flag.severity.value,
            "columns": list(f.flag.columns),
            "problem": f.flag.message,
            "suggested_fix": f.flag.fix.kind,
        }
        for f in findings
    ]


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


def explanation_key(check: str, columns: Sequence[str]) -> str:
    """ "A1:refund_issued", or just "A12" for a flag without columns (no trailing colon)."""
    return f"{check}:{','.join(columns)}" if columns else check


def flag_key(flag: AuditFlag) -> str:
    return explanation_key(flag.check.value, flag.columns)


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
        plan = default_plan(ctx.profile, ctx.task)
        if not ctx.overrides:
            return plan
        return ModelPlan.model_validate({**plan.model_dump(mode="json"), **ctx.overrides})

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
    #: Replaces the A11 lexicon on the generated report (the benchmark's LLM-judge
    #: ablation): returns causal-claim flags for a filled draft. None = the lexicon.
    causal_checker: Callable[[ReportDraft], list[AuditFlag]] | None = None

    def _flagged(self, draft: ReportDraft, ctx: "ReportContext") -> list[str] | None:
        """The statements the judge flags in ``draft`` (None: the lexicon decides)."""
        if self.causal_checker is None:
            return None
        return [
            s
            for f in self.causal_checker(fill_draft(draft, ctx.values))
            if isinstance(f.evidence, CausalEvidence)
            for s in f.evidence.statements
        ]

    @property
    def offline(self) -> bool:
        return False

    def propose_plan(self, ctx: PlanContext) -> ModelPlan:
        """Ask for a plan. Schema and data-aware validation are one check, so a plan gets
        exactly one repair for any problem; a second failure stops the run."""
        rendered = load_prompt("plan").render(
            goal=ctx.goal,
            target={"target": ctx.target, "task": ctx.task.value},
            profile=profile_summary(ctx.profile),
            sample_rows=sample_rows(ctx.dev, ctx.seed),
            history=ctx.history,
            eda=ctx.eda,
            findings=findings_summary(ctx.findings),
        )
        try:
            return self.session.call(
                node="plan",
                prompt=rendered,
                schema=ModelPlan,
                check=lambda plan: [
                    *plan_errors(plan, ctx.profile, ctx.task),
                    *handling_problems(plan, ctx.findings),
                ],
            )
        except LLMOutputError as exc:
            if exc.last_answer is None:
                raise  # not even schema-valid twice
            raise AgentError(
                "the proposed plan was rejected twice by validation: " + "; ".join(exc.errors)
            ) from exc

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
            explained = {explanation_key(e.check.value, e.columns) for e in answer.explanations}
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
            # A numbered plain-text list: as JSON the quotes inside would arrive escaped.
            constraints="\n".join(
                f"{i}. {c}" for i, c in enumerate(fix_constraints(plan, fixes), start=1)
            ),
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
                explanation_key(e.check.value, e.columns): e.explanation
                for e in answer.explanations
            },
            substitution=None,
        )

    def draft_report(self, ctx: ReportContext) -> ReportOutcome:
        # Identifiers inside code-written caveats (run ids, column names) are run state.
        allowed = allowed_tokens(ctx.columns, ctx.plan) | state_identifiers(ctx.caveats)
        placeholders = set(ctx.values)
        required = sorted({f.check.value for f in _open_flags(ctx.flags)})

        #: A11 flags of the first draft that had any: recorded even when the repair fixes them.
        caught: list[AuditFlag] = []

        def form_problems(draft: ReportDraft) -> list[str]:
            return draft_problems(
                draft, allowed=allowed, placeholders=placeholders, required_checks=required
            )

        def check(draft: ReportDraft) -> list[str]:
            filled = fill_draft(draft, ctx.values)
            if self.causal_checker is None:
                causal = _causal_flags(filled)
                problems = causal_problems(draft)
            else:
                causal = self.causal_checker(filled)
                problems = [
                    f"a reviewer flagged this as a causal claim: {s!r}. Rephrase it without "
                    "claiming cause and effect"
                    for f in causal
                    if isinstance(f.evidence, CausalEvidence)
                    for s in f.evidence.statements
                ] or (["remove the causal claims"] if causal else [])
            if causal and not caught:
                caught.extend(causal)
            return form_problems(draft) + problems

        rendered = load_prompt("report").render(
            goal=ctx.goal,
            facts=ctx.facts,
            placeholders=placeholder_catalogue(ctx.values),
            flags=[
                {**_flag_brief(f), "explanation": ctx.explanations.get(flag_key(f), f.message)}
                for f in _open_flags(ctx.flags)
            ],
            caveats=ctx.caveats,
        )
        try:
            draft = self.session.call(
                node="report", prompt=rendered, schema=ReportDraft, check=check
            )
            return ReportOutcome(
                draft=fill_draft(draft, ctx.values), causal_flags=caught, substitution=None
            )
        except LLMOutputError as exc:
            last = exc.last_answer
            if not isinstance(last, ReportDraft) or form_problems(last):
                raise  # numbers, schema or missing flags: no safe deterministic fallback
            # Only causal wording remains: code strips it (not credited to the model).
            return ReportOutcome(
                draft=_strip_causal(fill_draft(last, ctx.values), self._flagged(last, ctx)),
                causal_flags=caught,
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


def causal_problems(draft: ReportDraft) -> list[str]:
    """Repair feedback naming each causal sentence and the exact words that triggered A11,
    so the model can rephrase precisely (checked on the draft as written, placeholders and
    all)."""
    problems = []
    if draft.causal_claim:
        problems.append(
            "causal_claim is true: a predictive model shows association, not cause and "
            "effect; set causal_claim to false"
        )
    for flag in check_causal_claims(full_text(draft)):
        if isinstance(flag.evidence, CausalEvidence):
            words = ", ".join(f'"{p}"' for p in flag.evidence.phrases)
            for statement in flag.evidence.statements:
                problems.append(
                    f"causal wording ({words}) in: {statement!r}. Rephrase without causal "
                    "verbs (causes, leads to, drives, results in, because of, effect of); "
                    "use neutral verbs such as 'is', 'gives', 'reaches', 'is associated with'"
                )
    return problems


def _causal_flags(draft: ReportDraft) -> list[AuditFlag]:
    structured = (
        [StructuredCausalClaim(cause="(model)", effect="(target)", statement="causal_claim=true")]
        if draft.causal_claim
        else []
    )
    return check_causal_claims(full_text(draft), structured)


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _strip_causal(draft: ReportDraft, flagged: list[str] | None = None) -> ReportDraft:
    """Code fallback: drop causal sentences and state the limitation.

    ``flagged`` are the statements a judge flagged (LLM-judge ablation); without it the
    A11 lexicon decides sentence by sentence.
    """

    def causal(sentence: str) -> bool:
        if flagged is None:
            return bool(check_causal_claims(sentence))
        text = sentence.strip()
        return bool(text) and any(s.strip() and (s.strip() in text or text in s) for s in flagged)

    def clean(text: str) -> str:
        kept = [s for s in _SENTENCE.split(text) if not causal(s)]
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
