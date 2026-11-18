"""The LLM-judge ablation (DS-STAR style): Holdout with its audit layer replaced by Gemini.

Same compiler, seal, fix loop, loop guards and unseal gate; only the verdicts change. After
each cross-validated plan the judge receives the profile, the plan, the CV result with
per-fold scores and the dummy baseline, and returns issues in Holdout's categories and
severities, each with a typed fix (drop columns, change metric, change split, deduplicate,
none). They become ordinary audit flags (owned by the check that covers the category, with
:class:`JudgeEvidence`), so the rest of the graph treats them exactly like audit flags. The
report's causal-claim check (A11) is replaced too: a second judge prompt lists the causal
sentences of each draft.

The judge's calls go through the run's own ``LLMSession`` (same caps, retries, ledger).
"""

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from holdout.agent.graph import AgentDeps, AuditInputs
from holdout.agent.llm_inputs import profile_summary
from holdout.agent.planner import LLMPlanner
from holdout.agent.prompts import PromptTemplate
from holdout.agent.report import full_text
from holdout.agent.schemas import ReportDraft
from holdout.audit.types import (
    AuditFlag,
    AuditReport,
    Category,
    CausalEvidence,
    ChangeMetric,
    ChangeSplit,
    CheckId,
    CheckRun,
    Deduplicate,
    DropColumns,
    Fix,
    JudgeEvidence,
    NoAction,
    RemoveCausalClaim,
    Severity,
)
from holdout.core.schemas import MetricName, Task
from holdout.llm.structured import LLMSession, RenderedPrompt
from holdout.plan.metrics import get_metric
from holdout.runner import parse_split_option

PROMPTS = Path(__file__).resolve().parent / "prompts"
VERIFIER: Final = "gemini judge"

#: Which audit check "owns" each category (for the flag's check id and the report table).
CATEGORY_CHECK: Final[dict[Category, CheckId]] = {
    Category.TARGET_LEAKAGE: CheckId.A1,
    Category.POST_OUTCOME_COLUMN: CheckId.A2,
    Category.ID_SIGNAL: CheckId.A3,
    Category.DUPLICATE_ROWS: CheckId.A4,
    Category.GROUP_LEAKAGE: CheckId.A5,
    Category.TIME_LEAKAGE: CheckId.A6,
    Category.METRIC_MISMATCH: CheckId.A7,
    Category.FORMAT_ISSUE: CheckId.A8,
    Category.ENCODING_LEAKAGE: CheckId.A9,
    Category.SUSPICIOUS_SCORE: CheckId.A10,
    Category.CAUSAL_CLAIM: CheckId.A11,
    Category.UNSTABLE_VALIDATION: CheckId.A12,
}


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class JudgeFix(_Strict):
    kind: Literal["drop_columns", "change_metric", "change_split", "deduplicate", "none"]
    columns: list[str] = Field(default_factory=list, max_length=50)
    metric: MetricName | None = None
    split: str | None = Field(default=None, max_length=120)


class JudgeIssue(_Strict):
    category: Category
    severity: Severity
    columns: list[str] = Field(default_factory=list, max_length=50)
    explanation: str = Field(min_length=10, max_length=600)
    fix: JudgeFix


class JudgeVerdict(_Strict):
    issues: list[JudgeIssue] = Field(max_length=20)


class CausalVerdict(_Strict):
    causal_statements: list[str] = Field(max_length=20)


def load_template(name: str) -> PromptTemplate:
    raw = (PROMPTS / f"{name}.md").read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    header = re.match(r"\A---\n(?P<h>.*?)\n---\n", text, re.DOTALL)
    sections = dict(re.findall(r"<(system|user)>\n(.*?)\n</\1>", text, re.DOTALL))
    if header is None or set(sections) != {"system", "user"}:
        raise ValueError(f"prompt {name}: missing header or sections")
    fields = {
        k.strip(): v.strip()
        for k, v in (line.split(":", 1) for line in header.group("h").splitlines() if ":" in line)
    }
    return PromptTemplate(
        name=fields["name"],
        version=fields["version"],
        purpose=fields["purpose"],
        sha256=hashlib.sha256(raw).hexdigest(),
        system=sections["system"],
        user=sections["user"],
    )


def _prompt(template: PromptTemplate, **data: object) -> RenderedPrompt:
    r = template.render(**data)
    return RenderedPrompt(r.name, r.version, r.sha256, r.system, r.user)


def _split_text(split: str | None) -> str:
    return (split or "").strip()


def verdict_problems(verdict: JudgeVerdict, columns: set[str], task: Task) -> list[str]:
    """Checks that make a verdict actionable (repaired once if any fail)."""
    problems = []
    for n, issue in enumerate(verdict.issues, start=1):
        unknown = [c for c in [*issue.columns, *issue.fix.columns] if c not in columns]
        if unknown:
            problems.append(f"issue {n}: unknown columns {unknown}")
        if issue.fix.kind == "drop_columns" and not issue.fix.columns:
            problems.append(f"issue {n}: drop_columns needs columns")
        if issue.fix.kind == "change_metric":
            if issue.fix.metric is None:
                problems.append(f"issue {n}: change_metric needs a metric")
            elif get_metric(issue.fix.metric).task is not task:
                problems.append(f"issue {n}: {issue.fix.metric.value} is not a {task.value} metric")
        if issue.fix.kind == "change_split":
            split = _split_text(issue.fix.split)
            kind, _, column = split.partition(":")
            if kind not in {"group", "time"} or not column:
                problems.append(f"issue {n}: change_split needs group:COLUMN or time:COLUMN")
            elif column != "<row>" and column not in columns:
                problems.append(f"issue {n}: split column {column!r} does not exist")
    return problems


def to_fix(fix: JudgeFix, reason: str, seed: int) -> Fix:
    if fix.kind == "drop_columns":
        return DropColumns(columns=list(fix.columns), reason=reason[:200])
    if fix.kind == "change_metric" and fix.metric is not None:
        return ChangeMetric(metric=fix.metric)
    if fix.kind == "change_split":
        split = _split_text(fix.split)
        if split.endswith(":<row>"):
            return ChangeSplit(
                new_spec=None,
                instruction="time is only in the row order: add a row-number column, then "
                "run again with --split time:<that column>",
            )
        return ChangeSplit(new_spec=parse_split_option(split, seed), instruction=f"--split {split}")
    if fix.kind == "deduplicate":
        return Deduplicate(scope="before_split")
    return NoAction(reason=reason[:200] or "the judge proposed no action")


def to_flags(verdict: JudgeVerdict, seed: int) -> list[AuditFlag]:
    return [
        AuditFlag(
            check=CATEGORY_CHECK[issue.category],
            category=issue.category,
            severity=issue.severity,
            columns=list(issue.columns),
            evidence=JudgeEvidence(verifier=VERIFIER, rationale=issue.explanation),
            message=issue.explanation,
            fix=to_fix(issue.fix, issue.explanation, seed),
        )
        for issue in verdict.issues
    ]


class Judge:
    """Installed into an agent run by :func:`install` (``run_agent(customize=...)``)."""

    def __init__(self, session: LLMSession) -> None:
        self.session = session
        self.audit_template = load_template("judge")
        self.report_template = load_template("judge_report")

    def audit(self, inputs: AuditInputs) -> AuditReport:
        columns = set(map(str, inputs.dev.columns))
        prompt = _prompt(
            self.audit_template,
            goal=inputs.goal,
            target={"target": inputs.target, "task": inputs.plan.task.value},
            profile=profile_summary(inputs.profile),
            split=inputs.spec.model_dump(mode="json"),
            plan=inputs.plan.model_dump(mode="json"),
            cv={
                "metric": inputs.plan.metric.value,
                "plan": {
                    "mean": inputs.cv.mean,
                    "std": inputs.cv.std,
                    "fold_scores": inputs.cv.fold_scores,
                },
                "dummy": {"mean": inputs.baseline.mean, "fold_scores": inputs.baseline.fold_scores},
            },
        )
        verdict = self.session.call(
            node="audit",
            prompt=prompt,
            schema=JudgeVerdict,
            check=lambda v: verdict_problems(v, columns, inputs.plan.task),
            send_length_bounds=False,  # Gemini: "too many states" otherwise (HTTP 400)
        )
        return AuditReport(
            flags=to_flags(verdict, inputs.spec.seed),
            checks_run=[
                CheckRun(check=c, status="ran", seconds=0.0, detail=VERIFIER) for c in CheckId
            ],
            thresholds_hash="llm_judge",
            deep_mode=False,
            entity_scope=inputs.spec.entity_scope,
        )

    def causal(self, draft: ReportDraft) -> list[AuditFlag]:
        text = full_text(draft)
        verdict = self.session.call(
            node="report",
            prompt=_prompt(self.report_template, report=text),
            schema=CausalVerdict,
            check=lambda v: [
                f"not a sentence of the report: {s!r}" for s in v.causal_statements if s not in text
            ],
            send_length_bounds=False,
        )
        if not verdict.causal_statements:
            return []
        statements = list(verdict.causal_statements)
        return [
            AuditFlag(
                check=CheckId.A11,
                category=Category.CAUSAL_CLAIM,
                severity=Severity.WARNING,
                columns=[],
                evidence=CausalEvidence(source="text", statements=statements, phrases=[VERIFIER]),
                message="The judge found causal claims a predictive model cannot support.",
                fix=RemoveCausalClaim(statements=statements),
            )
        ]


def install() -> Callable[[AgentDeps], None]:
    """``customize`` hook: replace the audit layer and the report's causal check."""

    def customize(deps: AgentDeps) -> None:
        if deps.session is None or not isinstance(deps.planner, LLMPlanner):
            raise ValueError("the LLM judge needs an LLM session (not an offline run)")
        judge = Judge(deps.session)
        deps.auditor = judge.audit
        deps.planner.causal_checker = judge.causal

    return customize
