"""Typed LLM outputs and the agent's records (decisions, substitutions, rounds, usage)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from holdout.audit.types import CheckId
from holdout.core.schemas import CVResult
from holdout.plan.schema import ModelPlan


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ----------------------------------------------------------------------------- LLM outputs


class FlagExplanation(_Strict):
    check: CheckId
    columns: list[str]
    explanation: str = Field(min_length=10, max_length=600)


class ExplainFixResponse(_Strict):
    explanations: list[FlagExplanation]
    revised_plan: ModelPlan


class ReportDraft(_Strict):
    summary: str = Field(min_length=20, max_length=1500)
    findings: list[str] = Field(max_length=12)
    caveats: list[str] = Field(max_length=12)
    recommendations: list[str] = Field(max_length=8)
    causal_claim: bool = Field(description="true only if the prose claims cause and effect")


class SuspectColumn(_Strict):
    column: str
    reason: str = Field(min_length=3, max_length=200)


class HintResponse(_Strict):
    suspects: list[SuspectColumn] = Field(max_length=50)


class EDASnippet(_Strict):
    purpose: str = Field(min_length=3, max_length=200)
    code: str = Field(min_length=1, max_length=4000)


class EDAProposal(_Strict):
    snippets: list[EDASnippet] = Field(max_length=4)


# ----------------------------------------------------------------------------- records

DecisionKind = Literal["split", "review", "unseal"]
DecisionMode = Literal["interactive", "autopilot"]


class Decision(_Strict):
    """One interrupt answer, recorded identically for a human and for autopilot."""

    kind: DecisionKind
    mode: DecisionMode
    accepted: bool
    detail: dict[str, Any] = Field(default_factory=dict)
    reason: str


class Substitution(_Strict):
    """The code-applied plan replaced the LLM's integrated plan (not credited to the LLM)."""

    node: str
    round: int
    missed_fixes: list[str]


class RoundRecord(_Strict):
    round: int
    plan: ModelPlan
    plan_source: Literal["llm", "code_substitution", "default"]
    cv: CVResult
    flag_counts: dict[str, int]
    flags: list[str] = Field(description="check id, severity, category and columns per flag")


class NodeUsage(_Strict):
    calls: int
    cache_hits: int
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    inr: float
    #: Transient failures retried (429 / 5xx / timeouts); they cost no tokens.
    retries: int = 0


class LLMOutage(_Strict):
    """The run paused because the LLM stayed unavailable after every retry."""

    node: str
    status: int | None = Field(description="last HTTP status; None for a timeout")
    retries: int
    at: str = Field(description="UTC ISO time the run paused")
    message: str


class SandboxResult(_Strict):
    purpose: str
    status: Literal["ok", "rejected", "timeout", "memory", "error"]
    output: str
    detail: str | None = None


class AgentRecord(_Strict):
    """The agent part of a run manifest."""

    mode: DecisionMode
    offline: bool
    llm_client: str = Field(description="client class, e.g. GeminiClient, FakeLLM, none")
    model: str | None
    model_overridden: bool = Field(
        default=False, description="--model replaced the configured model for this run"
    )
    model_verified: bool
    seed_passed_to_model: bool
    thinking_level: str | None
    max_output_tokens: int | None
    prompts: dict[str, dict[str, str]] = Field(description="prompt name -> version, sha256")
    usage_by_node: dict[str, NodeUsage]
    total_inr: float
    cache_hits: int
    retries: int = 0
    outages: list[LLMOutage] = Field(
        default_factory=list, description="pauses for LLM unavailability before resuming"
    )
    run_cap_inr: float
    project_cap_inr: float
    budget_stop: str | None
    decisions: list[Decision]
    rounds: list[RoundRecord]
    substitutions: list[Substitution]
    explanations: dict[str, str]
    eda: list[SandboxResult]
    report: ReportDraft | None
    stop_reason: str
    previous_run_id: str | None
    next_run_id: str | None
