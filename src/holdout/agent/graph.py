"""The agent graph (LangGraph, SQLite checkpointer).

intake -> confirm_split (interrupt) -> seal -> [eda] -> plan -> compile_and_fit -> audit
  audit -> end_split                 (contaminated holdout or a ChangeSplit fix: new run)
  audit -> explain_and_fix -> review (interrupt) -> compile_and_fit   (at most 3 rounds)
  audit / explain_and_fix / review -> confirm_unseal (interrupt) -> unseal -> report

State holds only JSON-serialisable values; development rows are rebuilt from the hashed CSV
and the split spec in each node (never stored, so a checkpoint never contains data).
Nodes with side effects are idempotent (seal, unseal, report), so a run killed mid-node
resumes from its last checkpoint without sealing or evaluating twice.

Loop guards: at most ``MAX_FIX_ROUNDS`` fix rounds; a revised plan identical to an earlier
one stops the loop; one plan retry with validation feedback; a per-node wall-clock limit
(checked when the node returns - Python cannot safely kill a running thread; LLM calls are
bounded by the SDK request timeout).
"""

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, TypedDict, cast

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from holdout.agent.planner import (
    PlanContext,
    Planner,
    ReportContext,
    plan_fixes,
)
from holdout.agent.report import placeholder_values
from holdout.agent.schemas import (
    QUALITY_NODES,
    AgentRecord,
    Decision,
    FirstAttempt,
    LLMOutage,
    LLMQuality,
    NodeUsage,
    RejectedAttempt,
    RoundRecord,
    SandboxResult,
    Substitution,
)
from holdout.agent.tracing import tracer
from holdout.audit.gate import UnsealDecision, decide_unseal
from holdout.audit.runner import run_audits, write_audit
from holdout.audit.thresholds import load_thresholds
from holdout.audit.types import AuditFlag, AuditReport, ChangeSplit, CheckId, CheckRun, Severity
from holdout.core.errors import AgentError, NodeTimeoutError, RunPausingError
from holdout.core.schemas import CVResult, DataType, EntityScope, HoldoutResult, Task
from holdout.core.state import RunState
from holdout.data.load import LoadedData, load_dataset
from holdout.data.profile import DatasetProfile, profile_schema
from holdout.data.seal import SealedHoldout
from holdout.data.split import SplitProposal, SplitSpec, apply_split, propose_split
from holdout.llm import config as llm_config
from holdout.llm.progress import NullProgress, Progress
from holdout.llm.structured import LLMSession
from holdout.plan.compiler import evaluate_cv, fit_final
from holdout.plan.defaults import baseline_plan
from holdout.plan.metrics import get_metric
from holdout.plan.schema import ModelPlan, validate_plan
from holdout.runner import RunFacts, finalize_run, parse_split_option
from holdout.sandbox.runner import run_snippets

MAX_FIX_ROUNDS: Final = 3

AgentGraph = StateGraph["AgentState", None, "AgentState", "AgentState"]
NODE_TIMEOUT_SECONDS: Final = 600.0


@dataclass(frozen=True)
class AgentOptions:
    """Everything that defines an agent run (stored in the run directory for --resume)."""

    csv: str
    target: str
    out_dir: str
    goal: str | None = None
    task: str | None = None
    data_type: str = "tabular"
    split: str = "auto"
    seed: int = 42
    entities: str | None = None
    mode: str = "interactive"
    offline: bool = False
    eda: bool = False
    use_cache: bool = True
    previous_run_id: str | None = None
    allow_followup: bool = True
    #: ``--model`` as given (None = the configured model).
    model: str | None = None
    #: The model this run is pinned to, resolved once at the start; resume and follow-up
    #: runs reuse it, so a run never switches models.
    pinned_model: str | None = None
    #: Plan fields the user stated, applied by the offline planner only (see PlanContext).
    plan_overrides: dict[str, Any] = field(default_factory=dict)
    #: Gemini service tier for every call of the run ("flex", "standard"; None = default).
    service_tier: str | None = None

    @property
    def model_overridden(self) -> bool:
        return self.model is not None

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class AgentState(TypedDict, total=False):
    options: dict[str, Any]
    run_id: str
    started_at: str
    data_sha256: str
    task: str
    goal: str
    proposal: dict[str, Any]
    spec: dict[str, Any]
    eda: list[dict[str, Any]]
    round: int
    current_plan: dict[str, Any]
    plan_source: str
    proposed_plan: dict[str, Any] | None
    proposed_source: str
    baseline: dict[str, Any]
    cv: dict[str, Any]
    audit: dict[str, Any]
    rounds: list[dict[str, Any]]
    explanations: dict[str, str]
    substitutions: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    unseal_request: dict[str, Any]
    unseal: dict[str, Any]
    holdout: dict[str, Any] | None
    stop_reason: str
    next_spec: dict[str, Any] | None
    status: str
    nodes_done: list[str]


@dataclass
class AgentDeps:
    planner: Planner
    session: LLMSession | None
    llm_client: str
    model_verified: bool = False
    node_start_index: int = 0
    timeout_seconds: float = field(default_factory=lambda: NODE_TIMEOUT_SECONDS)
    after_node: Callable[[str], None] | None = field(default=None)
    progress: Progress = field(default_factory=NullProgress)
    #: The node running now (named in an LLM-unavailable pause).
    current_node: str | None = None
    #: Calls made in nodes that later failed with LLM unavailability (lost from the
    #: checkpoint, but spent): carried from the paused manifest into the final one.
    carried_calls: list[dict[str, Any]] = field(default_factory=list)
    #: Earlier pauses of this run (LLMOutage records as JSON).
    outages: list[dict[str, Any]] = field(default_factory=list)
    #: Replaces the deterministic audit layer (the benchmark's LLM-judge ablation only).
    auditor: "Auditor | None" = None


@dataclass(frozen=True)
class AuditInputs:
    """What an auditor sees after a cross-validated plan (development rows only)."""

    dev: pd.DataFrame
    plan: ModelPlan
    cv: CVResult
    baseline: CVResult
    profile: DatasetProfile
    spec: SplitSpec
    target: str
    goal: str
    round: int


Auditor = Callable[[AuditInputs], AuditReport]


# ----------------------------------------------------------------------------- helpers


def _options(state: AgentState) -> AgentOptions:
    return AgentOptions(**state["options"])


@lru_cache(maxsize=8)
def _load(
    csv: str, target: str, task: str | None, sha: str | None
) -> tuple[LoadedData, DatasetProfile]:
    loaded = load_dataset(Path(csv), target)
    if sha is not None and loaded.sha256 != sha:
        raise AgentError(f"{csv} changed during the run (sha256 {loaded.sha256}, expected {sha})")
    return loaded, profile_schema(loaded.frame, target, task=Task(task) if task else None)


def _materialize(state: AgentState) -> tuple[LoadedData, DatasetProfile, pd.DataFrame, SplitSpec]:
    opts = _options(state)
    loaded, profile = _load(opts.csv, opts.target, state.get("task"), state.get("data_sha256"))
    spec = SplitSpec.model_validate(state["spec"])
    dev, _ = apply_split(loaded.frame, spec, opts.target)  # the holdout half is discarded
    return loaded, profile, dev, spec


def _run_state(state: AgentState) -> RunState:
    opts = _options(state)
    return RunState(
        run_id=state["run_id"],
        out_dir=Path(opts.out_dir),
        run_seed=opts.seed,
        created_at=datetime.fromisoformat(state["started_at"]),
    )


def _plan_hash(plan: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def _audit(state: AgentState) -> AuditReport:
    return AuditReport.model_validate(state["audit"])


def _record(decision: Decision, state: AgentState) -> list[dict[str, Any]]:
    return [*state.get("decisions", []), decision.model_dump(mode="json")]


# ----------------------------------------------------------------------------- graph


def check_node_time(name: str, elapsed: float, api_seconds: float, limit: float) -> None:
    """Raise if the node's own compute (wall time minus API wait) is over ``limit``."""
    compute = max(0.0, elapsed - api_seconds)
    if compute > limit:
        raise NodeTimeoutError(
            f"node {name} used {compute:.0f} s of compute ({api_seconds:.0f} s waiting on the "
            f"API not counted), over the {limit:.0f} s limit",
            node=name,
        )


def build_graph(deps: AgentDeps) -> AgentGraph:
    planner = deps.planner

    def wrap(
        name: str, fn: Callable[[AgentState], dict[str, Any]]
    ) -> Callable[[AgentState], dict[str, Any]]:
        def run(state: AgentState) -> dict[str, Any]:
            session = deps.session
            start_index = len(session.records) if session is not None else 0
            deps.node_start_index = start_index
            deps.current_node = name
            waited_before = session.api_seconds if session is not None else 0.0
            deps.progress.node_started(name)
            started = time.perf_counter()
            with tracer().start_as_current_span(
                f"holdout.agent.{name}", record_exception=False
            ) as span:
                try:
                    update = fn(state)
                except RunPausingError as exc:
                    exc.node = exc.node or name
                    raise
                new = session.records[start_index:] if session is not None else []
                span.set_attribute("holdout.node", name)
                span.set_attribute("llm.calls", len(new))
                span.set_attribute("llm.cache_hits", sum(r.cache_hit for r in new))
                span.set_attribute("llm.prompt_tokens", sum(r.prompt_tokens for r in new))
                span.set_attribute("llm.output_tokens", sum(r.output_tokens for r in new))
                span.set_attribute("llm.thinking_tokens", sum(r.thinking_tokens for r in new))
                span.set_attribute("llm.inr", sum(r.inr for r in new))
                span.set_attribute("llm.retries", sum(r.retries for r in new))
            elapsed = time.perf_counter() - started
            deps.progress.node_finished(name, elapsed)
            # Only our own compute counts against the limit: time waiting on the API (each
            # call, its queueing under Flex, its retry waits) is excluded. Each call is
            # bounded by the request timeout and the retry policy instead.
            waited = (session.api_seconds if session is not None else 0.0) - waited_before
            check_node_time(name, elapsed, waited, deps.timeout_seconds)
            if new and name != "report":
                update["calls"] = [*state.get("calls", []), *(dataclasses.asdict(r) for r in new)]
            update["nodes_done"] = [*state.get("nodes_done", []), name]
            if deps.after_node is not None:
                deps.after_node(name)
            return update

        return run

    # ------------------------------------------------------------------ nodes

    def intake(state: AgentState) -> dict[str, Any]:
        opts = _options(state)
        loaded, profile = _load(opts.csv, opts.target, opts.task, None)
        task = Task(opts.task) if opts.task else profile.task_hint
        declared = EntityScope(opts.entities) if opts.entities else None
        proposal = propose_split(
            profile, task, DataType(opts.data_type), seed=opts.seed, entity_scope=declared
        )
        explicit = parse_split_option(opts.split, opts.seed)
        spec = (
            explicit.model_copy(update={"entity_scope": declared})
            if explicit is not None
            else proposal.spec
        )
        return {
            "data_sha256": loaded.sha256,
            "task": task.value,
            "goal": opts.goal or f"predict {opts.target}",
            "proposal": proposal.model_dump(mode="json"),
            "spec": spec.model_dump(mode="json"),
            "round": 0,
            "rounds": [],
            "decisions": [],
            "substitutions": [],
            "explanations": {},
            "calls": [],
            "eda": [],
            "stop_reason": "completed",
        }

    def confirm_split(state: AgentState) -> dict[str, Any]:
        proposal = SplitProposal.model_validate(state["proposal"])
        payload = {
            "kind": "split",
            "spec": state["spec"],
            "reasons": proposal.reasons,
            "warnings": proposal.warnings,
            "alternatives": [a.model_dump(mode="json") for a in proposal.alternatives],
            "declared_entity_scope": _options(state).entities,
        }
        decision = Decision.model_validate(interrupt(payload))
        spec = SplitSpec.model_validate(state["spec"])
        if decision.detail.get("split"):
            override = parse_split_option(str(decision.detail["split"]), _options(state).seed)
            if override is not None:
                spec = override
        if decision.detail.get("entity_scope"):
            spec = spec.model_copy(
                update={"entity_scope": EntityScope(decision.detail["entity_scope"])}
            )
        update: dict[str, Any] = {
            "spec": spec.model_dump(mode="json"),
            "decisions": _record(decision, state),
        }
        if not decision.accepted:
            update.update(status="stopped", stop_reason="split rejected by the user")
        return update

    def seal(state: AgentState) -> dict[str, Any]:
        opts = _options(state)
        loaded, _ = _load(opts.csv, opts.target, state["task"], state["data_sha256"])
        run_state = _run_state(state)
        spec = SplitSpec.model_validate(state["spec"])
        if run_state.seal_state_path.exists():  # resumed after sealing: reattach, never re-seal
            existing = SealedHoldout.load(run_state).record
            if existing.split_fingerprint != spec.fingerprint():
                raise AgentError("this run is sealed with a different split; start a new run")
        else:
            SealedHoldout.seal(loaded.frame, opts.target, spec, run_state, loaded.sha256)
        return {}

    def eda(state: AgentState) -> dict[str, Any]:
        _, profile, dev, _ = _materialize(state)
        snippets = planner.propose_eda(profile, state["goal"])
        results = run_snippets(snippets, dev.copy(), figures_dir=_run_state(state).run_dir / "eda")
        return {"eda": [r.model_dump(mode="json") for r in results]}

    def plan(state: AgentState) -> dict[str, Any]:
        _, profile, dev, _ = _materialize(state)
        proposed = planner.propose_plan(
            PlanContext(
                goal=state["goal"],
                target=_options(state).target,
                task=Task(state["task"]),
                profile=profile,
                dev=dev,
                seed=_options(state).seed,
                history=[],
                eda=[
                    {"purpose": e["purpose"], "status": e["status"], "output": e["output"][:4000]}
                    for e in state.get("eda", [])
                ],
                overrides=_options(state).plan_overrides,
            )
        )
        return {
            "current_plan": proposed.model_dump(mode="json"),
            "plan_source": "default" if planner.offline else "llm",
            "round": 0,
        }

    def compile_and_fit(state: AgentState) -> dict[str, Any]:
        _, profile, dev, spec = _materialize(state)
        current = ModelPlan.model_validate(state["current_plan"])
        target = _options(state).target
        baseline = evaluate_cv(
            baseline_plan(current), dev, target, spec, profile, label="dummy baseline"
        )
        result = evaluate_cv(current, dev, target, spec, profile, label=current.model.name)
        return {"baseline": baseline.model_dump(mode="json"), "cv": result.model_dump(mode="json")}

    def audit(state: AgentState) -> dict[str, Any]:
        _, profile, dev, spec = _materialize(state)
        run_state = _run_state(state)
        sealed = SealedHoldout.load(run_state)
        current = ModelPlan.model_validate(state["current_plan"])
        if deps.auditor is not None:  # the LLM-judge ablation replaces the audit layer
            report = deps.auditor(
                AuditInputs(
                    dev=dev,
                    plan=current,
                    cv=CVResult.model_validate(state["cv"]),
                    baseline=CVResult.model_validate(state["baseline"]),
                    profile=profile,
                    spec=spec,
                    target=_options(state).target,
                    goal=state["goal"],
                    round=state["round"],
                )
            )
        else:
            report = run_audits(
                run_state,
                dev,
                current,
                CVResult.model_validate(state["cv"]),
                baseline=CVResult.model_validate(state["baseline"]),
                profile=profile,
                spec=spec,
                target=_options(state).target,
                thresholds=load_thresholds(),
                overlap_source=sealed,
                hint_provider=planner.hint_provider(state["goal"]),
                goal_text=state["goal"],
            )
        if report.holdout_invalid and not sealed.holdout_invalid:
            sealed.mark_invalid(
                [f"{f.check.value}: {f.message}" for f in report.flags if f.invalidates_holdout],
                caller="holdout agent",
            )
        record = RoundRecord(
            round=state["round"],
            plan=current,
            plan_source=state["plan_source"],  # type: ignore[arg-type]
            cv=CVResult.model_validate(state["cv"]),
            flag_counts=report.counts,
            flags=[
                " ".join(
                    [f.check.value, f.severity.value, f.category.value, ",".join(f.columns)]
                ).strip()
                for f in report.flags
            ],
        )
        return {
            "audit": report.model_dump(mode="json"),
            "rounds": [*state.get("rounds", []), record.model_dump(mode="json")],
        }

    def end_split(state: AgentState) -> dict[str, Any]:
        flags = _audit(state).flags
        splits = [
            f.fix
            for f in flags
            if f.severity is Severity.CRITICAL and isinstance(f.fix, ChangeSplit)
        ]
        corrected = next((s.new_spec for s in splits if s.new_spec is not None), None)
        if corrected is not None:
            reason = "the split leaks (a new run with the corrected split is needed)"
        elif splits:
            reason = "the split leaks and cannot be expressed as a column: " + splits[0].instruction
        else:
            reason = "the sealed holdout is contaminated; fix the data and start a new run"
        return {
            "stop_reason": reason,
            "next_spec": corrected.model_dump(mode="json") if corrected is not None else None,
            "unseal_request": {"requested": False, "ack_warnings": False, "accept_risk": None},
        }

    def explain_and_fix(state: AgentState) -> dict[str, Any]:
        _, profile, _, _ = _materialize(state)
        current = ModelPlan.model_validate(state["current_plan"])
        flags = _audit(state).flags
        outcome = planner.explain_and_fix(current, flags, profile, state["round"])
        update: dict[str, Any] = {
            "explanations": {**state.get("explanations", {}), **outcome.explanations},
            "proposed_plan": None,
        }
        if outcome.substitution is not None:
            update["substitutions"] = [
                *state.get("substitutions", []),
                outcome.substitution.model_dump(mode="json"),
            ]
        if not plan_fixes(flags):
            return update
        proposed = outcome.plan.model_dump(mode="json")
        seen = {_plan_hash(r["plan"]) for r in state.get("rounds", [])}
        if _plan_hash(proposed) in seen:
            update["stop_reason"] = (
                "the revised plan repeats an earlier plan; stopped instead of looping"
            )
        elif state["round"] + 1 > MAX_FIX_ROUNDS:
            update["stop_reason"] = f"stopped after the maximum of {MAX_FIX_ROUNDS} fix rounds"
        else:
            update["proposed_plan"] = proposed
            update["proposed_source"] = (
                "code_substitution"
                if outcome.substitution is not None or planner.offline
                else "llm"
            )
        return update

    def review(state: AgentState) -> dict[str, Any]:
        payload = {
            "kind": "review",
            "current_plan": state["current_plan"],
            "proposed_plan": state["proposed_plan"],
            "flags": [
                f"{f.check.value} {f.severity.value} {','.join(f.columns)}"
                for f in _audit(state).flags
            ],
            "explanations": state.get("explanations", {}),
        }
        decision = Decision.model_validate(interrupt(payload))
        update: dict[str, Any] = {"decisions": _record(decision, state), "proposed_plan": None}
        if decision.accepted:
            update.update(
                current_plan=state["proposed_plan"],
                plan_source=state.get("proposed_source", "llm"),
                round=state["round"] + 1,
            )
        else:
            update["stop_reason"] = "the proposed fixes were declined"
        return update

    def confirm_unseal(state: AgentState) -> dict[str, Any]:
        report = _audit(state)
        preview = decide_unseal(report, requested=True, ack_warnings=True)
        payload = {
            "kind": "unseal",
            "counts": report.counts,
            "holdout_invalid": report.holdout_invalid,
            "blocked_by": preview.blocked_by,
        }
        decision = Decision.model_validate(interrupt(payload))
        detail = decision.detail if decision.accepted else {"requested": False}
        return {
            "decisions": _record(decision, state),
            "unseal_request": {
                "requested": bool(detail.get("requested", False)),
                "ack_warnings": bool(detail.get("ack_warnings", False)),
                "accept_risk": detail.get("accept_risk"),
            },
        }

    def unseal(state: AgentState) -> dict[str, Any]:
        _, profile, dev, spec = _materialize(state)
        request = state["unseal_request"]
        try:
            decision = decide_unseal(
                _audit(state),
                requested=request["requested"],
                ack_warnings=request["ack_warnings"],
                accept_risk=request["accept_risk"],
            )
        except ValueError as exc:
            raise AgentError(str(exc)) from exc
        holdout: HoldoutResult | None = None
        if decision.allowed:
            sealed = SealedHoldout.load(_run_state(state))
            holdout = sealed.last_result()  # resumed after evaluating: never evaluate twice
            if holdout is None:
                current = ModelPlan.model_validate(state["current_plan"])
                fitted = fit_final(current, dev, _options(state).target, spec, profile)
                holdout = sealed.evaluate(
                    fitted,
                    get_metric(current.metric),
                    caller=f"holdout agent ({_options(state).mode})",
                )
        return {
            "unseal": decision.model_dump(mode="json"),
            "holdout": holdout.model_dump(mode="json") if holdout is not None else None,
        }

    def report(state: AgentState) -> dict[str, Any]:
        return _write_report(state, deps)

    # ------------------------------------------------------------------ edges

    graph: AgentGraph = StateGraph(AgentState)
    for name, fn in (
        ("intake", intake),
        ("confirm_split", confirm_split),
        ("seal", seal),
        ("eda", eda),
        ("plan", plan),
        ("compile_and_fit", compile_and_fit),
        ("audit", audit),
        ("end_split", end_split),
        ("explain_and_fix", explain_and_fix),
        ("review", review),
        ("confirm_unseal", confirm_unseal),
        ("unseal", unseal),
        ("report", report),
    ):
        # LangGraph's add_node overloads are stricter than its runtime contract.
        graph.add_node(name, cast(Any, wrap(name, fn)))
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "confirm_split")
    graph.add_conditional_edges(
        "confirm_split", lambda s: END if s.get("status") == "stopped" else "seal", ["seal", END]
    )
    graph.add_conditional_edges(
        "seal",
        lambda s: "eda" if _options(s).eda and not deps.planner.offline else "plan",
        ["eda", "plan"],
    )
    graph.add_edge("eda", "plan")
    graph.add_edge("plan", "compile_and_fit")
    graph.add_edge("compile_and_fit", "audit")
    graph.add_conditional_edges(
        "audit", _route_after_audit, ["end_split", "explain_and_fix", "confirm_unseal"]
    )
    graph.add_edge("end_split", "report")
    graph.add_conditional_edges(
        "explain_and_fix",
        lambda s: "review" if s.get("proposed_plan") else "confirm_unseal",
        ["review", "confirm_unseal"],
    )
    graph.add_conditional_edges(
        "review",
        lambda s: "compile_and_fit" if _last_decision_accepted(s) else "confirm_unseal",
        ["compile_and_fit", "confirm_unseal"],
    )
    graph.add_edge("confirm_unseal", "unseal")
    graph.add_edge("unseal", "report")
    graph.add_edge("report", END)
    return graph


def _last_decision_accepted(state: AgentState) -> bool:
    decisions = state.get("decisions", [])
    return bool(decisions) and bool(decisions[-1]["accepted"])


def _route_after_audit(state: AgentState) -> str:
    report = _audit(state)
    split_problem = report.holdout_invalid or any(
        f.severity is Severity.CRITICAL and isinstance(f.fix, ChangeSplit) for f in report.flags
    )
    if split_problem:
        return "end_split"
    if report.at_least(Severity.WARNING):
        return "explain_and_fix"
    return "confirm_unseal"


# ----------------------------------------------------------------------------- report


def _write_report(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    opts = _options(state)
    loaded, profile, dev, spec = _materialize(state)
    run_state = _run_state(state)
    sealed = SealedHoldout.load(run_state)
    current = ModelPlan.model_validate(state["current_plan"])
    audit_report = _audit(state)
    result = CVResult.model_validate(state["cv"])
    baseline = CVResult.model_validate(state["baseline"])
    holdout = HoldoutResult.model_validate(state["holdout"]) if state.get("holdout") else None
    unseal_decision = (
        decide_unseal(audit_report, requested=False)
        if not state.get("unseal")
        else _unseal_from_state(state)
    )
    values = placeholder_values(
        {
            "validation_score": result.mean,
            "validation_std": result.std,
            "dummy_score": baseline.mean,
            "holdout_score": holdout.score if holdout is not None else None,
            "optimism_gap": result.mean - holdout.score if holdout is not None else None,
            "n_rows": len(loaded.frame),
            "n_dev_rows": len(dev),
            "n_holdout_rows": sealed.n_rows,
            "n_fix_rounds": max(0, len(state.get("rounds", [])) - 1),
            "n_critical": audit_report.counts["critical"],
            "n_warning": audit_report.counts["warning"],
        }
    )
    metric = get_metric(current.metric)
    facts = {
        "metric": metric.name.value,
        "higher_is_better": metric.direction.value == "higher",
        "plan_beats_dummy": metric.improvement(result.mean, baseline.mean) > 0,
        "model": current.model.name,
        "holdout_evaluated": holdout is not None,
        "holdout_invalid": audit_report.holdout_invalid,
        "stop_reason": state.get("stop_reason", "completed"),
        "split": spec.strategy.value,
    }
    caveats = list(sealed.caveats)
    outcome = deps.planner.draft_report(
        ReportContext(
            goal=state["goal"],
            facts=facts,
            values=values,
            flags=audit_report.flags,
            explanations=state.get("explanations", {}),
            caveats=caveats,
            columns=[c.name for c in profile.columns],
            plan=current,
            round=state.get("round", 0),
        )
    )
    final_audit = _with_causal_check(
        audit_report, outcome.causal_flags, ran=not deps.planner.offline
    )
    if final_audit is not audit_report:
        write_audit(run_state.audit_path, final_audit)
    substitutions = [Substitution.model_validate(s) for s in state.get("substitutions", [])]
    if outcome.substitution is not None:
        substitutions.append(outcome.substitution)
    session = deps.session
    calls = [*state.get("calls", [])]
    if session is not None:
        calls += [dataclasses.asdict(r) for r in session.records[deps.node_start_index :]]
    spent_calls = [*deps.carried_calls, *calls]
    record = AgentRecord(
        mode=opts.mode,  # type: ignore[arg-type]
        offline=opts.offline,
        llm_client=deps.llm_client,
        model=None if session is None else session.client.model,
        model_overridden=opts.model_overridden and session is not None,
        model_verified=deps.model_verified,
        seed_passed_to_model=session is not None and session.client.supports_seed,
        thinking_level=None if session is None else llm_config.THINKING_LEVEL,
        max_output_tokens=None if session is None else llm_config.MAX_OUTPUT_TOKENS,
        prompts={
            c["prompt_name"]: {"version": c["prompt_version"], "sha256": c["prompt_sha256"]}
            for c in spent_calls
        },
        usage_by_node=usage_by_node(spent_calls),
        total_inr=sum(c["inr"] for c in spent_calls),
        cache_hits=sum(bool(c["cache_hit"]) for c in spent_calls),
        retries=sum(int(c.get("retries", 0)) for c in spent_calls),
        api_seconds=round(session.api_seconds, 2) if session is not None else 0.0,
        service_tier=opts.service_tier,
        quality=None if session is None else llm_quality(spent_calls, len(substitutions)),
        rejected_attempts=rejected_attempts(spent_calls),
        outages=[LLMOutage.model_validate(o) for o in deps.outages],
        run_cap_inr=session.budget.run_cap_inr if session is not None else llm_config.RUN_CAP_INR,
        project_cap_inr=(
            session.budget.project_cap_inr if session is not None else llm_config.PROJECT_CAP_INR
        ),
        budget_stop=None,
        decisions=[Decision.model_validate(d) for d in state.get("decisions", [])],
        rounds=[RoundRecord.model_validate(r) for r in state.get("rounds", [])],
        substitutions=substitutions,
        explanations=state.get("explanations", {}),
        eda=[SandboxResult.model_validate(e) for e in state.get("eda", [])],
        report=outcome.draft,
        stop_reason=state.get("stop_reason", "completed"),
        previous_run_id=opts.previous_run_id,
        next_run_id=None,
    )
    manifest = finalize_run(
        RunFacts(
            state=run_state,
            started_at=datetime.fromisoformat(state["started_at"]),
            loaded=loaded,
            target=opts.target,
            task=Task(state["task"]),
            goal=state["goal"],
            entity_scope=spec.entity_scope,
            profile=profile,
            proposal=SplitProposal.model_validate(state["proposal"]),
            spec=spec,
            n_dev_rows=len(dev),
            seal=sealed,
            plan=current,
            issues=validate_plan(current, profile),
            baseline=baseline,
            result=result,
            thresholds_hash=final_audit.thresholds_hash,
            audit=final_audit,
            unseal=unseal_decision,
            holdout=holdout,
            agent=record,
        )
    )
    return {
        "status": "completed",
        "calls": calls,
        "substitutions": [s.model_dump(mode="json") for s in substitutions],
        "audit": manifest.audit.model_dump(mode="json"),
    }


def _unseal_from_state(state: AgentState) -> UnsealDecision:
    return UnsealDecision.model_validate(state["unseal"])


def _with_causal_check(report: AuditReport, causal: list[AuditFlag], *, ran: bool) -> AuditReport:
    """A11 on the generated report (agent runs); offline runs keep the Week 2 skip."""
    if not ran:
        return report
    runs = [
        r
        if r.check is not CheckId.A11
        else CheckRun(
            check=CheckId.A11,
            status="ran",
            seconds=0.0,
            detail=f"generated report checked; {len(causal)} causal-claim flag(s)",
        )
        for r in report.checks_run
    ]
    return report.model_copy(update={"flags": [*report.flags, *causal], "checks_run": runs})


def llm_quality(calls: list[dict[str, Any]], substitutions: int) -> LLMQuality:
    """First-attempt validity per node, repairs and substitutions, from the call records.

    A first attempt is an answer to a fresh request (``attempt == 1``); cache hits replay an
    earlier answer and failed calls (unavailable, billing) got no answer, so neither is
    scored. Valid means it passed the schema and every code check of that node.
    """
    answered = [c for c in calls if not c.get("cache_hit") and not c.get("failed", False)]
    first_attempt = {}
    for node in QUALITY_NODES:
        firsts = [c for c in answered if c["node"] == node and c["attempt"] == 1]
        valid = sum(bool(c["valid"]) for c in firsts)
        first_attempt[node] = FirstAttempt(
            first_attempts=len(firsts),
            valid=valid,
            rate=valid / len(firsts) if firsts else None,
        )
    return LLMQuality(
        first_attempt=first_attempt,
        repairs=sum(c["attempt"] == 2 for c in answered),
        substitutions=substitutions,
    )


def rejected_attempts(calls: list[dict[str, Any]]) -> list[RejectedAttempt]:
    return [
        RejectedAttempt(
            node=c["node"],
            prompt=c["prompt_name"],
            prompt_version=c["prompt_version"],
            attempt=c["attempt"],
            errors=list(c["errors"]),
            text=c.get("rejected_text"),
        )
        for c in calls
        if not c["valid"] and not c.get("failed", False) and not c.get("cache_hit")
    ]


def _tier_counts(calls: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in calls:
        if c.get("cache_hit") or c.get("failed"):
            continue
        tier = str(c.get("tier") or "unreported")
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def usage_by_node(calls: list[dict[str, Any]]) -> dict[str, NodeUsage]:
    nodes: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        nodes.setdefault(call["node"], []).append(call)
    return {
        node: NodeUsage(
            calls=len(items),
            cache_hits=sum(bool(c["cache_hit"]) for c in items),
            prompt_tokens=sum(c["prompt_tokens"] for c in items),
            output_tokens=sum(c["output_tokens"] for c in items),
            thinking_tokens=sum(c["thinking_tokens"] for c in items),
            inr=sum(c["inr"] for c in items),
            retries=sum(int(c.get("retries", 0)) for c in items),
            cached_tokens=sum(int(c.get("cached_tokens", 0)) for c in items),
            inr_cache_adjusted=sum(
                float(c.get("inr_cache_adjusted", c["inr"])) for c in items if not c["cache_hit"]
            ),
            tiers=_tier_counts(items),
        )
        for node, items in sorted(nodes.items())
    }


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
