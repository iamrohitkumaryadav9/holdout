"""Run and resume the agent graph; answer interrupts by policy, by a person, or pause.

* ``run_agent`` allocates the run directory (with its SQLite checkpoint database and the
  options needed to resume), pins the model for the whole run, then drives the graph.
* Interrupts: autopilot answers with :func:`autopilot_decision`; interactive mode asks the
  given :class:`Asker`; with no asker (no terminal) the run pauses (CLI exit code 4) and
  ``resume_agent`` continues it later, in a new process if need be.
* LLM outages: when a call fails with 429 / 5xx / timeouts on every retry, the run pauses
  with status ``paused_llm_unavailable`` (CLI exit code 5). The checkpoint holds every
  completed node and ``run_manifest.json`` records what failed, where, what was spent and
  the exact resume command. A run is never switched to another model to get past an outage.
  A call refused by a spend cap (``paused_budget_cap``, exit 6; the call is never sent) and
  a billing / quota refusal from Gemini (``paused_billing``, exit 7) pause the same way.
* Autopilot runs that end because the split leaks start exactly one follow-up run with the
  corrected split; the two manifests link to each other.
"""

import dataclasses
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from holdout.agent.graph import AgentDeps, AgentOptions, AgentState, build_graph, usage_by_node
from holdout.agent.planner import LLMPlanner, OfflinePlanner, Planner
from holdout.agent.policy import Asker, autopilot_decision
from holdout.agent.schemas import Decision, LLMOutage, PauseKind, RoundRecord
from holdout.core.errors import (
    AgentError,
    BudgetExceededError,
    HoldoutError,
    LLMBillingError,
    RunPausingError,
)
from holdout.core.state import RunState
from holdout.data.split import SplitSpec
from holdout.export.manifest import (
    PausedRunManifest,
    PausedStatus,
    RunManifest,
    capture_environment,
    write_manifest,
    write_report,
)
from holdout.llm import config as llm_config
from holdout.llm.budget import Budget, balance_warning
from holdout.llm.cache import ResponseCache
from holdout.llm.client import GeminiClient, LLMClient
from holdout.llm.progress import NullProgress, Progress
from holdout.llm.structured import LLMSession

ClientFactory = Callable[[], LLMClient]
#: Adjusts a run's dependencies after they are built (the benchmark's LLM-judge ablation
#: installs its auditor here, on the run's own session, so costs land in the same ledger).
Customize = Callable[[AgentDeps], None]

PAUSED_LLM: Final = "paused_llm_unavailable"
PAUSED_BUDGET: Final = "paused_budget_cap"
PAUSED_BILLING: Final = "paused_billing"
PAUSED_STATUSES = frozenset({PAUSED_LLM, PAUSED_BUDGET, PAUSED_BILLING})
VERIFY_NODE = "verify_model"


@dataclass(frozen=True)
class AgentOutcome:
    run_id: str
    run_dir: Path
    status: str  # completed | paused | stopped | paused_llm_unavailable
    stop_reason: str | None
    pending: dict[str, Any] | None
    manifest: RunManifest | None
    followup: "AgentOutcome | None" = None
    #: Set when status is paused_llm_unavailable.
    paused: PausedRunManifest | None = None


def run_agent(
    options: AgentOptions,
    *,
    client_factory: ClientFactory | None = None,
    asker: Asker | None = None,
    after_node: Callable[[str], None] | None = None,
    progress: Progress | None = None,
    customize: Customize | None = None,
) -> AgentOutcome:
    """Start a new agent run.

    Raises:
        LLMConfigError: if the model (``--model`` or the configured one) has no price.
    """
    if not options.offline:
        pinned = options.pinned_model or options.model or llm_config.DEFAULT_MODEL
        llm_config.require_priced(pinned, date.today(), tier=options.service_tier or "standard")
        llm_config.project_cap_inr()  # a malformed HOLDOUT_PROJECT_CAP_INR fails before the run
        options = replace(options, pinned_model=pinned)
    started = datetime.now(UTC)
    state = RunState.create(Path(options.out_dir), options.seed, now=started)
    state.agent_options_path.write_text(json.dumps(options.to_json(), indent=2), encoding="utf-8")
    initial: AgentState = {
        "options": options.to_json(),
        "run_id": state.run_id,
        "started_at": started.isoformat(),
    }
    return _drive(state, options, initial, client_factory, asker, after_node, progress, customize)


def resume_agent(
    run_dir: Path,
    *,
    mode: str | None = None,
    model: str | None = None,
    client_factory: ClientFactory | None = None,
    asker: Asker | None = None,
    after_node: Callable[[str], None] | None = None,
    progress: Progress | None = None,
    customize: Customize | None = None,
) -> AgentOutcome:
    """Continue a paused or interrupted run from its last checkpoint, on the same model.

    Raises:
        AgentError: if ``model`` differs from the model the run is pinned to.
    """
    options_path = run_dir / "agent_options.json"
    if not options_path.is_file():
        raise HoldoutError(f"{run_dir} is not an agent run (no agent_options.json)")
    options = AgentOptions(**json.loads(options_path.read_text(encoding="utf-8")))
    if not options.offline and options.pinned_model is None:  # runs from before pinning
        options = replace(options, pinned_model=options.model or llm_config.DEFAULT_MODEL)
    if model is not None and model != options.pinned_model:
        raise AgentError(
            f"run {run_dir.name} is pinned to model {options.pinned_model!r}; a run never "
            f"switches models. Resume without --model, or start a new run with --model {model}"
        )
    if mode is not None:
        options = replace(options, mode=mode)
    state = RunState(
        run_id=run_dir.name,
        out_dir=run_dir.parent,
        run_seed=options.seed,
        created_at=datetime.now(UTC),
    )
    return _drive(state, options, None, client_factory, asker, after_node, progress, customize)


def resume_command(run_id: str, out_dir: Path) -> str:
    return f"holdout agent --resume {run_id} --out {out_dir}"


def _deps(
    options: AgentOptions,
    run_id: str,
    client_factory: ClientFactory | None,
    spent_inr: float,
    progress: Progress,
) -> AgentDeps:
    if options.offline:
        return AgentDeps(
            planner=OfflinePlanner(), session=None, llm_client="none", progress=progress
        )
    model = options.pinned_model or llm_config.DEFAULT_MODEL
    timeout = (
        llm_config.FLEX_REQUEST_TIMEOUT_SECONDS
        if options.service_tier == "flex"
        else llm_config.REQUEST_TIMEOUT_SECONDS
    )
    client = (
        client_factory()
        if client_factory is not None
        else GeminiClient(model=model, timeout_seconds=timeout)
    )
    if client.model != model:
        raise AgentError(
            f"the LLM client serves {client.model!r} but this run is pinned to {model!r}; "
            "a run never switches models"
        )
    verified = False
    if isinstance(client, GeminiClient):
        progress.node_started(VERIFY_NODE)
        started = datetime.now(UTC)
        try:
            client.verify_model(
                on_retry=lambda event: progress.retry_waiting(VERIFY_NODE, "list models", event)
            )
        except RunPausingError as exc:
            exc.node = VERIFY_NODE
            raise
        progress.node_finished(VERIFY_NODE, (datetime.now(UTC) - started).total_seconds())
        verified = client.model_verified
    session = LLMSession(
        client=client,
        budget=Budget(run_id=run_id, model=client.model, spent_run_inr=spent_inr),
        seed=options.seed,
        cache=ResponseCache(enabled=options.use_cache),
        progress=progress,
        service_tier=options.service_tier,
    )
    standing = balance_warning(session.budget.ledger.balance_left())
    if standing is not None:
        progress.warning(standing)
    planner: Planner = LLMPlanner(session)
    return AgentDeps(
        planner=planner,
        session=session,
        llm_client=type(client).__name__,
        model_verified=verified,
        progress=progress,
    )


def _load_paused(state: RunState) -> PausedRunManifest | None:
    if not state.manifest_path.is_file():
        return None
    raw = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    if raw.get("status") not in PAUSED_STATUSES:
        return None
    return PausedRunManifest.model_validate(raw)


def _drive(
    state: RunState,
    options: AgentOptions,
    initial: AgentState | None,
    client_factory: ClientFactory | None,
    asker: Asker | None,
    after_node: Callable[[str], None] | None,
    progress: Progress | None,
    customize: Customize | None = None,
) -> AgentOutcome:
    progress = progress if progress is not None else NullProgress()
    connection = sqlite3.connect(state.checkpoint_path, check_same_thread=False)
    try:
        config: Any = {"configurable": {"thread_id": state.run_id}}
        saver = SqliteSaver(connection)
        reader = build_graph(
            AgentDeps(planner=OfflinePlanner(), session=None, llm_client="none")
        ).compile(checkpointer=saver)
        paused = _load_paused(state)
        if initial is None:
            existing = reader.get_state(config).values
            if not existing:
                if paused is None:
                    raise HoldoutError(f"run {state.run_id} has no checkpoint to resume")
                # Paused before the first node (model verification): start from the top.
                initial = {
                    "options": options.to_json(),
                    "run_id": state.run_id,
                    "started_at": paused.started_at.isoformat(),
                }
        else:
            existing = {}
        carried = list(paused.interrupted_calls) if paused is not None else []
        outages = (
            [o.model_dump(mode="json") for o in [*paused.earlier_outages, paused.outage]]
            if paused is not None
            else []
        )
        spent = sum(c["inr"] for c in [*existing.get("calls", []), *carried])
        deps: AgentDeps | None = None
        try:
            deps = _deps(options, state.run_id, client_factory, spent, progress)
            deps.after_node = after_node
            if customize is not None:
                customize(deps)
            deps.carried_calls = carried
            deps.outages = outages
            app = build_graph(deps).compile(checkpointer=saver)
            result: dict[str, Any] = app.invoke(initial, config)
            while "__interrupt__" in result:
                payload = dict(result["__interrupt__"][0].value)
                if options.mode == "autopilot":
                    decision = autopilot_decision(payload)
                elif asker is not None:
                    decision = asker.ask(payload)
                else:
                    return AgentOutcome(
                        run_id=state.run_id,
                        run_dir=state.run_dir,
                        status="paused",
                        stop_reason=None,
                        pending=payload,
                        manifest=None,
                    )
                result = app.invoke(Command(resume=decision.model_dump(mode="json")), config)
            final = app.get_state(config).values
        except RunPausingError as exc:
            values = reader.get_state(config).values or dict(initial or {})
            record = _write_paused(state, options, exc, values, deps, carried, outages)
            return AgentOutcome(
                run_id=state.run_id,
                run_dir=state.run_dir,
                status=record.status,
                stop_reason=record.message,
                pending=None,
                manifest=None,
                paused=record,
            )
    finally:
        connection.close()
    if final.get("status") == "stopped":
        if paused is not None:
            state.manifest_path.unlink()  # the pause is over; the run ended without a result
        return AgentOutcome(
            run_id=state.run_id,
            run_dir=state.run_dir,
            status="stopped",
            stop_reason=final.get("stop_reason"),
            pending=None,
            manifest=None,
        )
    manifest = RunManifest.model_validate_json(state.manifest_path.read_text(encoding="utf-8"))
    outcome = AgentOutcome(
        run_id=state.run_id,
        run_dir=state.run_dir,
        status="completed",
        stop_reason=final.get("stop_reason"),
        pending=None,
        manifest=manifest,
    )
    next_spec = final.get("next_spec")
    if options.mode == "autopilot" and next_spec and options.allow_followup:
        return _followup(
            outcome,
            options,
            SplitSpec.model_validate(next_spec),
            client_factory,
            progress,
            customize,
        )
    return outcome


def _write_paused(
    state: RunState,
    options: AgentOptions,
    exc: RunPausingError,
    values: dict[str, Any],
    deps: AgentDeps | None,
    carried: list[dict[str, Any]],
    outages: list[dict[str, Any]],
) -> PausedRunManifest:
    """Record the pause in ``run_manifest.json``; the checkpoint already holds the rest."""
    node = exc.node or (deps.current_node if deps is not None else None) or VERIFY_NODE
    partial: list[dict[str, Any]] = []
    if deps is not None and deps.session is not None:
        partial = [dataclasses.asdict(r) for r in deps.session.records[deps.node_start_index :]]
    interrupted = [*carried, *partial]
    calls = [*values.get("calls", []), *interrupted]
    now = datetime.now(UTC)
    model = options.pinned_model or llm_config.DEFAULT_MODEL
    command = resume_command(state.run_id, state.run_dir.parent)
    done = list(values.get("nodes_done", []))
    where = (
        f"The run is checkpointed in {state.run_dir}"
        + (f" after node '{done[-1]}'" if done else " before its first node")
        + "; nothing completed is lost and the sealed holdout is untouched."
    )
    kind, status_text, message = _pause_text(exc, model, node, where, command)
    outage = LLMOutage(
        kind=kind,
        node=node,
        status=exc.status,
        retries=exc.retries,
        at=now.isoformat(),
        message=str(exc),
    )
    record = PausedRunManifest(
        status=status_text,
        run_id=state.run_id,
        started_at=datetime.fromisoformat(values["started_at"]),
        paused_at=now,
        target=options.target,
        goal_text=values.get("goal", options.goal),
        run_seed=options.seed,
        model=model,
        model_overridden=options.model_overridden,
        outage=outage,
        earlier_outages=[LLMOutage.model_validate(o) for o in outages],
        message=message,
        resume_command=command,
        completed_nodes=done,
        usage_by_node=usage_by_node(calls),
        total_inr=sum(c["inr"] for c in calls),
        retries=sum(int(c.get("retries", 0)) for c in calls),
        interrupted_calls=interrupted,
        decisions=[Decision.model_validate(d) for d in values.get("decisions", [])],
        rounds=[RoundRecord.model_validate(r) for r in values.get("rounds", [])],
        environment=capture_environment(),
    )
    write_manifest(state.manifest_path, record)
    return record


def _pause_text(
    exc: RunPausingError, model: str, node: str, where: str, command: str
) -> tuple[PauseKind, PausedStatus, str]:
    """(outage kind, manifest status, the one-paragraph message) for a pause."""
    if isinstance(exc, BudgetExceededError):
        if exc.scope == "run":
            limit = (
                f"would take this run's spend from Rs {exc.run_spent_inr:.2f} past the per-run "
                f"cap of Rs {exc.cap_inr:g}"
            )
            raise_it = "To allow more for this run, raise RUN_CAP_INR in holdout/llm/config.py"
        elif exc.scope == "reserve":
            limit = (
                f"would take project spend from Rs {exc.project_spent_inr:.2f} into the "
                f"benchmark reserve (spend other than the benchmark may reach Rs "
                f"{exc.cap_inr:g})"
            )
            raise_it = (
                "The reserve keeps money for the final benchmark run. Lower "
                "BENCHMARK_RESERVE_INR in holdout/llm/config.py only as a deliberate, "
                "documented decision"
            )
        else:
            limit = (
                f"would take project spend from Rs {exc.project_spent_inr:.2f} past the "
                f"project cap of Rs {exc.cap_inr:g}"
            )
            raise_it = (
                "To continue, check your prepaid balance with 'holdout spend' and, only if "
                f"it covers more, raise the cap for this shell with "
                f'$env:{llm_config.PROJECT_CAP_ENV} = "<rupees>"'
            )
        message = (
            f"Budget cap reached: the next Gemini call in node '{node}' could cost up to "
            f"Rs {exc.estimate_inr:.2f}, which {limit}, so it was not sent. Spend so far: "
            f"this run Rs {exc.run_spent_inr:.2f}, project Rs {exc.project_spent_inr:.2f}. "
            f"{where} {raise_it}, then resume with: {command}"
        )
        return "budget_cap", PAUSED_BUDGET, message
    status = "a timeout" if exc.status is None else f"HTTP {exc.status}"
    if isinstance(exc, LLMBillingError):
        message = (
            f"Gemini ({model}) refused the call in node '{node}' for billing or quota "
            f"reasons ({status}): the prepaid credit may be used up. {where} Top up in "
            "Google AI Studio, record the new balance with 'holdout spend --set-balance "
            f"<rupees>', then resume with: {command}"
        )
        return "billing", PAUSED_BILLING, message
    message = (
        f"Gemini ({model}) stayed unavailable: the call in node '{node}' failed with "
        f"{status} on every attempt ({exc.retries} retries with backoff). {where} When the "
        f"service recovers, resume with: {command}"
    )
    return "llm_unavailable", PAUSED_LLM, message


def _followup(
    first: AgentOutcome,
    options: AgentOptions,
    spec: SplitSpec,
    client_factory: ClientFactory | None,
    progress: Progress,
    customize: Customize | None = None,
) -> AgentOutcome:
    """Exactly one new run with the corrected split (same model); it carries the caveat."""
    column = spec.group_col or spec.time_col
    if column is None:
        raise AgentError("a follow-up run needs a group or time column")
    followup_options = replace(
        options,
        split=f"{spec.strategy.value}:{column}",
        entities=spec.entity_scope.value if spec.entity_scope is not None else options.entities,
        previous_run_id=first.run_id,
        allow_followup=False,
    )
    second = run_agent(
        followup_options, client_factory=client_factory, progress=progress, customize=customize
    )
    assert first.manifest is not None
    if first.manifest.agent is not None:
        linked = first.manifest.model_copy(
            update={"agent": first.manifest.agent.model_copy(update={"next_run_id": second.run_id})}
        )
        write_manifest(first.run_dir / "run_manifest.json", linked)
        write_report(first.run_dir / "report.md", linked)
        first = replace(first, manifest=linked)
    return replace(first, followup=second)
