"""Run and resume the agent graph; answer interrupts by policy, by a person, or pause.

* ``run_agent`` allocates the run directory (with its SQLite checkpoint database and the
  options needed to resume), then drives the graph.
* Interrupts: autopilot answers with :func:`autopilot_decision`; interactive mode asks the
  given :class:`Asker`; with no asker (no terminal) the run pauses (CLI exit code 4) and
  ``resume_agent`` continues it later, in a new process if need be.
* Autopilot runs that end because the split leaks start exactly one follow-up run with the
  corrected split; the two manifests link to each other.
"""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from holdout.agent.graph import AgentDeps, AgentOptions, AgentState, build_graph
from holdout.agent.planner import LLMPlanner, OfflinePlanner, Planner
from holdout.agent.policy import Asker, autopilot_decision
from holdout.core.errors import AgentError, HoldoutError
from holdout.core.state import RunState
from holdout.data.split import SplitSpec
from holdout.export.manifest import RunManifest, write_manifest, write_report
from holdout.llm.budget import Budget
from holdout.llm.cache import ResponseCache
from holdout.llm.client import GeminiClient, LLMClient
from holdout.llm.structured import LLMSession

ClientFactory = Callable[[], LLMClient]


@dataclass(frozen=True)
class AgentOutcome:
    run_id: str
    run_dir: Path
    status: str  # completed | paused | stopped
    stop_reason: str | None
    pending: dict[str, Any] | None
    manifest: RunManifest | None
    followup: "AgentOutcome | None" = None


def run_agent(
    options: AgentOptions,
    *,
    client_factory: ClientFactory | None = None,
    asker: Asker | None = None,
    after_node: Callable[[str], None] | None = None,
) -> AgentOutcome:
    """Start a new agent run."""
    started = datetime.now(UTC)
    state = RunState.create(Path(options.out_dir), options.seed, now=started)
    state.agent_options_path.write_text(json.dumps(options.to_json(), indent=2), encoding="utf-8")
    initial: AgentState = {
        "options": options.to_json(),
        "run_id": state.run_id,
        "started_at": started.isoformat(),
    }
    return _drive(state, options, initial, client_factory, asker, after_node)


def resume_agent(
    run_dir: Path,
    *,
    mode: str | None = None,
    client_factory: ClientFactory | None = None,
    asker: Asker | None = None,
    after_node: Callable[[str], None] | None = None,
) -> AgentOutcome:
    """Continue a paused or interrupted run from its last checkpoint."""
    options_path = run_dir / "agent_options.json"
    if not options_path.is_file():
        raise HoldoutError(f"{run_dir} is not an agent run (no agent_options.json)")
    options = AgentOptions(**json.loads(options_path.read_text(encoding="utf-8")))
    if mode is not None:
        options = replace(options, mode=mode)
    state = RunState(
        run_id=run_dir.name,
        out_dir=run_dir.parent,
        run_seed=options.seed,
        created_at=datetime.now(UTC),
    )
    return _drive(state, options, None, client_factory, asker, after_node)


def _deps(
    options: AgentOptions, run_id: str, client_factory: ClientFactory | None, spent_inr: float
) -> AgentDeps:
    if options.offline:
        return AgentDeps(planner=OfflinePlanner(), session=None, llm_client="none")
    client = client_factory() if client_factory is not None else GeminiClient()
    verified = False
    if isinstance(client, GeminiClient):
        client.verify_model()
        verified = client.model_verified
    session = LLMSession(
        client=client,
        budget=Budget(run_id=run_id, model=client.model, spent_run_inr=spent_inr),
        seed=options.seed,
        cache=ResponseCache(enabled=options.use_cache),
    )
    planner: Planner = LLMPlanner(session)
    return AgentDeps(
        planner=planner, session=session, llm_client=type(client).__name__, model_verified=verified
    )


def _drive(
    state: RunState,
    options: AgentOptions,
    initial: AgentState | None,
    client_factory: ClientFactory | None,
    asker: Asker | None,
    after_node: Callable[[str], None] | None,
) -> AgentOutcome:
    connection = sqlite3.connect(state.checkpoint_path, check_same_thread=False)
    try:
        config: Any = {"configurable": {"thread_id": state.run_id}}
        saver = SqliteSaver(connection)
        spent = 0.0
        if initial is None:
            snapshot = build_graph(
                AgentDeps(planner=OfflinePlanner(), session=None, llm_client="none")
            )
            existing = snapshot.compile(checkpointer=saver).get_state(config).values
            if not existing:
                raise HoldoutError(f"run {state.run_id} has no checkpoint to resume")
            spent = sum(c["inr"] for c in existing.get("calls", []))
        deps = _deps(options, state.run_id, client_factory, spent)
        deps.after_node = after_node
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
    finally:
        connection.close()
    if final.get("status") == "stopped":
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
        return _followup(outcome, options, SplitSpec.model_validate(next_spec), client_factory)
    return outcome


def _followup(
    first: AgentOutcome,
    options: AgentOptions,
    spec: SplitSpec,
    client_factory: ClientFactory | None,
) -> AgentOutcome:
    """Exactly one new run with the corrected split; it carries the re-drawn caveat."""
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
    second = run_agent(followup_options, client_factory=client_factory)
    assert first.manifest is not None
    if first.manifest.agent is not None:
        linked = first.manifest.model_copy(
            update={"agent": first.manifest.agent.model_copy(update={"next_run_id": second.run_id})}
        )
        write_manifest(first.run_dir / "run_manifest.json", linked)
        write_report(first.run_dir / "report.md", linked)
        first = replace(first, manifest=linked)
    return replace(first, followup=second)
