"""``holdout agent``: the LLM agent (or its offline twin) from the command line.

Exit codes: 0 done; 2 error; 3 unsealing was requested and refused; 4 paused at a question
with no terminal to ask it on; 5 paused because the LLM stayed unavailable after every
retry (reserved for that). Resume either pause with ``holdout agent --resume RUN_ID``.
"""

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.prompt import Confirm, Prompt

from holdout.agent.driver import PAUSED_LLM, AgentOutcome, resume_agent, run_agent
from holdout.agent.graph import AgentOptions
from holdout.agent.schemas import Decision
from holdout.agent.tracing import configure_tracing
from holdout.cli import EXIT_UNSEAL_REFUSED, TaskOption, _print_audit, _print_summary, app, console
from holdout.core.errors import HoldoutError
from holdout.core.schemas import DataType, EntityScope
from holdout.llm.progress import LineProgress

EXIT_PAUSED = 4
EXIT_LLM_UNAVAILABLE = 5


class RichAsker:
    """Answers interrupts at a terminal with rich prompts."""

    def ask(self, payload: dict[str, Any]) -> Decision:
        kind = payload["kind"]
        if kind == "split":
            return self._split(payload)
        if kind == "review":
            return self._review(payload)
        return self._unseal(payload)

    def _split(self, payload: dict[str, Any]) -> Decision:
        spec = payload["spec"]
        column = spec.get("group_col") or spec.get("time_col")
        console.print(
            f"[bold]Proposed split:[/bold] {spec['strategy']}{f' on {column!r}' if column else ''}"
        )
        for reason in payload["reasons"]:
            console.print(f"  reason: {reason}")
        for warning in payload["warnings"]:
            console.print(f"  [yellow]warning:[/yellow] {warning}")
        detail: dict[str, Any] = {}
        if payload.get("declared_entity_scope") is None and any(
            "--entities" in w for w in payload["warnings"]
        ):
            detail["entity_scope"] = Prompt.ask(
                "Predictions for new entities or entities already in the data?",
                choices=["new", "known"],
                default="new",
            )
        if Confirm.ask("Use this split?", default=True):
            return Decision(
                kind="split",
                mode="interactive",
                accepted=True,
                detail=detail,
                reason="accepted by the user",
            )
        override = Prompt.ask(
            "Split to use instead (random, stratified, group:COL, time:COL) or 'stop'"
        )
        if override.strip() == "stop":
            return Decision(
                kind="split",
                mode="interactive",
                accepted=False,
                detail={},
                reason="rejected by the user",
            )
        detail["split"] = override.strip()
        return Decision(
            kind="split",
            mode="interactive",
            accepted=True,
            detail=detail,
            reason="split chosen by the user",
        )

    def _review(self, payload: dict[str, Any]) -> Decision:
        console.print("[bold]The audit proposes fixes:[/bold]")
        for flag in payload["flags"]:
            console.print(f"  {flag}")
        for key, text in payload["explanations"].items():
            console.print(f"  [dim]{key}:[/dim] {text}")
        before = {d["column"] for d in payload["current_plan"]["drop_columns"]}
        after = {d["column"] for d in payload["proposed_plan"]["drop_columns"]}
        if after - before:
            console.print(f"  drop: {', '.join(sorted(after - before))}")
        old_metric = payload["current_plan"]["metric"]
        new_metric = payload["proposed_plan"]["metric"]
        if old_metric != new_metric:
            console.print(f"  metric: {old_metric} -> {new_metric}")
        accepted = Confirm.ask("Apply these fixes and re-run the audit?", default=True)
        return Decision(
            kind="review",
            mode="interactive",
            accepted=accepted,
            detail={},
            reason="answered by the user",
        )

    def _unseal(self, payload: dict[str, Any]) -> Decision:
        counts = payload["counts"]
        console.print(
            f"[bold]Audit:[/bold] {counts['critical']} critical, {counts['warning']} warning"
        )
        for blocker in payload["blocked_by"]:
            console.print(f"  [yellow]{blocker}[/yellow]")
        if payload["holdout_invalid"] or not Confirm.ask(
            "Evaluate the sealed holdout now (once)?", default=False
        ):
            return Decision(
                kind="unseal",
                mode="interactive",
                accepted=False,
                detail={"requested": False},
                reason="declined by the user",
            )
        ack = counts["warning"] == 0 or Confirm.ask("Acknowledge the warnings?", default=False)
        risk = None
        if counts["critical"]:
            text = Prompt.ask(
                "Critical flags are open. Reason to accept the risk (empty = no)", default=""
            )
            risk = text.strip() or None
        return Decision(
            kind="unseal",
            mode="interactive",
            accepted=True,
            detail={"requested": True, "ack_warnings": ack, "accept_risk": risk},
            reason="answered by the user",
        )


@app.command()
def agent(
    csv: Annotated[Path | None, typer.Option(help="Input CSV file.", dir_okay=False)] = None,
    target: Annotated[str | None, typer.Option(help="Target column name.")] = None,
    goal: Annotated[str | None, typer.Option(help="What you want to learn, in words.")] = None,
    task: Annotated[TaskOption, typer.Option(help="Task; auto infers it.")] = TaskOption.AUTO,
    data_type: Annotated[DataType, typer.Option(help="tabular or timeseries.")] = DataType.TABULAR,
    split: Annotated[
        str, typer.Option(help="auto | random | stratified | group:COL | time:COL")
    ] = "auto",
    seed: Annotated[int, typer.Option(help="Run seed (also the LLM generation seed).")] = 42,
    entities: Annotated[EntityScope | None, typer.Option(help="new or known entities.")] = None,
    autopilot: Annotated[
        bool,
        typer.Option("--autopilot", help="Answer every question by the logged autopilot policy."),
    ] = False,
    offline: Annotated[
        bool, typer.Option("--offline", help="No API calls: default plan, audits, template report.")
    ] = False,
    eda: Annotated[
        bool,
        typer.Option("--eda", help="Let the model run up to four sandboxed analysis snippets."),
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Do not use cached LLM responses.")
    ] = False,
    trace: Annotated[
        bool, typer.Option("--trace", help="Print OpenTelemetry spans to the console.")
    ] = False,
    resume: Annotated[
        str | None, typer.Option(help="Resume a paused or interrupted run by its id.")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            help="Gemini model for this run only (must be priced in holdout.llm.config); "
            "recorded in the manifest. A run never switches models, also not on --resume."
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", help="Print a timestamped line per graph node, LLM call and retry."
        ),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Show Python tracebacks for errors.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Directory for run outputs.")] = Path("runs"),
) -> None:
    """Run the agent: the model proposes, Holdout's code validates, audits and decides."""
    configure_tracing(console=trace)
    asker = RichAsker() if sys.stdin.isatty() else None
    progress = (
        LineProgress(lambda line: console.print(line, markup=False, highlight=False))
        if verbose
        else None
    )
    try:
        if resume is not None:
            outcome = resume_agent(
                out / resume,
                mode="autopilot" if autopilot else None,
                model=model,
                asker=asker,
                progress=progress,
            )
        else:
            if csv is None or target is None:
                raise HoldoutError("--csv and --target are required (or --resume RUN_ID)")
            if not csv.is_file():
                raise HoldoutError(f"CSV not found: {csv}")
            options = AgentOptions(
                csv=str(csv),
                target=target,
                out_dir=str(out),
                goal=goal,
                task=None if task is TaskOption.AUTO else task.value,
                data_type=data_type.value,
                split=split,
                seed=seed,
                entities=entities.value if entities is not None else None,
                mode="autopilot" if autopilot else "interactive",
                offline=offline,
                eda=eda,
                use_cache=not no_cache,
                model=model,
            )
            outcome = run_agent(options, asker=asker, progress=progress)
    except HoldoutError as exc:
        if debug:
            raise
        console.print(f"[red]error:[/red] {exc}", markup=True, highlight=False)
        raise typer.Exit(code=2) from None
    raise typer.Exit(code=_report(outcome))


def _report(outcome: AgentOutcome) -> int:
    if outcome.status == PAUSED_LLM and outcome.paused is not None:
        console.print("LLM unavailable - run paused.", style="bold yellow", highlight=False)
        console.print(outcome.paused.message, markup=False, highlight=False)
        return EXIT_LLM_UNAVAILABLE
    if outcome.status == "paused":
        kind = (outcome.pending or {}).get("kind", "?")
        console.print(
            f"[yellow]Paused at the {kind} question[/yellow] (no terminal to ask on). "
            "Resume with:\n"
            f"  holdout agent --resume {outcome.run_id} --out {outcome.run_dir.parent}"
        )
        return EXIT_PAUSED
    if outcome.status == "stopped" or outcome.manifest is None:
        console.print(f"[yellow]Stopped:[/yellow] {outcome.stop_reason}")
        return 0
    manifest = outcome.manifest
    _print_audit(manifest.audit)
    _print_summary(manifest, outcome.run_dir)
    record = manifest.agent
    if record is not None:
        console.print(
            f"agent: {record.mode}, model {record.model or 'none (offline)'}"
            f"{' (--model)' if record.model_overridden else ''}, "
            f"Rs {record.total_inr:.2f}, {record.cache_hits} cached, {record.retries} retries, "
            f"{len(record.substitutions)} substitution(s); stop: {record.stop_reason}"
        )
    code = 0
    if manifest.unseal.requested and not manifest.unseal.allowed:
        code = EXIT_UNSEAL_REFUSED
    if outcome.followup is not None:
        console.print(
            f"[bold]Started a new run with the corrected split:[/bold] {outcome.followup.run_id}"
        )
        code = max(code, _report(outcome.followup))
    return code
