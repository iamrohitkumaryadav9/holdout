"""Command-line interface: ``holdout run`` and ``holdout make-example``."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from holdout.core.errors import HoldoutError
from holdout.core.schemas import DataType, Direction, Task
from holdout.data.examples import write_examples
from holdout.data.split import SplitProposal, SplitSpec
from holdout.runner import RunOptions, RunOutcome, execute_run

app = typer.Typer(
    name="holdout",
    help="A self-auditing data-science agent: the holdout is sealed before anything sees it.",
    no_args_is_help=True,
    add_completion=False,
)
# soft_wrap: never hard-wrap long paths, so printed paths stay copy-pasteable.
console = Console(highlight=False, soft_wrap=True)


class TaskOption(StrEnum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class _ConsoleObserver:
    def on_stage(self, message: str) -> None:
        console.print(f"[dim]-> {message}[/dim]")

    def on_split(self, proposal: SplitProposal, spec: SplitSpec) -> None:
        column = spec.group_col or spec.time_col
        detail = f" on {column!r}" if column else ""
        console.print(
            f"[bold]Split:[/bold] {spec.strategy.value}{detail}, test_size={spec.test_size}"
        )
        if spec != proposal.spec:
            console.print(f"  (user override; proposed was {proposal.spec.strategy.value})")
        for reason in proposal.reasons:
            console.print(f"  reason: {reason}")
        for warning in proposal.warnings:
            console.print(f"  [yellow]warning:[/yellow] {warning}")
        for alternative in proposal.alternatives:
            alt_column = alternative.time_col or alternative.group_col
            console.print(f"  alternative: --split {alternative.strategy.value}:{alt_column}")


@app.command()
def run(
    csv: Annotated[Path, typer.Option(help="Input CSV file.", exists=True, dir_okay=False)],
    target: Annotated[str, typer.Option(help="Target column name.")],
    task: Annotated[
        TaskOption, typer.Option(help="Task; auto infers it from the target.")
    ] = TaskOption.AUTO,
    data_type: Annotated[
        DataType,
        typer.Option(
            help=(
                "tabular, or timeseries: time-ordered tabular data (time-based split; "
                "forecasting is not supported yet)."
            )
        ),
    ] = DataType.TABULAR,
    split: Annotated[
        str, typer.Option(help="auto | random | stratified | group:COL | time:COL")
    ] = "auto",
    seed: Annotated[int, typer.Option(help="Run seed; every random state derives from it.")] = 42,
    plan: Annotated[
        Path | None,
        typer.Option(help="ModelPlan JSON; default plan if omitted.", exists=True, dir_okay=False),
    ] = None,
    unseal: Annotated[
        bool, typer.Option("--unseal", help="Evaluate the sealed holdout once, at the end.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Directory for run outputs.")] = Path("runs"),
) -> None:
    """Profile, split, seal, cross-validate, and (with --unseal) score the holdout once."""
    options = RunOptions(
        csv=csv,
        target=target,
        task=None if task is TaskOption.AUTO else Task(task.value),
        data_type=data_type,
        split=split,
        seed=seed,
        plan_path=plan,
        unseal=unseal,
        out_dir=out,
    )
    try:
        outcome = execute_run(options, observer=_ConsoleObserver())
    except HoldoutError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    _print_summary(outcome)


@app.command("make-example")
def make_example(
    out: Annotated[Path, typer.Option(help="Directory for the example CSVs.")] = Path("examples"),
    seed: Annotated[int, typer.Option(help="Generator seed.")] = 0,
) -> None:
    """Write two synthetic CSVs: churn classification and time-ordered store sales."""
    classification, timeseries = write_examples(out, seed=seed)
    console.print(f"wrote {classification}  (target: churned, classification)")
    console.print(f"wrote {timeseries}  (target: units_sold, time-ordered regression)")


def _print_summary(outcome: RunOutcome) -> None:
    manifest = outcome.manifest
    for warning in manifest.split.reuse_warnings:
        console.print(f"[yellow]ledger warning:[/yellow] {warning}")
    for issue in outcome.plan_issues:
        console.print(f"[yellow]plan {issue.severity.value}:[/yellow] {issue.message}")
    metric = f"{manifest.metric.value} ({manifest.direction.value} is better)"
    table = Table(title=f"Run {manifest.run_id}: {metric}")
    table.add_column("")
    table.add_column("score", justify="right")
    table.add_row(
        "Dummy baseline (CV)", f"{outcome.baseline.mean:.4f} +/- {outcome.baseline.std:.4f}"
    )
    table.add_row(
        f"Validation: {outcome.result.plan_label} (CV)",
        f"{outcome.result.mean:.4f} +/- {outcome.result.std:.4f}",
    )
    table.add_row(
        "Holdout (single evaluation)",
        "not unsealed" if outcome.holdout is None else f"{outcome.holdout.score:.4f}",
    )
    gap = manifest.optimism_gap
    table.add_row("Optimism gap (validation - holdout)", "n/a" if gap is None else f"{gap:+.4f}")
    console.print(table)
    if gap is not None and manifest.direction is Direction.LOWER:
        console.print("[dim]lower is better: a negative gap means validation was optimistic[/dim]")
    if outcome.holdout is None:
        console.print("[dim]holdout still sealed; re-run with --unseal to evaluate it once[/dim]")
    console.print(f"artifacts: {outcome.state.run_dir}")
    for name in ("pipeline.py", "run_manifest.json", "report.md"):
        console.print(f"  {outcome.state.run_dir / name}")


if __name__ == "__main__":
    app()
