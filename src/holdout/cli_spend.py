"""``holdout spend``: Gemini spend from the project ledger, caps and prepaid balance."""

from datetime import date, datetime
from typing import Annotated

import typer

from holdout.cli import app, console
from holdout.core.errors import HoldoutError
from holdout.llm.budget import SpendLedger
from holdout.llm.spend import build_report, render


@app.command()
def spend(
    set_balance: Annotated[
        float | None,
        typer.Option(
            "--set-balance",
            help="Record the prepaid balance (rupees) Google AI Studio shows now.",
        ),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            help="Date the balance was shown (YYYY-MM-DD; default today). Spend from the "
            "start of that day is subtracted.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Show Gemini spend: today, 7 days, all time, per run and model, caps and balance."""
    ledger = SpendLedger()
    try:
        if set_balance is not None:
            day = date.fromisoformat(as_of) if as_of else datetime.now().astimezone().date()
            if day > datetime.now().astimezone().date():
                raise HoldoutError(f"--as-of {day} is in the future")
            ledger.record_balance(set_balance, day)
            if not as_json:
                console.print(
                    f"Recorded a prepaid balance of Rs {set_balance:.2f} as of {day}.",
                    highlight=False,
                )
        elif as_of is not None:
            raise HoldoutError("--as-of only applies with --set-balance")
        report = build_report(ledger)
    except (HoldoutError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}", highlight=False)
        raise typer.Exit(code=2) from None
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    for line in render(report):
        console.print(line, markup=False, highlight=False)
    if report.balance is not None and report.balance.warning is not None:
        console.print(report.balance.warning, style="bold red", highlight=False)
