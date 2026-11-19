"""``holdout spend``: Gemini spend from the project ledger, caps and prepaid balance."""

from datetime import date, datetime
from typing import Annotated

import typer

from holdout.cli import app, console
from holdout.core.errors import HoldoutError
from holdout.llm.budget import SpendLedger
from holdout.llm.spend import build_report, render


def _as_of(text: str | None) -> date | datetime | None:
    """None (now), an exact datetime (naive = local time), or a bare date (start of day)."""
    if not text:
        return None
    if "T" in text or " " in text.strip():
        try:
            moment = datetime.fromisoformat(text.strip())
        except ValueError as exc:
            raise HoldoutError(f"--as-of {text!r} is not an ISO datetime") from exc
        moment = moment if moment.tzinfo is not None else moment.astimezone()
        if moment > datetime.now().astimezone():
            raise HoldoutError(f"--as-of {text} is in the future")
        return moment
    try:
        day = date.fromisoformat(text.strip())
    except ValueError as exc:
        raise HoldoutError(f"--as-of {text!r} is not an ISO date or datetime") from exc
    if day > datetime.now().astimezone().date():
        raise HoldoutError(f"--as-of {day} is in the future")
    return day


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
            help="When the balance was shown: an ISO datetime such as 2026-11-19T18:13 "
            "(local time unless an offset is given; default now). Spend strictly after it is "
            "subtracted. A bare date (YYYY-MM-DD) means the start of that day.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Show Gemini spend: today, 7 days, all time, per run and model, caps and balance."""
    ledger = SpendLedger()
    try:
        if set_balance is not None:
            when = _as_of(as_of)
            record = ledger.record_balance(set_balance, when)
            if not as_json:
                shown = (record.at.astimezone().isoformat(timespec="seconds") if record.at
                         else f"the start of {record.as_of}")  # fmt: skip
                console.print(
                    f"Recorded a prepaid balance of Rs {set_balance:.2f} as of {shown}; spend "
                    "after that is subtracted.",
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
