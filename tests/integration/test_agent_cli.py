"""``holdout agent`` from the command line, and the interactive prompts."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from holdout import cli_agent
from holdout.cli import app
from holdout.data.examples import CLASSIFICATION_FILENAME, write_examples

runner = CliRunner()


@pytest.fixture(scope="module")
def churn_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("examples")
    write_examples(out)
    return out / CLASSIFICATION_FILENAME


def test_offline_autopilot_runs_end_to_end(churn_csv: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "agent",
            "--csv",
            str(churn_csv),
            "--target",
            "churned",
            "--offline",
            "--autopilot",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "agent: autopilot, model none (offline), Rs 0.00" in result.output
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Agent" in report
    assert "**Substitutions:** 0" in report


def test_without_a_terminal_an_interactive_run_pauses_then_resumes(
    churn_csv: Path, tmp_path: Path
) -> None:
    paused = runner.invoke(
        app,
        [
            "agent",
            "--csv",
            str(churn_csv),
            "--target",
            "churned",
            "--offline",
            "--out",
            str(tmp_path),
        ],
    )
    assert paused.exit_code == cli_agent.EXIT_PAUSED, paused.output
    assert "Paused at the split question" in paused.output
    run_id = next(p for p in tmp_path.iterdir() if p.is_dir()).name
    resumed = runner.invoke(
        app, ["agent", "--resume", run_id, "--out", str(tmp_path), "--autopilot"]
    )
    assert resumed.exit_code == 0, resumed.output
    assert (tmp_path / run_id / "run_manifest.json").is_file()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["agent", "--offline"], "--csv and --target are required"),
        (["agent", "--csv", "missing.csv", "--target", "y", "--offline"], "CSV not found"),
        (["agent", "--resume", "nope"], "not an agent run"),
    ],
)
def test_agent_errors_exit_with_code_2(args: list[str], message: str, tmp_path: Path) -> None:
    result = runner.invoke(app, [*args, "--out", str(tmp_path)])
    assert result.exit_code == 2
    assert message in result.output


def test_unseal_refusal_exits_with_code_3(churn_csv: Path, tmp_path: Path) -> None:
    from holdout.agent.driver import run_agent
    from holdout.agent.graph import AgentOptions
    from holdout.audit.gate import UnsealDecision
    from holdout.cli import EXIT_UNSEAL_REFUSED

    outcome = run_agent(
        AgentOptions(
            csv=str(churn_csv),
            target="churned",
            out_dir=str(tmp_path),
            mode="autopilot",
            offline=True,
        )
    )
    assert outcome.manifest is not None
    refused = UnsealDecision(
        requested=True,
        allowed=False,
        blocked_by=["critical flag(s): A1 x"],
        acknowledged_warnings=True,
        accepted_risk=None,
        overridden_criticals=[],
    )
    blocked = replace(outcome, manifest=outcome.manifest.model_copy(update={"unseal": refused}))
    assert cli_agent._report(blocked) == EXIT_UNSEAL_REFUSED
    assert cli_agent._report(outcome) == 0


# ------------------------------------------------------------------ interactive prompts


class _Answers:
    def __init__(self, answers: list[Any]) -> None:
        self.answers = answers

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.answers.pop(0)


def test_rich_asker_split_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "kind": "split",
        "spec": {"strategy": "stratified"},
        "reasons": ["classification target"],
        "warnings": ["'account' looks like ... (--entities new) ..."],
        "declared_entity_scope": None,
    }
    monkeypatch.setattr(cli_agent.Prompt, "ask", _Answers(["known", "group:account"]))
    monkeypatch.setattr(cli_agent.Confirm, "ask", _Answers([False]))
    decision = cli_agent.RichAsker().ask(payload)
    assert decision.detail == {"entity_scope": "known", "split": "group:account"}
    assert decision.mode == "interactive"
    monkeypatch.setattr(cli_agent.Prompt, "ask", _Answers(["new", "stop"]))
    monkeypatch.setattr(cli_agent.Confirm, "ask", _Answers([False]))
    assert not cli_agent.RichAsker().ask(payload).accepted


def test_rich_asker_review_and_unseal(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {"drop_columns": [], "metric": "accuracy"}
    review = {
        "kind": "review",
        "flags": ["A7 critical metric_mismatch converted"],
        "explanations": {"A7:converted": "accuracy hides the minority class"},
        "current_plan": plan,
        "proposed_plan": {"drop_columns": [{"column": "x"}], "metric": "balanced_accuracy"},
    }
    monkeypatch.setattr(cli_agent.Confirm, "ask", _Answers([True]))
    assert cli_agent.RichAsker().ask(review).accepted
    unseal = {
        "kind": "unseal",
        "counts": {"critical": 1, "warning": 1},
        "blocked_by": ["x"],
        "holdout_invalid": False,
    }
    monkeypatch.setattr(cli_agent.Confirm, "ask", _Answers([True, True]))
    monkeypatch.setattr(cli_agent.Prompt, "ask", _Answers(["approved for a demo by the owner"]))
    decision = cli_agent.RichAsker().ask(unseal)
    assert decision.detail == {
        "requested": True,
        "ack_warnings": True,
        "accept_risk": "approved for a demo by the owner",
    }
    monkeypatch.setattr(cli_agent.Confirm, "ask", _Answers([False]))
    assert cli_agent.RichAsker().ask(unseal).detail == {"requested": False}
