"""LLM outages: a run pauses cleanly (checkpoint + paused manifest, exit code 5), resumes
on the same model, counts its retries, and never switches models."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from bench.traps import build_case
from holdout.agent import driver
from holdout.agent.driver import PAUSED_LLM, resume_agent
from holdout.agent.graph import AgentOptions
from holdout.cli import app
from holdout.core.errors import AgentError, HoldoutError, LLMConfigError
from holdout.llm import config as llm_config
from holdout.llm.client import FakeLLM, GeminiClient
from holdout.llm.progress import LineProgress
from holdout.llm.retry import TransientLLMError
from holdout.runner import reaudit_run
from tests.agent_harness import fake_for, run_case, write_case

runner = CliRunner()


def _down(working: FakeLLM, prompt: str = "plan", failures: int = 6, **kw: Any) -> FakeLLM:
    return FakeLLM(
        default=working.default,
        scripts={prompt: [TransientLLMError(503) for _ in range(failures)]},
        **kw,
    )


def test_an_outage_pauses_with_a_checkpoint_and_resume_completes(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))
    outcome = run_case(case, tmp_path, llm=_down(working))

    assert outcome.status == PAUSED_LLM
    paused = outcome.paused
    assert paused is not None
    assert (paused.outage.node, paused.outage.status, paused.outage.retries) == ("plan", 503, 5)
    assert paused.completed_nodes == ["intake", "confirm_split", "seal"]
    assert paused.usage_by_node["plan"].retries == 5
    assert paused.usage_by_node["plan"].inr == 0.0  # retries cost no tokens
    expected = f"holdout agent --resume {outcome.run_id} --out {outcome.run_dir.parent}"
    assert paused.resume_command == expected
    assert expected in paused.message
    assert "node 'plan'" in paused.message
    assert "HTTP 503" in paused.message
    raw = json.loads((outcome.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert raw["status"] == "paused_llm_unavailable"
    assert (outcome.run_dir / "checkpoints.sqlite").is_file()
    with pytest.raises(HoldoutError, match="paused_llm_unavailable"):
        reaudit_run(outcome.run_dir)

    lines: list[str] = []
    resumed = resume_agent(
        outcome.run_dir, client_factory=lambda: working, progress=LineProgress(lines.append)
    )
    assert resumed.status == "completed"
    manifest = resumed.manifest
    assert manifest is not None
    assert manifest.agent is not None
    agent = manifest.agent
    assert [o.node for o in agent.outages] == ["plan"]
    assert agent.retries == 5
    assert agent.usage_by_node["plan"].calls == 2  # the failed call and the successful one
    assert agent.usage_by_node["plan"].retries == 5
    assert any("A1 critical" in f and "refund_issued" in f for f in agent.rounds[0].flags)
    # Resumed after the seal: the remaining nodes ran once, sealing did not run again.
    assert "node seal: start" not in "\n".join(lines)
    assert any("node plan: start" in line for line in lines)
    report = (resumed.run_dir / "report.md").read_text(encoding="utf-8")
    assert "**Paused**" in report
    assert "5 transient-error retries" in report


@pytest.mark.full_only
def test_retries_that_clear_are_counted_in_the_manifest(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))
    outcome = run_case(case, tmp_path, llm=_down(working, prompt="report", failures=2))
    assert outcome.manifest is not None
    assert outcome.manifest.agent is not None
    assert outcome.manifest.agent.retries == 2
    assert outcome.manifest.agent.usage_by_node["report"].retries == 2
    assert outcome.manifest.agent.outages == []


@pytest.mark.full_only
def test_a_run_is_pinned_to_one_model(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))
    other = FakeLLM(default=working.default, model_name="gemini-3.6-flash")
    down = _down(working, model_name="gemini-3.6-flash")
    outcome = run_case(case, tmp_path, llm=down, model="gemini-3.6-flash")
    assert outcome.status == PAUSED_LLM
    assert outcome.paused is not None
    assert (outcome.paused.model, outcome.paused.model_overridden) == ("gemini-3.6-flash", True)
    with pytest.raises(AgentError, match="never switches models"):
        resume_agent(outcome.run_dir, model="gemini-3.8-flash", client_factory=lambda: working)
    with pytest.raises(AgentError, match="never switches models"):
        resume_agent(outcome.run_dir, client_factory=lambda: working)  # serves 3.8
    resumed = resume_agent(outcome.run_dir, client_factory=lambda: other)
    assert resumed.manifest is not None
    assert resumed.manifest.agent is not None
    assert resumed.manifest.agent.model == "gemini-3.6-flash"
    assert resumed.manifest.agent.model_overridden
    assert "Model override" in (resumed.run_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.full_only
def test_an_outage_while_verifying_the_model_pauses_before_the_first_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))

    def listing() -> list[Any]:
        raise TransientLLMError(503)

    sdk = SimpleNamespace(models=SimpleNamespace(list=listing))

    class _Gemini(GeminiClient):
        def __init__(self, model: str) -> None:
            super().__init__(model, sdk_client=sdk, sleep=lambda s: None)

    monkeypatch.setattr(driver, "GeminiClient", _Gemini)
    csv = tmp_path / "data.csv"
    outcome = driver.run_agent(
        AgentOptions(csv=str(csv), target=case.target, out_dir=str(tmp_path / "runs"))
    )
    assert outcome.status == PAUSED_LLM
    assert outcome.paused is not None
    assert outcome.paused.outage.node == "verify_model"
    assert outcome.paused.completed_nodes == []
    assert "before its first node" in outcome.paused.message
    resumed = resume_agent(outcome.run_dir, mode="autopilot", client_factory=lambda: working)
    assert resumed.status == "completed"
    assert resumed.manifest is not None
    assert resumed.manifest.agent is not None
    assert [o.node for o in resumed.manifest.agent.outages] == ["verify_model"]


def test_the_smoke_script_reports_what_completed_when_gemini_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import smoke_gemini

    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))
    models = [SimpleNamespace(name=f"models/{llm_config.DEFAULT_MODEL}")]
    sdk = SimpleNamespace(models=SimpleNamespace(list=lambda: models))

    class _Gemini(GeminiClient):
        def __init__(self, model: str) -> None:
            super().__init__(model, sdk_client=sdk, sleep=lambda s: None)

    def fake_run(options: AgentOptions, **kw: Any) -> driver.AgentOutcome:
        return driver.run_agent(
            replace(options, csv=str(tmp_path / "data.csv"), target=case.target),
            client_factory=lambda: _down(working),
            **kw,
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(smoke_gemini, "GeminiClient", _Gemini)
    monkeypatch.setattr(smoke_gemini, "run_agent", fake_run)
    monkeypatch.setattr("sys.argv", ["smoke_gemini", "--out", str(tmp_path / "smoke")])
    assert smoke_gemini.main() == 5
    printed = capsys.readouterr().out
    assert printed.startswith("before: project spend Rs")
    assert "\nafter:  project spend Rs" in printed
    assert "node verify_model: done in" in printed
    assert "HTTP 503 on attempt 5; retrying in" in printed  # progress shown by default
    assert "== churn example: paused_llm_unavailable" in printed
    assert "completed nodes: intake, confirm_split, seal" in printed
    assert "holdout agent --resume" in printed
    assert "skipped (Gemini unavailable): direct_leak trap" in printed


def test_the_smoke_script_exits_5_when_the_model_list_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import smoke_gemini

    def listing() -> list[Any]:
        raise TransientLLMError(503)

    sdk = SimpleNamespace(models=SimpleNamespace(list=listing))

    class _Gemini(GeminiClient):
        def __init__(self, model: str) -> None:
            super().__init__(model, sdk_client=sdk, sleep=lambda s: None)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(smoke_gemini, "GeminiClient", _Gemini)
    monkeypatch.setattr("sys.argv", ["smoke_gemini", "--model", "gemini-3.6-flash"])
    assert smoke_gemini.main() == 5
    printed = capsys.readouterr().out
    assert "Gemini is unavailable" in printed
    assert "No run was started" in printed


def test_an_unpriced_model_is_refused_before_any_run(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    with pytest.raises(LLMConfigError, match="no price"):
        run_case(case, tmp_path, model="gemini-9-ultra")
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_the_cli_exits_5_with_a_resume_command_and_no_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_case("direct_leak", 0)
    csv = write_case(case, tmp_path)
    working = fake_for(case, csv)
    clients = {"current": _down(working)}

    class _Client:  # stands in for GeminiClient(model=...) inside the driver
        def __new__(cls, model: str) -> FakeLLM:  # type: ignore[misc]
            assert model == clients["current"].model
            return clients["current"]

    monkeypatch.setattr(driver, "GeminiClient", _Client)
    out = tmp_path / "runs"
    base = ["agent", "--csv", str(csv), "--target", case.target, "--autopilot", "--out", str(out)]
    result = runner.invoke(app, [*base, "--no-cache", "--verbose"])
    assert result.exit_code == 5, result.output
    assert "Traceback" not in result.output
    assert "LLM unavailable - run paused." in result.output
    assert "node 'plan'" in result.output
    assert "HTTP 503" in result.output
    assert "retrying in" in result.output  # --verbose shows each retry wait
    assert "node seal: done in" in result.output
    run_id = next(p for p in out.iterdir() if p.is_dir()).name
    assert f"holdout agent --resume {run_id} --out {out}" in " ".join(result.output.split())

    mismatch = runner.invoke(
        app, ["agent", "--resume", run_id, "--out", str(out), "--model", "gemini-3.6-flash"]
    )
    assert mismatch.exit_code == 2
    assert "never switches models" in " ".join(mismatch.output.split())

    clients["current"] = working
    resumed = runner.invoke(app, ["agent", "--resume", run_id, "--out", str(out), "--autopilot"])
    assert resumed.exit_code == 0, resumed.output
    assert "5 retries" in resumed.output
