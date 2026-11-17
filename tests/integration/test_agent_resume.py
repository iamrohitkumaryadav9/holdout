"""Resume: a run killed mid-graph, or paused at a question, continues in a new process.

The kill test hard-exits the process (``os._exit``) right after the audit node, before its
checkpoint is written, then resumes in a fresh process. The final manifest must equal an
uninterrupted run's apart from timestamps, run ids and timings.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench.traps import build_case
from tests.agent_harness import run_case, write_case

ROOT = Path(__file__).resolve().parents[2]

VOLATILE = {
    "run_id",
    "started_at",
    "finished_at",
    "evaluated_at",
    "seconds",
    "fit_seconds",
    "timestamp",
    "environment",
    "previous_run_id",
    "next_run_id",
}

_RUN = """
import os, sys
from pathlib import Path
from bench.traps import build_case
from holdout.agent.driver import run_agent, resume_agent
from holdout.agent.graph import AgentOptions
from tests.agent_harness import fake_for
case = build_case("direct_leak", 0)
csv = Path(sys.argv[1]); out = Path(sys.argv[2]); action = sys.argv[3]
fake = fake_for(case, csv)
def crash(node):
    if node == "audit":
        os._exit(9)
if action == "kill":
    run_agent(AgentOptions(csv=str(csv), target=case.target, out_dir=str(out), goal=case.goal_text,
              mode="autopilot", use_cache=False), client_factory=lambda: fake, after_node=crash)
elif action == "pause":
    options = AgentOptions(csv=str(csv), target=case.target, out_dir=str(out),
                           goal=case.goal_text, mode="interactive", use_cache=False)
    outcome = run_agent(options, client_factory=lambda: fake)
    print(outcome.status, outcome.pending["kind"])
else:
    run_dir = next(p for p in out.iterdir() if p.is_dir())
    outcome = resume_agent(run_dir, mode="autopilot", client_factory=lambda: fake)
    print(outcome.status)
"""


def _python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _RUN, *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _normalised(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalised(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_normalised(v) for v in value]
    return value


def _manifest(out: Path) -> dict[str, Any]:
    run_dir = next(p for p in out.iterdir() if p.is_dir())
    data: dict[str, Any] = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return data


@pytest.mark.full_only
def test_a_killed_run_resumes_to_an_identical_manifest(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    csv = write_case(case, tmp_path)
    reference = run_case(case, tmp_path / "reference")
    assert reference.manifest is not None

    out = tmp_path / "killed" / "runs"
    killed = _python(str(csv), str(out), "kill")
    assert killed.returncode == 9, killed.stderr
    run_dir = next(p for p in out.iterdir() if p.is_dir())
    assert (run_dir / "checkpoints.sqlite").is_file()
    assert not (run_dir / "run_manifest.json").exists()  # it really stopped mid-run

    resumed = _python(str(csv), str(out), "resume")
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.strip() == "completed"
    expected = _normalised(reference.manifest.model_dump(mode="json"))
    actual = _normalised(_manifest(out))
    expected["data"].pop("path")
    actual["data"].pop("path")
    assert actual == expected
    access = (run_dir / "holdout_access.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["outcome"] for line in access] == ["evaluated"]  # once


@pytest.mark.full_only
def test_a_paused_run_resumes_in_a_new_process(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    csv = write_case(case, tmp_path)
    out = tmp_path / "runs"
    paused = _python(str(csv), str(out), "pause")
    assert paused.returncode == 0, paused.stderr
    assert paused.stdout.strip() == "paused split"
    resumed = _python(str(csv), str(out), "resume")
    assert resumed.returncode == 0, resumed.stderr
    manifest = _manifest(out)
    decisions = manifest["agent"]["decisions"]
    assert [d["kind"] for d in decisions] == ["split", "review", "unseal"]
    assert {d["mode"] for d in decisions} == {"autopilot"}
    assert manifest["agent"]["mode"] == "interactive"  # how the run started
