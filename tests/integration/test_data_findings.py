"""v1.1 data findings through the agent graph (fake LLM; the Gemini planner path is not
exercised live in v1.1).

Post-benchmark fixes from docs/error_analysis.md, section 1: findings are recorded on the
raw development rows before the planner (fix 1) and a split-level finding is never masked
by dropping a column (fix 2). Not part of the pre-registered evaluation."""

from pathlib import Path
from typing import Any

import pytest

from bench.traps import build_case
from holdout.agent.driver import run_agent
from holdout.agent.graph import AgentDeps, AgentOptions, AuditInputs
from holdout.audit.types import AuditReport, CheckId, CheckRun, Severity
from holdout.llm.client import FakeLLM, LLMRequest
from tests.agent_fakes import answer_findings, data
from tests.agent_harness import fake_for, run_case, write_case
from tests.unit.audit.test_findings import _group_flag


def test_an_offline_run_reports_every_problem_found(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    outcome = run_case(case, tmp_path, offline=True)
    manifest = outcome.manifest
    assert manifest is not None
    agent = manifest.agent
    assert agent is not None
    leak = [f for f in agent.data_findings if "refund_issued" in f.flag.columns]
    assert leak, [f.flag.columns for f in agent.data_findings]
    assert {f.finding_id for f in agent.findings} == {f.id for f in agent.data_findings}
    status = next(s for s in agent.findings if s.finding_id == leak[0].id)
    assert status.resolved  # the code fix dropped the leak
    assert manifest.plans[-1].plan.findings_handled  # the offline planner answered too
    report = (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Problems found and how they were handled" in report
    assert f"| {leak[0].id} |" in report


def test_the_first_plan_reuses_the_probe_audit(tmp_path: Path) -> None:
    """The offline plan is the probe: compile_and_fit and the round-0 audit are not rerun."""
    case = build_case("direct_leak", 0)
    outcome = run_case(case, tmp_path, offline=True)
    manifest = outcome.manifest
    assert manifest is not None
    assert manifest.agent is not None
    first = manifest.agent.rounds[0]
    assert any(text.startswith("A1 critical") for text in first.flags)


def _forgetful(case: Any, csv: Path) -> FakeLLM:
    """Answers the plan without findings_handled first, then answers it on repair."""
    working = fake_for(case, csv)
    plain = working.default["plan"](None)
    return FakeLLM(
        default=working.default,
        scripts={"plan": [plain, working.default["plan"]]},
    )


def test_a_plan_that_ignores_the_findings_is_repaired(tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    fake = _forgetful(case, write_case(case, tmp_path))
    outcome = run_case(case, tmp_path, llm=fake)
    assert outcome.manifest is not None
    agent = outcome.manifest.agent
    assert agent is not None
    assert agent.quality is not None
    assert agent.quality.first_attempt["plan"].rate == 0.0
    assert agent.quality.repairs >= 1
    (rejected,) = [r for r in agent.rejected_attempts if r.node == "plan"]
    assert any("answer every finding" in e for e in rejected.errors)
    first_plan = outcome.manifest.plans[1].plan
    assert {h.finding_id for h in first_plan.findings_handled} == {
        f.id for f in agent.data_findings
    }
    (_, second) = fake.calls_for("plan")
    assert "<repair>" in second.user


def _masking(case: Any, csv: Path) -> FakeLLM:
    """The v1.0 failure: the model drops the entity key and keeps the row-level split."""
    working = fake_for(case, csv)

    def plan(request: LLMRequest) -> dict[str, Any]:
        payload = dict(working.default["plan"](None))
        payload["drop_columns"] = [
            *payload["drop_columns"],
            {"column": "account", "reason": "an identifier, not a feature"},
        ]
        found = data(request, "findings")
        answers = answer_findings(payload, found)
        for answer, finding in zip(answers, found, strict=True):
            if finding["level"] == "split":  # claims the drop handles it
                answer.update(action="drop_columns", reason="the key is dropped")
        return {**payload, "findings_handled": answers}

    return FakeLLM(default={**working.default, "plan": plan})


def _key_sensitive_auditor(deps: AgentDeps) -> None:
    """An audit layer that sees the entity leak only while the plan uses the key column
    (what A5 does when no entity-constant feature remains to re-identify the entity)."""

    def audit(inputs: AuditInputs) -> AuditReport:
        flags = [] if "account" in inputs.plan.dropped else [_group_flag()]
        return AuditReport(
            flags=flags,
            checks_run=[CheckRun(check=CheckId.A5, status="ran", seconds=0.0)],
            thresholds_hash="t" * 64,
            deep_mode=False,
            entity_scope=None,
        )

    deps.auditor = audit


def test_the_graph_reasserts_a_masked_entity_finding(tmp_path: Path) -> None:
    """Fix 2: entity key dropped + row-level split kept => still critical, holdout invalid."""
    case = build_case("group_leakage", 0, n_rows=400)
    csv = write_case(case, tmp_path)
    fake = _masking(case, csv)
    outcome = run_agent(
        AgentOptions(csv=str(csv), target=case.target, out_dir=str(tmp_path / "runs"),
                     goal=case.goal_text, mode="autopilot", use_cache=False),
        client_factory=lambda: fake,
        customize=_key_sensitive_auditor,
    )  # fmt: skip
    first = outcome.manifest
    assert first is not None
    assert first.agent is not None
    assert "account" in first.plans[1].plan.dropped
    (persisted,) = [f for f in first.audit.flags if f.check is CheckId.A5]
    assert persisted.severity is Severity.CRITICAL
    assert persisted.invalidates_holdout
    assert "persists" in persisted.message
    assert "Dropping the column does not remove it" in persisted.message
    assert first.holdout_invalid
    (status,) = [s for s in first.agent.findings if s.check is CheckId.A5]
    assert not status.resolved
    assert outcome.followup is not None  # autopilot starts the corrected run


@pytest.mark.full_only
def test_dropping_the_entity_key_does_not_mask_the_group_leak(tmp_path: Path) -> None:
    case = build_case("group_leakage", 0, n_rows=600)
    outcome = run_case(case, tmp_path, llm=_masking(case, write_case(case, tmp_path)))
    first = outcome.manifest
    assert first is not None
    assert first.agent is not None
    assert "account" in first.plans[1].plan.dropped  # the masking plan was accepted as a plan
    entity = [f for f in first.agent.data_findings if f.flag.check is CheckId.A5]
    assert entity
    assert entity[0].flag.severity is Severity.CRITICAL
    # ...but the finding persists: still critical, holdout invalid, a corrected follow-up.
    # (On this trap the entity-constant features let A5 re-detect the leak on the masked
    # plan by itself; the re-assertion when A5 does not is covered in
    # tests/unit/audit/test_findings.py.)
    assert any(f.check is CheckId.A5 and f.severity is Severity.CRITICAL for f in first.audit.flags)
    assert first.holdout_invalid
    status = next(s for s in first.agent.findings if s.finding_id == entity[0].id)
    assert not status.resolved
    assert status.planner_action == "drop_columns"
    assert outcome.followup is not None
    assert outcome.followup.manifest is not None
    assert outcome.followup.manifest.split.spec.group_col == "account"
