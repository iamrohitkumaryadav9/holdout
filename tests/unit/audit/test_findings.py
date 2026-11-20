"""v1.1 data findings: recorded before the planner, answered by it, decided by code.

Post-benchmark fix from docs/error_analysis.md, section 1 (the LLM planner silently avoided
16 of 20 problems its own audit layer would have reported). Not part of the pre-registered
evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holdout.agent.schemas import AgentRecord
from holdout.audit.findings import (
    collect_findings,
    handling_problems,
    offline_handling,
    persisting_split_flags,
    resolution,
    with_handling,
)
from holdout.audit.types import (
    AuditFlag,
    AuditReport,
    Category,
    ChangeSplit,
    CheckId,
    CheckRun,
    DropColumns,
    GroupEvidence,
    IdLikeEvidence,
    Severity,
)
from holdout.core.schemas import EntityScope, SplitStrategy, Task
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec
from holdout.export.manifest import RunManifest, findings_section, render_report
from holdout.llm.structured import llm_schema
from holdout.plan.defaults import default_plan
from holdout.plan.schema import FindingHandling, ModelPlan

FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "v1_0" / "run_manifest_group_leakage_telco_s101.json"
)
ROW_SPLIT = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1)


def _plan() -> ModelPlan:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "account": [f"A{i % 40}" for i in range(400)],
        "ticket": rng.integers(0, 50, 400),
        "x": rng.normal(size=400),
        "y": rng.integers(0, 2, 400),
    })  # fmt: skip
    return default_plan(profile_schema(frame, "y", task=Task.CLASSIFICATION), Task.CLASSIFICATION)


def _id_flag() -> AuditFlag:
    return AuditFlag(
        check=CheckId.A3,
        category=Category.ID_SIGNAL,
        severity=Severity.CRITICAL,
        columns=["ticket"],
        evidence=IdLikeEvidence(
            signals=["integer_sequence"],
            unique_ratio=1.0,
            row_order_spearman=None,
            single_feature_relative_lift=0.9,
            single_feature_normalized_score=0.9,
            used_by_plan=True,
        ),
        message="'ticket' is an identifier that predicts the target.",
        fix=DropColumns(columns=["ticket"], reason="identifier"),
    )


def _group_flag(severity: Severity = Severity.CRITICAL) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A5,
        category=Category.GROUP_LEAKAGE,
        severity=severity,
        columns=["account"],
        evidence=GroupEvidence(
            column="account",
            n_groups=40,
            mean_rows_per_group=10.0,
            constant_feature_share=0.2,
            entity_scope=None,
            comparison=None,
        ),
        message="Grouping by 'account' drops the CV score: the plan memorises entities.",
        fix=ChangeSplit(
            new_spec=SplitSpec(strategy=SplitStrategy.GROUP, group_col="account", seed=1),
            instruction="start a new run with --split group:account",
        ),
        invalidates_holdout=severity is Severity.CRITICAL,
    )


def _report(*flags: AuditFlag) -> AuditReport:
    return AuditReport(
        flags=list(flags),
        checks_run=[CheckRun(check=CheckId.A1, status="ran", seconds=0.1)],
        thresholds_hash="t" * 64,
        deep_mode=False,
        entity_scope=None,
    )


# --------------------------------------------------------------------- collection


def test_findings_are_numbered_and_leveled() -> None:
    findings = collect_findings(_report(_id_flag(), _group_flag()))
    assert [(f.id, f.level, f.flag.check) for f in findings] == [
        ("F1", "column", CheckId.A3),
        ("F2", "split", CheckId.A5),
    ]


def test_info_flags_and_plan_level_checks_are_not_findings() -> None:
    info = _id_flag().model_copy(update={"severity": Severity.INFO})
    metric = _id_flag().model_copy(update={"check": CheckId.A7})
    assert collect_findings(_report(info, metric)) == []


def test_an_entity_key_is_a_finding_at_any_severity() -> None:
    """v1.1.3: A5 info (the split already groups by the key, or entities are known) is a
    finding, so the report says how the entity key was handled."""
    (finding,) = collect_findings(_report(_group_flag(Severity.INFO)))
    assert finding.flag.check is CheckId.A5
    assert finding.level == "split"
    assert finding.flag.severity is Severity.INFO


# --------------------------------------------------------------------- handling


def test_every_finding_needs_exactly_one_consistent_answer() -> None:
    plan = _plan()
    findings = collect_findings(_report(_id_flag(), _group_flag()))
    problems = handling_problems(plan, findings)
    assert any("no entry for F1" in p for p in problems)
    assert any("no entry for F2" in p for p in problems)

    lying = with_handling(plan, [
        FindingHandling(finding_id="F1", action="drop_columns", reason="identifier"),
        FindingHandling(finding_id="F2", action="decline_causal_claim", reason="n/a"),
        FindingHandling(finding_id="F9", action="keep_with_reason", reason="n/a"),
    ])  # fmt: skip
    problems = handling_problems(lying, findings)
    assert any("F1: action drop_columns, but the plan does not drop ['ticket']" in p
               for p in problems)  # fmt: skip
    assert any("F2: decline_causal_claim applies to the goal" in p for p in problems)
    assert any("'F9', which is not a finding" in p for p in problems)


def test_the_offline_answers_are_valid() -> None:
    plan = _plan()
    findings = collect_findings(_report(_id_flag(), _group_flag()))
    answered = with_handling(plan, offline_handling(plan, findings))
    assert handling_problems(answered, findings, offline=True) == []
    assert [h.action for h in answered.findings_handled] == ["deferred_to_audit"] * 2
    assert answered.findings_handled[0].reason == (
        "the offline planner does not act on findings; code applies the audit's fixes"
    )
    # A model may not defer: it must say what it does.
    assert any(
        "reserved for the offline planner" in p for p in handling_problems(answered, findings)
    )


# --------------------------------------------------------------------- no masking (fix 2)


def test_dropping_the_entity_key_with_a_row_split_kept_is_still_critical() -> None:
    """The exact masking pattern: entity key dropped + random split kept => still critical."""
    findings = collect_findings(_report(_group_flag()))
    plan = _plan()
    masked = ModelPlan.model_validate({
        **plan.model_dump(mode="json"),
        "drop_columns": [*plan.model_dump(mode="json")["drop_columns"],
                         {"column": "account", "reason": "identifier"}],
    })  # fmt: skip
    # The audit of the masked plan no longer raises A5 (the key is not a feature any more).
    final = _report()
    persisted = persisting_split_flags(findings, final, ROW_SPLIT)
    assert len(persisted) == 1
    flag = persisted[0]
    assert flag.severity is Severity.CRITICAL
    assert flag.invalidates_holdout
    assert "Data finding F1 persists" in flag.message
    assert _report(*persisted).holdout_invalid

    (status,) = resolution(findings, masked, ROW_SPLIT, _report(*persisted))
    assert not status.resolved
    assert status.outcome.startswith("open: the split still leaks")


def test_a_group_split_or_known_entities_resolve_the_entity_finding() -> None:
    findings = collect_findings(_report(_group_flag()))
    group = SplitSpec(strategy=SplitStrategy.GROUP, group_col="account", seed=1)
    known = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1, entity_scope=EntityScope.KNOWN)
    for spec in (group, known):
        assert persisting_split_flags(findings, _report(), spec) == []
        (status,) = resolution(findings, _plan(), spec, _report())
        assert status.resolved


def test_a_warning_split_finding_is_reported_but_not_reasserted() -> None:
    findings = collect_findings(_report(_group_flag(Severity.WARNING)))
    assert persisting_split_flags(findings, _report(), ROW_SPLIT) == []
    (status,) = resolution(findings, _plan(), ROW_SPLIT, _report())
    assert not status.resolved  # still listed as open in the report table


def test_an_already_raised_critical_is_not_duplicated() -> None:
    findings = collect_findings(_report(_group_flag()))
    assert persisting_split_flags(findings, _report(_group_flag()), ROW_SPLIT) == []


# --------------------------------------------------------------------- outcome and report


def test_a_column_finding_is_resolved_by_the_drop() -> None:
    findings = collect_findings(_report(_id_flag()))
    plan = _plan()
    dropped = ModelPlan.model_validate({
        **plan.model_dump(mode="json"),
        "drop_columns": [*plan.model_dump(mode="json")["drop_columns"],
                         {"column": "ticket", "reason": "identifier"}],
    })  # fmt: skip
    (kept,) = resolution(findings, plan, ROW_SPLIT, _report(_id_flag()))
    assert not kept.resolved
    assert "still raises it" in kept.outcome
    (done,) = resolution(findings, dropped, ROW_SPLIT, _report())
    assert done.resolved
    assert done.outcome == "resolved: the final plan does drop columns"


def test_the_report_table_lists_every_finding() -> None:
    findings = collect_findings(_report(_id_flag(), _group_flag()))
    plan = with_handling(_plan(), offline_handling(_plan(), findings))
    lines = findings_section(resolution(findings, plan, ROW_SPLIT, _report(_id_flag())))
    text = "\n".join(lines)
    assert "## Problems found and how they were handled" in text
    assert "| F1 | A3 | critical | `ticket` |" in text
    assert "| F2 | A5 | critical | `account` |" in text
    assert "deferred to audit: the offline planner does not act on findings" in text
    assert "0 handled, 2 open." in text


# --------------------------------------------------------------------- compatibility


def test_the_llm_schema_has_no_one_of() -> None:
    """findings_handled must not reintroduce oneOf (Gemini's structured output)."""
    schema = json.dumps(llm_schema(ModelPlan))
    assert "oneOf" not in schema
    assert "findings_handled" in schema


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_a_v1_0_manifest_still_loads() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert "thresholds_v11_hash" not in raw
    assert "data_findings" not in raw["agent"]
    manifest = RunManifest.model_validate(raw)
    assert manifest.thresholds_v11_hash is None
    assert manifest.agent is not None
    assert manifest.agent.data_findings == []
    assert manifest.agent.findings == []
    assert all(p.plan.findings_handled == [] for p in manifest.plans)
    report = render_report(manifest)
    assert "Problems found" not in report  # v1.0 runs have no findings table
    AgentRecord.model_validate(raw["agent"])
