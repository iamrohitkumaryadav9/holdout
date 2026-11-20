"""The artifacts of an offline run say what happened, consistently, in neutral words.

Found by reading a deployed app run on the direct_leak demo: a "Week 3" status text, CLI
flag wording for a split the user never chose, plan_source "code_substitution" next to
"Substitutions: 0", and an offline findings row that read as if the planner kept a leak."""

import json
import re
from pathlib import Path

from bench.traps import build_case
from holdout.webapp import core

MILESTONE = re.compile(r"\bWeek \d")


def _offline_direct_leak(tmp_path: Path):  # type: ignore[no-untyped-def]
    case = build_case("direct_leak", 0)
    return core.run_holdout(case.df, case.target, "predict which customers convert", tmp_path)


def test_offline_planner_with_a_code_applied_drop(tmp_path: Path) -> None:
    outcome = _offline_direct_leak(tmp_path)
    manifest = outcome.manifest
    assert manifest is not None
    agent = manifest.agent
    assert agent is not None
    raw = (outcome.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    report = (outcome.run_dir / "report.md").read_text(encoding="utf-8")

    # 4. One definition of substitution: the offline planner has no model answer to replace.
    assert [r.plan_source for r in agent.rounds] == ["default", "code_fix"]
    assert agent.substitutions == []
    assert "code_substitution" not in raw
    assert (
        "**Substitutions:** none (offline run: no model answers to replace); code applied "
        "the audit's fixes in 1 round(s)"
    ) in report
    assert "refund_issued" in manifest.plans[-1].plan.dropped

    # 2. No milestone wording; A11's goal part and report part are stated separately.
    assert not MILESTONE.search(raw)
    assert not MILESTONE.search(report)
    (a11,) = [r for r in manifest.audit.checks_run if r.check.value == "A11"]
    assert a11.detail == (
        "goal (at intake): no causal question found; report: not checked (offline runs "
        "write template prose, not generated text)"
    )
    assert f"- Causal claims (A11): {a11.detail}" in report

    # 3. Autopilot only declared the entity scope: the split is the proposed one.
    assert manifest.split.source == "proposed"
    assert "entity scope declared: new" in manifest.split.reasons
    assert "--split" not in json.dumps(manifest.split.model_dump(mode="json"))

    # 5. The findings table says what the planner did apart from what code did.
    leak = next(f for f in agent.findings if "refund_issued" in f.columns)
    assert leak.planner_action == "deferred_to_audit"
    assert leak.planner_reason == (
        "the offline planner does not act on findings; code applies the audit's fixes"
    )
    assert leak.outcome == "resolved: the final plan does drop columns"
    assert "keep with reason" not in report
    assert "deferred to audit: the offline planner does not act on findings" in report


def test_every_demo_goal_names_its_target(tmp_path: Path) -> None:
    """1. A demo's goal, target and columns agree (the app no longer carries a goal over
    from another demo either: tests/app/test_webapp.py)."""
    for demo in core.demos():
        frame, target = demo.build()
        assert target in frame.columns, demo.label
        assert demo.concept in target.lower(), (demo.label, target)
        assert demo.concept in demo.goal.lower(), (demo.label, demo.goal)
    causal = next(d for d in core.demos() if "causal" in d.label.lower())
    frame, _ = causal.build()
    assert "channel" in frame.columns  # the column the causal goal asks about
    assert "channel" in causal.goal
