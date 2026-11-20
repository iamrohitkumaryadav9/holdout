"""Regression: the app's group_leakage demo, first run and corrected follow-up run.

Found by reading the deployed app's artifacts: the follow-up's model was below the dummy
with no warning, 'account' was no data finding although the split proposer called it an
entity, and 'account' stayed a (target-encoded) feature. v1.1 code paths only."""

from pathlib import Path

from holdout.audit.checks.a12_stability import BELOW_BASELINE
from holdout.audit.types import CheckId, Severity
from holdout.webapp import core


def test_the_group_leakage_demo_first_and_followup_runs(tmp_path: Path) -> None:
    demo = next(d for d in core.demos() if "group_leakage" in d.label)
    frame, target = demo.build()
    outcome = core.run_holdout(frame, target, demo.goal, tmp_path)

    # 4. The first run: invalid holdout, the follow-up named, no irrelevant blocker.
    first = outcome.manifest
    assert first is not None
    assert first.holdout_invalid
    assert outcome.followup is not None
    report = (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    assert (
        f"Holdout started a new run with the corrected split: `{outcome.followup.run_id}`."
    ) in report
    assert len(first.unseal.blocked_by) == 1
    assert first.unseal.blocked_by[0].startswith("holdout invalid (A5)")

    second = outcome.followup.manifest
    assert second is not None
    assert second.split.spec.group_col == "account"
    assert second.agent is not None
    follow_report = (outcome.followup.run_dir / "report.md").read_text(encoding="utf-8")

    # 2. The entity key is a data finding, resolved by the group split.
    (entity,) = [f for f in second.agent.findings if f.check is CheckId.A5]
    assert entity.columns == ["account"]
    assert entity.resolved
    assert entity.outcome.startswith("resolved: group split on 'account'")
    assert "## Problems found and how they were handled" in follow_report
    assert "resolved: group split on 'account'" in follow_report

    # 3. The key is not a feature, and the finding says so.
    assert "account" in second.plans[-1].plan.dropped
    assert entity.outcome.endswith("the key is not a feature (predictions are for new entities)")

    # 1. A model that does not beat the dummy is a warning and the summary says so first.
    assert second.validation_score <= second.plans[0].cv.mean
    (a12,) = [f for f in second.audit.flags if f.check is CheckId.A12]
    assert a12.severity is Severity.WARNING
    assert second.agent.report is not None
    assert second.agent.report.summary.startswith(BELOW_BASELINE)
    assert second.unseal.allowed  # warned, not blocked (docs/design.md)
