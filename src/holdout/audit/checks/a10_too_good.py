"""A10 too good to be true: a near-ceiling score switches the audit into deep mode.

A near-perfect score is not a flag on its own (some problems are easy), so A10 never
raises severity. It changes how hard the other checks look: A1 runs on every feature
without row sampling (up to the deep-mode caps), and A5/A6 test more candidates. The run
records an info entry saying so.
"""

from holdout.audit.checks import CheckOutcome
from holdout.audit.context import AuditContext
from holdout.audit.scoring import normalized
from holdout.audit.types import AuditFlag, Category, CheckId, NoAction, ScoreEvidence, Severity


def deep_mode(ctx: AuditContext) -> bool:
    score = normalized(ctx.metric, ctx.full_score, ctx.dummy_score)
    return score >= ctx.thresholds.a10_too_good.deep_mode_normalized_score


def run(ctx: AuditContext) -> CheckOutcome:
    if not deep_mode(ctx):
        return CheckOutcome(detail="score within normal range; standard audit depth")
    score = normalized(ctx.metric, ctx.full_score, ctx.dummy_score)
    t = ctx.thresholds.a10_too_good
    flag = AuditFlag(
        check=CheckId.A10,
        category=Category.SUSPICIOUS_SCORE,
        severity=Severity.INFO,
        columns=[],
        evidence=ScoreEvidence(
            metric=ctx.metric.name,
            score=ctx.full_score,
            dummy_score=ctx.dummy_score,
            normalized_score=score,
            deep_mode=True,
        ),
        message=(
            f"The plan scores {ctx.full_score:.3f} {ctx.metric.name.value} ({score:.0%} of the "
            "way from dummy to perfect). Not a flag by itself; the audit ran in deep mode "
            f"(all features up to {t.deep_max_features}, up to {t.deep_max_rows} rows, more "
            "group and time candidates)."
        ),
        fix=NoAction(reason="deep mode applied; see the other checks"),
    )
    return CheckOutcome(flags=[flag], detail="deep mode")
