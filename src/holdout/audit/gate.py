"""The unseal gate: may the holdout be evaluated after this audit?

Rules, in order of strength:

1. A contaminated holdout (any flag with ``invalidates_holdout``) is never evaluated.
   Nothing overrides this; the fix is a new run with a corrected split.
2. An incomplete audit (a check errored) blocks unsealing; it is not overridable either,
   because an unknown is not an accepted risk.
3. Critical flags block unsealing unless the user accepts the risk with a written reason
   (``--accept-risk "reason"``), which is logged in the manifest and shown in the report.
4. Warnings must be acknowledged (``--ack-warnings``), even when a risk is accepted.

The decision is made in the run orchestrator, not in the CLI, so programmatic callers get
the same rules.
"""

from pydantic import BaseModel, ConfigDict, Field

from holdout.audit.types import AuditReport, Severity

#: Minimum length of an accepted-risk reason: a reason, not a keystroke.
MIN_RISK_REASON_CHARS = 10


class UnsealDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: bool
    allowed: bool
    blocked_by: list[str] = Field(description="why unsealing is (or would be) refused")
    acknowledged_warnings: bool
    accepted_risk: str | None
    overridden_criticals: list[str] = Field(
        description="critical flags unsealed past with --accept-risk"
    )


def validate_risk_reason(accept_risk: str | None) -> str | None:
    """The stripped reason, or None; raise `ValueError` when it is too short to be one."""
    if accept_risk is None:
        return None
    reason = accept_risk.strip()
    if len(reason) < MIN_RISK_REASON_CHARS:
        raise ValueError(
            f"--accept-risk needs a reason of at least {MIN_RISK_REASON_CHARS} characters"
        )
    return reason


def decide_unseal(
    report: AuditReport,
    *,
    requested: bool,
    ack_warnings: bool = False,
    accept_risk: str | None = None,
) -> UnsealDecision:
    reason = validate_risk_reason(accept_risk)
    blocked: list[str] = []
    if report.holdout_invalid:
        invalidating = sorted({f.check.value for f in report.flags if f.invalidates_holdout})
        blocked.append(
            f"holdout invalid ({', '.join(invalidating)}): it shares rows, entities or time with "
            "development data; start a new run with the corrected split (not overridable)"
        )
    if not report.complete:
        failed = [r.check.value for r in report.checks_run if r.status == "error"]
        blocked.append(f"audit incomplete: {', '.join(failed)} failed (not overridable)")
    criticals = [
        f"{f.check.value} {', '.join(f.columns) or f.category.value}"
        for f in report.flags
        if f.severity is Severity.CRITICAL and not f.invalidates_holdout
    ]
    if criticals and reason is None:
        blocked.append(
            f'critical flag(s): {"; ".join(criticals)} (override only with --accept-risk "reason")'
        )
    warnings = [f for f in report.flags if f.severity is Severity.WARNING]
    if warnings and not ack_warnings:
        blocked.append(f"{len(warnings)} warning(s) not acknowledged (use --ack-warnings)")
    return UnsealDecision(
        requested=requested,
        allowed=requested and not blocked,
        blocked_by=blocked,
        acknowledged_warnings=ack_warnings,
        accepted_risk=reason,
        overridden_criticals=criticals if reason is not None and not blocked else [],
    )
