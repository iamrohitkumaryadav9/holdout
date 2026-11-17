"""Interrupt decisions: interactive (a person answers) or autopilot (a scripted policy).

Autopilot is the unattended mode used by the Week 4 benchmark. Its policy, applied at each
interrupt and recorded in the manifest exactly like a human decision:

* split: accept the proposed split; entity scope "new" unless the user declared one;
* review: accept the agent's fixes;
* unseal: request unsealing with warnings acknowledged by policy and **no** accepted risk,
  so the gate unseals only when no critical flag is open.
"""

from typing import Any, Protocol

from holdout.agent.schemas import Decision

AUTOPILOT_REASONS = {
    "split": "autopilot: accept the proposed split; entity scope new unless declared",
    "review": "autopilot: accept the agent's fixes",
    "unseal": "autopilot: unseal only with no open critical flags; warnings acknowledged by policy",
}


class Asker(Protocol):
    """Answers interrupts for a person (the CLI implements it with rich prompts)."""

    def ask(self, payload: dict[str, Any]) -> Decision: ...


def autopilot_decision(payload: dict[str, Any]) -> Decision:
    kind = payload["kind"]
    if kind == "split":
        declared = payload.get("declared_entity_scope")
        return Decision(
            kind="split",
            mode="autopilot",
            accepted=True,
            detail={"entity_scope": declared or "new"},
            reason=AUTOPILOT_REASONS["split"],
        )
    if kind == "review":
        return Decision(
            kind="review",
            mode="autopilot",
            accepted=True,
            detail={},
            reason=AUTOPILOT_REASONS["review"],
        )
    if kind == "unseal":
        return Decision(
            kind="unseal",
            mode="autopilot",
            accepted=True,
            detail={"requested": True, "ack_warnings": True, "accept_risk": None},
            reason=AUTOPILOT_REASONS["unseal"],
        )
    raise ValueError(f"unknown interrupt kind {kind!r}")
