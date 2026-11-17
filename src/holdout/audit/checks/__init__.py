"""The twelve audit checks, one module each. See ``docs/audits.md``."""

from dataclasses import dataclass, field

from holdout.audit.types import AuditFlag, SamplingInfo


@dataclass(frozen=True)
class CheckOutcome:
    """Flags raised by one check, plus how much data it used and a short note."""

    flags: list[AuditFlag] = field(default_factory=list)
    sampling: SamplingInfo | None = None
    detail: str | None = None
    skipped: bool = False
