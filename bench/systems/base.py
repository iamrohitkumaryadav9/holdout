"""What every benchmark system receives and returns.

Fairness rules (docs/benchmark.md): every system gets the same development partition (the
harness keeps the holdout), the same goal text, the same run seed, the same pinned model
and the same per-run budget cap. A system returns its reported issues (Category and
Severity), a description of its final pipeline, the fitted model (scored once on the
holdout by the harness) and its own validation claim.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from holdout.audit.types import Category, Severity
from holdout.core.schemas import EntityScope, Task
from holdout.llm.progress import NullProgress, Progress

SystemName = Literal["holdout", "audits_only", "llm_judge", "react", "flaml"]
Variant = Literal["original", "anonymised", "planted"]
UnitStatus = Literal["completed", "paused", "failed", "stopped"]


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReportedIssue(_Record):
    category: Category
    severity: Severity
    columns: list[str] = Field(default_factory=list)
    source: str = Field(description="what reported it: an audit check id, judge, react, ...")
    message: str = ""


class FinalPipeline(_Record):
    """What the final pipeline does, in the terms the scorer checks."""

    dropped_columns: list[str] = Field(default_factory=list)
    metric: str | None = None
    #: random | stratified | group:COL | time:COL | time:<row> | unknown
    split: str = "unknown"
    deduplicated: bool = False
    model: str = ""


class NodeQuality(_Record):
    first_attempts: int
    valid: int


class BenchResult(_Record):
    """One run unit's outcome (one line of ``results/<suite>/<system>.jsonl``)."""

    suite: str
    task_id: str
    system: SystemName
    seed: int
    variant: Variant
    status: UnitStatus
    error: str | None = None
    issues: list[ReportedIssue] = Field(default_factory=list)
    pipeline: FinalPipeline | None = None
    validation_metric: str | None = None
    validation_score: float | None = None
    holdout_score_own_metric: float | None = None
    benchmark_metric: str
    holdout_score: float | None = None
    dummy_holdout_score: float | None = None
    partition: str = ""
    report_text: str | None = None
    report_a11_flags: int = 0
    cost_inr: float = 0.0
    seconds: float = 0.0
    calls: int = 0
    retries: int = 0
    cache_hits: int = 0
    first_attempt: dict[str, NodeQuality] = Field(default_factory=dict)
    repairs: int = 0
    substitutions: int = 0
    model: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime

    @property
    def unit_key(self) -> tuple[str, str, str, int, str]:
        return (self.suite, self.task_id, self.system, self.seed, self.variant)


@dataclass
class SystemContext:
    """The inputs of one run: development rows only, never the holdout."""

    unit_id: str
    dev: pd.DataFrame
    target: str
    task: Task
    goal: str
    seed: int
    workdir: Path
    model: str
    run_cap_inr: float
    entity_scope: EntityScope | None = None
    #: The planted task's plan overrides (e.g. ``metric = accuracy`` for the imbalance
    #: trap). Only the deterministic ``audits_only`` path uses them: it cannot read the goal,
    #: and the trap needs the stated metric to exist. LLM systems read the goal text.
    plan_overrides: dict[str, Any] = field(default_factory=dict)
    progress: Progress = field(default_factory=NullProgress)


@dataclass
class SystemOutput:
    """What a system hands back; the harness scores ``estimator`` on the holdout."""

    status: UnitStatus
    issues: list[ReportedIssue] = field(default_factory=list)
    pipeline: FinalPipeline | None = None
    estimator: Any = None
    validation_metric: str | None = None
    validation_score: float | None = None
    report_text: str | None = None
    report_a11_flags: int = 0
    cost_inr: float = 0.0
    calls: int = 0
    retries: int = 0
    cache_hits: int = 0
    first_attempt: dict[str, NodeQuality] = field(default_factory=dict)
    repairs: int = 0
    substitutions: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class System(Protocol):
    name: SystemName
    uses_llm: bool

    def run(self, ctx: SystemContext) -> SystemOutput: ...

    def resume(self, ctx: SystemContext) -> SystemOutput:
        """Continue a paused run from its checkpoint (systems without one rerun)."""
        ...
