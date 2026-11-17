"""What a check may see: development rows and run metadata, never holdout rows.

The only path toward the holdout is :class:`OverlapSource`, which the seal implements and
which answers with counts (see ``holdout.data.seal``).
"""

import dataclasses
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from holdout.audit.hints import HintProvider, LexiconHintProvider
from holdout.audit.thresholds import Thresholds
from holdout.core.schemas import CVResult, Task
from holdout.core.seeds import derive_seed
from holdout.data.profile import DatasetProfile
from holdout.data.seal import HoldoutOverlap
from holdout.data.split import SplitSpec
from holdout.plan.metrics import Metric, get_metric
from holdout.plan.schema import ModelPlan


class OverlapSource(Protocol):
    """Counts-only view of the sealed holdout (implemented by ``SealedHoldout``)."""

    def holdout_overlap(self, dev_features: pd.DataFrame) -> HoldoutOverlap | None: ...


@dataclass(frozen=True)
class AuditContext:
    dev: pd.DataFrame
    target: str
    profile: DatasetProfile
    spec: SplitSpec
    plan: ModelPlan
    cv_result: CVResult
    baseline: CVResult
    thresholds: Thresholds
    overlap_source: OverlapSource | None = None
    hint_provider: HintProvider = dataclasses.field(default_factory=LexiconHintProvider)
    goal_text: str = ""
    deep_mode: bool = False

    @property
    def task(self) -> Task:
        return self.plan.task

    @property
    def metric(self) -> Metric:
        return get_metric(self.plan.metric)

    @property
    def dummy_score(self) -> float:
        return self.baseline.mean

    @property
    def full_score(self) -> float:
        return self.cv_result.mean

    @property
    def features(self) -> list[str]:
        return [str(c) for c in self.dev.columns if c != self.target]

    @property
    def used_features(self) -> set[str]:
        """Features the plan's pipeline actually consumes (not dropped)."""
        return {c for c in self.features if c not in self.plan.dropped}

    def seed(self, name: str) -> int:
        """A seed for one audit stream, derived from the run seed."""
        return derive_seed(self.spec.seed, f"audit/{name}")

    def with_deep_mode(self) -> "AuditContext":
        return dataclasses.replace(self, deep_mode=True)
