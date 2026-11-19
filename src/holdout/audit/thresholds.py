"""Load ``thresholds.toml`` into typed, validated models and hash it.

Every section is a frozen pydantic model with ``extra="forbid"``, so a misspelt or
missing key fails at load time instead of silently falling back to a default. The sha256
of the file's bytes is recorded in every run manifest.
"""

import hashlib
import tomllib
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SamplingThresholds(_Section):
    max_rows: int = Field(ge=100)
    max_features: int = Field(ge=1)


class SingleFeatureThresholds(_Section):
    tree_max_depth: int = Field(ge=1, le=10)
    warn_relative_lift: float = Field(gt=0.0, le=1.5)
    warn_normalized_score: float = Field(gt=0.0, le=1.0)
    min_full_normalized_lift: float = Field(ge=0.0, le=1.0)


class HintThresholds(_Section):
    min_normalized_score_for_warning: float = Field(ge=0.0)


class IdLikeThresholds(_Section):
    near_unique_ratio: float = Field(gt=0.0, le=1.0)
    integer_sequence_max_span_ratio: float = Field(ge=1.0)
    uniform_code_min_share: float = Field(gt=0.0, le=1.0)
    row_order_min_abs_spearman: float = Field(gt=0.0, le=1.0)
    signal_min_normalized_score: float = Field(ge=0.0)


class DuplicateThresholds(_Section):
    holdout_overlap_invalidate_share: float = Field(gt=0.0, le=1.0)
    cross_fold_critical_share: float = Field(gt=0.0, le=1.0)


class GroupThresholds(_Section):
    min_mean_rows_per_value: float = Field(ge=1.0)
    min_groups: int = Field(ge=6)
    max_unique_ratio: float = Field(gt=0.0, le=1.0)
    max_candidates: int = Field(ge=1)
    deep_max_candidates: int = Field(ge=1)
    warn_relative_drop: float = Field(gt=0.0)
    critical_relative_drop: float = Field(gt=0.0)
    min_drop_to_noise: float = Field(ge=0.0)


class TimeThresholds(_Section):
    autocorrelation_min_z: float = Field(gt=0.0)
    autocorrelation_min_effect: float = Field(gt=0.0)
    n_permutations: int = Field(ge=20)
    sorted_min_abs_spearman: float = Field(gt=0.0, le=1.0)
    sorted_max_runs_per_value: float = Field(ge=1.0)
    max_candidates: int = Field(ge=1)
    deep_max_candidates: int = Field(ge=1)
    warn_relative_drop: float = Field(gt=0.0)
    critical_relative_drop: float = Field(gt=0.0)
    min_drop_to_noise: float = Field(ge=0.0)


class MetricThresholds(_Section):
    heavy_tail_ratio: float = Field(gt=1.0)


class FormatThresholds(_Section):
    numeric_parse_min_share: float = Field(gt=0.0, le=1.0)
    sentinel_values: list[float]
    sentinel_min_share: float = Field(gt=0.0, le=1.0)
    sentinel_min_iqr_distance: float = Field(gt=0.0)


class TooGoodThresholds(_Section):
    deep_mode_normalized_score: float = Field(gt=0.0, le=1.0)
    deep_max_rows: int = Field(ge=100)
    deep_max_features: int = Field(ge=1)


class StabilityThresholds(_Section):
    confidence: float = Field(gt=0.5, lt=1.0)


class Thresholds(_Section):
    sampling: SamplingThresholds
    a01_single_feature: SingleFeatureThresholds
    a02_hints: HintThresholds
    a03_id_like: IdLikeThresholds
    a04_duplicates: DuplicateThresholds
    a05_group: GroupThresholds
    a06_time: TimeThresholds
    a07_metric: MetricThresholds
    a08_formats: FormatThresholds
    a10_too_good: TooGoodThresholds
    a12_stability: StabilityThresholds


class IdLikeV11(_Section):
    key_name_tokens: list[str] = Field(min_length=1)


class GroupV11(_Section):
    min_entity_constant_features: int = Field(ge=1)


class ThresholdsV11(_Section):
    """The v1.1 additions (``thresholds_v11.toml``), layered on :class:`Thresholds`."""

    a03_id_like: IdLikeV11
    a05_group: GroupV11


def default_thresholds_v11_bytes() -> bytes:
    """The packaged ``thresholds_v11.toml`` as bytes."""
    return resources.files("holdout.audit").joinpath("thresholds_v11.toml").read_bytes()


def load_thresholds_v11(path: Path | None = None) -> tuple[ThresholdsV11, str]:
    """Load the v1.1 additions with their sha256 (v1.0 behaviour does not call this)."""
    raw = path.read_bytes() if path is not None else default_thresholds_v11_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    return ThresholdsV11.model_validate(data), hashlib.sha256(raw).hexdigest()


def default_thresholds_bytes() -> bytes:
    """The packaged ``thresholds.toml`` as bytes (LF line endings via .gitattributes)."""
    return resources.files("holdout.audit").joinpath("thresholds.toml").read_bytes()


def parse_thresholds(raw: bytes) -> tuple[Thresholds, str]:
    """Parse TOML bytes into ``Thresholds`` and return them with the bytes' sha256."""
    data = tomllib.loads(raw.decode("utf-8"))
    return Thresholds.model_validate(data), hashlib.sha256(raw).hexdigest()


def load_thresholds(path: Path | None = None) -> tuple[Thresholds, str]:
    """Load thresholds from ``path`` (default: the packaged file) with their sha256."""
    raw = path.read_bytes() if path is not None else default_thresholds_bytes()
    return parse_thresholds(raw)
