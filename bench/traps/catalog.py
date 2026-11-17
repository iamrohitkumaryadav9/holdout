"""Registry of traps: build a case from a name and seed, and build its clean twin."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bench.traps import leaks, quality, structure, synthetic
from bench.traps.types import Provenance, TrapCase
from holdout.core.schemas import Task

TrapFn = Callable[[pd.DataFrame, str, Task, int, dict[str, Any]], TrapCase]


@dataclass(frozen=True)
class TrapSpec:
    apply: TrapFn
    clean: TrapFn
    base: str
    base_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Base overrides for the clean twin (e.g. no drift for the time-order trap).
    clean_base_kwargs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


TRAPS: dict[str, TrapSpec] = {
    "direct_leak": TrapSpec(leaks.direct_leak, leaks.clean_direct_leak, "classification"),
    "noisy_proxy": TrapSpec(leaks.noisy_proxy, leaks.clean_noisy_proxy, "classification"),
    "id_with_signal": TrapSpec(leaks.id_with_signal, leaks.clean_id_with_signal, "classification"),
    "cross_split_duplicates": TrapSpec(
        structure.cross_split_duplicates,
        structure.clean_cross_split_duplicates,
        "classification",
    ),
    "group_leakage": TrapSpec(
        structure.group_leakage, structure.clean_group_leakage, "classification"
    ),
    "time_order": TrapSpec(
        structure.time_order,
        structure.clean_time_order,
        "time_ordered",
        base_kwargs={"drift": 1.0},
        clean_base_kwargs={"drift": 0.0},
    ),
    "imbalance_accuracy": TrapSpec(
        quality.imbalance_accuracy,
        quality.clean_imbalance_accuracy,
        "classification",
        base_kwargs={"n_rows": 2000},
    ),
    "messy_formats": TrapSpec(quality.messy_formats, quality.clean_messy_formats, "classification"),
    "causal_question": TrapSpec(
        quality.causal_question, quality.clean_causal_question, "classification"
    ),
}


def build_case(
    name: str, seed: int, params: dict[str, Any] | None = None, **base_kwargs: Any
) -> TrapCase:
    """Build trap ``name`` on its synthetic base, deterministically for ``seed``."""
    spec = TRAPS[name]
    merged_params = {**spec.params, **(params or {})}
    kwargs = {**spec.base_kwargs, **base_kwargs}
    frame, target, task = synthetic.BUILDERS[spec.base](seed=seed, **kwargs)
    case = spec.apply(frame, target, task, seed, merged_params)
    return dataclasses.replace(
        case, provenance=Provenance(name, spec.base, kwargs, seed, merged_params)
    )


def clean_twin(case: TrapCase) -> TrapCase:
    """The same base data and seed with the trap's mechanism neutralised."""
    if case.provenance is None:
        raise ValueError("clean_twin needs a case built by build_case (it has no provenance)")
    origin = case.provenance
    spec = TRAPS[origin.trap]
    kwargs = {**origin.base_kwargs, **spec.clean_base_kwargs}
    frame, target, task = synthetic.BUILDERS[origin.base](seed=origin.seed, **kwargs)
    twin = spec.clean(frame, target, task, origin.seed, origin.params)
    return dataclasses.replace(
        twin, provenance=Provenance(origin.trap, origin.base, kwargs, origin.seed, origin.params)
    )
