"""v1.1 A3: near-uniqueness alone is not an identifier; the message matches the evidence.

Post-benchmark fix from docs/error_analysis.md, section 3 (A3 false positives on integer
measurements: cpu_act ``freeswap``, steel_plates ``V3``/``V4``/``V8``)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holdout.audit.checks.a03_id_like import key_like_name, name_tokens
from holdout.audit.runner import run_audits
from holdout.audit.thresholds import load_thresholds, load_thresholds_v11
from holdout.audit.types import CheckId, Severity
from holdout.core.schemas import SplitStrategy, Task
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec
from holdout.plan.compiler import evaluate_cv
from holdout.plan.defaults import baseline_plan, default_plan
from holdout.plan.schema import DropColumn

V11 = load_thresholds_v11()
SPEC = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=3)
TOKENS = V11[0].a03_id_like.key_name_tokens


def test_key_like_names_need_a_key_token_at_the_end() -> None:
    assert name_tokens("patientNbr") == ["patient", "nbr"]
    for name in ("record_number", "encounter_id", "patientNbr", "id", "order-no", "SKU code"):
        assert key_like_name(name, TOKENS), name
    for name in ("number_vmail_messages", "freeswap", "V3", "identity", "idle_time", "body"):
        assert not key_like_name(name, TOKENS), name


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame({
        # An integer measurement, near-unique, informative: NOT an identifier.
        "freeswap": (1_000_000 + 50_000 * y + rng.integers(0, 40_000, n)).astype("int64"),
        # A consecutive record number issued in outcome order: an identifier with signal.
        "record_number": np.argsort(np.argsort(y + rng.random(n) * 0.5)).astype("int64") + 1,
        "noise": rng.normal(size=n),
        "converted": y,
    })  # fmt: skip


def _a3(
    frame: pd.DataFrame, v11: bool, drop: str | None = None
) -> dict[str, tuple[Severity, list[str], str]]:
    profile = profile_schema(frame, "converted", task=Task.CLASSIFICATION)
    plan = default_plan(profile, Task.CLASSIFICATION)
    if drop is not None:
        extra = DropColumn(column=drop, reason="dropped for the test")
        plan = plan.model_copy(update={"drop_columns": [*plan.drop_columns, extra]})
    dev = frame.sample(frac=0.8, random_state=1)
    report = run_audits(
        None, dev, plan, evaluate_cv(plan, dev, "converted", SPEC, profile),
        baseline=evaluate_cv(baseline_plan(plan), dev, "converted", SPEC, profile),
        profile=profile, spec=SPEC, target="converted", thresholds=load_thresholds(),
        thresholds_v11=V11 if v11 else None,
        checks=frozenset({CheckId.A1, CheckId.A2, CheckId.A3}),
    )  # fmt: skip
    return {f.columns[0]: (f.severity, f.evidence.signals, f.message)  # type: ignore[union-attr]
            for f in report.flags if f.check is CheckId.A3}  # fmt: skip


def test_an_integer_measurement_is_no_longer_an_identifier() -> None:
    frame = _frame()
    v10, v11 = _a3(frame, v11=False), _a3(frame, v11=True)
    assert "freeswap" in v10  # the v1.0 false positive
    assert "freeswap" not in v11
    severity, signals, _ = v11["record_number"]
    assert severity is Severity.CRITICAL  # the real identifier is still caught
    assert "integer_sequence" in signals
    assert v10["record_number"][0] is Severity.CRITICAL


def test_the_message_claims_an_ordering_only_with_row_order_evidence() -> None:
    # Shuffled rows: record_number is a sequence with signal but no row-order correlation.
    shuffled = _frame().sample(frac=1.0, random_state=5).reset_index(drop=True)
    severity, signals, message = _a3(shuffled, v11=True, drop="record_number")["record_number"]
    assert severity is Severity.WARNING  # dropped by the plan
    assert "row_order_correlation" not in signals
    assert "ordering encodes the outcome" not in message
    assert "the plan drops it" in message
    v10_message = _a3(shuffled, v11=False, drop="record_number")["record_number"][2]
    assert "ordering encodes the outcome" in v10_message  # the v1.0 text, kept for v1.0
    # Rows in record order: the correlation is there, and so is the claim.
    ordered = shuffled.sort_values("record_number").reset_index(drop=True)
    severity, signals, message = _a3(ordered, v11=True, drop="record_number")["record_number"]
    assert "row_order_correlation" in signals
    assert "ordering encodes the outcome" in message


# --------------------------------------------------------------- the benchmark's frames


def _datasets_cached() -> bool:
    try:
        from bench.data.openml import cached_path
        from bench.data.registry import load_registry
    except ImportError:
        return False
    registry = load_registry()
    return all(cached_path(s).exists() for s in registry.values() if s.pool == "final")


def _raw_a3(task_id: str, seed: int) -> dict[str, Severity]:
    from bench.harness import partition
    from bench.run import Unit, prepare

    unit = Unit("final_v1", task_id, "audits_only", seed, "planted")
    p = prepare(unit)
    dev, _, _ = partition(p.frame, p.target, p.rule, seed)
    profile = profile_schema(dev, p.target, task=p.task)
    plan = default_plan(profile, p.task)
    spec = SplitSpec(strategy=SplitStrategy.STRATIFIED if p.task is Task.CLASSIFICATION
                     else SplitStrategy.RANDOM, seed=seed)  # fmt: skip
    report = run_audits(
        None, dev, plan, evaluate_cv(plan, dev, p.target, spec, profile),
        baseline=evaluate_cv(baseline_plan(plan), dev, p.target, spec, profile),
        profile=profile, spec=spec, target=p.target, thresholds=load_thresholds(),
        thresholds_v11=V11, checks=frozenset({CheckId.A1, CheckId.A2, CheckId.A3}),
    )  # fmt: skip
    return {f.columns[0]: f.severity for f in report.flags if f.check is CheckId.A3}


@pytest.mark.full_only
@pytest.mark.skipif(not _datasets_cached(), reason="needs the OpenML files (no network in CI)")
def test_the_benchmark_false_positives_are_gone_and_the_identifiers_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flagged = _raw_a3("cross_split_duplicates:clean@cpu_act", 202)
    assert "freeswap" not in flagged
    steel = _raw_a3("noisy_proxy:clean@steel_plates_fault", 101)
    assert not {"V3", "V4", "V8"} & set(steel)
    for task in ("id_with_signal@credit_card_default", "id_with_signal@wilt"):
        assert _raw_a3(task, 101).get("record_number") is Severity.CRITICAL, task
