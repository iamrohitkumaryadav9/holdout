"""Amendment 1: the pre-flight prices the observed standard-fallback share at the standard
rate (0% until 10 LLM units have completed); the unit worst case stays at the caps."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from bench.budget import (
    FALLBACK_MIN_UNITS,
    Measured,
    estimate,
    observed_fallback_share,
    unit_worst_case,
)
from bench.config import load_bench_config
from bench.run import append_result
from bench.systems.base import BenchResult, SystemName
from holdout.llm import config
from holdout.llm.budget import SpendLedger

DAY = date(2026, 11, 20)
FLEX_MEASURED = {"react": Measured(6.0, 3, DAY, "flex")}


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)
    SpendLedger().record_balance(5000.0, datetime.now().astimezone().date())


def test_the_fallback_share_is_priced_at_standard() -> None:
    cfg = load_bench_config()  # pins flex
    flex = estimate({"react": 10}, cfg, purpose="benchmark", measured=FLEX_MEASURED, today=DAY)
    half = estimate({"react": 10}, cfg, purpose="benchmark", measured=FLEX_MEASURED, today=DAY,
                    fallback_share=0.5, fallback_note="50.0% measured")  # fmt: skip
    every = estimate({"react": 10}, cfg, purpose="benchmark", measured=FLEX_MEASURED,
                     today=DAY, fallback_share=1.0)  # fmt: skip
    assert flex.total_inr == pytest.approx(10 * 6.0 * 1.2)
    assert half.total_inr == pytest.approx(flex.total_inr * 1.5)  # standard costs 2x Flex
    assert every.total_inr == pytest.approx(flex.total_inr * 2)
    assert "fallback share priced at standard: 50.0% measured" in half.render()
    # The worst case per unit (and so the guard and the headroom) stays at the caps.
    assert half.headroom_inr == flex.headroom_inr == unit_worst_case("react", 15.0)
    with pytest.raises(ValueError, match="fallback_share"):
        estimate({"react": 1}, cfg, purpose="benchmark", measured=FLEX_MEASURED,
                 fallback_share=1.5)  # fmt: skip


def _done(i: int, system: SystemName, calls: int, fallbacks: int,
          status: str = "completed") -> BenchResult:  # fmt: skip
    now = datetime.now(UTC)
    return BenchResult(suite="final_v1", task_id=f"t{i}", system=system, seed=101,
                       variant="planted", status=status, benchmark_metric="roc_auc",  # type: ignore[arg-type]
                       calls=calls, fallbacks=fallbacks, started_at=now,
                       finished_at=now)  # fmt: skip


def test_the_share_is_measured_from_completed_llm_units_only_after_ten(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for i in range(FALLBACK_MIN_UNITS - 1):
        append_result(root, _done(i, "react", 10, 4))
    share, note = observed_fallback_share(root, ["final_v1", "real_world_v1"])
    assert share == 0.0
    assert "0% assumed: 9 completed LLM unit(s)" in note
    append_result(root, _done(99, "holdout", 10, 0))
    append_result(root, _done(98, "react", 10, 10, status="paused"))  # not completed
    append_result(root, _done(97, "flaml", 0, 0))  # no LLM
    share, note = observed_fallback_share(root, ["final_v1", "real_world_v1"])
    assert share == pytest.approx(36 / 100)
    assert "36 of 100 calls in 10 LLM units" in note
