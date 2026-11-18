"""The runner's per-invocation spend cap and the pilot report."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bench.config import load_bench_config
from bench.pilot import FULL_UNITS, pilot_report
from bench.run import Unit, append_result, run_all
from bench.systems.base import BenchResult, SystemContext, SystemOutput
from holdout.llm import config
from holdout.llm.budget import SpendLedger


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SpendLedger:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)
    return SpendLedger()


class _Spender:
    """A system that spends ``inr`` per run (recorded in the ledger) and returns nothing."""

    name = "holdout"
    uses_llm = True

    def __init__(self, ledger: SpendLedger, inr: float) -> None:
        self.ledger = ledger
        self.inr = inr
        self.runs = 0

    def run(self, ctx: SystemContext) -> SystemOutput:
        self.runs += 1
        self.ledger.append(
            {"kind": "call", "timestamp": datetime.now(UTC).isoformat(), "inr": self.inr}
        )
        return SystemOutput(status="stopped", cost_inr=self.inr)

    def resume(self, ctx: SystemContext) -> SystemOutput:
        return self.run(ctx)


def _prepared(unit: Unit) -> Any:
    from bench.harness import PartitionRule
    from bench.run import PreparedUnit, normalise
    from bench.traps import build_case

    case = build_case("direct_leak", 0)
    return PreparedUnit(
        normalise(case.df), case.target, case.task, case.goal_text, None, {}, PartitionRule()
    )


def test_the_spend_cap_stops_after_the_unit_that_reaches_it(
    tmp_path: Path, ledger: SpendLedger
) -> None:
    spender = _Spender(ledger, 12.0)
    units = [Unit("dev_v1", f"t{i}", "holdout", 11, "planted") for i in range(5)]
    lines: list[str] = []
    done = run_all(
        units,
        {"holdout": spender},
        tmp_path / "r",
        model="m",
        run_cap_inr=15.0,
        write=lines.append,
        prepare_fn=_prepared,
        max_spend_inr=30.0,
        ledger=ledger,
    )
    assert spender.runs == 3  # 12, 24 (< 30: continue), 36 (>= 30: stop after this unit)
    assert len(done) == 3
    assert "spend cap reached: Rs 36.00 of Rs 30.00" in lines[-1]
    assert "stopped cleanly" in lines[-1]


def test_retry_failed_reruns_failed_units_only_when_asked(tmp_path: Path) -> None:
    from bench.run import pending

    root = tmp_path / "r"
    units = [Unit("dev_v1", "t", "holdout", 11, "planted")]
    append_result(root, _result(status="failed", error="HTTP 400"))
    assert pending(units, root) == []
    assert pending(units, root, retry_failed=True) == [(units[0], False)]
    append_result(root, _result(status="completed"))
    assert pending(units, root, retry_failed=True) == []


def test_bench_schemas_are_sent_without_length_bounds_but_still_validated() -> None:
    import json

    from pydantic import BaseModel, Field, ValidationError

    from bench.systems.llm_judge import JudgeVerdict
    from bench.systems.react import ReactStep
    from holdout.llm.structured import llm_schema

    for schema in (JudgeVerdict, ReactStep):
        sent = json.dumps(llm_schema(schema, length_bounds=False))
        for keyword in ('"maxItems"', '"minItems"', '"maxLength"', '"minLength"'):
            assert keyword not in sent
        assert '"maxLength"' in json.dumps(llm_schema(schema))  # default keeps them
    with pytest.raises(ValidationError):
        ReactStep.model_validate({"thought": "", "action": "python", "code": "x"})

    class Odd(BaseModel):
        maxLength: int = Field(max_length=None)  # noqa: N815 - a property with a keyword name

    assert "maxLength" in llm_schema(Odd, length_bounds=False)["properties"]


def _result(**kw: Any) -> BenchResult:
    now = datetime.now(UTC)
    base: dict[str, Any] = {
        "suite": "dev_v1", "task_id": "t", "system": "holdout", "seed": 11,
        "variant": "planted", "status": "completed", "benchmark_metric": "roc_auc",
        "started_at": now, "finished_at": now,
    }  # fmt: skip
    base.update(kw)
    return BenchResult(**base)


def test_the_pilot_report_explains_spend_and_projects_the_full_run(
    tmp_path: Path, ledger: SpendLedger
) -> None:
    ledger.record_balance(740.0, datetime.now().astimezone().date())
    results = tmp_path / "pilot"
    append_result(
        results,
        _result(cost_inr=1.0, seconds=60, extra={"inr_by_node": {"plan": 0.3, "report": 0.7}}),
    )
    append_result(
        results,
        _result(
            system="react",
            cost_inr=4.0,
            seconds=120,
            extra={
                "inr_by_node": {"react": 4.0},
                "inr_by_step": [0.5, 1.5, 2.0],
                "steps": 3,
                "prompt_tokens": 9000,
                "observation_tokens_sent_estimate": 1000,
                "observation_tokens_sent_untruncated_estimate": 5000,
            },
        ),
    )
    ledger.append({"kind": "call", "timestamp": datetime.now(UTC).isoformat(), "inr": 5.5})
    text = pilot_report(results, "dev_v1", load_bench_config(), ledger=ledger)
    assert "| react | react | 4.000 |" in text
    assert "| holdout | report | 0.700 |" in text
    assert "Rs per LLM call: 0.500, 1.500, 2.000" in text
    assert "unattributed (e.g. a unit that failed or paused after spending" in text
    assert "Rs 0.500" in text.split("unattributed")[1][:200]
    assert f"| holdout | {1.0 * FULL_UNITS * 1.2:.2f} | {1.0 * FULL_UNITS * 1.2:.2f} |" in text
    react_row = next(line for line in text.splitlines() if line.startswith("| react | 5"))
    with_trunc, without = (float(c) for c in react_row.split("|")[2:4])
    assert without > with_trunc  # the untruncated history costs more
    assert "balance left (estimate): Rs" in text
