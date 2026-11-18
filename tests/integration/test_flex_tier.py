"""The Flex tier: requested per call from the pinned config, a 15-minute request timeout,
priced at the tier that served each call, and API wait kept apart from compute time (the
node limit and "time per run" count compute only)."""

import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from bench.config import CONFIG_PATH, load_bench_config
from bench.harness import PartitionRule
from bench.run import PreparedUnit, Unit, normalise, run_unit
from bench.systems import registry
from bench.systems.react import ReactSystem
from bench.traps import build_case
from holdout.agent.graph import check_node_time
from holdout.core.errors import AgentError, LLMRequestError
from holdout.llm import config as llm_config
from holdout.llm.budget import Budget
from holdout.llm.cache import ResponseCache
from holdout.llm.client import FakeLLM, LLMRequest
from holdout.llm.structured import LLMSession, RenderedPrompt
from tests.agent_harness import fake_for, run_case, write_case
from tests.integration.test_bench_systems import REACT_CODE, REACT_FINAL

CASE = build_case("direct_leak", 0)
MODEL = "gemini-3.8-flash"
DELAY = 0.15  # seconds each fake call "waits on the API"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(llm_config.TRUST_CACHE_DISCOUNT_ENV, raising=False)
    return tmp_path / "home"


def _prepared(unit: Unit) -> PreparedUnit:
    return PreparedUnit(
        frame=normalise(CASE.df),
        target=CASE.target,
        task=CASE.task,
        goal=CASE.goal_text,
        entity_scope=None,
        plan_overrides=dict(CASE.plan_overrides),
        rule=PartitionRule(),
    )


def _slow(item: dict[str, Any]) -> Any:
    def respond(request: LLMRequest) -> dict[str, Any]:
        time.sleep(DELAY)
        return item

    return respond


def _react(tmp_path: Path, tier: str | None) -> tuple[Any, list[LLMRequest]]:
    step = {"thought": "Fit and submit.", "action": "python", "code": REACT_CODE}
    fake = FakeLLM(scripts={"react": [_slow(step), _slow(REACT_FINAL)]})
    unit = Unit("dev_v1", "direct_leak@synthetic", "react", 11, "planted")
    result = run_unit(
        unit, ReactSystem(lambda: fake), tmp_path / (tier or "default"), model=MODEL,
        run_cap_inr=15.0, prepare_fn=_prepared, service_tier=tier,
    )  # fmt: skip
    return result, fake.calls


def test_flex_is_requested_on_every_call_recorded_and_priced_at_half(tmp_path: Path) -> None:
    flex, flex_calls = _react(tmp_path, "flex")
    standard, standard_calls = _react(tmp_path, None)
    assert flex.status == standard.status == "completed", (flex.error, standard.error)
    assert [c.service_tier for c in flex_calls] == ["flex", "flex"]
    assert [c.service_tier for c in standard_calls] == [None, None]  # the API default
    assert flex.tiers_served == {"flex": 2}
    assert standard.tiers_served == {"standard": 2}
    assert flex.prompt_tokens == standard.prompt_tokens > 0
    assert flex.cost_inr == pytest.approx(standard.cost_inr / 2)


def test_api_wait_is_recorded_apart_from_compute_time(tmp_path: Path) -> None:
    result, _ = _react(tmp_path, "flex")
    assert result.api_seconds >= 2 * DELAY
    assert result.compute_seconds == pytest.approx(result.seconds - result.api_seconds, abs=0.02)
    assert result.time_per_run == result.compute_seconds
    # Results written before API wait was recorded report their wall time.
    old = result.model_copy(update={"api_seconds": 0.0, "compute_seconds": 0.0})
    assert old.time_per_run == result.seconds


def test_api_time_counts_even_when_the_request_is_refused() -> None:
    class _Refusing(FakeLLM):
        def generate(self, request: LLMRequest, on_retry: Any = None) -> Any:
            time.sleep(DELAY)
            raise LLMRequestError("HTTP 400")

    session = LLMSession(
        client=_Refusing(), budget=Budget(run_id="r", model=MODEL), seed=1,
        cache=ResponseCache(enabled=False), service_tier="flex",
    )  # fmt: skip
    prompt = RenderedPrompt("t", "1", "x", "s", "u")

    class Word(BaseModel):
        word: str

    with pytest.raises(LLMRequestError):
        session.call(node="n", prompt=prompt, schema=Word)
    assert session.api_seconds >= DELAY


def test_the_node_limit_counts_compute_only() -> None:
    check_node_time("plan", elapsed=900.0, api_seconds=850.0, limit=600.0)  # 50 s compute
    with pytest.raises(AgentError, match="used 601 s of compute"):
        check_node_time("plan", elapsed=1401.0, api_seconds=800.0, limit=600.0)
    check_node_time("plan", elapsed=1.0, api_seconds=2.0, limit=0.5)  # clock noise: 0 compute


def test_the_agent_requests_and_records_the_tier(tmp_path: Path) -> None:
    fake = fake_for(CASE, write_case(CASE, tmp_path / "case"))
    outcome = run_case(CASE, tmp_path, llm=fake, service_tier="flex")
    assert outcome.manifest is not None
    agent = outcome.manifest.agent
    assert agent is not None
    assert agent.service_tier == "flex"
    assert {c.service_tier for c in fake.calls} == {"flex"}
    tiers = {t for u in agent.usage_by_node.values() for t in u.tiers}
    assert tiers == {"flex"}
    assert agent.api_seconds >= 0.0


def test_flex_clients_wait_up_to_fifteen_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []

    class _Gemini:
        model = MODEL

        def __init__(self, model: str, *, timeout_seconds: float) -> None:
            seen.append(timeout_seconds)

        def verify_model(self) -> None:
            return None

    monkeypatch.setattr(registry, "GeminiClient", _Gemini)
    registry.gemini_factory(MODEL, "flex")()
    registry.gemini_factory(MODEL, None)()
    assert llm_config.FLEX_REQUEST_TIMEOUT_SECONDS == 15 * 60
    assert seen == [15 * 60, llm_config.REQUEST_TIMEOUT_SECONDS]


def test_the_bench_config_pins_flex_and_rejects_unknown_tiers(tmp_path: Path) -> None:
    assert load_bench_config().service_tier == "flex"
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        'service_tier = "flex"', 'service_tier = "priority"'
    )
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown service tier"):
        load_bench_config(path)
