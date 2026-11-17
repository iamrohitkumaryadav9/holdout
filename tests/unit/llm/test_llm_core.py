"""Prices, budget caps and ledger, cache keys, the structured-call contract, GeminiClient."""

import json
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, Field

from holdout.core.errors import (
    BudgetExceededError,
    LLMConfigError,
    LLMOutputError,
    LLMRequestError,
    LLMUnavailableError,
)
from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger, cost_inr, estimate_inr
from holdout.llm.cache import ResponseCache, cache_key
from holdout.llm.client import FakeLLM, GeminiClient, LLMRequest, Usage
from holdout.llm.structured import LLMSession, RenderedPrompt
from holdout.plan.schema import ModelPlan
from tests.helpers import plan_dict
from tests.tiers import examples


class Answer(BaseModel):
    word: str = Field(min_length=1)
    count: int = Field(ge=0)


PROMPT = RenderedPrompt(name="t", version="1", sha256="abc", system="sys", user="data")


def _session(tmp_path: Path, llm: FakeLLM, seed: int = 1, cap: float = 15.0) -> LLMSession:
    budget = Budget(
        run_id="r", model=llm.model, run_cap_inr=cap, ledger=SpendLedger(tmp_path / "ledger.jsonl")
    )
    return LLMSession(client=llm, budget=budget, seed=seed, cache=ResponseCache(tmp_path / "cache"))


# ------------------------------------------------------------------ config and budget


def test_prices_double_from_2027() -> None:
    late_2026 = config.price_for(config.DEFAULT_MODEL, date(2026, 12, 31))
    early_2027 = config.price_for(config.DEFAULT_MODEL, date(2027, 1, 1))
    assert early_2027.input_usd_per_million == 2 * late_2026.input_usd_per_million
    assert early_2027.output_usd_per_million == 2 * late_2026.output_usd_per_million
    with pytest.raises(KeyError):
        config.price_for("unknown-model", date(2026, 12, 1))
    with pytest.raises(KeyError, match="no price"):
        config.price_for(config.DEFAULT_MODEL, date(2025, 1, 1))


def test_cost_counts_thinking_as_output() -> None:
    on = date(2026, 11, 17)
    usage = Usage(prompt_tokens=1_000_000, output_tokens=0, thinking_tokens=1_000_000)
    expected = (0.75 + 3.75) * config.USD_TO_INR
    assert cost_inr(config.DEFAULT_MODEL, usage, on) == pytest.approx(expected)


def test_holdout_home_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path))
    assert config.holdout_home() == tmp_path
    monkeypatch.delenv("HOLDOUT_HOME")
    assert config.holdout_home() == Path.home() / ".holdout"


def test_run_cap_stops_before_the_call(tmp_path: Path) -> None:
    llm = FakeLLM(default={"t": lambda r: {"word": "a", "count": 1}})
    session = _session(tmp_path, llm, cap=0.01)  # below one worst-case call
    with pytest.raises(BudgetExceededError, match="run cap"):
        session.call(node="n", prompt=PROMPT, schema=Answer)
    assert llm.calls == []  # never sent


def test_project_cap_reads_the_ledger(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    ledger.append({"inr": 749.99})
    budget = Budget(run_id="r", model=config.DEFAULT_MODEL, ledger=ledger)
    request = LLMRequest("t", "1", "s", "u", {}, seed=1)
    with pytest.raises(BudgetExceededError, match="project cap"):
        budget.check(request, "n")


def test_ledger_persists_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    code = (
        "from pathlib import Path; from holdout.llm.budget import SpendLedger; "
        f"SpendLedger(Path({str(path)!r})).append({{'inr': 2.5}})"
    )
    for _ in range(2):
        subprocess.run([sys.executable, "-c", code], check=True)
    assert SpendLedger(path).total_inr() == pytest.approx(5.0)


def test_estimate_is_a_worst_case(tmp_path: Path) -> None:
    request = LLMRequest("t", "1", "s" * 400, "u" * 400, {}, seed=1, max_output_tokens=100)
    real = Usage(prompt_tokens=200, output_tokens=100, thinking_tokens=0)
    on = date(2026, 11, 17)
    assert estimate_inr(config.DEFAULT_MODEL, request, on) >= cost_inr(
        config.DEFAULT_MODEL, real, on
    )


# ------------------------------------------------------------------ cache


def test_cache_key_covers_seed_version_model_and_text() -> None:
    base = LLMRequest("t", "1", "s", "u", {"type": "object"}, seed=1)
    key = cache_key("m", base)
    assert key != cache_key("m", LLMRequest("t", "1", "s", "u", {"type": "object"}, seed=2))
    assert key != cache_key("m", LLMRequest("t", "2", "s", "u", {"type": "object"}, seed=1))
    assert key != cache_key("other", base)
    assert key != cache_key("m", LLMRequest("t", "1", "s", "u!", {"type": "object"}, seed=1))
    assert key == cache_key("m", LLMRequest("t", "1", "s", "u", {"type": "object"}, seed=1))


def test_cache_hit_costs_nothing_and_seeds_never_share(tmp_path: Path) -> None:
    llm = FakeLLM(default={"t": lambda r: {"word": "a", "count": r.seed}})
    first = _session(tmp_path, llm, seed=1)
    assert first.call(node="n", prompt=PROMPT, schema=Answer).count == 1
    again = _session(tmp_path, llm, seed=1)
    assert again.call(node="n", prompt=PROMPT, schema=Answer).count == 1
    assert len(llm.calls) == 1  # served from the cache
    assert again.records[0].cache_hit
    assert again.records[0].inr == 0.0
    other_seed = _session(tmp_path, llm, seed=2)
    assert other_seed.call(node="n", prompt=PROMPT, schema=Answer).count == 2
    assert len(llm.calls) == 2  # a different seed never reuses seed 1's response


def test_disabled_cache_always_calls(tmp_path: Path) -> None:
    llm = FakeLLM(default={"t": lambda r: {"word": "a", "count": 1}})
    for _ in range(2):
        session = _session(tmp_path, llm)
        session.cache = ResponseCache(tmp_path / "cache", enabled=False)
        session.call(node="n", prompt=PROMPT, schema=Answer)
    assert len(llm.calls) == 2


# ------------------------------------------------------------------ structured calls


def test_invalid_answer_is_repaired_once(tmp_path: Path) -> None:
    llm = FakeLLM(scripts={"t": ['{"word": ""}', {"word": "ok", "count": 2}]})
    session = _session(tmp_path, llm)
    assert session.call(node="n", prompt=PROMPT, schema=Answer).word == "ok"
    repair = llm.calls[1].user
    assert "<repair>" in repair
    assert "word" in repair
    assert [r.attempt for r in session.records] == [1, 2]
    assert [r.valid for r in session.records] == [False, True]


def test_second_invalid_answer_fails_loudly(tmp_path: Path) -> None:
    llm = FakeLLM(scripts={"t": ["not json", '```json\n{"count": -1}\n```']})
    with pytest.raises(LLMOutputError, match="invalid after one repair"):
        _session(tmp_path, llm).call(node="n", prompt=PROMPT, schema=Answer)
    assert len(llm.calls) == 2  # exactly one repair


def test_semantic_check_triggers_repair(tmp_path: Path) -> None:
    llm = FakeLLM(scripts={"t": [{"word": "bad", "count": 1}, {"word": "good", "count": 1}]})
    answer = _session(tmp_path, llm).call(
        node="n",
        prompt=PROMPT,
        schema=Answer,
        check=lambda a: ["word must not be 'bad'"] if a.word == "bad" else [],
    )
    assert answer.word == "good"
    assert "word must not be 'bad'" in llm.calls[1].user


def test_invalid_answers_are_not_cached(tmp_path: Path) -> None:
    llm = FakeLLM(scripts={"t": ["nope", {"word": "x", "count": 0}]})
    _session(tmp_path, llm).call(node="n", prompt=PROMPT, schema=Answer)
    replay = FakeLLM(scripts={"t": ["still nope", {"word": "y", "count": 0}]})
    _session(tmp_path, replay).call(node="n", prompt=PROMPT, schema=Answer)
    # The invalid first answer was never cached, so attempt 1 is sent again; its repair quotes
    # a different previous answer, so it is a different request and is sent too.
    assert replay.calls[0].user == "data"
    assert len(replay.calls) == 2
    cached = list((tmp_path / "cache").rglob("*.json"))
    assert len(cached) == 2  # only the two valid (repaired) answers


@settings(max_examples=examples(60, 20), deadline=None)
@given(
    payload=st.one_of(
        st.text(max_size=80),
        st.dictionaries(
            st.text(max_size=12), st.one_of(st.integers(), st.text(max_size=8)), max_size=5
        ),
        st.fixed_dictionaries(
            {
                "task": st.sampled_from(["classification", "regression", "ranking"]),
                "metric": st.text(max_size=10),
                "model": st.dictionaries(st.sampled_from(["name", "C", "max_iter"]), st.integers()),
            }
        ),
    )
)
def test_fuzzed_plan_json_is_rejected_or_valid_never_unvalidated(payload: Any) -> None:
    import tempfile

    text = payload if isinstance(payload, str) else json.dumps(payload)
    llm = FakeLLM(scripts={"t": [text, text]})
    with tempfile.TemporaryDirectory() as tmp:
        try:
            plan = _session(Path(tmp), llm).call(node="plan", prompt=PROMPT, schema=ModelPlan)
        except LLMOutputError:
            return
    assert isinstance(plan, ModelPlan)  # only a fully validated plan ever comes back
    ModelPlan.model_validate(plan.model_dump())


def test_a_valid_plan_passes_through(tmp_path: Path) -> None:
    llm = FakeLLM(scripts={"t": [plan_dict()]})
    plan = _session(tmp_path, llm).call(node="plan", prompt=PROMPT, schema=ModelPlan)
    assert plan.model.name == "hist_gb"


# ------------------------------------------------------------------ GeminiClient (no network)


class _FakeModels:
    def __init__(self, outcomes: list[Any], names: list[str]) -> None:
        self.outcomes = outcomes
        self.names = names
        self.configs: list[Any] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.configs.append(config)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def list(self) -> list[Any]:
        return [SimpleNamespace(name=f"models/{n}") for n in self.names]


class _StatusError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"status {code}")
        self.code = code


def _response(text: str) -> Any:
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=7)
    return SimpleNamespace(text=text, usage_metadata=usage)


def _gemini(
    outcomes: list[Any], names: list[str] | None = None
) -> tuple[GeminiClient, _FakeModels]:
    models = _FakeModels(outcomes, names or [config.DEFAULT_MODEL, "gemini-other-flash"])
    client = GeminiClient(sdk_client=SimpleNamespace(models=models), sleep=lambda s: None)
    return client, models


def test_gemini_request_uses_schema_seed_and_thinking_cap() -> None:
    client, models = _gemini([_response('{"word": "a", "count": 1}')])
    request = LLMRequest("t", "1", "sys", "user", {"type": "object"}, seed=42)
    response = client.generate(request)
    sent = models.configs[0]
    assert sent.seed == 42
    assert sent.response_mime_type == "application/json"
    assert sent.response_json_schema == {"type": "object"}
    assert sent.thinking_config.thinking_level.value == config.THINKING_LEVEL.upper()
    assert sent.max_output_tokens == config.MAX_OUTPUT_TOKENS
    assert response.usage == Usage(prompt_tokens=10, output_tokens=5, thinking_tokens=7)
    assert client.supports_seed


def test_gemini_retries_transient_errors_then_fails_typed() -> None:
    client, models = _gemini([_StatusError(429), _StatusError(503), _response("{}")])
    response = client.generate(LLMRequest("t", "1", "s", "u", {}, seed=1))
    assert response.text == "{}"
    assert response.retries == 2
    assert len(models.configs) == 3  # two transient failures, then success
    client, models = _gemini([_StatusError(400)])
    with pytest.raises(LLMRequestError, match="HTTP 400") as rejected:
        client.generate(LLMRequest("t", "1", "s", "u", {}, seed=1))
    assert rejected.value.status == 400
    assert models.outcomes == []  # sent exactly once
    client, models = _gemini([_StatusError(503)] * 6)
    with pytest.raises(LLMUnavailableError, match="HTTP 503 on all 6 attempts") as down:
        client.generate(LLMRequest("t", "1", "s", "u", {}, seed=1))
    assert (down.value.status, down.value.retries) == (503, 5)
    assert len(models.configs) == 6


def test_gemini_model_verification() -> None:
    client, _ = _gemini([])
    assert config.DEFAULT_MODEL in client.verify_model()
    assert client.model_verified
    missing, _ = _gemini([], names=["gemini-1-flash"])
    with pytest.raises(LLMConfigError, match="not available"):
        missing.verify_model()


def test_gemini_needs_a_key_from_the_environment() -> None:
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY is not set"):
        GeminiClient()


def test_sockets_are_blocked_in_tests() -> None:
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("example.com", 443))
