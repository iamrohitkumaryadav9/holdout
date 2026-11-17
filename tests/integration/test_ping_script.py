"""scripts/ping_gemini.py: one minimal request per model, 30 s timeout, one attempt, no
cache, same caps and ledger as agent runs (no network: the SDK is stubbed)."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import ping_gemini

from holdout.llm import config
from holdout.llm.budget import SpendLedger
from holdout.llm.client import GeminiClient
from holdout.llm.retry import BillingLLMError, RetryPolicy, TransientLLMError


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)
    return tmp_path / "home"


def _install(monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, Any]) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    class _Gemini(GeminiClient):
        def __init__(self, model: str, *, policy: RetryPolicy, timeout_seconds: float) -> None:
            def generate_content(*, model: str, contents: str, config: Any) -> Any:
                seen.append(
                    {"model": model, "contents": contents, "mime": config.response_mime_type}
                )
                outcome = outcomes[model]
                if isinstance(outcome, Exception):
                    raise outcome
                usage = SimpleNamespace(
                    prompt_token_count=8, candidates_token_count=1, thoughts_token_count=40
                )
                return SimpleNamespace(text="OK", usage_metadata=usage)

            seen.append({"policy": policy, "timeout": timeout_seconds})
            sdk = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
            super().__init__(model, sdk_client=sdk, policy=policy, sleep=lambda s: None)

    monkeypatch.setattr(ping_gemini, "GeminiClient", _Gemini)
    return seen


def test_ping_sends_one_plain_request_per_model_and_charges_the_ledger(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    models = sorted(config.PRICE_TABLE)
    seen = _install(monkeypatch, dict.fromkeys(models, "ok"))
    SpendLedger().record_balance(747.0, datetime.now().astimezone().date())
    monkeypatch.setattr("sys.argv", ["ping_gemini"])
    assert ping_gemini.main() == 0
    out = capsys.readouterr().out
    requests = [s for s in seen if "contents" in s]
    assert [r["model"] for r in requests] == models  # one request per priced model
    assert {r["contents"] for r in requests} == {"Reply with the single word OK"}
    assert {r["mime"] for r in requests} == {None}  # plain text, not JSON
    settings = [s for s in seen if "policy" in s]
    assert {(s["policy"].max_attempts, s["timeout"]) for s in settings} == {(1, 30.0)}
    assert out.startswith("before: project spend Rs 0.00 of Rs 750")
    assert "balance left about Rs 747.00" in out.splitlines()[0]
    assert out.count("OK HTTP 200") == len(models)
    assert "tokens in 8 out 1 thinking 40" in out
    assert "| project Rs" in out
    assert "reply 'OK'" in out
    after = [line for line in out.splitlines() if line.startswith("after:")]
    assert len(after) == 1
    assert len(SpendLedger().calls()) == len(models)
    assert SpendLedger().total_inr() > 0


def test_ping_reports_failures_with_exit_codes(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        monkeypatch,
        {
            "gemini-3.8-flash": TransientLLMError(503),
            "gemini-3.6-flash": BillingLLMError(429, "prepayment credits are depleted"),
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["ping_gemini", "--model", "gemini-3.8-flash", "--model", "gemini-3.6-flash"],
    )
    assert ping_gemini.main() == 7  # billing outranks unavailability
    out = capsys.readouterr().out
    assert "gemini-3.8-flash         UNAVAILABLE HTTP 503" in out
    assert "gemini-3.6-flash         BILLING HTTP 429" in out
    assert [e["kind"] for e in SpendLedger().entries()] == ["failed", "failed"]
    assert SpendLedger().total_inr() == 0.0


def test_ping_obeys_the_project_cap(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _install(monkeypatch, {"gemini-3.8-flash": "ok"})
    SpendLedger().append(
        {"kind": "call", "timestamp": datetime.now(UTC).isoformat(), "inr": 749.999}
    )
    monkeypatch.setattr("sys.argv", ["ping_gemini", "--model", "gemini-3.8-flash"])
    assert ping_gemini.main() == 6
    assert "NOT SENT: project cap" in capsys.readouterr().out
    assert seen == []  # no client was even built


def test_ping_refuses_an_unpriced_model(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["ping_gemini", "--model", "gemini-9-ultra"])
    assert ping_gemini.main() == 2
    assert "no price" in capsys.readouterr().err
