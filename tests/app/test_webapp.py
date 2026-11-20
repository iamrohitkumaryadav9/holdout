"""Smoke tests of the Streamlit app with AppTest (no browser, no network).

The app is offline by default; a visitor's Gemini key lives in the session only (never in
``os.environ``, never in a file). Critical findings and warnings are shown apart, with
different wording; the Benchmark page leads with the pre-registered v1.0 results."""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from bench.traps import build_case
from holdout import __version__
from holdout.llm import config as llm_config
from holdout.webapp import core, text
from tests.agent_harness import fake_for, write_case

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")
FAKE_KEY = "FAKE-test-key_not-real_0123456789"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(llm_config.API_KEY_ENV, raising=False)
    return tmp_path / "home"


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_every_page_renders_and_shows_the_version(home: Path) -> None:
    at = _app()
    assert at.title[0].value == "Run Holdout on a CSV"
    assert at.radio(key="engine").value.startswith("Offline")  # offline by default
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert f"[{__version__}]({text.TAG_URL})" in sidebar
    assert text.TAG_URL.endswith(f"/tree/v{__version__}")
    for page in ("Benchmark", "About"):
        at.radio(key="page").set_value(page).run()
        assert not at.exception, [e.value for e in at.exception]
    assert f"tag v{__version__}" in at.markdown[1].value or any(
        f"tag v{__version__}" in m.value for m in at.markdown
    )


def test_the_benchmark_page_leads_with_the_preregistered_results(home: Path) -> None:
    at = _app()
    at.radio(key="page").set_value("Benchmark").run()
    headers = [h.value for h in at.header]
    assert headers == [text.PREREGISTERED_LABEL, text.POSTHOC_LABEL]
    body = "\n".join(m.value for m in at.markdown)
    assert body.index("51.8% [33.9%, 69.6%]") < body.index("77.8% -> 88.9%")
    assert "exact McNemar p = 0.27, not significant" in body
    assert body.count(text.OPTIMISTIC) >= 2
    for tradeoff in text.TRADEOFFS:
        assert tradeoff in body
    assert "### Detection recall" in body  # the generated tables, not retyped numbers


def test_a_gemini_key_stays_in_the_session(home: Path, tmp_path: Path) -> None:
    at = _app()
    at.radio(key="engine").set_value("Gemini (your API key)").run()
    assert any("Gemini runs left in this session" in i.value for i in at.info)
    at.text_input(key="gemini_key").input(FAKE_KEY).run()
    assert not at.exception
    assert FAKE_KEY not in os.environ.values()
    assert all(FAKE_KEY not in p.read_text(errors="ignore")
               for p in tmp_path.rglob("*") if p.is_file())  # fmt: skip


def test_a_malformed_key_is_refused_before_any_request(home: Path) -> None:
    at = _app()
    at.radio(key="engine").set_value("Gemini (your API key)").run()
    at.text_input(key="gemini_key").input("not a key").run()
    assert any("does not look like an API key" in e.value for e in at.error)
    assert not at.button  # no Run button is shown


@pytest.mark.full_only
def test_an_offline_run_separates_critical_findings_from_warnings(home: Path) -> None:
    at = _app()
    at.selectbox(key="demo").set_value("Leaky column (direct_leak)").run()
    at.button(key="run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    heads = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert any(h.startswith("#### :red[Critical") for h in heads)
    assert any(h.startswith("#### :orange[Warnings") for h in heads)
    assert any(text.CRITICAL_FINDING_EXPLAINED in e.value for e in at.error)
    assert any(text.WARNING_EXPLAINED in w.value for w in at.warning)
    assert [m.label for m in at.metric] == [
        "Validation (roc_auc)", "Dummy baseline", "Sealed holdout"
    ]  # fmt: skip


@pytest.mark.full_only
def test_a_gemini_run_passes_the_key_only_to_the_client(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Gemini path with a fake client: the key reaches the client and nothing else."""
    case = build_case("direct_leak", 0)
    fake = fake_for(case, write_case(case, tmp_path / "data"))
    received: list[str | None] = []

    def client(model: str, api_key: str | None = None) -> object:
        received.append(api_key)
        assert model == fake.model
        return fake

    monkeypatch.setattr(core, "GeminiClient", client)
    lines: list[str] = []
    outcome = core.run_holdout(
        case.df, case.target, case.goal_text, tmp_path / "work", api_key=FAKE_KEY,
        on_line=lines.append,
    )  # fmt: skip
    assert received == [FAKE_KEY]
    assert outcome.manifest is not None
    assert outcome.manifest.agent is not None
    assert not outcome.manifest.agent.offline
    assert FAKE_KEY not in os.environ.values()
    assert all(FAKE_KEY not in line for line in lines)
    for root in (tmp_path, home):
        for path in root.rglob("*"):
            if path.is_file():
                assert FAKE_KEY.encode() not in path.read_bytes(), path
