"""The web app's logic without Streamlit."""

from pathlib import Path

import pytest

from holdout.llm import config as llm_config
from holdout.llm.client import GeminiClient
from holdout.webapp import core, text

ROOT = Path(__file__).resolve().parents[2]


def test_the_session_cap() -> None:
    for used in range(core.MAX_GEMINI_RUNS_PER_SESSION):
        allowed, note = core.gemini_allowed(used)
        assert allowed
        assert f"Rs {llm_config.RUN_CAP_INR:g}" in note
    allowed, note = core.gemini_allowed(core.MAX_GEMINI_RUNS_PER_SESSION)
    assert not allowed
    assert "Offline runs are unlimited" in note


@pytest.mark.parametrize(
    ("value", "ok"),
    [("AIzaSyA-abcdefghijklmnopqrstuvwxyz012345", True), ("short", False),
     ("has spaces in it but is long enough", False), ("", False)],
)  # fmt: skip
def test_the_key_shape_check(value: str, ok: bool) -> None:
    assert core.looks_like_key(value) is ok


def test_an_explicit_key_is_used_without_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(llm_config.API_KEY_ENV, raising=False)
    seen: dict[str, object] = {}

    class StubClient:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    import google.genai

    monkeypatch.setattr(google.genai, "Client", StubClient)
    client = GeminiClient(api_key="session-key-0123456789abcdef")
    assert seen["api_key"] == "session-key-0123456789abcdef"
    assert llm_config.API_KEY_ENV not in __import__("os").environ
    assert "session-key" not in repr(vars(client))


def test_the_demos_build_without_a_download() -> None:
    labels = [d.label for d in core.demos()]
    assert labels[0] == "Churn (clean)"
    frame, target, _ = core.demos()[0].build()
    assert target in frame.columns


def test_results_sections_are_cut_from_the_generated_document() -> None:
    doc = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
    sections = core.markdown_sections(doc, "final_v1", ("Detection recall", "False alarms"))
    assert [s.splitlines()[0] for s in sections] == [
        "### Detection recall (task = unit, seeds averaged; 95% bootstrap CI)",
        "### False alarms on clean controls (task = unit: any seed or variant alarming or "
        "failing counts)",
    ]
    assert "51.8% [33.9%, 69.6%]" in sections[0]


@pytest.mark.parametrize("doc", ["README.md", "docs/results_v11.md"])
def test_the_documents_carry_the_apps_wording(doc: str) -> None:
    """The app, README and results_v11.md state the same results, labels and caveats."""
    body = (ROOT / doc).read_text(encoding="utf-8")
    for sentence in (*text.TRADEOFFS, *text.POSTHOC_SUMMARY):
        assert sentence in body, sentence[:60]
    if doc == "README.md":
        for sentence in text.PREREGISTERED_SUMMARY:
            assert sentence in body, sentence[:60]
        assert body.index(text.PREREGISTERED_LABEL) < body.index(text.POSTHOC_LABEL)
