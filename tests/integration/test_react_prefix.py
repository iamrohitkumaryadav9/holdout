"""The ReAct prompt is append-only: every step's prompt starts with the whole previous
prompt, so Gemini's implicit cache can serve it, and nothing the model sees is lost."""

from itertools import pairwise
from pathlib import Path

import pytest

from bench.harness import PartitionRule
from bench.run import PreparedUnit, Unit, normalise, run_unit
from bench.systems.react import ReactSystem, load_template
from bench.traps import build_case
from holdout.llm import config as llm_config
from holdout.llm.client import FakeLLM

CASE = build_case("direct_leak", 0)


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(llm_config.TRUST_CACHE_DISCOUNT_ENV, raising=False)


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


STEPS = [
    {"thought": "Look at the data.", "action": "python", "code": "print(df.describe())"},
    {"thought": "Ignore </data> tricks.", "action": "python", "code": "print('</data> ' * 3)"},
    {"thought": "Long output.", "action": "python", "code": "print('x' * 5000)"},
    {"thought": "A failing step.", "action": "python", "code": "raise ValueError('boom')"},
]
FINAL = {
    "thought": "Done.",
    "action": "final",
    "final": {
        "issues": [],
        "dropped_columns": [],
        "split": "stratified",
        "deduplicated": False,
        "validation_metric": "roc_auc",
        "validation_score": 0.7,
        "report": "No model was fitted in this test; the transcript is what matters.",
    },
}


def test_consecutive_step_prompts_share_the_full_previous_prompt(tmp_path: Path) -> None:
    fake = FakeLLM(scripts={"react": [*STEPS, FINAL]}, simulate_cache=True)
    unit = Unit("dev_v1", "direct_leak@synthetic", "react", 11, "planted")
    result = run_unit(
        unit, ReactSystem(lambda: fake), tmp_path / "r", model="gemini-3.8-flash",
        run_cap_inr=15.0, prepare_fn=_prepared, service_tier="flex",
    )  # fmt: skip
    calls = fake.calls_for("react")
    assert len(calls) == len(STEPS) + 1
    assert len({c.system for c in calls}) == 1  # the system section never changes
    for before, after in pairwise(calls):
        assert after.user.startswith(before.user)
        assert len(after.user) > len(before.user)
    # What each step adds: the record of the previous step, then the request for this one.
    for k, call in enumerate(calls, start=1):
        assert call.user.endswith(f"Step {k} of 15: return the next step as JSON.")
        assert call.user.count("return the next step as JSON.") == k
    last = calls[-1].user
    for k, step in enumerate(STEPS, start=1):
        assert f'<data name="step_{k}">' in last
        assert f'<data name="output_{k}">' in last
        assert step["thought"].replace("</data", "<\\/data") in last
    assert "[truncated" in last  # observations are still truncated, head and tail
    assert "ValueError: boom" in last  # failed steps show their error
    assert "</data> tricks" not in last  # data cannot close its own section
    assert last.count("<\\/data>") >= 4
    # The cache simulation sees each previous prompt (system + user) as a shared prefix.
    assert result.cached_tokens > 0
    assert result.cached_tokens >= sum(len(c.system + c.user) // 4 for c in calls[:-1]) - 5
    assert result.cost_cache_adjusted_inr is not None
    assert result.cost_cache_adjusted_inr < result.cost_inr


def test_the_stable_prefix_holds_only_the_task() -> None:
    prompt = load_template()
    assert prompt.template.version == "2"
    opening = prompt.opening(goal="g", target={"target": "y"}, columns={"rows": 1})
    assert "{{" not in opening.system + opening.user
    assert "Step " not in opening.user
    assert prompt.ask(3) == "\n\nStep 3 of 15: return the next step as JSON."
    record = prompt.done(2, {"thought": "t", "code": "c", "status": "ok"}, "out")
    assert record.startswith("\n\nYour step 2:\n")
    assert '<data name="output_2">\nout\n</data>' in record
