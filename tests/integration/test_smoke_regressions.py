"""Regressions from the first paid smoke runs (2026-11-17, gemini-3.8-flash).

1. The first plan used an invalid model tag (the tagged union reached Gemini as oneOf/const,
   which its structured output does not document); fixed in the LLM-facing schema and the
   plan prompt, without loosening validation.
2. The report took three calls: A11 flagged "resulting in an optimism gap" (arithmetic on
   the run's own numbers), and the generic "remove the causal claims" feedback could not
   be acted on; fixed in A11 and in the repair feedback.
3. The substitution was that report fallback, not explain_and_fix (whose answer was valid);
   the explain_fix prompt now states each fix as an explicit constraint anyway.
4. Answer-quality metrics are recorded per run.

Fixtures (reconstructed where the raw answer was not stored) are described in
tests/fixtures/smoke_2026_11_17/README.md.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bench.traps import build_case
from holdout.agent.planner import LLMPlanner, ReportContext, causal_problems
from holdout.agent.prompts import load_prompt
from holdout.agent.report import placeholder_values
from holdout.agent.schemas import ReportDraft
from holdout.audit.checks.a11_causal import check_causal_claims
from holdout.core.errors import LLMConfigError, LLMOutputError
from holdout.core.schemas import MetricName
from holdout.llm import config
from holdout.llm.budget import Budget, SpendLedger
from holdout.llm.cache import ResponseCache
from holdout.llm.client import FakeLLM
from holdout.llm.structured import LLMSession, llm_schema
from holdout.plan.schema import MODEL_NAMES, ModelPlan
from tests.agent_harness import fake_for, run_case, write_case

FIXTURES = Path(__file__).parents[1] / "fixtures" / "smoke_2026_11_17"
SMOKE_ERRORS = {
    "churn": "Input tag 'HistGradientBoostingClassifier' found using 'name' does not match any "
    "of the expected tags: 'dummy', 'logreg', 'ridge', 'random_forest', 'hist_gb', 'lightgbm'",
    "direct_leak": "Input tag 'hist_gradient_boosting' found using 'name' does not match any "
    "of the expected tags: 'dummy', 'logreg', 'ridge', 'random_forest', 'hist_gb', 'lightgbm'",
}


def _fixture(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return loaded


# ------------------------------------------------------------------ 1. plan tags


@pytest.mark.parametrize("label", ["churn", "direct_leak"])
def test_the_smoke_plans_fail_exactly_as_recorded_and_validation_is_not_loosened(
    label: str,
) -> None:
    with pytest.raises(ValidationError) as rejected:
        ModelPlan.model_validate(_fixture(f"plan_{label}_rejected.json"))
    (error,) = rejected.value.errors()
    assert error["msg"] == SMOKE_ERRORS[label]
    ModelPlan.model_validate(_fixture(f"plan_{label}_repaired.json"))


def test_the_llm_schema_uses_only_documented_keywords_and_spells_out_the_tags() -> None:
    schema = llm_schema(ModelPlan)
    text = json.dumps(schema)
    for keyword in ('"oneOf"', '"const"', '"discriminator"'):
        assert keyword not in text
    tags = sorted(
        definition["properties"]["name"]["enum"][0]
        for definition in schema["$defs"].values()
        if "name" in definition.get("properties", {})
    )
    assert tags == sorted(MODEL_NAMES)
    assert "hist_gb" in schema["properties"]["model"]["description"]
    assert len(schema["properties"]["model"]["anyOf"]) == len(MODEL_NAMES)


def test_model_names_match_the_union() -> None:
    for name in MODEL_NAMES:
        plan = _fixture("plan_churn_repaired.json")
        plan["model"] = {"name": name}
        plan["task"] = "regression" if name == "ridge" else "classification"
        assert ModelPlan.model_validate(plan).model.name == name


def test_the_plan_prompt_lists_every_tag_and_metric_and_its_example_validates() -> None:
    prompt = load_prompt("plan")
    assert prompt.version == "2"
    for name in MODEL_NAMES:
        assert f'"{name}"' in prompt.system
    listed = set(re.findall(r'"([a-z0-9_]+)"', prompt.system.split("- metric:")[1].split("- ")[0]))
    assert listed == {m.value for m in MetricName}
    example = prompt.system.split("<example_plan>")[1].split("</example_plan>")[0]
    ModelPlan.model_validate(json.loads(example))
    assert 'not "HistGradientBoostingClassifier" or "hist_gradient_boosting"' in prompt.system


@pytest.mark.parametrize("label", ["churn", "direct_leak"])
def test_a_rejected_tag_is_repaired_once_and_counted(label: str, tmp_path: Path) -> None:
    case = build_case("direct_leak", 0)
    working = fake_for(case, write_case(case, tmp_path))
    repaired = _fixture("plan_direct_leak_repaired.json")
    rejected = {**repaired, "model": {**repaired["model"], "name": _bad_tag(label)}}
    replay = FakeLLM(default=working.default, scripts={"plan": [rejected, repaired]})
    outcome = run_case(case, tmp_path, llm=replay)
    assert outcome.manifest is not None
    agent = outcome.manifest.agent
    assert agent is not None
    assert agent.quality is not None
    assert agent.quality.first_attempt["plan"].rate == 0.0
    assert agent.quality.repairs == 1
    (kept,) = [r for r in agent.rejected_attempts if r.node == "plan"]
    assert kept.errors == [f"model: {SMOKE_ERRORS[label]}"]
    assert kept.text is not None
    assert _bad_tag(label) in kept.text  # the raw answer is kept now
    report = (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    assert "**Answer quality:** plan 0/1" in report


def _bad_tag(label: str) -> str:
    return "HistGradientBoostingClassifier" if label == "churn" else "hist_gradient_boosting"


# ------------------------------------------------------------------ 2. the report


VALUES = placeholder_values(
    {
        "validation_score": 0.6602,
        "validation_std": 0.0462,
        "dummy_score": 0.5,
        "holdout_score": 0.7221,
        "optimism_gap": -0.0619,
        "n_rows": 1200,
        "n_dev_rows": 960,
        "n_holdout_rows": 240,
        "n_fix_rounds": 1,
        "n_critical": 0,
        "n_warning": 0,
    }
)


def _report_context() -> ReportContext:
    plan = ModelPlan.model_validate(_fixture("plan_direct_leak_repaired.json"))
    return ReportContext(
        goal="predict converted",
        facts={"metric": "roc_auc", "holdout_evaluated": True},
        values=VALUES,
        flags=[],
        explanations={},
        caveats=[],
        columns=["refund_issued", "score_a", "score_b", "converted"],
        plan=plan,
        round=1,
    )


def _planner(tmp_path: Path, script: list[Any]) -> tuple[LLMPlanner, FakeLLM]:
    llm = FakeLLM(scripts={"report": script})
    budget = Budget(run_id="r", model=llm.model, ledger=SpendLedger(tmp_path / "ledger.jsonl"))
    session = LLMSession(
        client=llm, budget=budget, seed=1, cache=ResponseCache(tmp_path / "c", enabled=False)
    )
    return LLMPlanner(session), llm


def test_the_smoke_report_is_accepted_in_one_call(tmp_path: Path) -> None:
    planner, llm = _planner(tmp_path, [_fixture("report_direct_leak_first_draft.json")])
    outcome = planner.draft_report(_report_context())
    assert len(llm.calls) == 1
    assert outcome.substitution is None
    assert outcome.causal_flags == []
    assert "resulting in an optimism gap of -0.0619" in outcome.draft.summary


@pytest.mark.parametrize(
    "sentence",
    [
        "On the independent holdout set, the model achieved a score of 0.7221, resulting in "
        "an optimism gap of -0.0619.",
        "Dropping the leaking column led to a lower validation score.",
        "Stronger regularisation results in a smaller gap between folds.",
        # Verbatim from the second paid smoke run's churn report (kept in rejected_attempts):
        "This model reflects statistical associations in historical data and cannot confirm "
        "whether specific factors cause churn.",
        "It is unclear if tenure drives churn.",
    ],
)
def test_a11_ignores_run_statistics_and_unasserted_verbs(sentence: str) -> None:
    assert check_causal_claims(sentence) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Phone contact leads to higher conversion, improving the score.",
        "Being contacted by phone results in conversion.",
        "A refund drives churn.",
        "Customers convert because of the discount.",
        "The effect of price on demand is large.",
        "Discounts cause higher sales and a better score.",
        "We checked whether the data was clean, and discounts cause higher sales.",
    ],
)
def test_a11_still_flags_claims_about_the_data(sentence: str) -> None:
    assert check_causal_claims(sentence) != []


def _causal_draft() -> dict[str, Any]:
    draft = _fixture("report_direct_leak_first_draft.json")
    draft["summary"] += " Being contacted by phone results in conversion."
    return draft


def test_causal_feedback_quotes_the_exact_words(tmp_path: Path) -> None:
    clean = _fixture("report_direct_leak_first_draft.json")
    planner, llm = _planner(tmp_path, [_causal_draft(), clean])
    outcome = planner.draft_report(_report_context())
    assert len(llm.calls) == 2  # one draft, one repair
    repair = llm.calls[1].user
    assert '"results in"' in repair
    assert "Being contacted by phone results in conversion." in repair
    assert outcome.substitution is None
    assert len(outcome.causal_flags) == 1  # caught on the first draft, recorded
    problems = causal_problems(ReportDraft.model_validate(_causal_draft()))
    assert len(problems) == 1


def test_a_causal_claim_that_survives_the_repair_is_stripped_by_code(tmp_path: Path) -> None:
    planner, llm = _planner(tmp_path, [_causal_draft(), _causal_draft()])
    outcome = planner.draft_report(_report_context())
    assert len(llm.calls) == 2
    assert outcome.substitution is not None
    assert outcome.substitution.missed_fixes == ["remove_causal_claim"]
    assert "results in conversion" not in outcome.draft.summary
    assert "resulting in an optimism gap" in outcome.draft.summary  # not causal: kept


def test_a_report_with_typed_numbers_twice_still_fails_loudly(tmp_path: Path) -> None:
    bad = {**_fixture("report_direct_leak_first_draft.json")}
    bad["summary"] = bad["summary"].replace("{{validation_score}}", "0.66")
    planner, _ = _planner(tmp_path, [bad, bad])
    with pytest.raises(LLMOutputError, match="invalid after one repair"):
        planner.draft_report(_report_context())


# ------------------------------------------------------------------ 3. explain_fix + 4. metrics


def test_explain_fix_states_each_fix_as_a_constraint_and_metrics_are_recorded(
    tmp_path: Path,
) -> None:
    case = build_case("direct_leak", 0)
    fake = fake_for(case, write_case(case, tmp_path))
    outcome = run_case(case, tmp_path, llm=fake)
    (call,) = fake.calls_for("explain_fix")
    assert call.prompt_version == "2"
    assert 'revised_plan.drop_columns MUST contain an entry {"column": "refund_issued"' in call.user
    assert "revised_plan.model.name MUST be exactly one of" in call.user
    assert outcome.manifest is not None
    agent = outcome.manifest.agent
    assert agent is not None
    assert agent.quality is not None
    rates = {n: f.rate for n, f in agent.quality.first_attempt.items()}
    assert rates == {"plan": 1.0, "explain_and_fix": 1.0, "report": 1.0}
    assert (agent.quality.repairs, agent.quality.substitutions) == (0, 0)
    assert agent.rejected_attempts == []
    assert (
        "plan 1/1, explain_and_fix 1/1, report 1/1 valid at first attempt; 0 repair(s); "
        "0 substitution(s)" in (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    )


# ------------------------------------------------------------------ 5. price table


def test_the_404_model_is_no_longer_priced() -> None:
    assert "gemini-3-flash-preview" not in config.PRICE_TABLE
    with pytest.raises(LLMConfigError, match="no price"):
        config.require_priced("gemini-3-flash-preview", config.PRICES_AS_OF)
