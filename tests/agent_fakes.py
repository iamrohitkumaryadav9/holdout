"""A deterministic, cooperative fake LLM for agent tests.

It reads the same ``<data>`` sections a real model sees and answers like a well-behaved
model: the default plan (plus the trap's overrides), fix-applying revised plans, digit-free
report prose with placeholders. For a causal goal its first report draft makes a causal
claim (as LLMs tend to), so A11 has something to catch; with feedback it removes it.
"""

import json
import re
from typing import Any

from holdout.core.schemas import Task
from holdout.data.profile import DatasetProfile
from holdout.llm.client import FakeLLM, LLMRequest
from holdout.plan.defaults import default_plan
from holdout.plan.schema import ModelPlan

_SECTION = re.compile(r'<data name="(?P<name>[a-z_]+)">\n(?P<body>.*?)\n</data>', re.DOTALL)


def data(request: LLMRequest, name: str) -> Any:
    """The value of one data section of a rendered prompt (JSON when possible)."""
    for match in _SECTION.finditer(request.user):
        if match.group("name") == name:
            body = match.group("body").replace("<\\/data", "</data").replace("<\\data", "<data")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
    raise KeyError(name)


def plan_responder(profile: DatasetProfile, task: Task, overrides: dict[str, Any]) -> Any:
    plan = default_plan(profile, task)
    payload = {**plan.model_dump(mode="json"), **overrides}
    ModelPlan.model_validate(payload)

    def respond(request: LLMRequest) -> dict[str, Any]:
        return payload

    return respond


def explain_fix(request: LLMRequest) -> dict[str, Any]:
    plan = data(request, "plan")
    drops = list(plan["drop_columns"])
    for label in data(request, "fixes"):
        kind, value = label.split(":", 1)
        if kind == "drop" and value not in {d["column"] for d in drops}:
            drops.append({"column": value, "reason": "the audit found this column leaking"})
        if kind == "metric":
            plan["metric"] = value
            plan["rationale"]["metric"] = f"{value}: chosen after the audit"
    plan["drop_columns"] = drops
    explanations = [
        {
            "check": flag["check"],
            "columns": flag["columns"],
            "explanation": (
                f"{flag['check']} found a {flag['category'].replace('_', ' ')} problem; it can "
                "make the validation score look better than it will be in use."
            ),
        }
        for flag in data(request, "flags")
    ]
    return {"explanations": explanations, "revised_plan": plan}


def report(request: LLMRequest) -> dict[str, Any]:
    catalogue = data(request, "placeholders")
    names = {key.strip("{}") for key in catalogue}
    flags = data(request, "flags")
    goal = str(data(request, "goal")).lower()
    feedback = data(request, "feedback")
    score = "{{validation_score}}" if "validation_score" in names else "a validation score"
    summary = f"The final plan reaches {score} in cross-validation"
    summary += " against {{dummy_score}} for the dummy baseline." if "dummy_score" in names else "."
    causal = "cause" in goal and feedback == "none"
    if causal:
        summary += " Yes: being contacted by phone causes customers to convert."
    findings = []
    for flag in flags:
        where = ", ".join(flag["columns"]) or "the data"
        category = flag["category"].replace("_", " ")
        findings.append(f"{flag['check']} ({flag['severity']}) on {where}: {category}.")
    return {
        "summary": summary,
        "findings": findings,
        "caveats": data(request, "caveats"),
        "recommendations": ["Review the audit flags before relying on the score."],
        "causal_claim": causal,
    }


def eda(request: LLMRequest) -> dict[str, Any]:
    target = data(request, "target")
    return {
        "snippets": [
            {
                "purpose": "target balance",
                "code": f"print(df[{target!r}].value_counts().to_dict())",
            },
            {"purpose": "missingness", "code": "print(df.isna().mean().round(3).to_dict())"},
        ]
    }


def cooperative_fake(
    profile: DatasetProfile, task: Task, overrides: dict[str, Any] | None = None
) -> FakeLLM:
    return FakeLLM(
        default={
            "plan": plan_responder(profile, task, overrides or {}),
            "explain_fix": explain_fix,
            "report": report,
            "hints": lambda request: {"suspects": []},
            "eda": eda,
        }
    )
