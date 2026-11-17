"""Run the agent on a trap case with the cooperative fake, and read the results."""

from pathlib import Path
from typing import Any

from bench.traps import TrapCase
from holdout.agent.driver import AgentOutcome, run_agent
from holdout.agent.graph import AgentOptions
from holdout.agent.report import allowed_tokens, placeholder_values, prose_numbers
from holdout.audit.types import Category, Severity
from holdout.data.load import load_dataset
from holdout.data.profile import profile_schema
from holdout.export.manifest import RunManifest
from holdout.llm.client import FakeLLM
from tests.agent_fakes import cooperative_fake


def write_case(case: TrapCase, root: Path) -> Path:
    csv = root / "data.csv"
    case.df.to_csv(csv, index=False)
    return csv


def fake_for(case: TrapCase, csv: Path) -> FakeLLM:
    profile = profile_schema(load_dataset(csv, case.target).frame, case.target, task=case.task)
    return cooperative_fake(profile, case.task, case.plan_overrides)


def run_case(
    case: TrapCase,
    root: Path,
    *,
    offline: bool = False,
    mode: str = "autopilot",
    llm: FakeLLM | None = None,
    **options: Any,
) -> AgentOutcome:
    csv = write_case(case, root)
    fake = llm if llm is not None else fake_for(case, csv)
    return run_agent(
        AgentOptions(
            csv=str(csv),
            target=case.target,
            out_dir=str(root / "runs"),
            goal=case.goal_text,
            mode=mode,
            offline=offline,
            **options,
        ),
        client_factory=lambda: fake,
    )


def detected(manifest: RunManifest, category: Category, minimum: Severity) -> bool:
    """Expected category at >= minimum severity in any audit round or the final audit."""
    flags: list[tuple[str, str]] = []
    if manifest.agent is not None:
        for round_ in manifest.agent.rounds:
            for text in round_.flags:
                parts = text.split(" ")
                flags.append((parts[2], parts[1]))
    flags += [(f.category.value, f.severity.value) for f in manifest.audit.flags]
    return any(c == category.value and Severity(s).at_least(minimum) for c, s in flags)


def report_numbers_outside_state(manifest: RunManifest) -> list[str]:
    """Numbers in the agent's report prose that are not formatted run-state values."""
    assert manifest.agent is not None
    report = manifest.agent.report
    assert report is not None
    result, baseline = manifest.plans[-1].cv, manifest.plans[0].cv
    holdout = None if isinstance(manifest.holdout, str) else manifest.holdout
    values = placeholder_values(
        {
            "validation_score": result.mean,
            "validation_std": result.std,
            "dummy_score": baseline.mean,
            "holdout_score": holdout.score if holdout else None,
            "optimism_gap": result.mean - holdout.score if holdout else None,
            "n_rows": manifest.data.n_rows,
            "n_dev_rows": manifest.split.n_dev_rows,
            "n_holdout_rows": manifest.split.n_holdout_rows,
            "n_fix_rounds": max(0, len(manifest.agent.rounds) - 1),
            "n_critical": manifest.audit.counts["critical"],
            "n_warning": manifest.audit.counts["warning"],
        }
    )
    allowed = allowed_tokens(list(manifest.profile.column_kinds), manifest.plans[-1].plan)
    text = " ".join([report.summary, *report.findings, *report.caveats, *report.recommendations])
    state_numbers = set(values.values())
    return [n for n in prose_numbers(text, allowed) if n not in state_numbers]
