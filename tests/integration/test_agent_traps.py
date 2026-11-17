"""The full agent graph, autopilot, cooperative fake LLM: every trap is detected exactly as
the deterministic Week 2 runs detect it, every clean twin stays free of critical flags, and
every report states only numbers taken from run state."""

from pathlib import Path
from typing import Any

import pytest

from bench.traps import TRAPS, build_case, clean_twin
from holdout.audit.types import CheckId
from tests.agent_harness import detected, report_numbers_outside_state, run_case

FAST_TRAPS = {"direct_leak"}


def _tiered(names: set[str]) -> list[Any]:
    return [
        n if n in FAST_TRAPS else pytest.param(n, marks=pytest.mark.full_only)
        for n in sorted(names)
    ]


@pytest.mark.parametrize("name", _tiered(set(TRAPS)))
def test_agent_detects_every_trap(name: str, tmp_path: Path) -> None:
    case = build_case(name, 0)
    outcome = run_case(case, tmp_path)
    assert outcome.status == "completed"
    manifest = outcome.manifest
    assert manifest is not None
    truth = case.truth
    assert truth.expected_category is not None
    assert detected(manifest, truth.expected_category, truth.min_expected_severity)
    assert report_numbers_outside_state(manifest) == []
    a11 = next(r for r in manifest.audit.checks_run if r.check is CheckId.A11)
    assert a11.status == "ran"  # the generated report is checked in agent runs


@pytest.mark.parametrize("name", _tiered(set(TRAPS)))
@pytest.mark.full_only
def test_agent_clean_twins_raise_no_critical(name: str, tmp_path: Path) -> None:
    outcome = run_case(clean_twin(build_case(name, 0)), tmp_path)
    manifest = outcome.manifest
    assert manifest is not None
    assert manifest.agent is not None
    first_round = manifest.agent.rounds[0].flags
    assert not [f for f in first_round if " critical " in f]
    assert not manifest.holdout_invalid
    assert report_numbers_outside_state(manifest) == []
