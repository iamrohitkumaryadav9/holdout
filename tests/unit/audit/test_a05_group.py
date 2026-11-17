"""A5: entity memorisation is critical and invalidates the holdout, unless entities are
declared known; clean twins (same entities, no entity effect) stay quiet."""

import pytest

from bench.traps import build_case, clean_twin
from holdout.audit.checks import a05_group as a5
from holdout.audit.types import Severity
from holdout.core.schemas import EntityScope, SplitStrategy
from holdout.data.split import SplitSpec
from tests.audit_harness import context_for, trap_context, twin_context
from tests.tiers import SEEDS


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_entity_leakage_is_critical_and_invalidates(seed: int) -> None:
    outcome = a5.run(trap_context("group_leakage", seed))
    flag = next(f for f in outcome.flags if f.columns == ["account"])
    assert flag.severity is Severity.CRITICAL
    assert flag.invalidates_holdout
    assert flag.fix.kind == "change_split"
    spec = flag.fix.new_spec  # type: ignore[union-attr]
    assert spec is not None
    assert (spec.strategy, spec.group_col, spec.entity_scope) == (
        SplitStrategy.GROUP,
        "account",
        EntityScope.NEW,
    )
    comparison = flag.evidence.comparison  # type: ignore[union-attr]
    assert comparison.relative_drop >= 0.35
    assert comparison.alternative_splitter == "GroupKFold(account)"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trap", ["group_leakage", "direct_leak"])
def test_quiet_on_clean_twins(trap: str, seed: int) -> None:
    assert [
        f for f in a5.run(twin_context(trap, seed)).flags if f.severity is not Severity.INFO
    ] == []


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_known_entities_record_info_and_never_invalidate(seed: int) -> None:
    ctx = context_for(build_case("group_leakage", seed), seed, entity_scope=EntityScope.KNOWN)
    assert ctx.spec.entity_scope is EntityScope.KNOWN
    outcome = a5.run(ctx)
    flag = next(f for f in outcome.flags if f.columns == ["account"])
    assert flag.severity is Severity.INFO
    assert not flag.invalidates_holdout
    assert "--entities known" in flag.message
    assert flag.evidence.entity_scope is EntityScope.KNOWN  # type: ignore[union-attr]


def test_new_entities_declared_explicitly_split_by_group_and_stay_quiet() -> None:
    case = build_case("group_leakage", 0)
    ctx = context_for(case, 0, entity_scope=EntityScope.NEW)
    assert ctx.spec.strategy is SplitStrategy.GROUP  # the proposal grouped by 'account'
    assert a5.run(ctx).flags == []  # the run's own group is excluded from candidates


def test_time_split_runs_skip_a5() -> None:
    case = clean_twin(build_case("group_leakage", 0))
    case.df["when"] = range(len(case.df))
    spec = SplitSpec(strategy=SplitStrategy.TIME, time_col="when", seed=0)
    outcome = a5.run(context_for(case, 0, spec=spec))
    assert outcome.skipped


def test_repeated_integer_measurements_are_not_candidates() -> None:
    ctx = twin_context("direct_leak", 0)
    assert [c.column for c in a5.candidates(ctx)] == []  # age, tenure repeat but are measures
