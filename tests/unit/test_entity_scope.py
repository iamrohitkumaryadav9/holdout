"""Pre-seal entity question and the new/known entity-scope paths of the split proposal."""

import numpy as np
import pandas as pd

from holdout.core.schemas import DataType, EntityScope, SplitStrategy, Task
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec, propose_split


def _accounts(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_accounts, rows = 60, 5
    return pd.DataFrame(
        {
            "account": np.repeat([f"a{i:03d}" for i in range(n_accounts)], rows),
            "tier": np.repeat(rng.choice(["gold", "silver"], n_accounts), rows),
            "x": rng.normal(size=n_accounts * rows),
            "y": rng.integers(0, 2, n_accounts * rows),
        }
    )


def test_profile_finds_structural_entities_target_blind() -> None:
    profile = profile_schema(_accounts(), "y")
    assert [e.column for e in profile.structural_entities] == ["account"]
    assert profile.entity_candidates == []  # neutral name: not found by name


def test_unanswered_scope_asks_and_offers_a_group_split() -> None:
    profile = profile_schema(_accounts(), "y")
    proposal = propose_split(profile, Task.CLASSIFICATION, DataType.TABULAR, seed=1)
    assert proposal.spec.strategy is SplitStrategy.STRATIFIED
    question = next(w for w in proposal.warnings if "entity scope new or known" in w)
    assert "'account' looks like an entity identifier" in question
    assert (
        SplitSpec(strategy=SplitStrategy.GROUP, group_col="account", seed=1)
        in proposal.alternatives
    )


def test_new_entities_get_a_group_split() -> None:
    profile = profile_schema(_accounts(), "y")
    proposal = propose_split(
        profile, Task.CLASSIFICATION, DataType.TABULAR, seed=1, entity_scope=EntityScope.NEW
    )
    assert proposal.spec.strategy is SplitStrategy.GROUP
    assert proposal.spec.group_col == "account"
    assert proposal.spec.entity_scope is EntityScope.NEW
    assert not any("entity scope new or known" in w for w in proposal.warnings)


def test_known_entities_keep_a_row_level_split() -> None:
    frame = _accounts().rename(columns={"account": "account_id"})  # name-detected too
    profile = profile_schema(frame, "y")
    assert profile.entity_candidates == ["account_id"]
    proposal = propose_split(
        profile, Task.CLASSIFICATION, DataType.TABULAR, seed=1, entity_scope=EntityScope.KNOWN
    )
    assert proposal.spec.strategy is SplitStrategy.STRATIFIED
    assert any("entities declared known" in r for r in proposal.reasons)


def test_entity_scope_does_not_change_the_holdout_fingerprint() -> None:
    plain = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1)
    declared = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1, entity_scope=EntityScope.KNOWN)
    assert plain.fingerprint() == declared.fingerprint()
