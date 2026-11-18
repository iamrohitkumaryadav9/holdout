"""The harness: the iid and deployment-faithful holdouts, their rule per trap type, and the
seal."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from bench.harness import (
    DEPLOY_RULES,
    BenchSeal,
    PartitionRule,
    SealError,
    partition,
    planted_rule,
    real_world_rule,
)
from bench.traps.catalog import TRAPS
from holdout.core.schemas import Task


def _frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "t": range(n),
            "account": [f"a{i % 60}" for i in range(n)],
            "x": rng.normal(size=n),
            "leak": y,
            "y": y,
        }
    )


def _rows(frame: pd.DataFrame) -> set[int]:
    return set(frame["t"])


def test_every_trap_type_has_a_documented_deploy_rule() -> None:
    assert set(DEPLOY_RULES) == set(TRAPS)


def test_both_holdouts_are_disjoint_from_development() -> None:
    for rule in (
        PartitionRule(),
        PartitionRule("group", group_column="account"),
        PartitionRule("time"),
    ):
        dev, iid, deploy = partition(_frame(), "y", rule, 101)
        assert not _rows(dev) & _rows(iid)
        assert not _rows(dev) & _rows(deploy)


def test_group_deploy_holds_unseen_entities_and_iid_shares_them() -> None:
    dev, iid, deploy = partition(_frame(), "y", PartitionRule("group", group_column="account"), 101)
    assert not set(dev["account"]) & set(deploy["account"])
    assert set(iid["account"]) <= set(dev["account"])  # iid: entities seen in training


def test_time_deploy_is_the_future_and_iid_is_drawn_from_the_past() -> None:
    dev, iid, deploy = partition(_frame(), "y", PartitionRule("time"), 101)
    assert deploy["t"].min() > max(dev["t"].max(), iid["t"].max())
    assert len(deploy) == 120
    assert iid["t"].min() < dev["t"].max()  # interleaved with training rows in time


def test_duplicates_deploy_drops_copies_of_dev_rows_and_iid_keeps_them() -> None:
    frame = _frame(300)
    doubled = pd.concat([frame, frame.iloc[:90]]).reset_index(drop=True)  # exact copies
    rule = planted_rule("cross_split_duplicates", Task.CLASSIFICATION, [])
    dev, iid, deploy = partition(doubled, "y", rule, 101)
    features = ["t", "account", "x", "leak"]
    seen = set(map(tuple, dev[features].astype(str).to_numpy()))

    def copies(holdout: pd.DataFrame) -> int:
        return sum(tuple(r) in seen for r in holdout[features].astype(str).to_numpy())

    assert copies(iid) > 0
    assert copies(deploy) == 0
    assert _rows(deploy) <= _rows(iid)


@pytest.mark.parametrize("trap", ["direct_leak", "noisy_proxy", "id_with_signal"])
def test_leak_columns_are_permuted_in_the_deploy_holdout_only(trap: str) -> None:
    rule = planted_rule(trap, Task.CLASSIFICATION, ["leak"])
    assert rule.unavailable_columns == ("leak",)
    dev, iid, deploy = partition(_frame(1000), "y", rule, 101)
    assert (dev["leak"] == dev["y"]).all()  # systems still see the leak in development
    assert (iid["leak"] == iid["y"]).all()  # leak intact in the iid holdout
    pd.testing.assert_frame_equal(iid.drop(columns=["leak"]), deploy.drop(columns=["leak"]))
    assert 0.35 < (deploy["leak"] == deploy["y"]).mean() < 0.65  # no link any more
    assert sorted(deploy["leak"]) == sorted(iid["leak"])  # same values, permuted


@pytest.mark.parametrize("trap", ["imbalance_accuracy", "messy_formats", "causal_question"])
def test_other_traps_have_identical_holdouts(trap: str) -> None:
    rule = planted_rule(trap, Task.CLASSIFICATION, ["x"])
    assert rule.deploy_is_iid
    _, iid, deploy = partition(_frame(), "y", rule, 101)
    pd.testing.assert_frame_equal(iid, deploy)


@pytest.mark.parametrize("trap", sorted(TRAPS))
def test_clean_twins_have_deploy_equal_to_iid(trap: str) -> None:
    rule = planted_rule(trap, Task.CLASSIFICATION, ["account", "leak"], clean=True)
    assert rule.deploy_is_iid
    _, iid, deploy = partition(_frame(), "y", rule, 202)
    pd.testing.assert_frame_equal(iid, deploy)


def test_partitions_are_deterministic_per_seed() -> None:
    rule = planted_rule("direct_leak", Task.CLASSIFICATION, ["leak"])
    a = partition(_frame(), "y", rule, 101)
    b = partition(_frame(), "y", rule, 101)
    c = partition(_frame(), "y", rule, 202)
    for left, right in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(left, right)
    assert not a[1].equals(c[1])
    assert a[1]["y"].mean() == pytest.approx(_frame()["y"].mean(), abs=0.03)  # stratified


def test_real_world_rules_follow_the_recorded_deploy_holdout() -> None:
    assert real_world_rule("group:patient_nbr", Task.CLASSIFICATION).group_column == "patient_nbr"
    assert real_world_rule("time", Task.CLASSIFICATION).deploy == "time"
    assert real_world_rule("iid", Task.REGRESSION).deploy_is_iid
    with pytest.raises(ValueError, match="unknown deploy holdout"):
        real_world_rule("random", Task.CLASSIFICATION)


def test_the_seal_scores_both_holdouts_logs_and_detects_tampering(tmp_path: Path) -> None:
    rule = planted_rule("direct_leak", Task.CLASSIFICATION, ["leak"])
    dev, iid, deploy = partition(_frame(), "y", rule, 101)
    seal = BenchSeal.create(tmp_path / "seal", iid, deploy, "y")
    model = DummyClassifier(strategy="prior").fit(dev.drop(columns=["y"]), dev["y"])
    for which in ("iid", "deploy"):
        assert seal.evaluate(model, "roc_auc", caller="dummy", which=which) == pytest.approx(0.5)
    assert [e["holdout"] for e in seal.access_log()] == ["iid", "deploy"]
    assert BenchSeal.load(tmp_path / "seal").sha256 == seal.sha256
    iid.iloc[:5].to_parquet(seal.path("iid"), index=False)
    with pytest.raises(SealError, match="iid holdout"):
        seal.evaluate(model, "roc_auc", caller="x", which="iid")
