"""The real-world suite: validation against the registry and enums, the frozen hash, and the
anonymiser's mapping."""

import json
from pathlib import Path

import pandas as pd
import pytest

from bench.data.registry import load_registry
from bench.suites import frozen
from bench.suites.real_world import (
    ANON_TARGET,
    ANONYMISED_PATH,
    REAL_WORLD_PATH,
    Anonymisation,
    SuiteError,
    anonymisation_for,
    load_anonymisations,
    load_real_world,
    suite_sha256,
)
from holdout.audit.types import Category


def test_the_frozen_real_world_suite_has_not_changed() -> None:
    assert suite_sha256(REAL_WORLD_PATH) == frozen.REAL_WORLD_V1_SHA256
    recorded = json.loads(ANONYMISED_PATH.read_text(encoding="utf-8"))
    assert recorded["suite_sha256"] == frozen.REAL_WORLD_V1_SHA256
    assert suite_sha256(ANONYMISED_PATH) == frozen.REAL_WORLD_V1_ANONYMISED_SHA256


def test_the_suite_is_valid_and_cited() -> None:
    cases = load_real_world()
    kinds = [c.kind for c in cases]
    assert kinds.count("issues") == 5
    assert kinds.count("clean") == 4
    scored = [i for c in cases for i in c.scored_issues]
    assert len([i for i in scored if not i.secondary]) == 7
    for issue in scored:
        assert issue.citation_url.startswith("https://")
        assert issue.categories
        assert all(isinstance(c, Category) for c in issue.categories)
    diabetes = next(c for c in cases if c.id == "diabetes_readmission")
    assert diabetes.entity_scope is not None
    assert diabetes.entity_scope.value == "new"
    unscored = [i.id for c in cases for i in c.issues if not i.scored]
    assert unscored == ["death_makes_readmission_impossible"]


def test_real_world_datasets_are_never_trap_bases() -> None:
    registry = load_registry()
    for case in load_real_world():
        assert registry[case.dataset].pool == "real_world"


def _mutated(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "suite.toml"
    path.write_text(REAL_WORLD_PATH.read_text(encoding="utf-8").replace(old, new, 1), "utf-8")
    return path


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('dataset = "titanic"', 'dataset = "qsar_biodeg"', "'final' pool"),
        ("openml_version = 3", "openml_version = 4", "id/version differ"),
        ('remediation = "drop:boat"', 'remediation = "remove boat"', "bad remediation"),
        (
            'citation_url = "https://arxiv.org/abs/1301.3524"',
            'citation_url = ""',
            "needs a citation URL",
        ),
        ('categories = ["group_leakage"]', 'categories = ["leakage"]', "bad case"),
    ],
)
def test_an_inconsistent_suite_is_refused(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(SuiteError, match=message):
        load_real_world(_mutated(tmp_path, old, new))


def test_the_anonymiser_renames_in_file_order_and_keeps_the_values() -> None:
    frame = pd.DataFrame({"boat": ["2", None], "survived": [1, 0], "age": [30.0, 41.0]})
    mapping = anonymisation_for(list(frame.columns), "survived")
    assert mapping.columns == {"boat": "col_1", "survived": ANON_TARGET, "age": "col_2"}
    renamed = mapping.apply(frame)
    assert list(renamed.columns) == ["col_1", "target", "col_2"]
    assert renamed["col_2"].tolist() == [30.0, 41.0]  # only names change
    with pytest.raises(SuiteError, match="columns differ"):
        mapping.apply(frame.drop(columns=["age"]))


def test_the_recorded_mappings_cover_every_case_and_name_every_issue_column() -> None:
    cases = {c.id: c for c in load_real_world()}
    mappings = load_anonymisations()
    assert set(mappings) == set(cases)
    for case_id, mapping in mappings.items():
        case = cases[case_id]
        assert isinstance(mapping, Anonymisation)
        assert mapping.rename(case.target) == ANON_TARGET
        names = list(mapping.columns.values())
        features = {n for n in names if n != ANON_TARGET}
        assert features == {f"col_{i}" for i in range(1, len(features) + 1)}
        assert len(names) == len(set(names))  # one-to-one
        for issue in case.issues:
            for column in issue.columns:
                assert mapping.rename(column).startswith("col_")
