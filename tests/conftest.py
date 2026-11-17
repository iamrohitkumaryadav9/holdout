"""Shared fixtures and the test-tier switch.

Tiers: ``pytest`` runs the fast tier (2 seeds, no ``full_only`` or ``slow`` tests);
``pytest --tier full`` (or ``HOLDOUT_TEST_TIER=full``) runs every test and seed. The tier
is set here, before collection, because seed parametrisations depend on it
(``tests/tiers.py``; do not import that module from this file).
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from holdout.data.examples import make_classification_example, make_timeseries_example
from tests.helpers import AccuracyScorer, make_grouped_df


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tier",
        choices=("fast", "full"),
        default=None,
        help="fast (default): fewer seeds, no full_only/slow tests; full: everything",
    )


def pytest_configure(config: pytest.Config) -> None:
    tier = config.getoption("--tier") or os.environ.get("HOLDOUT_TEST_TIER", "fast")
    os.environ["HOLDOUT_TEST_TIER"] = tier


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("HOLDOUT_TEST_TIER") == "full":
        return
    skip = pytest.mark.skip(reason="full tier only (run with --tier full)")
    for item in items:
        if "full_only" in item.keywords or "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def churn_df() -> pd.DataFrame:
    return make_classification_example(n_rows=400, seed=3)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return make_timeseries_example(n_days=60, seed=3)


@pytest.fixture
def grouped_df() -> pd.DataFrame:
    return make_grouped_df()


@pytest.fixture
def accuracy_scorer() -> AccuracyScorer:
    return AccuracyScorer()


@pytest.fixture
def csv_path(tmp_path: Path, churn_df: pd.DataFrame) -> Path:
    path = tmp_path / "churn.csv"
    churn_df.to_csv(path, index=False)
    return path
