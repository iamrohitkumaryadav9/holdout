"""Shared fixtures and the test-tier switch.

Tiers: ``pytest`` runs the fast tier (2 seeds, no ``full_only`` or ``slow`` tests);
``pytest --tier full`` (or ``HOLDOUT_TEST_TIER=full``) runs every test and seed. The tier
is set here, before collection, because seed parametrisations depend on it
(``tests/tiers.py``; do not import that module from this file).
"""

import os
import socket
from collections.abc import Iterator
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


class NetworkBlockedError(RuntimeError):
    """Raised by any socket use during tests: CI never touches the network."""


@pytest.fixture(autouse=True, scope="session")
def _no_network() -> Iterator[None]:
    def refuse(*args: object, **kwargs: object) -> None:
        raise NetworkBlockedError("network access is blocked in tests (use FakeLLM)")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", refuse)
        patch.setattr(socket.socket, "connect_ex", refuse)
        patch.setattr(socket, "create_connection", refuse)
        patch.setattr(socket, "getaddrinfo", refuse)
        yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_holdout_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Spend ledger and LLM cache go to a temporary HOLDOUT_HOME, never ~/.holdout."""
    home = tmp_path_factory.mktemp("holdout_home")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOLDOUT_HOME", str(home))
        patch.delenv("GEMINI_API_KEY", raising=False)
        yield home


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
