"""Shared fixtures: small, deterministic datasets that exercise each split strategy."""

from pathlib import Path

import pandas as pd
import pytest

from holdout.data.examples import make_classification_example, make_timeseries_example
from tests.helpers import AccuracyScorer, make_grouped_df


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
