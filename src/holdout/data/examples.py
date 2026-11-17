"""Synthetic example datasets for demos and end-to-end tests.

Both datasets carry a known signal so a sensible plan must beat the dummy baseline, and
each exercises one branch of the split proposal: the churn data is plain tabular
classification (stratified split); the sales data is time-ordered regression (time split,
or the time-order warning when declared tabular).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from holdout.core.seeds import derive_seed

CLASSIFICATION_FILENAME = "churn_classification.csv"
TIMESERIES_FILENAME = "store_sales_timeseries.csv"


def make_classification_example(n_rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Customer churn: numeric, categorical and boolean features with missing values.

    The target ``churned`` (about 25% positive) depends on tenure, spend, support tickets,
    plan tier and a promo flag, plus noise.
    """
    rng = np.random.default_rng(derive_seed(seed, "example_classification"))
    tenure = rng.integers(0, 73, n_rows)
    spend = np.round(rng.lognormal(mean=3.6, sigma=0.5, size=n_rows), 2)
    tickets = rng.poisson(1.2, n_rows)
    tier = rng.choice(["basic", "standard", "premium"], n_rows, p=[0.5, 0.35, 0.15])
    channel = rng.choice(["web", "store", "partner", "referral"], n_rows).astype(object)
    region = rng.choice([f"region_{i:02d}" for i in range(12)], n_rows)
    promo = rng.choice(["yes", "no"], n_rows, p=[0.3, 0.7])
    age = rng.integers(18, 81, n_rows).astype(float)

    logit = (
        -0.45
        - 0.045 * tenure
        + 0.012 * (spend - 40)
        + 0.55 * tickets
        + np.select([tier == "basic", tier == "premium"], [0.5, -0.6], 0.0)
        - 0.6 * (promo == "yes")
        + rng.normal(0, 0.6, n_rows)
    )
    churned = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    age[rng.random(n_rows) < 0.03] = np.nan
    channel[rng.random(n_rows) < 0.05] = None
    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "plan_tier": tier,
            "signup_channel": channel,
            "region": region,
            "has_promo": promo,
            "churned": churned,
        }
    )


def make_timeseries_example(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Daily unit sales for five stores, rows in date order.

    ``units_sold`` has an upward trend, weekly seasonality and a promo effect, so a random
    split lets a model "see the future" of the trend; a time split does not.
    """
    rng = np.random.default_rng(derive_seed(seed, "example_timeseries"))
    stores = ["north", "south", "east", "west", "central"]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frame = pd.DataFrame(
        [(date, store) for date in dates for store in stores], columns=["date", "store"]
    )
    n_rows = len(frame)
    day_index = (frame["date"] - dates[0]).dt.days.to_numpy()
    weekday = frame["date"].dt.dayofweek.to_numpy()
    promo = (rng.random(n_rows) < 0.2).astype(int)
    temperature = np.round(
        12 + 10 * np.sin(2 * np.pi * day_index / 365) + rng.normal(0, 3, n_rows), 1
    )
    price = np.round(rng.uniform(4.0, 6.0, n_rows), 2)
    store_effect = frame["store"].map(dict(zip(stores, [0, 15, -10, 5, 25], strict=True)))
    units = (
        100
        + 0.25 * day_index
        + 12 * (weekday >= 5)
        + 20 * promo
        - 8 * (price - 5)
        + 0.6 * temperature
        + store_effect.to_numpy()
        + rng.normal(0, 8, n_rows)
    )
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["weekday"] = pd.Series(weekday).map(
        dict(enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]))
    )
    frame["promo"] = promo
    frame["temperature"] = temperature
    frame["price"] = price
    frame["units_sold"] = np.round(units, 1)
    return frame


def write_examples(out_dir: Path, seed: int = 0) -> tuple[Path, Path]:
    """Write both example CSVs to ``out_dir``; return (classification, timeseries) paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    classification_path = out_dir / CLASSIFICATION_FILENAME
    timeseries_path = out_dir / TIMESERIES_FILENAME
    make_classification_example(seed=seed).to_csv(classification_path, index=False)
    make_timeseries_example(seed=seed).to_csv(timeseries_path, index=False)
    return classification_path, timeseries_path
