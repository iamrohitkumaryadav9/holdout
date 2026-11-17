"""Synthetic base datasets for trap generation (no downloads needed).

Every builder is deterministic per seed and returns ``(frame, target, task)``. Column
names are deliberately neutral (no outcome-timing words, no id-like names), the data has
no duplicates, no entity structure and no time order unless a builder is asked for it, so
the clean bases stay quiet under every audit.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from holdout.core.schemas import Task
from holdout.core.seeds import derive_seed

Base = tuple[pd.DataFrame, str, Task]


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, f"bench/base/{name}"))


def classification(n_rows: int = 1200, seed: int = 0) -> Base:
    """Binary target ``converted`` (~30% positive) from a few informative features."""
    rng = _rng(seed, "classification")
    age = rng.integers(18, 75, n_rows)
    income = np.round(rng.lognormal(10.3, 0.45, n_rows), 2)
    tenure = rng.integers(0, 120, n_rows)
    visits = rng.poisson(3.0, n_rows)
    segment = rng.choice(["retail", "small_business", "enterprise", "public"], n_rows)
    channel = rng.choice(["web", "phone", "branch"], n_rows).astype(object)
    score_a = rng.normal(size=n_rows)
    score_b = rng.normal(size=n_rows)
    logit = (
        -1.1
        + 0.02 * (tenure - 60) / 10
        + 0.35 * (visits - 3)
        + 0.6 * score_a
        + np.select([segment == "enterprise", segment == "public"], [0.5, -0.4], 0.0)
        + 0.4 * np.log(income / 30000)
        + rng.normal(0, 0.5, n_rows)
    )
    converted = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    channel[rng.random(n_rows) < 0.03] = None
    frame = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "tenure_months": tenure,
            "visits": visits,
            "segment": segment,
            "channel": channel,
            "score_a": np.round(score_a, 4),
            "score_b": np.round(score_b, 4),
            "converted": converted,
        }
    )
    return frame, "converted", Task.CLASSIFICATION


def regression(n_rows: int = 1200, seed: int = 0) -> Base:
    """Continuous target ``spend`` from the same kind of features."""
    frame, _, _ = classification(n_rows, seed)
    rng = _rng(seed, "regression")
    spend = (
        200
        + 3.0 * frame["visits"]
        + 40 * frame["score_a"]
        + 0.002 * frame["income"]
        + np.where(frame["segment"] == "enterprise", 60, 0)
        + rng.normal(0, 25, len(frame))
    )
    frame = frame.drop(columns=["converted"]).assign(spend=np.round(spend, 2))
    return frame, "spend", Task.REGRESSION


def time_ordered(n_days: int = 1200, seed: int = 0, drift: float = 1.0) -> Base:
    """One row per day in date order; ``drift`` adds a trend and a changing promo effect.

    With ``drift=0`` the rows are exchangeable in time (no trend, no concept drift), which
    makes it the clean twin of the time-order trap.
    """
    rng = _rng(seed, "time_ordered")
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    progress = np.linspace(0.0, 1.0, n_days)
    temperature = np.round(rng.normal(15, 6, n_days), 1)
    promo = (rng.random(n_days) < 0.25).astype(int)
    price = np.round(rng.uniform(4.0, 6.0, n_days), 2)
    weekday = dates.dayofweek.to_numpy()
    promo_effect = 20 + drift * 40 * progress
    demand = (
        150
        + drift * 120 * progress
        + promo_effect * promo
        - 10 * (price - 5)
        + 1.2 * temperature
        + 8 * (weekday >= 5)
        + rng.normal(0, 8, n_days)
    )
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temperature": temperature,
            "promo": promo,
            "price": price,
            "weekday": pd.Series(weekday).map(
                dict(enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]))
            ),
            "demand": np.round(demand, 1),
        }
    )
    return frame, "demand", Task.REGRESSION


def easy_clean(n_rows: int = 1500, seed: int = 0) -> Base:
    """Near-separable but clean, modelled on the mushroom data: ``odor`` alone almost
    determines the class, honestly (it is measured before the outcome)."""
    rng = _rng(seed, "easy_clean")
    odor = rng.choice(
        ["almond", "anise", "none", "foul", "fishy", "spicy", "pungent", "creosote", "musty"],
        n_rows,
        p=[0.08, 0.08, 0.36, 0.2, 0.06, 0.06, 0.06, 0.05, 0.05],
    )
    poisonous_odor = np.isin(odor, ["foul", "fishy", "spicy", "pungent", "creosote", "musty"])
    ambiguous = odor == "none"
    edible = np.where(poisonous_odor, "p", "e").astype(object)
    flip = ambiguous & (rng.random(n_rows) < 0.04)
    edible[flip] = "p"
    return (
        pd.DataFrame(
            {
                "cap_shape": rng.choice(["bell", "conical", "convex", "flat", "knobbed"], n_rows),
                "cap_color": rng.choice(["brown", "buff", "gray", "green", "pink", "red"], n_rows),
                "odor": odor,
                "gill_size": rng.choice(["broad", "narrow"], n_rows),
                "ring_type": rng.choice(["evanescent", "flaring", "large", "pendant"], n_rows),
                "habitat": rng.choice(["grasses", "leaves", "meadows", "paths", "woods"], n_rows),
                "stem_height": np.round(rng.normal(6.0, 1.5, n_rows), 3),
                "edible": edible,
            }
        ),
        "edible",
        Task.CLASSIFICATION,
    )


BUILDERS: dict[str, Callable[..., Base]] = {
    "classification": classification,
    "regression": regression,
    "time_ordered": time_ordered,
    "easy_clean": easy_clean,
}
