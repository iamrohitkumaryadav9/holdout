"""Amendment 1: FLAML (LightGBM) refuses feature names with JSON special characters, which
failed all 8 magic_telescope units; the adapter renames them for FLAML only."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.systems.base import SystemContext
from bench.systems.flaml_system import FlamlSystem, RenamedColumns, safe_names
from holdout.core import portable
from holdout.core.schemas import Task


def test_safe_names_replace_refused_characters_and_stay_unique() -> None:
    names = safe_names(["fLength:", "fLength_", 'q"x', "plain", "a[1]", "{b}", "c,d"])
    assert names == {"fLength:": "fLength__1", "fLength_": "fLength_", 'q"x': "q_x",
                     "plain": "plain", "a[1]": "a_1_", "{b}": "_b_", "c,d": "c_d"}  # fmt: skip
    assert len(set(names.values())) == len(names)


def test_flaml_fits_and_scores_with_colon_names(tmp_path: Path) -> None:
    pytest.importorskip("flaml")
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"fLength:": x1, "fWidth:": rng.normal(size=n),
                          "fAlpha:": rng.normal(size=n)})  # fmt: skip
    frame["class:"] = np.where(x1 + rng.normal(0, 0.5, n) > 0, "g", "h")
    ctx = SystemContext(unit_id="t", dev=frame, target="class:", task=Task.CLASSIFICATION,
                        goal="g", seed=1, workdir=tmp_path, model="m",
                        run_cap_inr=15.0)  # fmt: skip
    out = FlamlSystem(time_budget=3).run(ctx)
    assert out.status == "completed"
    assert isinstance(out.estimator, RenamedColumns)
    assert out.extra["renamed_columns"] == 3
    holdout = frame.sample(100, random_state=1)
    auc = portable.score("roc_auc", out.estimator, holdout.drop(columns=["class:"]),
                         holdout["class:"])  # fmt: skip
    assert auc > 0.7  # scored on the original column names
