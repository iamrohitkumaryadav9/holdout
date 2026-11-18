"""The ReAct baseline sandbox and the restricted unpickler."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bench.systems.sandbox import (
    FIT_STEP_SECONDS,
    MODEL_FILE,
    STEP_SECONDS,
    check_step,
    is_fitting_step,
    prepare_workdir,
    run_step,
    truncate_observation,
)
from bench.systems.unpickle import UnsafePickleError, restricted_loads


def _dev(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {"x": x, "c": rng.choice(["a", "b"], n), "y": (x + rng.normal(0, 0.5, n) > 0).astype(int)}
    )


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        ("import os", "import of 'os'"),
        ("import subprocess", "import of 'subprocess'"),
        ("open('x')", "use of 'open'"),
        ("df.to_csv('x.csv')", "file I/O attribute 'to_csv'"),
        ("pd.read_parquet('../seal/holdout.parquet')", "file I/O attribute 'read_parquet'"),
        ("x = df.__class__", "private or dunder"),
        ("from sklearn import *", "star imports"),
        ("eval('1')", "use of 'eval'"),
    ],
)
def test_the_baseline_allowlist_rejects_escapes(code: str, fragment: str) -> None:
    assert any(fragment in p for p in check_step(code))


def test_the_baseline_allowlist_allows_the_modelling_stack() -> None:
    code = (
        "import numpy as np\nimport scipy.stats\nfrom sklearn.ensemble import "
        "RandomForestClassifier\nfrom lightgbm import LGBMClassifier\nimport math\n"
    )
    assert check_step(code) == []


def test_only_model_fitting_steps_get_60_seconds() -> None:
    assert (STEP_SECONDS, FIT_STEP_SECONDS) == (20.0, 60.0)
    assert is_fitting_step("model.fit(X, y)")
    assert is_fitting_step("scores = cross_val_score(m, X, y)")
    assert is_fitting_step("search = GridSearchCV(m, grid)")
    assert not is_fitting_step("print(df.describe())")


def test_objects_persist_between_steps_and_the_model_is_submitted(tmp_path: Path) -> None:
    prepare_workdir(tmp_path, _dev())
    first = run_step(
        "from sklearn.linear_model import LogisticRegression\n"
        "m = LogisticRegression().fit(df[['x']], df['y'])\nsave_object('m', m)\nprint('ok')",
        tmp_path,
    )
    assert (first.status, first.seconds_limit) == ("ok", 60.0)
    second = run_step("m = load_object('m')\nsubmit_model(m)", tmp_path)
    assert second.status == "ok", second.output
    model = restricted_loads((tmp_path / MODEL_FILE).read_bytes())
    assert list(model.predict(pd.DataFrame({"x": [-2.0, 2.0]}))) == [0, 1]


def test_a_step_that_runs_too_long_is_killed(tmp_path: Path) -> None:
    prepare_workdir(tmp_path, _dev(10))
    result = run_step("while True:\n    pass", tmp_path, step_seconds=1.0)
    assert result.status == "timeout"


def test_a_rejected_or_failing_step_never_raises(tmp_path: Path) -> None:
    prepare_workdir(tmp_path, _dev(10))
    assert run_step("import os", tmp_path).status == "rejected"
    failed = run_step("raise ValueError('boom')", tmp_path)
    assert failed.status == "error"
    assert "ValueError: boom" in failed.output
    assert run_step("submit_model(42)", tmp_path).status == "error"


@pytest.mark.full_only
def test_the_memory_limit_is_enforced(tmp_path: Path) -> None:
    prepare_workdir(tmp_path, _dev(10))
    result = run_step(
        "import numpy as np\nx = np.ones((400, 1024, 1024))\nprint(x.sum())", tmp_path
    )
    assert result.status in {"memory", "error"}


def test_the_react_prompt_is_published_verbatim_in_the_benchmark_docs() -> None:
    root = Path(__file__).resolve().parents[2]
    prompt = (root / "bench/systems/prompts/react.md").read_text("utf-8").replace("\r\n", "\n")
    docs = (root / "docs/benchmark.md").read_text("utf-8").replace("\r\n", "\n")
    assert prompt.strip() in docs


def test_observations_are_truncated_head_and_tail() -> None:
    text = "A" * 3000 + "Z" * 3000
    shown, removed = truncate_observation(text, 2000)
    assert len(shown) <= 2000
    assert removed == 4000
    assert "[truncated 4000 chars]" in shown
    assert shown.startswith("A")
    assert shown.endswith("Z")
    assert truncate_observation("short", 2000) == ("short", 0)


# ------------------------------------------------------------------ restricted unpickling


@pytest.mark.parametrize(
    "model",
    [
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer(
                (
                    OneHotEncoder(handle_unknown="ignore"),
                    make_column_selector(dtype_include=object),
                ),
                remainder="passthrough",
            ),
            RandomForestClassifier(n_estimators=5, random_state=0),
        ),
        LGBMClassifier(n_estimators=5, verbose=-1),
        HistGradientBoostingClassifier(max_iter=10),
    ],
    ids=["logreg", "column_transformer_forest", "lightgbm", "hist_gb"],
)
def test_real_pipelines_load_through_the_restricted_unpickler(model: object) -> None:
    dev = _dev()
    X = dev[["x", "c"]] if "column" in repr(model) else dev[["x"]]
    fitted = model.fit(X, dev["y"])  # type: ignore[attr-defined]
    loaded = restricted_loads(pickle.dumps(fitted))
    assert list(loaded.predict(X[:5])) == list(fitted.predict(X[:5]))


class _Evil:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        import os

        return (os.system, ("echo pwned",))


class _EvalEvil:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (eval, ("1+1",))


@pytest.mark.parametrize("payload", [_Evil(), _EvalEvil()], ids=["os.system", "eval"])
def test_malicious_pickles_are_refused(payload: object) -> None:
    with pytest.raises(UnsafePickleError, match="refusing to load"):
        restricted_loads(pickle.dumps(payload))
