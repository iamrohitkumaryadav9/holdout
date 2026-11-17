"""EDA sandbox: the allowlist rejects every blocked construct; limits kill on this OS;
the holdout path never reaches the sandbox directory or environment."""

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from holdout.agent.schemas import EDASnippet
from holdout.sandbox import runner
from holdout.sandbox.limits import Limits
from holdout.sandbox.policy import check_snippet

DEV = pd.DataFrame({"a": np.arange(200), "b": np.arange(200) * 0.5, "y": np.arange(200) % 2})


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import sys",
        "import subprocess",
        "import socket",
        "import shutil",
        "import pathlib",
        "import urllib.request",
        "import requests",
        "import http.client",
        "from os import path",
        "from . import x",
        "from pandas import *",
        "open('x.txt')",
        "eval('1+1')",
        "exec('x=1')",
        "__import__('os')",
        "compile('1', 'f', 'eval')",
        "getattr(df, 'shape')",
        "df.__class__",
        "df._data",
        "x = __builtins__",
        "pd.read_csv('other.csv')",
        "df.to_pickle('x.pkl')",
        "df.to_csv('x.csv')",
        "np.load('x.npy')",
        "np.save('x.npy', df.values)",
        "df.eval('a + b')",
        "df.query('a > 1')",
        "global x",
        "def f(:",
    ],
)
def test_blocked_constructs_are_rejected(code: str) -> None:
    assert check_snippet(code), f"not rejected: {code!r}"


@pytest.mark.parametrize(
    "code",
    [
        "print(df.describe().to_string())",
        "import numpy as np\nprint(np.corrcoef(df['a'], df['b'])[0, 1])",
        "from scipy import stats\nprint(stats.ttest_ind(df['a'], df['b']).pvalue)",
        "import scipy.stats\nprint(scipy.stats.skew(df['a']))",
        "import matplotlib.pyplot as plt\nplt.hist(df['a'])\nplt.savefig('a.png')",
        "print(df['y'].value_counts().to_dict())",
    ],
)
def test_allowed_snippets_pass_the_check(code: str) -> None:
    assert check_snippet(code) == []


def _run(code: str, limits: Limits | None = None) -> Any:
    return runner.run_snippet(EDASnippet(purpose="probe", code=code), DEV, limits)


def test_a_snippet_runs_on_the_dev_copy() -> None:
    result = _run("print(df.shape)\nprint(int(df['y'].sum()))")
    assert result.status == "ok"
    assert "(200, 3)" in result.output
    assert "100" in result.output


def test_runtime_guard_backs_up_the_static_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "check_snippet", lambda code: [])  # bypass layer one
    result = _run("import os\nprint(os.getcwd())")
    assert result.status == "error"
    assert "not allowed in the sandbox" in result.output
    result = _run("open('dev.parquet')")
    assert result.status == "error"
    assert "NameError" in result.output


def test_rejected_snippets_never_start_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_process(*args: object, **kwargs: object) -> None:
        raise AssertionError("a rejected snippet must not run")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    result = _run("import os")
    assert result.status == "rejected"
    assert "import of 'os'" in str(result.detail)


def test_timeout_kills_the_process() -> None:
    result = _run("while True:\n    pass", Limits(timeout_seconds=3))
    assert result.status == "timeout"


def test_memory_limit_kills_the_process() -> None:
    result = _run(
        "big = np.ones((80, 1024, 1024))\nprint(big.sum())",  # ~640 MB
        Limits(timeout_seconds=15, memory_bytes=300 * 1024**2),
    )
    assert result.status == "memory"


def test_output_is_capped() -> None:
    result = _run("print('x' * 500000)", Limits(output_bytes=1024))
    assert result.status == "ok"
    assert len(result.output) <= 1024


def test_figures_are_collected(tmp_path: Path) -> None:
    snippet = EDASnippet(
        purpose="histogram",
        code="import matplotlib.pyplot as plt\nplt.hist(df['a'])\nplt.savefig('a.png')",
    )
    result = runner.run_snippet(snippet, DEV, figures_dir=tmp_path / "eda")
    assert result.status == "ok"
    assert (tmp_path / "eda" / "a.png").is_file()


def test_sandbox_holds_only_dev_rows_and_no_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdout_path = tmp_path / "runs" / "r1" / "holdout.parquet"
    monkeypatch.setenv("GEMINI_API_KEY", "sk-sentinel-should-not-leak")
    seen: dict[str, Any] = {}
    real_popen = subprocess.Popen

    def spy(*args: Any, **kwargs: Any) -> Any:
        cwd = Path(kwargs["cwd"])
        seen["files"] = sorted(p.name for p in cwd.iterdir())
        seen["contents"] = b"".join(p.read_bytes() for p in cwd.iterdir())
        seen["env"] = dict(kwargs["env"])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", spy)
    assert _run("print(len(df))").status == "ok"
    assert seen["files"] == ["bootstrap.py", "dev.parquet", "snippet.py"]
    assert str(holdout_path).encode() not in seen["contents"]
    assert b"holdout" not in seen["contents"].lower()
    assert "GEMINI_API_KEY" not in seen["env"]
    assert "sk-sentinel-should-not-leak" not in str(seen["env"])
    assert os.environ["GEMINI_API_KEY"] == "sk-sentinel-should-not-leak"


def test_at_most_four_snippets_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runner, "run_snippet", lambda s, dev, figures_dir=None: calls.append(s.purpose) or None
    )
    runner.run_snippets([EDASnippet(purpose=f"snippet{i}", code="print(1)") for i in range(6)], DEV)
    assert calls == ["snippet0", "snippet1", "snippet2", "snippet3"]
