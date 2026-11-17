"""Run checked EDA snippets in a fresh ``python -I`` process on a copy of development rows.

The sandbox directory contains only the bootstrap script, the snippet and ``dev.parquet``
(development rows only; the holdout file and its path are never copied or mentioned). The
child gets a minimal environment (no API key, no inherited secrets), pruned builtins and an
import guard, the limits from :mod:`holdout.sandbox.limits`, and its output is capped.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from holdout.agent.schemas import EDASnippet, SandboxResult
from holdout.sandbox.limits import Limits, MemoryWatchdog, preexec_for
from holdout.sandbox.policy import ALLOWED_IMPORTS, BLOCKED_NAMES, check_snippet

#: At most this many snippets per run.
MAX_SNIPPETS_PER_RUN: Final = 4

_BOOTSTRAP = r"""
import builtins, io, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LIMIT = {limit}
ALLOWED = {allowed!r}
BLOCKED = {blocked!r}

class Capped(io.StringIO):
    def write(self, text):
        room = LIMIT - self.tell()
        if room > 0:
            super().write(text[:room])
        return len(text)

out = Capped()
sys.stdout = sys.stderr = out
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED:
        raise ImportError(f"import of {{name!r}} is not allowed in the sandbox")
    return real_import(name, globals, locals, fromlist, level)

safe = {{k: v for k, v in vars(builtins).items() if k not in BLOCKED}}
safe["__import__"] = guarded_import
df = pd.read_parquet("dev.parquet")
namespace = {{"__builtins__": safe, "df": df, "pd": pd, "np": np, "plt": plt}}
source = open("snippet.py", encoding="utf-8").read()
try:
    exec(compile(source, "snippet", "exec"), namespace)
    status = 0
except BaseException as exc:
    print(f"{{type(exc).__name__}}: {{exc}}")
    status = 3
sys.__stdout__.write(out.getvalue()[:LIMIT])
sys.__stdout__.flush()
sys.exit(status)
"""


def minimal_env(workdir: Path) -> dict[str, str]:
    env = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(workdir),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    for key in ("SYSTEMROOT", "PATH") if sys.platform == "win32" else ("PATH",):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_snippet(
    snippet: EDASnippet,
    dev: pd.DataFrame,
    limits: Limits | None = None,
    figures_dir: Path | None = None,
) -> SandboxResult:
    """Check and run one snippet; never raises for snippet failures."""
    limits = limits if limits is not None else Limits()
    violations = check_snippet(snippet.code)
    if violations:
        return SandboxResult(
            purpose=snippet.purpose,
            status="rejected",
            output="",
            detail="; ".join(str(v) for v in violations),
        )
    with tempfile.TemporaryDirectory(prefix="holdout-sandbox-") as tmp:
        workdir = Path(tmp)
        dev.to_parquet(workdir / "dev.parquet", index=False)
        (workdir / "snippet.py").write_text(snippet.code, encoding="utf-8")
        bootstrap = _BOOTSTRAP.format(
            limit=limits.output_bytes,
            allowed=sorted(ALLOWED_IMPORTS),
            blocked=sorted(BLOCKED_NAMES - {"__import__"}),
        )
        (workdir / "bootstrap.py").write_text(bootstrap, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-I", "bootstrap.py"],
            cwd=workdir,
            env=minimal_env(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec_for(limits),
        )
        watchdog = MemoryWatchdog(process.pid, limits.memory_bytes).start()
        try:
            raw, _ = process.communicate(timeout=limits.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return SandboxResult(
                purpose=snippet.purpose,
                status="timeout",
                output="",
                detail=f"killed after {limits.timeout_seconds:g} s",
            )
        finally:
            watchdog.stop()
        output = raw[: limits.output_bytes].decode("utf-8", errors="replace")
        if watchdog.tripped or "MemoryError" in output:
            return SandboxResult(
                purpose=snippet.purpose,
                status="memory",
                output=output,
                detail=f"memory limit {limits.memory_bytes // 1024**2} MB",
            )
        figures = sorted(p.name for p in workdir.glob("*.png"))
        if figures_dir is not None and figures:
            figures_dir.mkdir(parents=True, exist_ok=True)
            for name in figures:
                shutil.copy2(workdir / name, figures_dir / name)
        if process.returncode != 0:
            return SandboxResult(
                purpose=snippet.purpose,
                status="error",
                output=output,
                detail=f"exit code {process.returncode}",
            )
        detail = f"figures: {', '.join(figures)}" if figures else None
        return SandboxResult(purpose=snippet.purpose, status="ok", output=output, detail=detail)


def run_snippets(
    snippets: Sequence[EDASnippet], dev: pd.DataFrame, figures_dir: Path | None = None
) -> list[SandboxResult]:
    """Run at most :data:`MAX_SNIPPETS_PER_RUN` snippets, each in its own process."""
    return [run_snippet(s, dev, figures_dir=figures_dir) for s in snippets[:MAX_SNIPPETS_PER_RUN]]
