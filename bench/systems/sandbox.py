"""The "baseline" sandbox profile for the ReAct system.

Same layered design and limits as Holdout's EDA sandbox (``holdout.sandbox``): a static AST
allowlist, a fresh ``python -I`` process per step with a minimal environment (no API key),
pruned builtins, an import guard, the memory watchdog (1 GB) and the 64 KB output cap.
Differences, all for a fair, charitable baseline:

* imports: pandas, numpy, scipy, scikit-learn and LightGBM (any submodule), math, statistics;
* persistence: each step runs in a fresh process, so the namespace provides
  ``save_object(name, obj)`` / ``load_object(name)`` (pickles inside the run's own work
  directory) and ``submit_model(pipeline)`` (the final sklearn pipeline for the harness);
  the snippet itself still cannot call ``open`` or any file I/O;
* time: 20 s per step, or **60 s for model-fitting steps** (code that calls ``fit*``,
  ``cross_val*`` or a ``*SearchCV``), a deliberate fairness choice documented in
  docs/benchmark.md.

The work directory holds ``dev.parquet`` (development rows only) and the agent's own
objects; the sealed holdout and its path are never there.
"""

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from holdout.sandbox.limits import Limits, MemoryWatchdog, preexec_for
from holdout.sandbox.policy import (
    ALLOWED_TO_ATTRIBUTES,
    BLOCKED_ATTRIBUTE_PREFIXES,
    BLOCKED_ATTRIBUTES,
    BLOCKED_NAMES,
)
from holdout.sandbox.runner import minimal_env

ALLOWED_ROOTS: Final = frozenset(
    {"pandas", "numpy", "scipy", "sklearn", "lightgbm", "math", "statistics"}
)
STEP_SECONDS: Final = 20.0
FIT_STEP_SECONDS: Final = 60.0
OBJECT_NAME: Final = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
MODEL_FILE: Final = "final_model.pkl"
_FIT_NAME: Final = re.compile(r"^(fit\w*|cross_val\w*|\w*SearchCV)$")


@dataclass(frozen=True)
class StepResult:
    status: str  # ok | rejected | error | timeout | memory
    output: str
    seconds_limit: float
    detail: str | None = None


def is_fitting_step(source: str) -> bool:
    """True when the code fits a model: a call to ``fit*``, ``cross_val*`` or ``*SearchCV``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if _FIT_NAME.match(name):
                return True
    return False


def check_step(source: str) -> list[str]:
    """Every construct outside the baseline allowlist (empty when the step may run)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"line {exc.lineno or 0}: syntax error: {exc.msg}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_ROOTS:
                    problems.append(f"line {line}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module.split(".")[0] not in ALLOWED_ROOTS:
                problems.append(f"line {line}: import from {module!r} is not allowed")
            if any(a.name == "*" for a in node.names):
                problems.append(f"line {line}: star imports are not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                problems.append(f"line {line}: use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                problems.append(f"line {line}: private or dunder attribute {attr!r}")
            elif attr in BLOCKED_ATTRIBUTES:
                problems.append(f"line {line}: attribute {attr!r} is not allowed")
            elif attr.startswith(BLOCKED_ATTRIBUTE_PREFIXES) and attr not in ALLOWED_TO_ATTRIBUTES:
                problems.append(f"line {line}: file I/O attribute {attr!r} is not allowed")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            problems.append(f"line {line}: global and nonlocal are not allowed")
    return problems


_BOOTSTRAP = r"""
import builtins, io, pickle, re, sys
import numpy as np
import pandas as pd

LIMIT = {limit}
ROOTS = {roots!r}
BLOCKED = {blocked!r}
NAME = re.compile(r"^[a-z][a-z0-9_]{{0,40}}$")

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
    if level or name.split(".")[0] not in ROOTS:
        raise ImportError(f"import of {{name!r}} is not allowed in the sandbox")
    return real_import(name, globals, locals, fromlist, level)

def save_object(name, obj):
    if not NAME.match(str(name)):
        raise ValueError("object names: lowercase letters, digits and _")
    with open(f"obj_{{name}}.pkl", "wb") as handle:
        pickle.dump(obj, handle)

def load_object(name):
    if not NAME.match(str(name)):
        raise ValueError("object names: lowercase letters, digits and _")
    with open(f"obj_{{name}}.pkl", "rb") as handle:
        return pickle.load(handle)

def submit_model(pipeline):
    if not (hasattr(pipeline, "fit") and hasattr(pipeline, "predict")):
        raise TypeError("submit_model needs a fitted scikit-learn estimator or Pipeline")
    with open("{model}", "wb") as handle:
        pickle.dump(pipeline, handle)
    print("model submitted")

safe = {{k: v for k, v in vars(builtins).items() if k not in BLOCKED}}
safe["__import__"] = guarded_import
df = pd.read_parquet("dev.parquet")
namespace = {{
    "__builtins__": safe, "df": df, "pd": pd, "np": np,
    "save_object": save_object, "load_object": load_object, "submit_model": submit_model,
}}
source = open("step.py", encoding="utf-8").read()
try:
    exec(compile(source, "step", "exec"), namespace)
    status = 0
except BaseException as exc:
    print(f"{{type(exc).__name__}}: {{exc}}")
    status = 3
sys.__stdout__.write(out.getvalue()[:LIMIT])
sys.__stdout__.flush()
sys.exit(status)
"""


def prepare_workdir(workdir: Path, dev: pd.DataFrame) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(workdir / "dev.parquet", index=False)


def run_step(
    source: str,
    workdir: Path,
    limits: Limits | None = None,
    *,
    step_seconds: float = STEP_SECONDS,
    fit_seconds: float = FIT_STEP_SECONDS,
) -> StepResult:
    """Check and run one ReAct step in ``workdir`` (which persists between steps)."""
    base = limits if limits is not None else Limits()
    seconds = fit_seconds if is_fitting_step(source) else step_seconds
    problems = check_step(source)
    if problems:
        return StepResult("rejected", "", seconds, "; ".join(problems))
    step_limits = Limits(
        timeout_seconds=seconds, memory_bytes=base.memory_bytes, output_bytes=base.output_bytes
    )
    (workdir / "step.py").write_text(source, encoding="utf-8")
    bootstrap = _BOOTSTRAP.format(
        limit=step_limits.output_bytes,
        roots=sorted(ALLOWED_ROOTS),
        blocked=sorted(BLOCKED_NAMES - {"__import__"}),
        model=MODEL_FILE,
    )
    (workdir / "bootstrap.py").write_text(bootstrap, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-I", "bootstrap.py"],
        cwd=workdir,
        env=minimal_env(workdir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=preexec_for(step_limits),
    )
    watchdog = MemoryWatchdog(process.pid, step_limits.memory_bytes).start()
    try:
        raw, _ = process.communicate(timeout=step_limits.timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return StepResult("timeout", "", seconds, f"killed after {seconds:g} s")
    finally:
        watchdog.stop()
    output = raw[: step_limits.output_bytes].decode("utf-8", errors="replace")
    if watchdog.tripped or "MemoryError" in output:
        return StepResult(
            "memory", output, seconds, f"memory limit {step_limits.memory_bytes // 1024**2} MB"
        )
    if process.returncode != 0:
        return StepResult("error", output, seconds, f"exit code {process.returncode}")
    return StepResult("ok", output, seconds)


def truncate_observation(text: str, limit: int = 2000) -> tuple[str, int]:
    """Head and tail of ``text`` within ``limit`` characters, with a marker.

    Returns the shown text and the number of characters removed.
    """
    if len(text) <= limit:
        return text, 0
    removed = len(text) - limit
    marker = f"\n[truncated {removed} chars]\n"
    keep = max(0, limit - len(marker))
    head = keep // 2
    tail = keep - head
    shown = text[:head] + marker + (text[-tail:] if tail else "")
    return shown, removed
