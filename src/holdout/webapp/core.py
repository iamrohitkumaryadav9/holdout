"""The web app's logic, without Streamlit (unit-tested directly).

Key handling: a visitor's Gemini key lives only in their Streamlit session. It is passed to
:class:`GeminiClient` as an argument, never put in ``os.environ`` (shared by every session
on the server), never logged, cached or written to a run's files. Offline is the default.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from holdout.agent.driver import AgentOutcome, run_agent
from holdout.agent.graph import AgentOptions
from holdout.audit.types import AuditFlag, Severity
from holdout.data.examples import make_classification_example
from holdout.llm import config as llm_config
from holdout.llm.client import GeminiClient
from holdout.llm.progress import LineProgress

#: Gemini runs one browser session may start (each is also capped at RUN_CAP_INR, and a
#: corrected-split follow-up is a second run under the same cap).
MAX_GEMINI_RUNS_PER_SESSION: Final = 3
MAX_UPLOAD_ROWS: Final = 50_000


@dataclass(frozen=True)
class Demo:
    label: str
    description: str
    build: Callable[[], tuple[pd.DataFrame, str, str]]  # frame, target, goal


def _churn() -> tuple[pd.DataFrame, str, str]:
    return make_classification_example(), "churned", "predict which customers churn"


def _trap(name: str, **kwargs: Any) -> Callable[[], tuple[pd.DataFrame, str, str]]:
    def build() -> tuple[pd.DataFrame, str, str]:
        from bench.traps import build_case  # the repository's trap generators

        case = build_case(name, 0, **kwargs)
        return case.df, case.target, case.goal_text

    return build


def demos() -> list[Demo]:
    """Synthetic demo datasets (no download). The planted traps need the repository's
    ``bench`` package, present when the app runs from a checkout."""
    out = [Demo("Churn (clean)", "Synthetic customer churn; no planted problem.", _churn)]
    try:
        import bench.traps  # noqa: F401
    except ImportError:
        return out
    return [
        *out,
        Demo("Leaky column (direct_leak)",
             "`refund_issued` is derived from the target: a critical leak.",
             _trap("direct_leak")),
        Demo("Repeated customers (group_leakage)",
             "Rows share customers across the split: the holdout is contaminated.",
             _trap("group_leakage", n_rows=600)),
        Demo("Causal question (causal_question)",
             "The goal asks whether X causes Y; Holdout declines a causal conclusion.",
             _trap("causal_question")),
    ]  # fmt: skip


def gemini_allowed(runs_started: int) -> tuple[bool, str]:
    left = MAX_GEMINI_RUNS_PER_SESSION - runs_started
    if left <= 0:
        return False, (
            f"This session has used its {MAX_GEMINI_RUNS_PER_SESSION} Gemini runs. "
            "Offline runs are unlimited."
        )
    return True, (
        f"{left} of {MAX_GEMINI_RUNS_PER_SESSION} Gemini runs left in this session; each "
        f"run is capped at Rs {llm_config.RUN_CAP_INR:g} of spend (a follow-up run with a "
        "corrected split has its own cap)."
    )


def looks_like_key(text: str) -> bool:
    """A cheap shape check so an obviously wrong paste fails before any request."""
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]{20,200}", text.strip()))


def run_holdout(
    frame: pd.DataFrame,
    target: str,
    goal: str,
    workdir: Path,
    *,
    entities: str | None = None,
    api_key: str | None = None,
    on_line: Callable[[str], None] = lambda _: None,
) -> AgentOutcome:
    """One autopilot run: offline unless ``api_key`` is given (then Gemini, under the caps).

    Autopilot answers every question with the logged policy: it accepts the proposed split
    and the audit's fixes, and asks to unseal with warnings acknowledged. It never overrides
    a critical flag, so a critical finding still blocks the holdout.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    csv = workdir / "data.csv"
    frame.to_csv(csv, index=False)
    offline = api_key is None
    model = llm_config.DEFAULT_MODEL

    def client() -> GeminiClient:
        return GeminiClient(model=model, api_key=api_key)

    return run_agent(
        AgentOptions(
            csv=str(csv),
            target=target,
            out_dir=str(workdir / "runs"),
            goal=goal or None,
            entities=entities,
            mode="autopilot",
            offline=offline,
            use_cache=False,  # a shared server must not answer one visitor from another's
            model=None if offline else model,
        ),
        client_factory=None if offline else client,
        progress=LineProgress(on_line),
    )


def by_severity(flags: Sequence[AuditFlag]) -> dict[Severity, list[AuditFlag]]:
    out: dict[Severity, list[AuditFlag]] = {s: [] for s in Severity}
    for flag in flags:
        out[flag.severity].append(flag)
    return out


_SUITE = re.compile(r"^## (?P<name>.+)$", re.MULTILINE)


def markdown_sections(text: str, suite: str, titles: Sequence[str]) -> list[str]:
    """The ``### <title>`` subsections (title prefix match) of ``## <suite>`` in a
    generated results document, in document order, each with its heading."""
    starts = [(m.group("name").strip(), m.start()) for m in _SUITE.finditer(text)]
    body = ""
    for i, (name, start) in enumerate(starts):
        if name == suite:
            end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
            body = text[start:end]
    parts = re.split(r"^(?=### )", body, flags=re.MULTILINE)
    return [p.strip() for p in parts if any(p.startswith(f"### {t}") for t in titles)]
