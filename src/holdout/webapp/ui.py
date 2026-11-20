"""Streamlit pages: Run, Benchmark, About. Started by ``app/streamlit_app.py``."""

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import holdout
from holdout import __version__
from holdout.agent.driver import AgentOutcome
from holdout.audit.findings import FindingStatus
from holdout.audit.render import fix_text
from holdout.audit.types import AuditFlag, Severity
from holdout.core.errors import HoldoutError
from holdout.export.manifest import NOT_UNSEALED, RunManifest
from holdout.llm import config as llm_config
from holdout.webapp import core, text

PAGES = ("Run", "Benchmark", "About")
#: docs/ of the checkout: next to src/ for an editable install, else the working directory
#: (the repository root on Streamlit Community Cloud).
DOCS = next(
    (p for p in (Path(holdout.__file__).resolve().parents[2] / "docs", Path.cwd() / "docs")
     if (p / "results.md").exists()),
    Path.cwd() / "docs",
)  # fmt: skip


def main() -> None:
    st.set_page_config(page_title="Holdout", layout="wide")
    with st.sidebar:
        st.markdown(f"### Holdout [{__version__}]({text.TAG_URL})")
        st.caption("A self-auditing data-science agent for tabular CSVs.")
        page = st.radio("Page", PAGES, key="page")
        st.caption(f"[Source]({text.REPO_URL}) · MIT licence")
    {"Run": run_page, "Benchmark": benchmark_page, "About": about_page}[page]()


# ----------------------------------------------------------------------------- run


def _workdir() -> Path:
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="holdout-app-")
    return Path(st.session_state["workdir"])


@st.cache_data(show_spinner=False)
def _demo_frame(label: str) -> tuple[pd.DataFrame, str, str]:
    demo = next(d for d in core.demos() if d.label == label)
    return demo.build()


def _data() -> tuple[pd.DataFrame, str, str] | None:
    source = st.radio("Data", ["Demo dataset", "Upload a CSV"], horizontal=True, key="source")
    if source == "Demo dataset":
        options = core.demos()
        label = st.selectbox("Demo", [d.label for d in options], key="demo")
        st.caption(next(d.description for d in options if d.label == label))
        return _demo_frame(label)
    upload = st.file_uploader("CSV file", type=["csv"], key="upload")
    if upload is None:
        return None
    frame = pd.read_csv(upload)
    if len(frame) > core.MAX_UPLOAD_ROWS:
        st.error(f"This app takes at most {core.MAX_UPLOAD_ROWS:,} rows; use the CLI for more.")
        return None
    return frame, str(frame.columns[-1]), ""


def run_page() -> None:
    st.title("Run Holdout on a CSV")
    st.markdown(
        "Holdout seals part of your data first, proposes a model, audits it for leakage and "
        "misleading validation, and opens the sealed rows once, only if nothing critical is "
        "found. **Offline by default:** no API calls, a default plan and template prose."
    )
    loaded = _data()
    if loaded is None:
        return
    frame, default_target, default_goal = loaded
    st.caption(f"{len(frame):,} rows x {frame.shape[1]} columns")
    st.dataframe(frame.head(5), hide_index=True)
    columns = [str(c) for c in frame.columns]
    target = st.selectbox(
        "Target column", columns, index=columns.index(default_target), key="target"
    )
    goal = st.text_input("Goal (optional)", value=default_goal, key="goal")
    entities = st.selectbox(
        "Will predictions be for new entities (customers, patients) or ones already in the data?",
        ["unanswered (treated as new)", "new", "known"],
        key="entities",
    )
    engine = st.radio(
        "Planner", ["Offline (no API calls)", "Gemini (your API key)"], key="engine",
        horizontal=True,
    )  # fmt: skip
    api_key: str | None = None
    if engine.startswith("Gemini"):
        allowed, note = core.gemini_allowed(st.session_state.get("gemini_runs", 0))
        st.info(note)
        st.caption(
            "Your key stays in this browser session only: it is passed to the Gemini client "
            "and never stored, logged or written to the run's files. Only the schema, "
            "summary statistics and up to 5 development rows are sent, never the full data. "
            "Under Gemini's free tier Google may use inputs to improve its products."
        )
        key = st.text_input("Gemini API key", type="password", key="gemini_key")
        if not allowed:
            return
        if key and not core.looks_like_key(key):
            st.error("That does not look like an API key.")
            return
        api_key = key or None
        if api_key is None:
            st.warning("Enter a key, or switch to Offline.")
            return
    if st.button("Run", type="primary", key="run"):
        _run(frame, target, goal, entities, api_key)
    outcome = st.session_state.get("outcome")
    if outcome is not None:
        show_outcome(outcome)


def _run(frame: pd.DataFrame, target: str, goal: str, entities: str, api_key: str | None) -> None:
    n = st.session_state.get("n_runs", 0) + 1
    st.session_state["n_runs"] = n
    if api_key is not None:
        st.session_state["gemini_runs"] = st.session_state.get("gemini_runs", 0) + 1
    lines: list[str] = []
    with st.status("Running...", expanded=False) as status:
        log = st.empty()

        def on_line(line: str) -> None:
            lines.append(line if api_key is None else line.replace(api_key, "***"))
            log.code("\n".join(lines[-12:]))

        try:
            outcome = core.run_holdout(
                frame, target, goal, _workdir() / f"run_{n}",
                entities=None if entities.startswith("unanswered") else entities,
                api_key=api_key, on_line=on_line,
            )  # fmt: skip
        except (HoldoutError, ValueError) as exc:
            message = str(exc) if api_key is None else str(exc).replace(api_key, "***")
            status.update(label="Run failed", state="error")
            st.error(message)
            return
        status.update(label="Done", state="complete")
    st.session_state["outcome"] = outcome


def show_outcome(outcome: AgentOutcome) -> None:
    runs = [outcome] + ([outcome.followup] if outcome.followup is not None else [])
    for i, run in enumerate(runs):
        st.divider()
        title = "Follow-up run with the corrected split" if i else "Result"
        st.header(title)
        if run.manifest is None:
            st.warning(f"The run did not complete ({run.status}): {run.stop_reason}")
            continue
        show_manifest(run.manifest, run.run_dir)


def show_manifest(m: RunManifest, run_dir: Path) -> None:
    baseline, selected = m.plans[0].cv, m.plans[-1].cv
    holdout_text = "not opened" if isinstance(m.holdout, str) else f"{m.holdout.score:.4f}"
    cols = st.columns(3)
    cols[0].metric(f"Validation ({m.metric.value})", f"{selected.mean:.4f}")
    cols[1].metric("Dummy baseline", f"{baseline.mean:.4f}")
    cols[2].metric("Sealed holdout", holdout_text)
    if m.holdout_invalid:
        st.error(
            "**Holdout invalid.** The sealed rows share rows, entities or time with the "
            "development data, so they can never give an honest score. A new run with the "
            "corrected split is needed."
        )
    elif m.holdout == NOT_UNSEALED:
        st.warning("The holdout stayed sealed: " + "; ".join(m.unseal.blocked_by or ["-"]))
    else:
        st.success("No critical problem remained, so the holdout was opened once and scored.")
    agent = m.agent
    if agent is not None and agent.findings:
        st.subheader("Problems found before planning")
        show_findings(agent.findings)
    st.subheader("Audit of the final plan")
    show_flags(m.audit.flags)
    if agent is not None and agent.report is not None:
        with st.expander("Report", expanded=False):
            st.markdown(agent.report.summary)
            for caveat in agent.report.caveats:
                st.markdown(f"- {caveat}")
        if agent.decisions:
            with st.expander("Decisions (autopilot policy, logged)"):
                st.table(pd.DataFrame([
                    {"question": d.kind, "accepted": d.accepted, "reason": d.reason}
                    for d in agent.decisions
                ]))  # fmt: skip
    _downloads(run_dir)


def _severity_block(
    severity: Severity, rows: list[dict[str, Any]], *, findings: bool = False
) -> None:
    if severity is Severity.CRITICAL:
        st.markdown(f"#### :red[Critical ({len(rows)})]")
        explained = text.CRITICAL_FINDING_EXPLAINED if findings else text.CRITICAL_EXPLAINED
        st.error(explained, icon="⛔")
    elif severity is Severity.WARNING:
        st.markdown(f"#### :orange[Warnings ({len(rows)})]")
        st.warning(text.WARNING_EXPLAINED, icon="⚠️")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def show_findings(findings: list[FindingStatus]) -> None:
    for severity in (Severity.CRITICAL, Severity.WARNING):
        rows = [
            {
                "finding": f.finding_id,
                "status": "resolved" if f.resolved else "OPEN",
                "check": f.check.value,
                "columns": ", ".join(f.columns) or "-",
                "problem": f.problem,
                "outcome": f.outcome,
            }
            for f in findings
            if f.severity is severity
        ]
        if rows:
            _severity_block(severity, rows, findings=True)


def show_flags(flags: list[AuditFlag]) -> None:
    groups = core.by_severity(flags)
    if not groups[Severity.CRITICAL] and not groups[Severity.WARNING]:
        st.success("No critical findings and no warnings.")
    for severity in (Severity.CRITICAL, Severity.WARNING):
        rows = [
            {"check": f.check.value, "columns": ", ".join(f.columns) or "-",
             "message": f.message, "fix": fix_text(f.fix)}
            for f in groups[severity]
        ]  # fmt: skip
        if rows:
            _severity_block(severity, rows)
    if groups[Severity.INFO]:
        with st.expander(f"Info ({len(groups[Severity.INFO])})"):
            st.caption(text.INFO_EXPLAINED)
            for f in groups[Severity.INFO]:
                st.markdown(f"- **{f.check.value}**: {f.message}")


def _downloads(run_dir: Path) -> None:
    cols = st.columns(3)
    for col, name, mime in zip(
        cols,
        ("report.md", "run_manifest.json", "pipeline.py"),
        ("text/markdown", "application/json", "text/x-python"),
        strict=True,
    ):
        path = run_dir / name
        if path.exists():
            col.download_button(
                name, path.read_bytes(), file_name=name, mime=mime, key=f"dl-{run_dir}-{name}"
            )


# ----------------------------------------------------------------------------- benchmark


def _doc(name: str) -> str | None:
    path = DOCS / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def benchmark_page() -> None:
    st.title("Benchmark")
    st.header(text.PREREGISTERED_LABEL)
    st.markdown("\n".join(text.PREREGISTERED_SUMMARY))
    results = _doc("results.md")
    if results is None:
        st.info(f"See docs/results.md in the [repository]({text.REPO_URL}).")
    else:
        for suite, titles in (
            ("final_v1", ("Detection recall", "False alarms", "Pairwise detection",
                          "Cost and answer quality")),
            ("real_world_v1", ("Real-world issues", "False alarms", "Cost and answer quality")),
        ):  # fmt: skip
            st.subheader(suite)
            for section in core.markdown_sections(results, suite, titles):
                st.markdown(section)
    st.divider()
    st.header(text.POSTHOC_LABEL)
    st.warning(
        "Not part of the pre-registered evaluation. Every v1.1 fix came from errors on the "
        "benchmark itself, so final_v1 and real_world_v1 are optimistic; dev_v1 is the "
        "primary comparison. Only the offline audits_only system was re-run."
    )
    st.markdown("\n".join(text.POSTHOC_SUMMARY))
    st.markdown("**Trade-offs**")
    st.markdown("\n".join(f"- {t}" for t in text.TRADEOFFS))
    posthoc = _doc("results_v11.md")
    if posthoc is not None:
        with st.expander("Full post-hoc tables (docs/results_v11.md)"):
            st.markdown(posthoc.split("\n", 1)[1])


# ----------------------------------------------------------------------------- about


DOC_URL = f"{text.REPO_URL}/blob/v{__version__}/docs"


def about_page() -> None:
    st.title("About Holdout")
    st.markdown(
        f"Version **{__version__}** ([tag v{__version__}]({text.TAG_URL})). MIT licence.\n\n"
        "Holdout is a data-science agent whose correctness is enforced in code. The holdout "
        "rows are split off, written to disk and hashed before anything computes a "
        "feature-target statistic. Plans are typed and validated; deterministic code "
        "compiles them into leakage-safe pipelines and runs twelve audit checks. A critical "
        "finding blocks the single, logged holdout evaluation.\n\n"
        f"- Design and guarantees: [docs/design.md]({DOC_URL}/design.md)\n"
        f"- The checks and their blind spots: [docs/audits.md]({DOC_URL}/audits.md)\n"
        f"- Pre-registered results: [docs/results.md]({DOC_URL}/results.md)\n"
        f"- Post-hoc v1.1: [docs/results_v11.md]({DOC_URL}/results_v11.md)\n\n"
        f"Caps for Gemini runs in this app: Rs {llm_config.RUN_CAP_INR:g} per run and "
        f"{core.MAX_GEMINI_RUNS_PER_SESSION} runs per browser session; the server's ledger "
        "also enforces the project cap."
    )
