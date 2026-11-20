"""``run_manifest.json`` and ``report.md``: the auditable record of a run.

The manifest answers "what exactly happened?" months later: which bytes were analysed,
how they were split, every seed, every plan tried with its CV result, whether and how the
holdout was opened, and the software environment. The report is a short human summary
rendered from the manifest only, so the two can never disagree.
"""

import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from holdout.agent.schemas import AgentRecord, Decision, LLMOutage, NodeUsage, RoundRecord
from holdout.audit.findings import FindingStatus
from holdout.audit.gate import UnsealDecision
from holdout.audit.render import fix_text
from holdout.audit.types import AuditReport
from holdout.core.schemas import (
    ColumnKind,
    CVResult,
    Direction,
    EntityScope,
    HoldoutResult,
    MetricName,
    Task,
)
from holdout.data.profile import DatasetProfile
from holdout.data.split import SplitProposal, SplitSpec
from holdout.plan.schema import ModelPlan, PlanIssue

NOT_UNSEALED: Literal["not unsealed"] = "not unsealed"

#: Distributions whose versions are recorded in every manifest.
TRACKED_PACKAGES = (
    "holdout",
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "lightgbm",
    "pyarrow",
    "pydantic",
    "typer",
    "rich",
)


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataRecord(_Record):
    path: str
    sha256: str
    n_rows: int
    n_cols: int
    n_dropped_missing_target: int


class ProfileSummary(_Record):
    task_hint: Task
    column_kinds: dict[str, ColumnKind]
    target_class_frequencies: dict[str, float] | None
    datetime_candidates: list[str]
    monotonic_candidates: list[str]
    entity_candidates: list[str]
    looks_time_ordered: bool

    @classmethod
    def from_profile(cls, profile: DatasetProfile) -> "ProfileSummary":
        return cls(
            task_hint=profile.task_hint,
            column_kinds={c.name: c.kind for c in profile.columns},
            target_class_frequencies=profile.target_class_frequencies,
            datetime_candidates=profile.datetime_candidates,
            monotonic_candidates=profile.monotonic_candidates,
            entity_candidates=profile.entity_candidates,
            looks_time_ordered=profile.looks_time_ordered,
        )


class SplitRecord(_Record):
    spec: SplitSpec
    source: Literal["proposed", "user"]
    reasons: list[str]
    warnings: list[str]
    alternatives: list[SplitSpec]
    n_dev_rows: int
    n_holdout_rows: int
    holdout_sha256: str
    reuse_warnings: list[str]

    @classmethod
    def build(
        cls,
        spec: SplitSpec,
        proposal: SplitProposal,
        *,
        n_dev_rows: int,
        n_holdout_rows: int,
        holdout_sha256: str,
        reuse_warnings: list[str],
    ) -> "SplitRecord":
        # Declaring the entity scope answers a question about the proposal; it does not
        # choose another split. Only a different split (strategy, column, size) is "user".
        same_split = spec == proposal.spec.model_copy(update={"entity_scope": spec.entity_scope})
        if same_split:
            reasons = list(proposal.reasons)
            if spec.entity_scope is not None and proposal.spec.entity_scope is None:
                reasons.append(f"entity scope declared: {spec.entity_scope.value}")
        else:
            reasons = ["split chosen by the user instead of the proposed one"]
        return cls(
            spec=spec,
            source="proposed" if same_split else "user",
            reasons=reasons,
            warnings=proposal.warnings,
            alternatives=proposal.alternatives,
            n_dev_rows=n_dev_rows,
            n_holdout_rows=n_holdout_rows,
            holdout_sha256=holdout_sha256,
            reuse_warnings=reuse_warnings,
        )


class PlanRecord(_Record):
    label: str
    plan: ModelPlan
    issues: list[PlanIssue]
    cv: CVResult


class EnvironmentRecord(_Record):
    python: str
    platform: str
    packages: dict[str, str | None]
    git_commit: str | None
    git_dirty: bool | None


class RunManifest(_Record):
    schema_version: Literal[3] = 3
    status: Literal["completed"] = "completed"
    run_id: str
    started_at: datetime
    finished_at: datetime
    data: DataRecord
    target: str
    task: Task
    goal_text: str
    entity_scope: EntityScope | None = Field(description="user declaration; None = unanswered")
    metric: MetricName
    direction: Direction
    profile: ProfileSummary
    split: SplitRecord
    run_seed: int
    derived_seeds: dict[str, int]
    plans: list[PlanRecord] = Field(description="baseline first, then the selected plan")
    selected_plan: str
    #: v1.1: development rows the selected plan removed as duplicates (None: no dedup).
    dev_rows_deduplicated: int | None = Field(
        default=None,
        description="development rows removed by the plan's deduplicate step; None: not used",
    )
    validation_score: float
    thresholds_hash: str = Field(description="sha256 of the audit thresholds file")
    thresholds_v11_hash: str | None = Field(
        default=None,
        description="sha256 of thresholds_v11.toml, the v1.1 rule additions; None: v1.0",
    )
    audit: AuditReport
    unseal: UnsealDecision
    holdout_invalid: bool
    caveats: list[str]
    holdout: HoldoutResult | Literal["not unsealed"]
    optimism_gap: float | None = Field(description="validation_score - holdout score")
    artifacts: list[str]
    environment: EnvironmentRecord
    agent: AgentRecord | None = Field(
        default=None, description="present for agent runs (holdout agent), absent for holdout run"
    )


PausedStatus = Literal["paused_llm_unavailable", "paused_budget_cap", "paused_billing"]


class PausedRunManifest(_Record):
    """Written to ``run_manifest.json`` when an agent run pauses on an LLM problem (service
    unavailable, spend cap, billing); replaced by a :class:`RunManifest` once the resumed run
    completes."""

    schema_version: Literal[3] = 3
    status: PausedStatus = "paused_llm_unavailable"
    run_id: str
    started_at: datetime
    paused_at: datetime
    target: str
    goal_text: str | None
    run_seed: int
    model: str
    model_overridden: bool
    outage: LLMOutage = Field(description="this pause")
    earlier_outages: list[LLMOutage]
    message: str
    resume_command: str
    completed_nodes: list[str] = Field(description="graph nodes checkpointed before the pause")
    usage_by_node: dict[str, NodeUsage] = Field(description="including the failed call")
    total_inr: float
    retries: int
    interrupted_calls: list[dict[str, Any]] = Field(
        description="calls of nodes that did not finish (spent, but not in the checkpoint); "
        "carried into the final manifest on resume"
    )
    decisions: list[Decision]
    rounds: list[RoundRecord]
    environment: EnvironmentRecord


def capture_environment() -> EnvironmentRecord:
    """Record interpreter, OS, package versions and (if available) the git commit."""
    commit, dirty = _git_state(Path(__file__).resolve().parent)
    return EnvironmentRecord(
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages={name: _version(name) for name in TRACKED_PACKAGES},
        git_commit=commit,
        git_dirty=dirty,
    )


def write_manifest(path: Path, manifest: RunManifest | PausedRunManifest) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(manifest.model_dump_json(indent=2) + "\n")
    return path


def render_report(manifest: RunManifest) -> str:
    """A short Markdown summary, rendered from the manifest alone."""
    baseline, selected = manifest.plans[0], manifest.plans[-1]
    holdout = manifest.holdout
    holdout_text = NOT_UNSEALED if isinstance(holdout, str) else f"{holdout.score:.4f}"
    gap_text = "n/a" if manifest.optimism_gap is None else f"{manifest.optimism_gap:+.4f}"
    split = manifest.split
    lines = [f"# Holdout run {manifest.run_id}", ""]
    unseal = manifest.unseal
    if unseal.accepted_risk is not None and unseal.allowed:
        lines += [
            "> **RISK ACCEPTED.** The holdout was unsealed despite critical audit flag(s): "
            f"{'; '.join(unseal.overridden_criticals) or 'none'}.",
            f"> Reason given: *{unseal.accepted_risk}*",
            "",
        ]
    if manifest.holdout_invalid:
        lines += [
            "> **HOLDOUT INVALID.** The sealed holdout shares rows, entities or time with the "
            "development data, so it can never be evaluated. Start a new run with the "
            "corrected split (see the audit fixes below).",
            "",
        ]
    for caveat in manifest.caveats:
        lines += [f"> **Caveat:** {caveat}", ""]
    scope = (
        "unanswered (treated as new)"
        if manifest.entity_scope is None
        else manifest.entity_scope.value
    )
    lines += [
        f"- **Goal:** {manifest.goal_text}",
        f"- **Data:** `{Path(manifest.data.path).name}`, {manifest.data.n_rows} rows x "
        f"{manifest.data.n_cols} columns (sha256 `{manifest.data.sha256[:12]}...`)",
        f"- **Target:** `{manifest.target}` ({manifest.task.value}); "
        f"metric **{manifest.metric.value}** ({manifest.direction.value} is better)",
        f"- **Split:** {split.spec.strategy.value} ({split.source}), "
        f"{split.n_dev_rows} development / {split.n_holdout_rows} sealed holdout rows"
        + (
            f"; {manifest.dev_rows_deduplicated} duplicate development rows removed before "
            "cross-validation (the holdout is untouched)"
            if manifest.dev_rows_deduplicated is not None
            else ""
        ),
        f"- **Entities:** {scope}",
        f"- **Run seed:** {manifest.run_seed}; thresholds sha256 "
        f"`{manifest.thresholds_hash[:12]}...`"
        + (
            f"; v1.1 additions sha256 `{manifest.thresholds_v11_hash[:12]}...`"
            if manifest.thresholds_v11_hash
            else ""
        ),
        "",
        "## Results",
        "",
        "| | score |",
        "|---|---|",
        f"| Dummy baseline (CV) | {baseline.cv.mean:.4f} +/- {baseline.cv.std:.4f} |",
        f"| {selected.label} (CV) | {selected.cv.mean:.4f} +/- {selected.cv.std:.4f} |",
        f"| Holdout (single evaluation) | {holdout_text} |",
        f"| Optimism gap (validation - holdout) | {gap_text} |",
        "",
    ]
    if manifest.direction is Direction.LOWER:
        lines += [
            "For a lower-is-better metric, a *negative* gap means validation was optimistic.",
            "",
        ]
    if manifest.agent is not None and manifest.agent.findings:
        lines += findings_section(manifest.agent.findings)
    lines += _audit_section(manifest)
    lines += ["## Split reasoning", ""] + [f"- {r}" for r in split.reasons]
    warnings = split.warnings + split.reuse_warnings
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    plan_warnings = [i for i in selected.issues if i.severity.value == "warning"]
    if plan_warnings:
        lines += [""] + [f"- plan: {i.message}" for i in plan_warnings]
    lines += ["", "## Plan rationale", ""]
    lines += [f"- **{k}:** {v}" for k, v in selected.plan.rationale.model_dump().items()]
    if manifest.agent is not None:
        lines += _agent_section(manifest.agent)
    lines += ["", "## Artifacts", ""] + [f"- `{a}`" for a in manifest.artifacts]
    return "\n".join(lines) + "\n"


def findings_section(findings: Sequence[FindingStatus]) -> list[str]:
    """v1.1: every problem found on the raw data before planning, and its outcome."""
    lines = [
        "## Problems found and how they were handled",
        "",
        "Found on the raw development rows before planning (v1.1). The planner's action is "
        "what it said it would do; the outcome is decided by code from the final plan, split "
        "and audit.",
        "",
        "| finding | check | severity | columns | problem | planner action | outcome |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in findings:
        columns = ", ".join(f"`{c}`" for c in f.columns) or "-"
        action = "-" if f.planner_action is None else f.planner_action.replace("_", " ")
        if f.planner_reason:
            action += f": {f.planner_reason}"
        lines.append(
            f"| {f.finding_id} | {f.check.value} | {f.severity.value} | {columns} | "
            f"{_cell(f.problem)} | {_cell(action)} | "
            f"{'**' if not f.resolved else ''}{_cell(f.outcome)}{'**' if not f.resolved else ''} |"
        )
    open_count = sum(not f.resolved for f in findings)
    lines += ["", f"{len(findings) - open_count} handled, {open_count} open.", ""]
    return lines


def _audit_section(manifest: RunManifest) -> list[str]:
    audit = manifest.audit
    counts = audit.counts
    status = "complete" if audit.complete else "INCOMPLETE (a check failed)"
    lines = [
        "## Audit",
        "",
        f"{status}; {counts['critical']} critical, {counts['warning']} warning, "
        f"{counts['info']} info{'; deep mode' if audit.deep_mode else ''}.",
        "",
    ]
    if audit.flags:
        lines += ["| check | severity | columns | message | fix |", "|---|---|---|---|---|"]
        for flag in audit.flags:
            columns = ", ".join(f"`{c}`" for c in flag.columns) or "-"
            lines.append(
                f"| {flag.check.value} | {flag.severity.value} | {columns} | "
                f"{_cell(flag.message)} | {_cell(fix_text(flag.fix))} |"
            )
        lines.append("")
    errors = [r for r in audit.checks_run if r.status == "error"]
    lines += [f"- {r.check.value} failed: {r.detail}" for r in errors]
    a11 = next((r for r in audit.checks_run if r.check.value == "A11"), None)
    if a11 is not None and a11.status != "error" and a11.detail:
        lines += [f"- Causal claims (A11): {a11.detail}", ""]
    unseal = manifest.unseal
    if unseal.requested:
        verdict = "allowed" if unseal.allowed else "refused"
        lines += [f"**Unseal {verdict}.**"] + [f"- {b}" for b in unseal.blocked_by] + [""]
    return lines


def _agent_section(agent: AgentRecord) -> list[str]:
    model = agent.model or "none (offline)"
    lines = [
        "",
        "## Agent",
        "",
        f"- **Mode:** {agent.mode}{' (offline: no API calls)' if agent.offline else ''}; "
        f"model `{model}` via {agent.llm_client}",
        f"- **Spend:** Rs {agent.total_inr:.2f} of a Rs {agent.run_cap_inr:.2f} run cap; "
        f"{agent.cache_hits} cached response(s); {agent.retries} transient-error retr"
        f"{'y' if agent.retries == 1 else 'ies'}",
        f"- **Fix rounds:** {max(0, len(agent.rounds) - 1)}; stop reason: {agent.stop_reason}",
        substitutions_line(agent),
    ]
    if agent.quality is not None:
        lines.append(f"- **Answer quality:** {quality_line(agent)}")
    if agent.model_overridden:
        lines.append(f"- **Model override:** `{model}` was chosen with --model for this run")
    for outage in agent.outages:
        if outage.kind == "budget_cap":
            why = "a spend cap refused the next call"
        else:
            status = "a timeout" if outage.status is None else f"HTTP {outage.status}"
            what = "billing / quota refusal" if outage.kind == "billing" else status
            why = f"{what} after {outage.retries} retries"
        lines.append(f"- **Paused** at {outage.at} in node `{outage.node}`: {why}; resumed later")
    if agent.previous_run_id:
        lines.append(f"- **Follows run** `{agent.previous_run_id}` (corrected split)")
    if agent.next_run_id:
        lines.append(f"- **Continued in run** `{agent.next_run_id}` (corrected split)")
    if agent.decisions:
        lines += ["", "| decision | mode | accepted | reason |", "|---|---|---|---|"]
        lines += [
            f"| {d.kind} | {d.mode} | {'yes' if d.accepted else 'no'} | {_cell(d.reason)} |"
            for d in agent.decisions
        ]
    if agent.explanations:
        lines += ["", "### Flag explanations", ""]
        lines += [f"- **{k}:** {_cell(v)}" for k, v in sorted(agent.explanations.items())]
    report = agent.report
    if report is not None:
        lines += ["", "### Report", "", report.summary, ""]
        for title, items in (
            ("Findings", report.findings),
            ("Caveats", report.caveats),
            ("Recommendations", report.recommendations),
        ):
            if items:
                lines += [f"**{title}**", ""] + [f"- {item}" for item in items] + [""]
    return lines


def substitutions_line(agent: AgentRecord) -> str:
    """One definition: a substitution replaces a model answer with code's (listed in
    ``substitutions``, counted here, ``code_substitution`` rounds). Offline runs have no
    model answers; their code-applied fix rounds (``code_fix``) are counted separately."""
    code_fixes = sum(r.plan_source == "code_fix" for r in agent.rounds)
    if agent.offline:
        fixes = (
            f"code applied the audit's fixes in {code_fixes} round(s)"
            if code_fixes
            else "the audit asked for no plan fix"
        )
        return f"- **Substitutions:** none (offline run: no model answers to replace); {fixes}"
    return (
        f"- **Substitutions:** {len(agent.substitutions)} (model answers replaced by code "
        "because the model's integrated plan or report missed a fix; not credited to the "
        "model)"
    )


def quality_line(agent: AgentRecord) -> str:
    """'plan 1/1, explain_and_fix 0/0, report 1/1 valid at first attempt; 0 repairs; ...'"""
    if agent.quality is None:
        return "n/a (offline)"
    q = agent.quality
    firsts = ", ".join(
        f"{node} {f.valid}/{f.first_attempts}" for node, f in q.first_attempt.items()
    )
    return (
        f"{firsts} valid at first attempt; {q.repairs} repair(s); {q.substitutions} substitution(s)"
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_report(path: Path, manifest: RunManifest) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_report(manifest))
    return path


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git_state(directory: Path) -> tuple[str | None, bool | None]:
    """Return (commit, dirty) for the repository containing ``directory``, if any."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(status.strip())
