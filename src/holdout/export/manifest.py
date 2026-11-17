"""``run_manifest.json`` and ``report.md``: the auditable record of a run.

The manifest answers "what exactly happened?" months later: which bytes were analysed,
how they were split, every seed, every plan tried with its CV result, whether and how the
holdout was opened, and the software environment. The report is a short human summary
rendered from the manifest only, so the two can never disagree.
"""

import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
        user_chosen = spec != proposal.spec
        return cls(
            spec=spec,
            source="user" if user_chosen else "proposed",
            reasons=["split chosen by the user (--split)"] if user_chosen else proposal.reasons,
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
    schema_version: Literal[2] = 2
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
    validation_score: float
    thresholds_hash: str = Field(description="sha256 of the audit thresholds file")
    audit: AuditReport
    unseal: UnsealDecision
    holdout_invalid: bool
    caveats: list[str]
    holdout: HoldoutResult | Literal["not unsealed"]
    optimism_gap: float | None = Field(description="validation_score - holdout score")
    artifacts: list[str]
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


def write_manifest(path: Path, manifest: RunManifest) -> Path:
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
        f"{split.n_dev_rows} development / {split.n_holdout_rows} sealed holdout rows",
        f"- **Entities:** {scope}",
        f"- **Run seed:** {manifest.run_seed}; thresholds sha256 "
        f"`{manifest.thresholds_hash[:12]}...`",
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
    lines += ["", "## Artifacts", ""] + [f"- `{a}`" for a in manifest.artifacts]
    return "\n".join(lines) + "\n"


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
    unseal = manifest.unseal
    if unseal.requested:
        verdict = "allowed" if unseal.allowed else "refused"
        lines += [f"**Unseal {verdict}.**"] + [f"- {b}" for b in unseal.blocked_by] + [""]
    return lines


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
