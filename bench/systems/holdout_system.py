"""Holdout as a benchmark system: ``holdout`` (the agent), ``audits_only`` (the offline,
deterministic path) and ``llm_judge`` (the agent with its audit layer replaced by Gemini).

The unit's development rows are written as the run's CSV; the agent runs in autopilot with
the pinned model, the unit's seed, goal and entity scope. Holdout seals part of those rows
as its own internal holdout; the benchmark's holdout is separate and never seen.

Reported issues are every flag of every audit round and of the final audit, of the first
run and of the follow-up run with a corrected split if autopilot started one. The final
plan is the last run's selected plan; it is refit on all development rows the system
received (the same data every other system could use) and handed to the harness.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.systems.base import (
    FinalPipeline,
    NodeQuality,
    ReportedIssue,
    SystemContext,
    SystemName,
    SystemOutput,
)
from holdout.agent.driver import (
    PAUSED_STATUSES,
    AgentOutcome,
    ClientFactory,
    Customize,
    resume_agent,
    run_agent,
)
from holdout.agent.graph import AgentOptions
from holdout.agent.report import full_text
from holdout.agent.schemas import NodeUsage
from holdout.audit.types import Category, Severity
from holdout.data.profile import profile_schema
from holdout.export.manifest import RunManifest
from holdout.plan.compiler import fit_final


class HoldoutSystem:
    uses_llm: bool

    def __init__(
        self,
        name: SystemName,
        *,
        client_factory: ClientFactory | None = None,
        customize: Callable[[], Customize] | None = None,
    ) -> None:
        if name not in ("holdout", "audits_only", "llm_judge"):
            raise ValueError(f"not a Holdout-family system: {name}")
        self.name: SystemName = name
        self.uses_llm = name != "audits_only"
        self.client_factory = client_factory
        self.customize = customize

    # ------------------------------------------------------------------ running

    def run(self, ctx: SystemContext) -> SystemOutput:
        csv = ctx.workdir / "dev.csv"
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        ctx.dev.to_csv(csv, index=False)
        options = AgentOptions(
            csv=str(csv),
            target=ctx.target,
            out_dir=str(ctx.workdir / "runs"),
            goal=ctx.goal,
            task=ctx.task.value,
            seed=ctx.seed,
            entities=ctx.entity_scope.value if ctx.entity_scope is not None else None,
            mode="autopilot",
            offline=not self.uses_llm,
            use_cache=False,
            model=None if not self.uses_llm else ctx.model,
            plan_overrides=dict(ctx.plan_overrides) if not self.uses_llm else {},
            service_tier=ctx.service_tier if self.uses_llm else None,
        )
        outcome = run_agent(
            options,
            client_factory=self.client_factory,
            progress=ctx.progress,
            customize=self.customize() if self.customize is not None else None,
        )
        return self._collect(ctx, outcome)

    def resume(self, ctx: SystemContext) -> SystemOutput:
        """Resume the paused run (first, or the follow-up) through its checkpoint."""
        state = _read_state(ctx.workdir)
        paused = Path(state["paused_run_dir"]) if state.get("paused_run_dir") else None
        if paused is None:
            return self.run(ctx)
        outcome = resume_agent(
            paused,
            client_factory=self.client_factory,
            progress=ctx.progress,
            customize=self.customize() if self.customize is not None else None,
        )
        first = state.get("first_run_dir")
        if first and Path(first) != paused:  # the follow-up was paused: re-attach the first
            manifest = _manifest(Path(first))
            if manifest is not None:
                outcome = AgentOutcome(
                    run_id=Path(first).name,
                    run_dir=Path(first),
                    status="completed",
                    stop_reason=None,
                    pending=None,
                    manifest=manifest,
                    followup=outcome,
                )
        return self._collect(ctx, outcome)

    # ------------------------------------------------------------------ results

    def _collect(self, ctx: SystemContext, outcome: AgentOutcome) -> SystemOutput:
        runs = [outcome] + ([outcome.followup] if outcome.followup is not None else [])
        paused = next((r for r in runs if r.status in PAUSED_STATUSES), None)
        _write_state(
            ctx.workdir,
            {
                "first_run_dir": str(outcome.run_dir),
                "paused_run_dir": str(paused.run_dir) if paused is not None else None,
            },
        )
        if paused is not None:
            return SystemOutput(
                status="paused",
                error=paused.stop_reason,
                extra={"paused_status": paused.status, "run_dir": str(paused.run_dir)},
            )
        manifests = [r.manifest for r in runs if r.manifest is not None]
        if not manifests:
            return SystemOutput(status="stopped", error=outcome.stop_reason)
        final = manifests[-1]
        plan = final.plans[-1].plan
        spec = final.split.spec
        estimator = fit_final(
            plan, ctx.dev, ctx.target, spec, profile_schema(ctx.dev, ctx.target, task=ctx.task)
        )
        agent = final.agent
        report = agent.report if agent is not None else None
        return SystemOutput(
            status="completed",
            issues=_issues(manifests),
            pipeline=FinalPipeline(
                dropped_columns=sorted(plan.dropped),
                metric=plan.metric.value,
                split=_split_label(spec.strategy.value, spec.group_col or spec.time_col),
                deduplicated=False,  # Holdout reports duplicates; it never removes rows itself
                model=plan.model.name,
            ),
            estimator=estimator,
            validation_metric=plan.metric.value,
            validation_score=final.validation_score,
            report_text=full_text(report) if report is not None else None,
            report_a11_flags=sum(
                f.check.value == "A11" and f.severity is not Severity.INFO
                for f in final.audit.flags
            ),
            cost_inr=sum(m.agent.total_inr for m in manifests if m.agent is not None),
            cost_cache_adjusted_inr=sum(u.inr_cache_adjusted for u in _nodes(manifests)),
            cached_tokens=sum(u.cached_tokens for u in _nodes(manifests)),
            prompt_tokens=sum(u.prompt_tokens for u in _nodes(manifests)),
            tiers_served=_merge_tiers(manifests),
            api_seconds=sum(m.agent.api_seconds for m in manifests if m.agent is not None),
            timeouts=_eda_timeouts(manifests),
            calls=sum(
                u.calls for m in manifests if m.agent for u in m.agent.usage_by_node.values()
            ),
            retries=sum(m.agent.retries for m in manifests if m.agent is not None),
            cache_hits=sum(m.agent.cache_hits for m in manifests if m.agent is not None),
            first_attempt=_first_attempt(manifests),
            repairs=sum(m.agent.quality.repairs for m in manifests if m.agent and m.agent.quality),
            substitutions=sum(len(m.agent.substitutions) for m in manifests if m.agent),
            extra={
                "inr_by_node": _inr_by_node(manifests),
                "run_dirs": [str(r.run_dir) for r in runs],
                "stop_reason": agent.stop_reason if agent is not None else None,
                "followup": outcome.followup is not None,
                "fix_rounds": max(0, len(agent.rounds) - 1) if agent is not None else 0,
            },
        )


def _issues(manifests: list[RunManifest]) -> list[ReportedIssue]:
    """Every flag of every round and of the final audit, de-duplicated."""
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    issues: list[ReportedIssue] = []

    def add(check: str, severity: str, category: str, columns: list[str], message: str) -> None:
        key = (check, severity, category, tuple(columns))
        if key not in seen:
            seen.add(key)
            issues.append(
                ReportedIssue(
                    category=Category(category),
                    severity=Severity(severity),
                    columns=columns,
                    source=check,
                    message=message,
                )
            )

    for manifest in manifests:
        if manifest.agent is not None:
            for round_ in manifest.agent.rounds:
                for text in round_.flags:  # "check severity category col1,col2"
                    parts = text.split(" ", 3)
                    columns = parts[3].split(",") if len(parts) > 3 and parts[3] else []
                    add(parts[0], parts[1], parts[2], columns, "")
        for flag in manifest.audit.flags:
            add(
                flag.check.value,
                flag.severity.value,
                flag.category.value,
                flag.columns,
                flag.message,
            )
    return issues


def _eda_timeouts(manifests: list[RunManifest]) -> dict[str, int]:
    """EDA sandbox snippets killed at their time limit, all runs."""
    n = sum(r.status == "timeout" for m in manifests if m.agent is not None for r in m.agent.eda)
    return {"eda_step": n} if n else {}


def _nodes(manifests: list[RunManifest]) -> list[NodeUsage]:
    return [u for m in manifests if m.agent for u in m.agent.usage_by_node.values()]


def _merge_tiers(manifests: list[RunManifest]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for usage in _nodes(manifests):
        for tier, n in usage.tiers.items():
            merged[tier] = merged.get(tier, 0) + n
    return merged


def _inr_by_node(manifests: list[RunManifest]) -> dict[str, float]:
    """Rupees per graph node (plan, audit or judge, explain_and_fix, report), all runs."""
    totals: dict[str, float] = {}
    for manifest in manifests:
        if manifest.agent is not None:
            for node, usage in manifest.agent.usage_by_node.items():
                totals[node] = totals.get(node, 0.0) + usage.inr
    return {k: round(v, 4) for k, v in sorted(totals.items())}


def _first_attempt(manifests: list[RunManifest]) -> dict[str, NodeQuality]:
    totals: dict[str, list[int]] = {}
    for manifest in manifests:
        if manifest.agent is None or manifest.agent.quality is None:
            continue
        for node, first in manifest.agent.quality.first_attempt.items():
            acc = totals.setdefault(node, [0, 0])
            acc[0] += first.first_attempts
            acc[1] += first.valid
    return {n: NodeQuality(first_attempts=a, valid=v) for n, (a, v) in totals.items()}


def _split_label(strategy: str, column: str | None) -> str:
    if strategy in {"group", "time"}:
        return f"{strategy}:{column}" if column else f"{strategy}:<row>"
    return strategy


def _manifest(run_dir: Path) -> RunManifest | None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status", "completed") != "completed":
        return None
    return RunManifest.model_validate(raw)


def _write_state(workdir: Path, state: dict[str, Any]) -> None:
    (workdir / "system_state.json").write_text(json.dumps(state), encoding="utf-8")


def _read_state(workdir: Path) -> dict[str, Any]:
    path = workdir / "system_state.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return loaded
