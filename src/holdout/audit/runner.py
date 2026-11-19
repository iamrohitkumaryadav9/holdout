"""Run the twelve checks in a fixed order and assemble the audit report.

Order: A10 first (it decides deep mode), then the cheap plan/data checks (A7, A9, A8), then
A1 (its measurements feed A2 and A3), A2, A3, the A1 flags (upgraded to critical where A2
or A3 corroborate), the split-level checks (A4, A5, A6), A12, and A11 (skipped until a
generated report exists, Week 3).

One failing check never kills the run: its error is recorded, the report is marked
incomplete, and an incomplete audit blocks unsealing. A skip is recorded but is not a
failure.
"""

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from holdout.audit.checks import (
    CheckOutcome,
    a01_single_feature,
    a02_post_outcome,
    a03_id_like,
    a04_duplicates,
    a05_group,
    a06_time,
    a07_metric,
    a08_formats,
    a09_encoding,
    a10_too_good,
    a12_stability,
)
from holdout.audit.checks.a01_single_feature import SingleFeatureResult
from holdout.audit.context import AuditContext, OverlapSource
from holdout.audit.hints import HintProvider, LexiconHintProvider
from holdout.audit.thresholds import Thresholds, ThresholdsV11, load_thresholds
from holdout.audit.types import AuditFlag, AuditReport, CheckId, CheckRun
from holdout.core.schemas import CVResult
from holdout.core.state import RunState
from holdout.data.profile import DatasetProfile
from holdout.data.split import SplitSpec
from holdout.plan.compiler import plan_rows
from holdout.plan.schema import ModelPlan

_ORDER = list(CheckId)

A11_SKIP_DETAIL = "skipped: no report text (causal-claim check runs on generated reports, Week 3)"


class _Recorder:
    def __init__(self, only: frozenset[CheckId] | None = None) -> None:
        self.flags: list[AuditFlag] = []
        self.runs: list[CheckRun] = []
        self.only = only

    def run(self, check: CheckId, fn: Callable[[], CheckOutcome]) -> CheckOutcome | None:
        if self.only is not None and check not in self.only:
            return None
        started = time.perf_counter()
        try:
            outcome = fn()
        except Exception as exc:  # isolate: one broken check must not kill the run
            self.runs.append(
                CheckRun(
                    check=check,
                    status="error",
                    seconds=time.perf_counter() - started,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            return None
        self.add(check, outcome, time.perf_counter() - started)
        return outcome

    def add(self, check: CheckId, outcome: CheckOutcome, seconds: float) -> None:
        self.flags.extend(outcome.flags)
        self.runs.append(
            CheckRun(
                check=check,
                status="skipped" if outcome.skipped else "ran",
                seconds=seconds,
                sampling=outcome.sampling,
                detail=outcome.detail,
            )
        )


def run_audits(
    run_state: RunState | None,
    dev_df: pd.DataFrame,
    plan: ModelPlan,
    cv_result: CVResult,
    *,
    baseline: CVResult,
    profile: DatasetProfile,
    spec: SplitSpec,
    target: str,
    thresholds: tuple[Thresholds, str] | None = None,
    overlap_source: OverlapSource | None = None,
    hint_provider: HintProvider | None = None,
    goal_text: str = "",
    thresholds_v11: tuple[ThresholdsV11, str] | None = None,
    checks: frozenset[CheckId] | None = None,
) -> AuditReport:
    """Audit a cross-validated plan on development rows; write ``audit.json`` if a run
    state is given.

    ``thresholds_v11`` (v1.1) layers the post-benchmark rule additions on the frozen
    thresholds; None keeps the v1.0 behaviour. ``checks`` restricts the checks run (the v1.1
    raw-data findings); None runs all of them.
    """
    loaded, thresholds_hash = thresholds if thresholds is not None else load_thresholds()
    v11, v11_hash = thresholds_v11 if thresholds_v11 is not None else (None, None)
    ctx = AuditContext(
        # v1.1: a plan that deduplicates is audited on the rows it is evaluated on.
        dev=plan_rows(plan, dev_df, target),
        target=target,
        profile=profile,
        spec=spec,
        plan=plan,
        cv_result=cv_result,
        baseline=baseline,
        thresholds=loaded,
        overlap_source=overlap_source,
        hint_provider=hint_provider if hint_provider is not None else LexiconHintProvider(),
        goal_text=goal_text,
        thresholds_v11=v11,
    )
    recorder = _Recorder(checks)
    a10 = recorder.run(CheckId.A10, lambda: a10_too_good.run(ctx))
    if a10 is not None and a10.flags:
        ctx = ctx.with_deep_mode()
    audit = ctx  # final context for the closures below
    recorder.run(CheckId.A7, lambda: a07_metric.run(audit))
    recorder.run(CheckId.A9, lambda: a09_encoding.run(audit))
    recorder.run(CheckId.A8, lambda: a08_formats.run(audit))

    single: SingleFeatureResult | None = None
    single_started = time.perf_counter()
    try:
        single = a01_single_feature.measure(audit)
    except Exception as exc:
        recorder.runs.append(
            CheckRun(
                check=CheckId.A1,
                status="error",
                seconds=time.perf_counter() - single_started,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
    single_seconds = time.perf_counter() - single_started

    hinted: set[str] = set()
    id_like: set[str] = set()

    def a2() -> CheckOutcome:
        outcome, columns = a02_post_outcome.run(audit, single)
        hinted.update(columns)
        return outcome

    def a3() -> CheckOutcome:
        outcome, columns = a03_id_like.run(audit, single)
        id_like.update(columns)
        return outcome

    recorder.run(CheckId.A2, a2)
    recorder.run(CheckId.A3, a3)
    if single is not None:
        measured = single
        corroboration = {
            column: [
                c
                for c, hit in ((CheckId.A2, column in hinted), (CheckId.A3, column in id_like))
                if hit
            ]
            for column in measured.scores
        }
        flagging_started = time.perf_counter()
        a1_outcome = recorder.run(
            CheckId.A1,
            lambda: CheckOutcome(
                flags=a01_single_feature.flags(audit, measured, corroboration),
                sampling=measured.sampling,
                detail=f"{len(measured.scores)} feature(s) scored with {measured.splitter}",
            ),
        )
        if a1_outcome is not None:
            # Attribute the measurement time to A1 as well.
            last = recorder.runs[-1]
            recorder.runs[-1] = last.model_copy(
                update={"seconds": single_seconds + (time.perf_counter() - flagging_started)}
            )

    recorder.run(CheckId.A4, lambda: a04_duplicates.run(audit))
    recorder.run(CheckId.A5, lambda: a05_group.run(audit))
    recorder.run(CheckId.A6, lambda: a06_time.run(audit))
    recorder.run(CheckId.A12, lambda: a12_stability.run(audit))
    recorder.add(CheckId.A11, CheckOutcome(skipped=True, detail=A11_SKIP_DETAIL), 0.0)

    report = AuditReport(
        flags=sorted(recorder.flags, key=lambda f: (_ORDER.index(f.check), -f.severity.rank)),
        checks_run=sorted(recorder.runs, key=lambda r: _ORDER.index(r.check)),
        thresholds_hash=thresholds_hash,
        thresholds_v11_hash=v11_hash,
        deep_mode=audit.deep_mode,
        entity_scope=spec.entity_scope,
    )
    if run_state is not None:
        write_audit(run_state.audit_path, report)
    return report


def write_audit(path: Path, report: AuditReport) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
    return path
