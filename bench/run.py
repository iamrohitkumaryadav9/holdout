"""Run the benchmark: ``python -m bench.run --suite final_v1 --systems holdout react ...``

A run unit is (suite, task, system, seed, variant). Results are appended to
``results/<suite>/<system>.jsonl``, one :class:`BenchResult` per line. The runner is
resumable: a unit whose latest result is completed, stopped or failed is skipped; a paused
unit (Gemini unavailable, billing, project cap) is resumed through its checkpoint on the
next pass; any exception is recorded as a failed unit and the benchmark continues.

The seal is per (suite, task, seed, variant), shared by every system, so all systems face
the same holdout. Every planted frame is checked against the frozen suite's sha256 before
it is used.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from bench.data.registry import Pool
from bench.harness import (
    BenchSeal,
    PartitionRule,
    SealError,
    benchmark_metric,
    partition,
    planted_rule,
    real_world_rule,
)
from bench.suites.planted import (
    DEV_V1_PATH,
    FINAL_V1_PATH,
    TaskSpec,
    build_task,
    frame_sha256,
)
from bench.suites.real_world import case_frame, load_real_world
from bench.systems.base import (
    BenchResult,
    System,
    SystemContext,
    SystemName,
    SystemOutput,
    Variant,
)
from holdout.core.errors import RunPausingError
from holdout.core.schemas import EntityScope, Task
from holdout.llm import config as llm_config
from holdout.llm.budget import SpendLedger
from holdout.llm.progress import LineProgress, NullProgress, Progress

RESULTS_DIR = Path("results")
SUITE_FILES = {"final_v1": FINAL_V1_PATH, "dev_v1": DEV_V1_PATH}
REAL_WORLD_SUITE = "real_world_v1"
REAL_WORLD_SEEDS = (101, 202)
TERMINAL = frozenset({"completed", "stopped", "failed"})


@dataclass(frozen=True)
class Unit:
    suite: str
    task_id: str
    system: SystemName
    seed: int
    variant: Variant

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        return (self.suite, self.task_id, self.system, self.seed, self.variant)

    @property
    def slug(self) -> str:
        safe = self.task_id.replace("@", "_at_").replace(":", "_")
        return f"{safe}__{self.variant}__s{self.seed}"


@dataclass(frozen=True)
class PreparedUnit:
    frame: pd.DataFrame
    target: str
    task: Task
    goal: str
    entity_scope: EntityScope | None
    plan_overrides: dict[str, Any]
    rule: PartitionRule
    #: False for real cases scored on detection and remediation only.
    holdout_scored: bool = True


# ----------------------------------------------------------------------------- units


def suite_tasks(suite: str) -> list[dict[str, Any]]:
    doc = json.loads(SUITE_FILES[suite].read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = doc["tasks"]
    return tasks


def units_for(
    suite: str,
    systems: Sequence[SystemName],
    *,
    seeds: Sequence[int] | None = None,
    tasks: Sequence[str] | None = None,
) -> list[Unit]:
    if suite == REAL_WORLD_SUITE:
        ids = [c.id for c in load_real_world()]
        variants: tuple[Variant, ...] = ("original", "anonymised")
        chosen_seeds = tuple(seeds or REAL_WORLD_SEEDS)
    else:
        doc = json.loads(SUITE_FILES[suite].read_text(encoding="utf-8"))
        ids = [t["id"] for t in doc["tasks"]]
        variants = ("planted",)
        chosen_seeds = tuple(seeds or doc["seeds"])
    if tasks is not None:
        unknown = sorted(set(tasks) - set(ids))
        if unknown:
            raise ValueError(f"unknown tasks for {suite}: {unknown}")
        ids = [i for i in ids if i in set(tasks)]
    return [
        Unit(suite, task_id, system, seed, variant)
        for task_id in ids
        for seed in chosen_seeds
        for variant in variants
        for system in systems
    ]


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns become plain strings, once, before the partition: every system
    trains on, and is scored with, identically typed data (a CSV round trip would otherwise
    turn labels such as '1'/'2' into integers)."""
    out = frame.copy()
    for column in out.columns:
        if isinstance(out[column].dtype, pd.CategoricalDtype):
            out[column] = pd.Series(
                [None if pd.isna(v) else str(v) for v in out[column]],
                index=out.index,
                dtype=object,
            )
    return out


def prepare(unit: Unit) -> PreparedUnit:
    if unit.suite == REAL_WORLD_SUITE:
        case = next(c for c in load_real_world() if c.id == unit.task_id)
        frame, target, goal, mapping = case_frame(
            case, "anonymised" if unit.variant == "anonymised" else "original"
        )
        deploy = case.deploy_holdout
        if mapping is not None and deploy.startswith("group:"):  # renamed when anonymised
            deploy = "group:" + mapping.rename(deploy.removeprefix("group:"))
        return PreparedUnit(
            normalise(frame),
            target,
            case.task,
            goal,
            case.entity_scope,
            {},
            real_world_rule(deploy, case.task),
            holdout_scored=case.holdout_scored,
        )
    record = next(t for t in suite_tasks(unit.suite) if t["id"] == unit.task_id)
    spec = TaskSpec(record["trap"], record["base"], record["kind"], record["file_order"])
    pool: Pool = "final" if unit.suite == "final_v1" else "dev"
    built = build_task(spec, unit.seed, pool)
    expected = record["frames"][str(unit.seed)]["sha256"]
    if frame_sha256(built.case.df) != expected:
        raise RuntimeError(f"{unit.task_id} seed {unit.seed} differs from the frozen suite")
    return PreparedUnit(
        normalise(built.case.df),
        built.case.target,
        built.case.task,
        built.case.goal_text,
        None,
        dict(built.case.plan_overrides),
        planted_rule(
            record["trap"],
            built.case.task,
            record["truth"]["affected_columns"],
            clean=record["kind"] == "clean",
        ),
    )


# ----------------------------------------------------------------------------- results


def results_path(root: Path, suite: str, system: str) -> Path:
    return root / suite / f"{system}.jsonl"


def latest_results(root: Path, suite: str, system: str) -> dict[tuple[Any, ...], BenchResult]:
    path = results_path(root, suite, system)
    latest: dict[tuple[Any, ...], BenchResult] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = BenchResult.model_validate_json(line)
                latest[result.unit_key] = result
    return latest


def append_result(root: Path, result: BenchResult) -> None:
    path = results_path(root, result.suite, result.system)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(result.model_dump_json() + "\n")


# ----------------------------------------------------------------------------- one unit


def _dummy_scores(seal: BenchSeal, dev: pd.DataFrame, target: str, task: Task) -> dict[str, float]:
    """The dummy's benchmark-metric score on both holdouts (computed once per seal)."""
    cache = seal.directory / "dummy.json"
    if cache.exists():
        scores: dict[str, float] = json.loads(cache.read_text(encoding="utf-8"))
        return scores
    X, y = dev.drop(columns=[target]), dev[target]
    dummy = (
        DummyClassifier(strategy="prior") if task is Task.CLASSIFICATION else DummyRegressor()
    ).fit(X, y)
    scores = {
        which: seal.evaluate(dummy, benchmark_metric(task), caller="dummy", which=which)
        for which in ("iid", "deploy")
    }
    cache.write_text(json.dumps(scores), encoding="utf-8")
    return scores


def _score_both(
    seal: BenchSeal, output: SystemOutput, metric: str, caller: str, errors: list[str]
) -> dict[str, float | None]:
    """Benchmark-metric and own-metric scores on both holdouts (None where it failed)."""
    scores: dict[str, float | None] = {}
    own = output.validation_metric if output.validation_metric else None
    for which in ("iid", "deploy"):
        try:
            value = seal.evaluate(output.estimator, metric, caller=caller, which=which)
            scores[which] = value
            scores[f"{which}_own"] = (
                value
                if own in (None, metric)
                else seal.evaluate(output.estimator, own, caller=caller, which=which)
            )
        except (SealError, ValueError) as exc:
            errors.append(f"{which} holdout scoring failed: {exc}")
            scores.setdefault(which, None)
            scores.setdefault(f"{which}_own", None)
    return scores


def run_unit(
    unit: Unit,
    system: System,
    root: Path,
    *,
    model: str,
    run_cap_inr: float,
    progress: Progress | None = None,
    resume: bool = False,
    prepare_fn: Callable[[Unit], PreparedUnit] = prepare,
) -> BenchResult:
    """Run (or resume) one unit and return its result; never raises for system failures
    except :class:`RunPausingError` subclasses raised before the system could record a
    pause (they are turned into a paused result)."""
    started = datetime.now(UTC)
    clock = time.perf_counter()
    prepared = prepare_fn(unit)
    seal_dir = root / "seals" / unit.suite / unit.slug
    dev, iid, deploy = partition(prepared.frame, prepared.target, prepared.rule, unit.seed)
    seal = (
        BenchSeal.load(seal_dir)
        if (seal_dir / "seal.json").exists()
        else BenchSeal.create(seal_dir, iid, deploy, prepared.target)
    )
    ctx = SystemContext(
        unit_id=f"{unit.suite}/{unit.slug}/{unit.system}",
        dev=dev,
        target=prepared.target,
        task=prepared.task,
        goal=prepared.goal,
        seed=unit.seed,
        workdir=root / "work" / unit.suite / unit.slug / unit.system,
        model=model,
        run_cap_inr=run_cap_inr,
        entity_scope=prepared.entity_scope,
        plan_overrides=prepared.plan_overrides,
        progress=progress or NullProgress(),
    )
    metric = benchmark_metric(prepared.task)
    try:
        output = system.resume(ctx) if resume else system.run(ctx)
    except RunPausingError as exc:
        output = SystemOutput(status="paused", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # recorded, the benchmark continues
        output = SystemOutput(status="failed", error=f"{type(exc).__name__}: {exc}")
    errors = [output.error] if output.error else []
    scores: dict[str, float | None] = {}
    if output.estimator is not None:
        scores = _score_both(seal, output, metric, unit.system, errors)
    dummy = _dummy_scores(seal, dev, prepared.target, prepared.task)
    return BenchResult(
        suite=unit.suite,
        task_id=unit.task_id,
        system=unit.system,
        seed=unit.seed,
        variant=unit.variant,
        status=output.status,
        error="; ".join(errors) or None,
        issues=output.issues,
        pipeline=output.pipeline,
        validation_metric=output.validation_metric,
        validation_score=output.validation_score,
        benchmark_metric=metric,
        holdout_iid=scores.get("iid"),
        holdout_deploy=scores.get("deploy"),
        holdout_iid_own_metric=scores.get("iid_own"),
        holdout_deploy_own_metric=scores.get("deploy_own"),
        dummy_holdout_iid=dummy["iid"],
        dummy_holdout_deploy=dummy["deploy"],
        holdout_scored=prepared.holdout_scored,
        partition=prepared.rule.describe(),
        report_text=output.report_text,
        report_a11_flags=output.report_a11_flags,
        cost_inr=output.cost_inr,
        seconds=round(time.perf_counter() - clock, 2),
        calls=output.calls,
        retries=output.retries,
        cache_hits=output.cache_hits,
        first_attempt=output.first_attempt,
        repairs=output.repairs,
        substitutions=output.substitutions,
        model=model if system.uses_llm else None,
        extra={**output.extra, "dev_rows": len(dev), "holdout_rows": seal.rows},
        started_at=started,
        finished_at=datetime.now(UTC),
    )


# ----------------------------------------------------------------------------- the loop


def pending(
    units: Iterable[Unit], root: Path, *, retry_failed: bool = False
) -> list[tuple[Unit, bool]]:
    """Units still to run, each with ``resume`` (True when its latest result is paused).
    ``retry_failed`` reruns units whose latest result is failed (e.g. after fixing a
    harness bug); the failed result stays in the file, a new one is appended."""
    cache: dict[tuple[str, str], dict[tuple[Any, ...], BenchResult]] = {}
    todo = []
    for unit in units:
        results = cache.setdefault(
            (unit.suite, unit.system), latest_results(root, unit.suite, unit.system)
        )
        latest = results.get(unit.key)
        if latest is None:
            todo.append((unit, False))
        elif latest.status == "paused":
            todo.append((unit, True))
        elif latest.status == "failed" and retry_failed:
            todo.append((unit, False))
    return todo


def run_all(
    units: Sequence[Unit],
    systems: dict[str, System],
    root: Path,
    *,
    model: str,
    run_cap_inr: float,
    progress: Progress | None = None,
    stop_on_pause: bool = True,
    write: Callable[[str], None] = print,
    prepare_fn: Callable[[Unit], PreparedUnit] = prepare,
    max_spend_inr: float | None = None,
    ledger: SpendLedger | None = None,
    retry_failed: bool = False,
) -> list[BenchResult]:
    """Run every pending unit in order. A pause stops the pass (the cause - an outage, a
    cap, billing - affects every later LLM unit too); rerun the command to continue.

    ``max_spend_inr`` caps this invocation's spend, measured from the ledger (so failed and
    paused units count too): once it is reached, no further unit starts (the current one
    finishes; each unit is bounded by its own per-run cap).
    """
    ledger = ledger if ledger is not None else SpendLedger()
    start = ledger.total_inr()
    done = []
    for unit, resume in pending(units, root, retry_failed=retry_failed):
        spent = ledger.total_inr() - start
        if max_spend_inr is not None and spent >= max_spend_inr:
            write(
                f"spend cap reached: Rs {spent:.2f} of Rs {max_spend_inr:.2f} for this run; "
                "stopped before the next unit"
            )
            break
        write(f"{'resume' if resume else 'run'} {unit.suite} {unit.task_id} {unit.system} "
              f"seed {unit.seed} {unit.variant}")  # fmt: skip
        result = run_unit(
            unit,
            systems[unit.system],
            root,
            model=model,
            run_cap_inr=run_cap_inr,
            progress=progress,
            resume=resume,
            prepare_fn=prepare_fn,
        )
        append_result(root, result)
        done.append(result)
        write(f"  -> {result.status} in {result.seconds:.0f} s, Rs {result.cost_inr:.2f}"
              + (f" ({result.error})" if result.error else ""))  # fmt: skip
        if result.status == "paused" and stop_on_pause:
            write("paused: rerun the same command to resume from here")
            break
        spent = ledger.total_inr() - start
        if max_spend_inr is not None and spent >= max_spend_inr:
            write(
                f"spend cap reached: Rs {spent:.2f} of Rs {max_spend_inr:.2f} for this run "
                f"after {unit.system} on {unit.task_id}; stopped cleanly"
            )
            break
    return done


def main(argv: Sequence[str] | None = None) -> int:
    from bench.config import load_bench_config
    from bench.systems.registry import build_systems

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, choices=[*SUITE_FILES, REAL_WORLD_SUITE])
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plan", action="store_true", help="estimate cost only (pre-flight)")
    parser.add_argument("--purpose", choices=["general", "benchmark"], default="general")
    parser.add_argument(
        "--retry-failed", action="store_true", help="rerun units whose latest result failed"
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        default=None,
        help="stop before the next unit once this invocation has spent this many rupees",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write the pilot report here when done"
    )
    args = parser.parse_args(argv)
    config = load_bench_config()
    systems_wanted = args.systems or list(config.systems)
    units = units_for(args.suite, systems_wanted, seeds=args.seeds, tasks=args.tasks)
    from bench.budget import preflight

    report = preflight(
        units, args.results, config, purpose=args.purpose, retry_failed=args.retry_failed
    )
    print(report.render())
    if args.plan:
        return 0
    if not report.fits:
        print("refusing to start: the estimated cost does not fit (see above)")
        return 6
    # The spend purpose is per process: every Budget built from here on reads it.
    os.environ[llm_config.SPEND_PURPOSE_ENV] = args.purpose
    systems = build_systems(systems_wanted, config)
    progress = LineProgress(lambda line: print(line, flush=True))
    run_all(
        units,
        systems,
        args.results,
        model=config.model,
        run_cap_inr=config.run_cap_inr,
        progress=progress,
        max_spend_inr=args.max_spend,
        retry_failed=args.retry_failed,
    )
    if args.report is not None:
        from bench.pilot import write_pilot_report

        write_pilot_report(args.results, args.suite, args.report, config)
        print(f"pilot report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
