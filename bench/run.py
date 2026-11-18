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
import functools
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
from holdout.core.schemas import EntityScope, MetricName, Task
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
    own = output.validation_metric or None
    known = {m.value for m in MetricName}
    if own is not None and own not in known:
        errors.append(f"own metric {own!r} is not a benchmark metric: no own-metric score")
        own = "__unknown__"
    for which in ("iid", "deploy"):
        scores[which] = scores[f"{which}_own"] = None
        try:
            scores[which] = seal.evaluate(output.estimator, metric, caller=caller, which=which)
        except (SealError, ValueError) as exc:
            errors.append(f"{which} holdout scoring failed: {exc}")
            continue
        if own in (None, metric):
            scores[f"{which}_own"] = scores[which]
        elif own != "__unknown__":
            try:
                scores[f"{which}_own"] = seal.evaluate(
                    output.estimator, own, caller=caller, which=which
                )
            except (SealError, ValueError) as exc:
                errors.append(f"{which} own-metric scoring failed: {exc}")
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
    service_tier: str | None = None,
    prepare_fn: Callable[[Unit], PreparedUnit] = prepare,
) -> BenchResult:
    """Run (or resume) one unit and return its result; never raises for system failures
    except :class:`RunPausingError` subclasses raised before the system could record a
    pause (they are turned into a paused result)."""
    started = datetime.now(UTC)
    clock = time.perf_counter()
    ledger = SpendLedger()
    spent_before = ledger.total_inr()
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
        service_tier=service_tier if system.uses_llm else None,
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
    wall = time.perf_counter() - clock
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
        # What the ledger recorded during this unit: a unit that fails or pauses after
        # spending (e.g. a plan call before a rejected request) is charged, not shown as Rs 0.
        cost_inr=max(output.cost_inr, ledger.total_inr() - spent_before),
        cost_cache_adjusted_inr=output.cost_cache_adjusted_inr,
        cached_tokens=output.cached_tokens,
        prompt_tokens=output.prompt_tokens,
        tiers_served=output.tiers_served,
        seconds=round(wall, 2),
        api_seconds=round(output.api_seconds, 2),
        compute_seconds=round(max(0.0, wall - output.api_seconds), 2),
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


def ensure_seal(unit: Unit, root: Path, prepare_fn: Callable[[Unit], PreparedUnit]) -> None:
    """Create the unit's seal and dummy scores if missing. With several workers this runs
    in the parent before dispatch, so units of different systems that share a seal never
    race to create it."""
    seal_dir = root / "seals" / unit.suite / unit.slug
    if (seal_dir / "seal.json").exists() and (seal_dir / "dummy.json").exists():
        return
    prepared = prepare_fn(unit)
    dev, iid, deploy = partition(prepared.frame, prepared.target, prepared.rule, unit.seed)
    seal = (
        BenchSeal.load(seal_dir)
        if (seal_dir / "seal.json").exists()
        else BenchSeal.create(seal_dir, iid, deploy, prepared.target)
    )
    _dummy_scores(seal, dev, prepared.target, prepared.task)


@dataclass(frozen=True)
class _Job:
    """Everything a worker process needs to run one unit (all picklable)."""

    unit: Unit
    resume: bool
    root: Path
    model: str
    run_cap_inr: float
    service_tier: str | None
    system_factory: Callable[[list[str]], dict[str, System]]
    prepare_fn: Callable[[Unit], PreparedUnit]


_WORKER_SYSTEMS: dict[str, System] = {}


def _work(job: _Job) -> BenchResult:
    """Run one unit in a worker process; systems are built once per process."""
    if job.unit.system not in _WORKER_SYSTEMS:
        _WORKER_SYSTEMS.update(job.system_factory([job.unit.system]))
    tag = f"[{job.unit.system} {job.unit.task_id} s{job.unit.seed} {job.unit.variant}]"
    progress = LineProgress(lambda line: print(f"{tag} {line}", flush=True))
    return run_unit(
        job.unit,
        _WORKER_SYSTEMS[job.unit.system],
        job.root,
        model=job.model,
        run_cap_inr=job.run_cap_inr,
        progress=progress,
        resume=job.resume,
        prepare_fn=job.prepare_fn,
        service_tier=job.service_tier,
    )


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
    service_tier: str | None = None,
    workers: int = 1,
    system_factory: Callable[[list[str]], dict[str, System]] | None = None,
    purpose: str | None = None,
) -> list[BenchResult]:
    """Run every pending unit in order, up to ``workers`` at a time.

    A pause stops the pass (the cause - an outage, a cap, billing - affects every later
    LLM unit too): no further unit starts, running units finish, and rerunning the command
    continues.

    The unit-start guard: a unit starts only if every applicable limit (the project cap,
    the benchmark reserve for general spend, ``max_spend_inr`` for this invocation, the
    prepaid balance when recorded) still has at least the unit's worst case (per-run cap,
    twice for Holdout-family systems, which may start one follow-up run) on top of the
    worst case of every unit still running. If not, it waits for running units to finish
    (their actual spend is usually far below the worst case) and checks again; with
    nothing running, the pass stops cleanly. Spend is read from the ledger, so failed and
    paused units count too.

    With ``workers`` > 1, units run in separate processes; ``system_factory`` builds the
    systems there and must be picklable, like ``prepare_fn``. Seals and dummy scores are
    created here first; results are appended here, in completion order.
    """
    import concurrent.futures as cf
    import multiprocessing

    from bench.budget import limits_left, start_refusal, unit_worst_case

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers > 1 and system_factory is None:
        raise ValueError("several workers need a picklable system_factory")
    ledger = ledger if ledger is not None else SpendLedger()
    purpose = purpose or llm_config.spend_purpose()
    start = ledger.total_inr()
    done: list[BenchResult] = []
    running: dict[cf.Future[BenchResult], Unit] = {}
    errors: list[BaseException] = []
    stopping = False
    pool = (
        cf.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )

    def worst(unit: Unit) -> float:
        return unit_worst_case(unit.system, run_cap_inr)

    def collect(block: bool) -> None:
        nonlocal stopping
        finished, _ = cf.wait(
            list(running), timeout=None if block else 0, return_when=cf.FIRST_COMPLETED
        )
        for future in finished:
            unit = running.pop(future)
            error = future.exception()
            if error is not None:  # a harness error, not a system failure: stop, keep others
                write(f"  -> {unit.system} {unit.task_id}: harness error {error!r}; stopping")
                errors.append(error)
                stopping = True
                continue
            result = future.result()
            # Units share the CPU under several workers (FLAML's wall-clock budget): record it.
            result = result.model_copy(update={"extra": {**result.extra, "workers": workers}})
            append_result(root, result)
            done.append(result)
            write(f"  -> {unit.system} {unit.task_id} seed {unit.seed}: {result.status} in "
                  f"{result.time_per_run:.0f} s compute + {result.api_seconds:.0f} s API, "
                  f"Rs {result.cost_inr:.2f}"
                  + (f" ({result.error})" if result.error else ""))  # fmt: skip
            if result.status == "paused" and stop_on_pause and not stopping:
                stopping = True
                write("paused: no further unit starts; rerun the same command to resume")

    try:
        for unit, resume in pending(units, root, retry_failed=retry_failed):
            while running and len(running) >= workers and not stopping:
                collect(block=True)
            while not stopping:
                left = limits_left(
                    ledger, purpose=purpose, invocation_start_inr=start,
                    max_spend_inr=max_spend_inr,
                )  # fmt: skip
                in_flight = sum(worst(u) for u in running.values())
                refusal = start_refusal(worst(unit), in_flight, left)
                if refusal is None:
                    break
                if not running:
                    write(
                        f"not starting {unit.system} on {unit.task_id} seed {unit.seed}: "
                        f"{refusal}. Spent Rs {ledger.total_inr() - start:.2f} in this run; "
                        "stopped cleanly"
                    )
                    stopping = True
                    break
                collect(block=True)
            if stopping:
                break
            write(f"{'resume' if resume else 'run'} {unit.suite} {unit.task_id} {unit.system} "
                  f"seed {unit.seed} {unit.variant}")  # fmt: skip
            if pool is None:
                result = run_unit(
                    unit, systems[unit.system], root, model=model, run_cap_inr=run_cap_inr,
                    progress=progress, resume=resume, prepare_fn=prepare_fn,
                    service_tier=service_tier,
                )  # fmt: skip
                future: cf.Future[BenchResult] = cf.Future()
                future.set_result(result)
                running[future] = unit
                collect(block=True)
            else:
                assert system_factory is not None
                ensure_seal(unit, root, prepare_fn)
                job = _Job(unit, resume, root, model, run_cap_inr, service_tier,
                           system_factory, prepare_fn)  # fmt: skip
                running[pool.submit(_work, job)] = unit
        while running:
            collect(block=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if errors:
        raise errors[0]
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
    parser.add_argument(
        "--workers", type=int, default=1, help="units run at once, in separate processes"
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    config = load_bench_config()
    systems_wanted = args.systems or list(config.systems)
    units = units_for(args.suite, systems_wanted, seeds=args.seeds, tasks=args.tasks)
    from bench.budget import preflight

    report = preflight(
        units, args.results, config, purpose=args.purpose, retry_failed=args.retry_failed,
        workers=args.workers,
    )  # fmt: skip
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
        service_tier=config.service_tier,
        workers=args.workers,
        system_factory=functools.partial(build_systems, config=config),
        purpose=args.purpose,
    )
    if args.report is not None:
        from bench.pilot import write_pilot_report

        write_pilot_report(args.results, args.suite, args.report, config)
        print(f"pilot report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
