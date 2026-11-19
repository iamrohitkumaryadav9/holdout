"""Run the offline system (audits_only) on a suite, without the budget pre-flight.

Used for the post-hoc v1.1 evaluation (docs/results_v11.md), including against the v1.0.0
code in a git worktree, whose pre-flight refuses even offline runs once the project cap
is spent (fixed in v1.1, commit "a run with no LLM units always passes the budget
pre-flight"). audits_only makes no API call, so there is nothing to budget:

    git worktree add --detach ../holdout_v100 v1.0.0
    cd ../holdout_v100
    PYTHONPATH=$PWD/src:$PWD python ../holdout/scripts/run_offline_eval.py \\
        --suite dev_v1 --results ../holdout/results_v11/v1_0

Run with PYTHONPATH set to the checkout whose code is evaluated (for the current code:
``PYTHONPATH=$PWD python scripts/run_offline_eval.py --suite dev_v1 --results
results_v11/v1_1``). Only audits_only is run (it never calls an LLM), and it refuses to
write into results/.
"""

import argparse
import functools
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from bench.config import load_bench_config
    from bench.run import REAL_WORLD_SUITE, SUITE_FILES, run_all, units_for
    from bench.systems.base import SystemName
    from bench.systems.registry import build_systems
    from holdout.llm.progress import LineProgress

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, choices=[*SUITE_FILES, REAL_WORLD_SUITE])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    resolved = Path(args.results).resolve()
    if any(p.name == "results" for p in (resolved, *resolved.parents)):
        parser.error("refusing to write into results/ (the pre-registered v1 results)")
    config = load_bench_config()
    offline: list[SystemName] = ["audits_only"]
    units = units_for(args.suite, offline)
    systems = build_systems(offline, config)
    run_all(
        units,
        systems,
        args.results,
        model=config.model,
        run_cap_inr=config.run_cap_inr,
        progress=LineProgress(lambda line: print(line, flush=True)),
        service_tier=config.tiers(),
        workers=args.workers,
        system_factory=functools.partial(build_systems, config=config),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
