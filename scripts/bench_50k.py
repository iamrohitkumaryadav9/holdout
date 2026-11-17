"""Time a full ``holdout run --unseal`` on 50k-row datasets, end to end.

Measures wall-clock time of the real CLI in a fresh interpreter (imports included), so the
number is what a user waits for. Exits non-zero if any run exceeds the budget.

    python scripts/bench_50k.py                 # 50k rows, 60 s budget
    python scripts/bench_50k.py --rows 100000 --budget 120
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from holdout.data.examples import make_classification_example, make_timeseries_example


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--budget", type=float, default=60.0, help="seconds per run")
    parser.add_argument("--out", type=Path, default=Path("bench_runs"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    classification = args.out / f"churn_{args.rows}.csv"
    timeseries = args.out / f"sales_{args.rows}.csv"
    make_classification_example(n_rows=args.rows, seed=1).to_csv(classification, index=False)
    make_timeseries_example(n_days=args.rows // 5, seed=1).to_csv(timeseries, index=False)

    cases = [
        ("classification, stratified", classification, ["--target", "churned"]),
        (
            "regression, time split",
            timeseries,
            ["--target", "units_sold", "--data-type", "timeseries"],
        ),
    ]
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"budget: {args.budget:.0f} s per run, {args.rows} rows each")
    failed = False
    for label, csv, extra in cases:
        command = [
            sys.executable,
            "-m",
            "holdout",
            "run",
            "--csv",
            str(csv),
            *extra,
            "--unseal",
            "--out",
            str(args.out / "runs"),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
        seconds = time.perf_counter() - started
        ok = completed.returncode == 0 and seconds <= args.budget
        failed |= not ok
        status = "ok" if ok else "FAIL"
        print(f"{status:4}  {label:28} {seconds:6.1f} s  (exit {completed.returncode})")
        if completed.returncode != 0:
            print(completed.stdout[-2000:], completed.stderr[-2000:], sep="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
