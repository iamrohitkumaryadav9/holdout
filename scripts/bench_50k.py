"""Time a full ``holdout run --unseal`` (with audits) on 50k-row datasets, end to end.

Measures wall-clock time of the real CLI in a fresh interpreter (imports included), so the
number is what a user waits for. Exit code 3 (the audit refused to unseal) is a normal
outcome here. Exits non-zero if any run errors or exceeds the budget.

Run from the repository root (`bench` is a top-level dev package):

    python -m scripts.bench_50k                 # 50k rows, 90 s budget
    python -m scripts.bench_50k --rows 100000 --budget 180
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from bench.traps import build_case
from holdout.data.examples import make_classification_example, make_timeseries_example

#: Exit code of ``holdout run`` when the audit refused --unseal.
UNSEAL_REFUSED = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--budget", type=float, default=90.0, help="seconds per run")
    parser.add_argument("--out", type=Path, default=Path("bench_runs"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    classification = args.out / f"churn_{args.rows}.csv"
    timeseries = args.out / f"sales_{args.rows}.csv"
    leak = args.out / f"direct_leak_{args.rows}.csv"
    entities = args.out / f"group_leakage_{args.rows}.csv"
    make_classification_example(n_rows=args.rows, seed=1).to_csv(classification, index=False)
    make_timeseries_example(n_days=args.rows // 5, seed=1).to_csv(timeseries, index=False)
    build_case("direct_leak", 1, n_rows=args.rows).df.to_csv(leak, index=False)
    build_case("group_leakage", 1, n_rows=args.rows).df.to_csv(entities, index=False)

    cases = [
        ("classification, stratified", classification, ["--target", "churned"]),
        (
            "regression, time split",
            timeseries,
            ["--target", "units_sold", "--data-type", "timeseries"],
        ),
        ("deep mode (planted leak)", leak, ["--target", "converted"]),
        # Worst cases for the audit: they force the expensive split comparisons.
        ("time order, random split (A6)", timeseries, ["--target", "units_sold"]),
        ("entities, row split (A5)", entities, ["--target", "converted"]),
    ]
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"budget: {args.budget:.0f} s per run, {args.rows} rows each")
    failed = False
    for label, csv, extra in cases:
        out = args.out / "runs" / csv.stem
        command = [
            sys.executable, "-m", "holdout", "run", "--csv", str(csv), *extra,
            "--unseal", "--out", str(out),
        ]  # fmt: skip
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
        ran = completed.returncode in (0, UNSEAL_REFUSED)
        ok = ran and seconds <= args.budget
        failed |= not ok
        status = "ok" if ok else "FAIL"
        print(f"{status:4}  {label:28} {seconds:6.1f} s  (exit {completed.returncode})")
        if ran:
            print("      " + _audit_summary(out))
        else:
            print(completed.stdout[-2000:], completed.stderr[-2000:], sep="\n")
    return 1 if failed else 0


def _audit_summary(out: Path) -> str:
    run_dir = max(p for p in out.iterdir() if p.is_dir())
    audit = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    slowest = sorted(audit["checks_run"], key=lambda r: -r["seconds"])[:3]
    timings = ", ".join(f"{r['check']} {r['seconds']:.1f}s" for r in slowest)
    counts = audit["counts"]
    return (
        f"audit: {counts['critical']} critical / {counts['warning']} warning / "
        f"{counts['info']} info; deep mode {audit['deep_mode']}; slowest: {timings}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
