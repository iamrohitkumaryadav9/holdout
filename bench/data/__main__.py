"""``python -m bench.data verify [--offline]``: download (once) and verify every dataset.

Prints, per dataset: pool, prepared rows and columns, and the target's distribution head.
Exit 1 if any file is missing (offline) or fails its sha256.
"""

import argparse
import sys

from bench.data.openml import DatasetIntegrityError, DatasetUnavailableError, load_frame
from bench.data.registry import load_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("--offline", action="store_true", help="never download")
    args = parser.parse_args()
    failures = 0
    for spec in load_registry().values():
        try:
            frame, target, task = load_frame(spec, allow_download=not args.offline)
        except (DatasetIntegrityError, DatasetUnavailableError) as exc:
            print(f"FAIL {spec.name}: {exc}")
            failures += 1
            continue
        top = (
            frame[target].value_counts().head(3).to_dict() if task.value == "classification" else {}
        )
        print(
            f"ok   {spec.pool:10} {spec.name:26} {len(frame):6} rows x {frame.shape[1]:3} cols "
            f"{task.value:14} {top}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
