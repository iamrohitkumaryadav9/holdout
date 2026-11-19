"""The benchmark's cost ledger, exported so results regenerate on any machine.

``python -m bench.costs export`` writes ``results/cost_ledger.jsonl``: every paid ledger
line billed to a unit of the final benchmark (``final_v1``, ``real_world_v1``), latest or
superseded. Each line keeps only what cost attribution and reporting need - the run id,
system, unit, suite, model, node, tiers, token counts and rupees - and never an API key, a
path or a prepaid balance.

``bench.analyze`` attributes costs from the local spend ledger (``~/.holdout``) and, for
lines it lacks, from this export (:func:`cost_calls`). On a machine without the local
ledger the export alone reproduces the committed cost tables.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from bench.budget import unit_id_of, unit_lines
from bench.systems.base import BenchResult
from holdout.llm.budget import SpendLedger

EXPORT_NAME: Final = "cost_ledger.jsonl"
SUITES: Final = ("final_v1", "real_world_v1")
#: Ledger fields kept in the export (no keys, paths or balances exist among them).
FIELDS: Final[tuple[str, ...]] = (
    "timestamp", "run_id", "model", "node", "prompt", "purpose",
    "tier_requested", "tier_served", "tier_billed", "fell_back",
    "prompt_tokens", "output_tokens", "thinking_tokens", "cached_tokens", "retries",
    "inr", "inr_list", "inr_cache_adjusted",
)  # fmt: skip


def all_results(results_root: Path, suites: Sequence[str] = SUITES) -> list[BenchResult]:
    """Every result line of ``suites`` (superseded lines too: their spend was real)."""
    out = []
    for suite in suites:
        for path in sorted((results_root / suite).glob("*.jsonl")):
            lines = path.read_text(encoding="utf-8").splitlines()
            out += [BenchResult.model_validate_json(x) for x in lines if x.strip()]
    return out


def _key(line: dict[str, Any]) -> tuple[str, str, float]:
    return (str(line.get("timestamp")), str(line.get("run_id")), float(line["inr"]))


def export_lines(
    results: Sequence[BenchResult], calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The ledger lines billed to ``results``, each tagged with its unit and system, once."""
    seen: set[tuple[str, str, float]] = set()
    out: list[dict[str, Any]] = []
    for result in results:
        for line in unit_lines(result, calls):
            key = _key(line)
            if key in seen:
                continue
            seen.add(key)
            row: dict[str, Any] = {k: line[k] for k in FIELDS if k in line}
            row.update({"unit": unit_id_of(result), "system": result.system,
                        "suite": result.suite, "kind": "call"})  # fmt: skip
            out.append(row)
    return sorted(out, key=lambda r: (str(r["timestamp"]), str(r["unit"])))


def write_export(results_root: Path, ledger: SpendLedger | None = None) -> Path:
    ledger = ledger if ledger is not None else SpendLedger()
    lines = export_lines(all_results(results_root), ledger.calls())
    path = results_root / EXPORT_NAME
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines),
                    encoding="utf-8", newline="\n")  # fmt: skip
    return path


def load_export(results_root: Path) -> list[dict[str, Any]]:
    path = results_root / EXPORT_NAME
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def cost_calls(results_root: Path, ledger: SpendLedger | None = None) -> list[dict[str, Any]]:
    """Paid calls for cost attribution: the local ledger's, plus every exported line the
    local ledger does not have (matched on timestamp, run id and rupees)."""
    try:
        local = (ledger if ledger is not None else SpendLedger()).calls()
    except OSError:
        local = []
    have = {_key(c) for c in local}
    exported = [c for c in load_export(results_root) if _key(c) not in have]
    return local + exported


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the benchmark's cost ledger.")
    parser.add_argument("command", choices=["export"])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    path = write_export(args.results)
    lines = load_export(args.results)
    print(f"wrote {len(lines)} lines, Rs {sum(float(r['inr']) for r in lines):.2f}, to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
