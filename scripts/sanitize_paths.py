"""Write a copy of the committed benchmark results with portable, repo-relative paths.

    python scripts/sanitize_paths.py [--results results] [--out results_sanitized]

**It never modifies ``results/``.** The pre-registered results files stay byte-identical
(the ``prereg-v1*`` and ``v1.0.0`` tags point at them, and no history is rewritten). This
script writes a separate copy for sharing.

**What changes, and only this:**
- the path fields ``extra.run_dirs`` and ``extra.run_dir``;
- path tokens inside ``error`` messages (e.g. "The run is checkpointed in <path>").

Each path becomes repo-relative with ``/`` separators. A path that is absolute and inside
the repository is made relative to it. An absolute path outside the repository becomes
``<outside-repo>/<last component>``, so no user or machine name is kept.

**What it checks.** Every other field of every line is compared with the original and must
be equal, and the copy must parse as the same results. ``tests/unit/test_sanitize_paths.py``
also regenerates ``docs/results.md`` from the copy and checks it is byte-identical.

**What it found** (2026-11-19). The committed results hold no absolute paths: the run
directories were already repo-relative, written with Windows separators
(``results\\work\\...``). The copy normalises those separators.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PATH_FIELDS = ("run_dirs", "run_dir")  # inside ``extra``
#: A path token embedded in free text: a drive or root, or a ``results`` / ``runs`` tree.
_TOKEN = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users)/|results[\\/]|runs[\\/])[^\s'\"),;]*")


def portable(path: str, repo: Path = REPO) -> str:
    """``path`` as a repo-relative POSIX path (see the module docs)."""
    is_windows = bool(re.match(r"^[A-Za-z]:[\\/]", path)) or "\\" in path
    pure = PureWindowsPath(path) if is_windows else PurePosixPath(path)
    if pure.is_absolute():
        # Only a path native to this OS can be inside this checkout; a Windows path read on
        # POSIX (or the reverse) is outside it by definition.
        if isinstance(pure, PureWindowsPath) == (os.name == "nt"):
            try:
                return Path(path).resolve().relative_to(repo.resolve()).as_posix()
            except (ValueError, OSError):
                pass
        return f"<outside-repo>/{pure.name}"
    return PurePosixPath(*pure.parts).as_posix()


def sanitize_line(record: dict[str, Any], repo: Path = REPO) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(record))  # a deep copy
    extra = out.get("extra") or {}
    for field in PATH_FIELDS:
        value = extra.get(field)
        if isinstance(value, list):
            extra[field] = [portable(v, repo) if isinstance(v, str) else v for v in value]
        elif isinstance(value, str):
            extra[field] = portable(value, repo)
    if isinstance(out.get("error"), str):
        out["error"] = _TOKEN.sub(lambda m: portable(m.group(0), repo), out["error"])
    return out


def _without_paths(record: dict[str, Any]) -> dict[str, Any]:
    stripped: dict[str, Any] = json.loads(json.dumps(record))
    for field in PATH_FIELDS:
        (stripped.get("extra") or {}).pop(field, None)
    if isinstance(stripped.get("error"), str):
        stripped["error"] = _TOKEN.sub("<path>", stripped["error"])
    return stripped


def sanitize_tree(results: Path, out: Path, repo: Path = REPO) -> dict[str, int]:
    """Copy ``results`` to ``out`` with portable paths; return lines and changed counts.

    Raises:
        ValueError: if ``out`` is ``results`` or inside it, or a line changed outside its
            path fields.
    """
    results, out = results.resolve(), out.resolve()
    if out == results or results in out.parents:
        raise ValueError(f"refusing to write inside {results}: the originals are never modified")
    stats = {"files": 0, "lines": 0, "changed_lines": 0}
    for src in sorted(results.rglob("*")):
        rel = src.relative_to(results)
        if src.is_dir() or rel.parts[0] in ("work", "seals"):
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        stats["files"] += 1
        if src.suffix != ".jsonl" or src.name == "cost_ledger.jsonl":
            shutil.copyfile(src, dst)  # reports and the cost export hold no paths
            continue
        lines_out = []
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            clean = sanitize_line(record, repo)
            if _without_paths(clean) != _without_paths(record):
                raise ValueError(f"{rel}: a non-path field changed")
            stats["lines"] += 1
            stats["changed_lines"] += clean != record
            lines_out.append(json.dumps(clean) + "\n")
        dst.write_text("".join(lines_out), encoding="utf-8", newline="\n")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--out", type=Path, default=REPO / "results_sanitized")
    args = parser.parse_args(argv)
    stats = sanitize_tree(args.results, args.out)
    print(f"wrote {stats['files']} files ({stats['lines']} result lines, "
          f"{stats['changed_lines']} with paths normalised) to {args.out}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
