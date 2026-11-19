"""scripts/sanitize_paths.py: a portable copy of the results, the originals untouched, only
path fields changed, and docs/results.md regenerates identically from the copy."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from scripts.sanitize_paths import portable, sanitize_line, sanitize_tree

from bench import analyze
from holdout.llm import config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLDOUT_HOME", str(tmp_path / "home"))  # no local ledger
    monkeypatch.delenv(config.PROJECT_CAP_ENV, raising=False)


def test_paths_become_repo_relative_posix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "results" / "work").mkdir(parents=True)
    assert portable(r"results\work\final_v1\x\holdout\runs\R1", repo) == (
        "results/work/final_v1/x/holdout/runs/R1"
    )
    assert portable(str(repo / "results" / "work" / "R2"), repo) == "results/work/R2"
    assert portable(r"C:\Users\someone\elsewhere\R3", repo) == "<outside-repo>/R3"
    assert portable("/home/someone/elsewhere/R4", repo) == "<outside-repo>/R4"
    line: dict[str, Any] = {
            "status": "paused", "error": r"checkpointed in results\work\a\runs\R1. Resume.",
            "extra": {"run_dirs": [r"results\work\a\runs\R1"], "run_dir": r"results\work\b",
                      "steps": 3}}  # fmt: skip
    clean = sanitize_line(line, repo)
    assert clean["extra"]["run_dirs"] == ["results/work/a/runs/R1"]
    assert clean["extra"]["run_dir"] == "results/work/b"
    assert clean["error"] == "checkpointed in results/work/a/runs/R1. Resume."
    assert clean["extra"]["steps"] == 3
    assert line["extra"]["run_dir"] == r"results\work\b"  # the input is not modified


def _digests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*/*.jsonl"))}  # fmt: skip


def test_the_committed_results_copy_is_portable_and_analyses_identically(tmp_path: Path) -> None:
    results = ROOT / "results"
    before = _digests(results)
    out = tmp_path / "results_sanitized"
    stats = sanitize_tree(results, out)
    assert _digests(results) == before  # the originals are byte-identical
    assert stats["changed_lines"] > 0
    for f in out.glob("*/*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            extra = record.get("extra") or {}
            paths = [*extra.get("run_dirs", []), extra.get("run_dir") or ""]
            assert not any("\\" in p or ":" in p for p in paths), paths
            assert "results\\" not in (record.get("error") or "")
    md = tmp_path / "results.md"
    assert analyze.main(["--results", str(out), "--out", str(md)]) == 0
    expected = (ROOT / "docs" / "results.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert md.read_text(encoding="utf-8") == expected.replace(str(results), str(out))


def test_the_script_refuses_to_write_inside_the_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    (results / "final_v1").mkdir(parents=True)
    (results / "final_v1" / "x.jsonl").write_text(json.dumps({"extra": {}}) + "\n")
    with pytest.raises(ValueError, match="never modified"):
        sanitize_tree(results, results / "copy")
    with pytest.raises(ValueError, match="never modified"):
        sanitize_tree(results, results)
