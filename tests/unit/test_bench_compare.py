"""The v1.1-vs-v1.0 comparison lists every per-seed change, scored by the frozen rules."""

from datetime import UTC, datetime
from pathlib import Path

from bench.compare import changes, render, summarise
from bench.run import append_result
from bench.score import load_results, planted_truths
from bench.systems.base import BenchResult, ReportedIssue
from holdout.audit.types import Category, Severity

TRUTHS = planted_truths("final_v1")
TRAP = next(t for t in TRUTHS.values() if t.kind == "trap" and t.expectations[0].columns)
CLEAN = next(t for t in TRUTHS.values() if t.kind == "clean")


def _issue(category: Category, severity: Severity, columns: list[str]) -> ReportedIssue:
    return ReportedIssue(category=category, severity=severity, columns=columns, source="A1",
                         message="")  # fmt: skip


def _result(task: str, variant: str, seed: int, issues: list[ReportedIssue]) -> BenchResult:
    now = datetime.now(UTC)
    return BenchResult(suite="final_v1", task_id=task, system="audits_only", seed=seed,
                       variant=variant, status="completed", benchmark_metric="roc_auc",
                       issues=issues, started_at=now, finished_at=now)  # fmt: skip


def _write(
    root: Path, trap_seeds: dict[int, bool], clean_seeds: dict[int, Severity | None]
) -> None:
    exp = TRAP.expectations[0]
    hit = _issue(exp.categories[0], Severity.CRITICAL, list(exp.columns))
    for seed, found in trap_seeds.items():
        append_result(root, _result(TRAP.task_id, TRAP.variant, seed, [hit] if found else []))
    for seed, severity in clean_seeds.items():
        issues = [] if severity is None else [_issue(Category.ID_SIGNAL, severity, ["x"])]
        append_result(root, _result(CLEAN.task_id, CLEAN.variant, seed, issues))


def test_every_change_is_listed_per_seed(tmp_path: Path) -> None:
    before, after = tmp_path / "v1_0", tmp_path / "v1_1"
    _write(before, {101: True, 202: False}, {101: Severity.CRITICAL, 202: None})
    _write(after, {101: False, 202: True}, {101: None, 202: Severity.WARNING})
    b, a = load_results(before, "final_v1"), load_results(after, "final_v1")
    found = {(c.kind, c.seed) for c in changes(b, a, TRUTHS, "audits_only")}
    assert found == {
        ("detection lost", 101),
        ("detection gained", 202),
        ("false alarm gone (critical)", 101),
        ("false alarm gone (warning)", 101),
        ("false alarm new (warning)", 202),
    }
    s = summarise(a, TRUTHS, "audits_only")
    assert (s.clean_tasks, s.clean_critical, s.clean_warning) == (1, 0, 1)
    text = "\n".join(render("final_v1", before, after))
    assert "| false alarm new (warning) |" in text
    assert "ID_SIGNAL".lower() in text


def test_no_change_says_so(tmp_path: Path) -> None:
    for name in ("v1_0", "v1_1"):
        _write(tmp_path / name, {101: True}, {101: None})
    text = "\n".join(render("final_v1", tmp_path / "v1_0", tmp_path / "v1_1"))
    assert "No per-seed change" in text
