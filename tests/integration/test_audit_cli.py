"""End-to-end audit wiring: unseal gate, holdout integrity, accepted risk, re-audit."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bench.traps import build_case
from holdout.audit.checks import a08_formats
from holdout.cli import EXIT_UNSEAL_REFUSED, app
from holdout.core.errors import HoldoutInvalidError
from holdout.core.state import RunState
from holdout.data.seal import SealedHoldout
from tests.helpers import AccuracyScorer

runner = CliRunner()
RISK = "approved by the data owner for this demo"


@pytest.fixture(scope="module")
def csvs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("traps")
    paths = {}
    for name in ("group_leakage", "direct_leak"):
        path = out / f"{name}.csv"
        build_case(name, 0).df.to_csv(path, index=False)
        paths[name] = path
    return paths


def _run(csv: Path, out: Path, *extra: str) -> Any:
    return runner.invoke(
        app, ["run", "--csv", str(csv), "--target", "converted", "--out", str(out), *extra]
    )


def _manifest(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return data


def _runs(out: Path) -> list[Path]:
    return sorted(p for p in out.iterdir() if p.is_dir())


def test_group_leakage_invalidates_and_accept_risk_cannot_override(
    csvs: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "runs"
    result = _run(csvs["group_leakage"], out, "--unseal", "--accept-risk", RISK, "--ack-warnings")
    assert result.exit_code == EXIT_UNSEAL_REFUSED, result.output
    assert "Start a NEW run with the corrected split" in result.output
    assert "--split group:account --entities new" in result.output
    run_dir = _runs(out)[0]
    manifest = _manifest(run_dir)
    assert manifest["holdout_invalid"] is True
    assert manifest["holdout"] == "not unsealed"
    assert manifest["unseal"]["allowed"] is False
    assert manifest["unseal"]["overridden_criticals"] == []
    assert any("not overridable" in b for b in manifest["unseal"]["blocked_by"])
    assert (run_dir / "audit.json").is_file()
    # The seal itself refuses, even with force: the guarantee is not only in the CLI.
    raw = _manifest(run_dir)
    state = RunState(raw["run_id"], out, raw["run_seed"], raw["started_at"])
    with pytest.raises(HoldoutInvalidError):
        SealedHoldout.load(state).evaluate(object(), AccuracyScorer(), caller="test", force=True)
    log = (run_dir / "holdout_access.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["outcome"] for line in log] == ["marked_invalid", "refused_invalid"]


def test_the_corrected_run_carries_the_redrawn_caveat(
    csvs: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "runs"
    assert _run(csvs["group_leakage"], out).exit_code == 0
    corrected = _run(csvs["group_leakage"], out, "--split", "group:account", "--entities", "new")
    assert corrected.exit_code == 0, corrected.output
    newest = _runs(out)[-1]
    manifest = _manifest(newest)
    assert manifest["holdout_invalid"] is False
    assert manifest["split"]["spec"]["entity_scope"] == "new"
    assert any("re-drawn after the data had been inspected" in c for c in manifest["caveats"])
    report = (newest / "report.md").read_text(encoding="utf-8")
    assert "> **Caveat:** this holdout was re-drawn" in report
    assert "- **Entities:** new" in report


def test_known_entities_never_invalidate(csvs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "runs"
    result = _run(csvs["group_leakage"], out, "--entities", "known", "--unseal", "--ack-warnings")
    manifest = _manifest(_runs(out)[0])
    assert manifest["holdout_invalid"] is False
    assert manifest["entity_scope"] == "known"
    a5 = [f for f in manifest["audit"]["flags"] if f["check"] == "A5"]
    assert a5
    assert all(f["severity"] == "info" for f in a5)
    assert "--entities known" in a5[0]["message"]
    assert result.exit_code in (0, EXIT_UNSEAL_REFUSED)  # other checks may still object


def test_direct_leak_critical_is_refused_then_unsealed_with_accepted_risk(
    csvs: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "runs"
    refused = _run(csvs["direct_leak"], out, "--unseal", "--ack-warnings")
    assert refused.exit_code == EXIT_UNSEAL_REFUSED
    assert "--accept-risk" in refused.output
    accepted = _run(csvs["direct_leak"], out, "--unseal", "--ack-warnings", "--accept-risk", RISK)
    assert accepted.exit_code == 0, accepted.output
    assert "RISK ACCEPTED" in accepted.output
    run_dir = _runs(out)[-1]
    manifest = _manifest(run_dir)
    assert manifest["unseal"]["accepted_risk"] == RISK
    assert manifest["unseal"]["overridden_criticals"] == ["A1 refund_issued"]
    assert isinstance(manifest["holdout"], dict)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert report.splitlines()[2].startswith("> **RISK ACCEPTED.**")


def test_warnings_need_acknowledgement(csvs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(csvs["direct_leak"], tmp_path, "--unseal", "--accept-risk", RISK)
    assert result.exit_code == EXIT_UNSEAL_REFUSED
    assert "--ack-warnings" in result.output


def test_too_short_risk_reason_is_an_error(csvs: dict[str, Path], tmp_path: Path) -> None:
    result = _run(csvs["direct_leak"], tmp_path, "--unseal", "--accept-risk", "ok")
    assert result.exit_code == 2
    assert "at least 10 characters" in result.output


def test_an_incomplete_audit_blocks_unsealing(
    csvs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(ctx: object) -> None:
        raise RuntimeError("injected")

    monkeypatch.setattr(a08_formats, "run", broken)
    out = tmp_path / "runs"
    result = _run(csvs["direct_leak"], out, "--unseal", "--ack-warnings", "--accept-risk", RISK)
    assert result.exit_code == EXIT_UNSEAL_REFUSED
    assert "INCOMPLETE" in result.output
    manifest = _manifest(_runs(out)[0])
    assert manifest["audit"]["complete"] is False
    assert any(b.startswith("audit incomplete: A8") for b in manifest["unseal"]["blocked_by"])


def test_reaudit_an_existing_run(csvs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "runs"
    assert _run(csvs["direct_leak"], out).exit_code == 0
    run_dir = _runs(out)[0]
    first = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    result = runner.invoke(app, ["audit", "--run", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "refund_issued" in result.output
    again = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    assert [f["check"] for f in again["flags"]] == [f["check"] for f in first["flags"]]
    assert _manifest(run_dir)["holdout"] == "not unsealed"  # re-audit never unseals


def test_reaudit_refuses_a_changed_csv(csvs: dict[str, Path], tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_bytes(csvs["direct_leak"].read_bytes())
    out = tmp_path / "runs"
    assert _run(csv, out).exit_code == 0
    csv.write_bytes(csv.read_bytes() + b"\n")  # same rows, different bytes
    result = runner.invoke(app, ["audit", "--run", str(_runs(out)[0])])
    assert result.exit_code == 2
    assert "changed since the run" in result.output


def test_reaudit_needs_a_run_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["audit", "--run", str(tmp_path)])
    assert result.exit_code == 2
    assert "no run manifest" in result.output
