"""v1.1 thresholds layer on the frozen v1.0 file, which never changes; both hashes are
recorded in the audit report and the run manifest."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from holdout.audit.thresholds import (
    default_thresholds_bytes,
    default_thresholds_v11_bytes,
    load_thresholds,
    load_thresholds_v11,
)
from holdout.audit.types import AuditReport

ROOT = Path(__file__).resolve().parents[3]
#: thresholds.toml as frozen by the pre-registration (docs/preregistration.md, section 5).
FROZEN_V1_SHA256 = "99a8905a1004207b9c314a99d6ab10da88be232a5786fb35819e04e4348287d6"


def test_the_v1_thresholds_file_is_byte_identical_to_its_frozen_version() -> None:
    assert hashlib.sha256(default_thresholds_bytes()).hexdigest() == FROZEN_V1_SHA256
    assert load_thresholds()[1] == FROZEN_V1_SHA256


def test_the_v1_thresholds_file_matches_the_v1_0_0_tag_when_git_can_see_it() -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    shown = subprocess.run(
        [git, "show", "v1.0.0:src/holdout/audit/thresholds.toml"],
        cwd=ROOT, capture_output=True, check=False,
    )  # fmt: skip
    if shown.returncode != 0:
        pytest.skip("tag v1.0.0 not fetched (shallow CI checkout)")
    tagged = shown.stdout.replace(b"\r\n", b"\n")
    assert hashlib.sha256(tagged).hexdigest() == FROZEN_V1_SHA256


def test_the_v11_additions_load_strictly_with_their_own_hash(tmp_path: Path) -> None:
    v11, digest = load_thresholds_v11()
    assert digest == hashlib.sha256(default_thresholds_v11_bytes()).hexdigest()
    assert digest != FROZEN_V1_SHA256
    assert "id" in v11.a03_id_like.key_name_tokens
    assert v11.a05_group.min_entity_constant_features == 1
    assert b"\r\n" not in default_thresholds_v11_bytes()
    typo = tmp_path / "bad.toml"
    typo.write_text(
        default_thresholds_v11_bytes().decode().replace("key_name_tokens", "key_tokens")
    )
    with pytest.raises(ValueError, match=r"key_name_tokens|key_tokens"):
        load_thresholds_v11(typo)


def test_audit_reports_written_before_v11_still_load() -> None:
    old = {"flags": [], "checks_run": [], "thresholds_hash": FROZEN_V1_SHA256,
           "deep_mode": False, "entity_scope": None, "complete": True,
           "holdout_invalid": False, "counts": {}}  # fmt: skip
    report = AuditReport.model_validate(old)
    assert report.thresholds_v11_hash is None
