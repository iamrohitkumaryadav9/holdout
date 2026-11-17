"""Run identity and the on-disk layout of a run directory.

All artifact paths are defined here once, so the seal, the exporters and the CLI cannot
disagree about where a run's files live.
"""

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from holdout.core.seeds import derived_seeds

#: File name of the cross-run ledger, shared by all runs under one output directory.
LEDGER_FILENAME = "holdout_ledger.jsonl"


@dataclass(frozen=True)
class RunState:
    """Identity and file layout of one run.

    ``run_id`` combines a UTC timestamp (sortable, human-readable) with random hex (unique
    even for two runs started in the same second).
    """

    run_id: str
    out_dir: Path
    run_seed: int
    created_at: datetime = field(compare=False)

    @classmethod
    def create(cls, out_dir: Path, run_seed: int, now: datetime | None = None) -> "RunState":
        """Allocate a fresh run directory under ``out_dir``.

        Raises:
            FileExistsError: if the generated run directory already exists, which would
                mean two runs share a seal; it is never silently reused.
        """
        created_at = now if now is not None else datetime.now(UTC)
        run_id = f"{created_at:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"
        state = cls(run_id=run_id, out_dir=out_dir, run_seed=run_seed, created_at=created_at)
        state.run_dir.mkdir(parents=True, exist_ok=False)
        return state

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.run_id

    @property
    def holdout_path(self) -> Path:
        return self.run_dir / "holdout.parquet"

    @property
    def seal_state_path(self) -> Path:
        return self.run_dir / "holdout_seal.json"

    @property
    def access_log_path(self) -> Path:
        return self.run_dir / "holdout_access.jsonl"

    @property
    def ledger_path(self) -> Path:
        return self.out_dir / LEDGER_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def pipeline_path(self) -> Path:
        return self.run_dir / "pipeline.py"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.md"

    @property
    def seeds(self) -> dict[str, int]:
        return derived_seeds(self.run_seed)
