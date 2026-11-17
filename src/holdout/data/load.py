"""Load a CSV with its content hash.

The hash ties every artifact (manifest, ledger, exported script) to the exact bytes that
were analysed, so a re-run on a modified file is detectable.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from holdout.core.errors import DataValidationError
from holdout.core.portable import file_sha256, load_csv


@dataclass(frozen=True)
class LoadedData:
    """A loaded dataset and the facts about how it was loaded."""

    frame: pd.DataFrame
    path: Path
    sha256: str
    n_dropped_missing_target: int


def load_dataset(path: Path, target: str) -> LoadedData:
    """Load ``path``, drop rows with a missing target, and hash the file.

    Raises:
        DataValidationError: if the file is missing or empty, the target column is absent,
            or fewer than two labelled rows remain.
    """
    if not path.is_file():
        raise DataValidationError(f"CSV not found: {path}")
    try:
        frame, n_dropped = load_csv(path, target)
    except KeyError as exc:
        raise DataValidationError(str(exc.args[0])) from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"CSV is empty: {path}") from exc
    if len(frame) < 2:
        raise DataValidationError(
            f"only {len(frame)} row(s) with a non-missing {target!r}; need at least 2"
        )
    return LoadedData(
        frame=frame, path=path, sha256=file_sha256(path), n_dropped_missing_target=n_dropped
    )
