"""load_dataset: hashing and precise errors for unusable files."""

from pathlib import Path

import pandas as pd
import pytest

from holdout.core.errors import DataValidationError
from holdout.core.portable import file_sha256
from holdout.data.load import load_dataset


def test_load_dataset_hashes_and_drops_unlabelled_rows(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, None, 1]}).to_csv(path, index=False)
    loaded = load_dataset(path, "y")
    assert loaded.sha256 == file_sha256(path)
    assert loaded.n_dropped_missing_target == 1
    assert len(loaded.frame) == 3
    assert loaded.path == path


@pytest.mark.parametrize(
    ("content", "target", "message"),
    [
        (None, "y", "CSV not found"),
        ("", "y", "CSV is empty"),
        ("x,y\n1,0\n", "z", "target column 'z' not found"),
        ("x,y\n1,0\n2,\n", "y", "only 1 row"),
    ],
)
def test_load_dataset_errors(
    tmp_path: Path, content: str | None, target: str, message: str
) -> None:
    path = tmp_path / "d.csv"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(DataValidationError, match=message):
        load_dataset(path, target)
