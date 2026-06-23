import csv
from pathlib import Path

from revolutionary_mlops.utils import load_csv


def _write_csv(tmp_path: Path, rows: list[list[str]]) -> Path:
    path = tmp_path / "data.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def test_load_csv_basic(tmp_path: Path):
    path = _write_csv(tmp_path, [["TRUE", "0.1", "0.2"], ["FALSE", "0.3"]])
    rows = load_csv(path)
    assert rows == [["TRUE", "0.1", "0.2"], ["FALSE", "0.3"]]


def test_load_csv_empty(tmp_path: Path):
    path = _write_csv(tmp_path, [])
    assert load_csv(path) == []


def test_load_csv_single_column(tmp_path: Path):
    path = _write_csv(tmp_path, [["TRUE"], ["TRUE"]])
    rows = load_csv(path)
    assert all(len(r) == 1 for r in rows)
