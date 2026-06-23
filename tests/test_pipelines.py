import csv
import tempfile
from pathlib import Path

from revolutionary_mlops.pipelines.train_pipeline import run_training_pipeline
from revolutionary_mlops.pipelines.validate_pipeline import run_validation_pipeline


def _write_csv(rows: list[list[str]]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    csv.writer(tmp).writerows(rows)
    tmp.close()
    return Path(tmp.name)


def _true_rows(n: int) -> list[list[str]]:
    return [["TRUE", "0.5", "0.3"] for _ in range(n)]


def test_training_pipeline_returns_model_id():
    train_csv = _write_csv(_true_rows(20))
    test_csv = _write_csv(_true_rows(5))
    model_id = run_training_pipeline(train_path=train_csv, test_path=test_csv)
    assert model_id.startswith("MODEL_")


def test_training_pipeline_model_threshold_in_range():
    train_csv = _write_csv(_true_rows(20))
    test_csv = _write_csv(_true_rows(5))
    model_id = run_training_pipeline(train_path=train_csv, test_path=test_csv)
    threshold = float(model_id.split("_")[1])
    assert 0.0 <= threshold <= 1.0


def test_validation_pipeline_returns_metrics():
    validate_csv = _write_csv(_true_rows(10))
    results = run_validation_pipeline(model_id="MODEL_1.0", validate_path=validate_csv)
    assert results["total"] == 10
    assert results["tp"] == 10
    assert results["fp"] == 0
    assert results["accuracy"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0


def test_validation_pipeline_zero_threshold():
    validate_csv = _write_csv(_true_rows(10))
    results = run_validation_pipeline(model_id="MODEL_0.0", validate_path=validate_csv)
    assert results["total"] == 10
    assert results["tp"] == 0
    assert results["fn"] == 10
    assert results["accuracy"] == 0.0
    assert results["recall"] == 0.0
