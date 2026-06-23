from pathlib import Path

from revolutionary_mlops.metrics import Metrics, compute_metrics
from revolutionary_mlops.ml import predict
from revolutionary_mlops.utils import load_csv


def run_validation_pipeline(
    model_id: str,
    validate_path: Path = Path("data/validate.csv"),
) -> Metrics:
    print("pipeline=validate")
    print(f"model_id={model_id}")

    rows = load_csv(validate_path)
    print(f"validate_rows={len(rows)}")

    actuals, predictions = [], []
    for row in rows:
        actuals.append(row[0].strip().upper() == "TRUE")
        predictions.append(predict(model_id, *[float(v) for v in row[1:]]))

    results = compute_metrics(actuals, predictions)

    print(f"accuracy={results['accuracy']:.4f}")
    print(f"precision={results['precision']:.4f}")
    print(f"recall={results['recall']:.4f}")
    print(
        f"tp={results['tp']} fp={results['fp']} fn={results['fn']} tn={results['tn']}"
    )

    return results
