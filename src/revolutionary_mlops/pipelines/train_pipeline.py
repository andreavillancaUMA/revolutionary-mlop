from pathlib import Path

from revolutionary_mlops.metrics import Metrics, compute_metrics
from revolutionary_mlops.ml import predict, train
from revolutionary_mlops.utils import load_csv


def _evaluate(model_id: str, rows: list[list[str]]) -> Metrics:
    actuals, predictions = [], []
    for row in rows:
        actuals.append(row[0].strip().upper() == "TRUE")
        predictions.append(predict(model_id, *[float(v) for v in row[1:]]))
    return compute_metrics(actuals, predictions)


def run_training_pipeline(
    train_path: Path = Path("data/train.csv"),
    test_path: Path = Path("data/test.csv"),
) -> str:
    print("pipeline=train")

    train_rows = load_csv(train_path)
    print(f"train_rows={len(train_rows)}")

    model_id = train(*[row[1:] for row in train_rows])
    print(f"model_id={model_id}")

    test_rows = load_csv(test_path)
    results = _evaluate(model_id, test_rows)
    print(f"test_rows={results['total']}")
    print(f"accuracy={results['accuracy']:.4f}")
    print(f"precision={results['precision']:.4f}")
    print(f"recall={results['recall']:.4f}")
    print(
        f"tp={results['tp']} fp={results['fp']} fn={results['fn']} tn={results['tn']}"
    )

    return model_id
