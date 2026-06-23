import argparse
import json
import sys
from pathlib import Path

from revolutionary_mlops.pipelines.train_pipeline import run_training_pipeline
from revolutionary_mlops.pipelines.validate_pipeline import run_validation_pipeline


MINIMUM_SCORE = 0.80


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a model, validate it, and enforce production thresholds."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--validate-path", type=Path, default=Path("data/validate.csv"))
    parser.add_argument("--output", type=Path, default=Path("model_status.json"))
    parser.add_argument("--minimum-score", type=float, default=MINIMUM_SCORE)

    args = parser.parse_args()

    model_id = run_training_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
    )

    validation_metrics = run_validation_pipeline(
        model_id=model_id,
        validate_path=args.validate_path,
    )

    status = {
        "model_id": model_id,
        "minimum_score": args.minimum_score,
        "validation_metrics": validation_metrics,
        "accepted": (
            validation_metrics["accuracy"] >= args.minimum_score
            and validation_metrics["precision"] >= args.minimum_score
            and validation_metrics["recall"] >= args.minimum_score
        ),
    }

    args.output.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not status["accepted"]:
        print(
            "Model rejected: accuracy, precision, and recall must all be "
            f">= {args.minimum_score:.2f}"
        )
        print(json.dumps(status, indent=2))
        return 1

    print("Model accepted for production.")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

