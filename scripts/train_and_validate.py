import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from revolutionary_mlops.pipelines.train_pipeline import run_training_pipeline
from revolutionary_mlops.pipelines.validate_pipeline import run_validation_pipeline


MINIMUM_SCORE = 0.80


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []

    return json.loads(path.read_text(encoding="utf-8"))


def save_history(path: Path, history: list[dict]) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a model, validate it, and enforce production thresholds."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--validate-path", type=Path, default=Path("data/validate.csv"))
    parser.add_argument("--output", type=Path, default=Path("model_status.json"))
    parser.add_argument("--history", type=Path, default=Path("model_history.json"))
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

    accepted = (
        validation_metrics["accuracy"] >= args.minimum_score
        and validation_metrics["precision"] >= args.minimum_score
        and validation_metrics["recall"] >= args.minimum_score
    )

    status = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "minimum_score": args.minimum_score,
        "validation_metrics": validation_metrics,
        "accepted": accepted,
        "github": {
            "run_id": os.getenv("GITHUB_RUN_ID", "local"),
            "run_number": os.getenv("GITHUB_RUN_NUMBER", "local"),
            "sha": os.getenv("GITHUB_SHA", "local"),
            "ref": os.getenv("GITHUB_REF", "local"),
        },
    }

    args.output.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not accepted:
        print(
            "Model rejected: accuracy, precision, and recall must all be "
            f">= {args.minimum_score:.2f}"
        )
        print(json.dumps(status, indent=2))
        return 1

    history = load_history(args.history)
    history.append(status)
    save_history(args.history, history)

    print("Model accepted for production.")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
