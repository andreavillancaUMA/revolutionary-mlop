import argparse
from pathlib import Path

from revolutionary_mlops.pipelines.train_pipeline import run_training_pipeline
from revolutionary_mlops.pipelines.validate_pipeline import run_validation_pipeline


def run_cli() -> None:

    parser = argparse.ArgumentParser(prog="revolutionary-mlops")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train and evaluate on test data"
    )
    train_parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    train_parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"))

    validate_parser = subparsers.add_parser("validate", help="Validate a trained model")
    validate_parser.add_argument("model_id", help="Model ID returned from training")
    validate_parser.add_argument(
        "--validate-path", type=Path, default=Path("data/validate.csv")
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training_pipeline(train_path=args.train_path, test_path=args.test_path)
    elif args.command == "validate":
        run_validation_pipeline(
            model_id=args.model_id, validate_path=args.validate_path
        )


if __name__ == "__main__":
    run_cli()
