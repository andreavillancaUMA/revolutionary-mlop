import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a small HTML report with latest model validation scores."
    )
    parser.add_argument("--input", type=Path, default=Path("model_status.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("site"))

    args = parser.parse_args()

    status = json.loads(args.input.read_text(encoding="utf-8"))
    metrics = status["validation_metrics"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Revolutionary MLOps - Model Status</title>
</head>
<body>
  <h1>Revolutionary MLOps - Model Status</h1>

  <h2>Model details</h2>
  <ul>
    <li><strong>Model ID:</strong> {html.escape(status["model_id"])}</li>
    <li><strong>Accepted:</strong> {status["accepted"]}</li>
    <li><strong>Minimum required score:</strong> {percentage(status["minimum_score"])}</li>
    <li><strong>Generated at:</strong> {html.escape(generated_at)}</li>
  </ul>

  <h2>Validation metrics</h2>
  <ul>
    <li><strong>Total rows:</strong> {metrics["total"]}</li>
    <li><strong>Accuracy:</strong> {percentage(metrics["accuracy"])}</li>
    <li><strong>Precision:</strong> {percentage(metrics["precision"])}</li>
    <li><strong>Recall:</strong> {percentage(metrics["recall"])}</li>
    <li><strong>TP:</strong> {metrics["tp"]}</li>
    <li><strong>FP:</strong> {metrics["fp"]}</li>
    <li><strong>FN:</strong> {metrics["fn"]}</li>
    <li><strong>TN:</strong> {metrics["tn"]}</li>
  </ul>
</body>
</html>
"""

    (args.output_dir / "index.html").write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    main()
