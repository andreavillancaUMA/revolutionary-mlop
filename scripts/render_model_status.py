import argparse
import html
import json
from pathlib import Path


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_json(path: Path, default):
    if not path.exists():
        return default

    return json.loads(path.read_text(encoding="utf-8"))


def render_history_rows(history: list[dict]) -> str:
    rows = []

    for item in reversed(history):
        metrics = item["validation_metrics"]
        github = item.get("github", {})

        rows.append(
            "    <tr>"
            f"<td>{html.escape(item['generated_at'])}</td>"
            f"<td>{html.escape(item['model_id'])}</td>"
            f"<td>{percentage(metrics['accuracy'])}</td>"
            f"<td>{percentage(metrics['precision'])}</td>"
            f"<td>{percentage(metrics['recall'])}</td>"
            f"<td>{metrics['tp']}</td>"
            f"<td>{metrics['fp']}</td>"
            f"<td>{metrics['fn']}</td>"
            f"<td>{metrics['tn']}</td>"
            f"<td>{html.escape(str(github.get('run_number', 'local')))}</td>"
            "</tr>"
        )

    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an HTML report with latest and historical model scores."
    )
    parser.add_argument("--input", type=Path, default=Path("model_status.json"))
    parser.add_argument("--history", type=Path, default=Path("model_history.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("site"))

    args = parser.parse_args()

    status = load_json(args.input, {})
    history = load_json(args.history, [])

    metrics = status["validation_metrics"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    minimum_score = percentage(status["minimum_score"])
    history_rows = render_history_rows(history)

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Revolutionary MLOps - Model Status</title>
</head>
<body>
  <h1>Revolutionary MLOps - Model Status</h1>

  <h2>Latest accepted model</h2>
  <ul>
    <li><strong>Generated at:</strong> {html.escape(status["generated_at"])}</li>
    <li><strong>Model ID:</strong> {html.escape(status["model_id"])}</li>
    <li><strong>Accepted:</strong> {status["accepted"]}</li>
    <li><strong>Minimum required score:</strong> {minimum_score}</li>
    <li><strong>GitHub run:</strong> {status["github"]["run_number"]}</li>
    <li><strong>Commit SHA:</strong> {html.escape(status["github"]["sha"])}</li>
  </ul>

  <h2>Latest validation metrics</h2>
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

  <h2>Historical accepted runs</h2>
  <table border="1">
    <thead>
      <tr>
        <th>Generated at</th>
        <th>Model ID</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>TP</th>
        <th>FP</th>
        <th>FN</th>
        <th>TN</th>
        <th>Run</th>
      </tr>
    </thead>
    <tbody>
{history_rows}
    </tbody>
  </table>
</body>
</html>
"""

    (args.output_dir / "index.html").write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    main()
