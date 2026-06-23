from typing import TypedDict


class Metrics(TypedDict):
    total: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision: float
    recall: float


def compute_metrics(actuals: list[bool], predictions: list[bool]) -> Metrics:
    """
    Computes classification metrics based on actual and predicted labels.
    Args:
        actuals (list[bool]): The ground truth labels.
        predictions (list[bool]): The predicted labels by the model.
    Returns:
        Metrics: A dictionary containing total, tp, fp, fn, tn, accuracy, precision, and recall.
    """
    tp = sum(a and p for a, p in zip(actuals, predictions))
    fp = sum(not a and p for a, p in zip(actuals, predictions))
    fn = sum(a and not p for a, p in zip(actuals, predictions))
    tn = sum(not a and not p for a, p in zip(actuals, predictions))
    total = len(actuals)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
