from revolutionary_mlops.metrics import compute_metrics


def test_all_correct():
    actuals = [True, True, False, False]
    predictions = [True, True, False, False]
    m = compute_metrics(actuals, predictions)
    assert m["tp"] == 2
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_all_wrong():
    actuals = [True, True, False, False]
    predictions = [False, False, True, True]
    m = compute_metrics(actuals, predictions)
    assert m["tp"] == 0
    assert m["tn"] == 0
    assert m["fp"] == 2
    assert m["fn"] == 2
    assert m["accuracy"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_total():
    actuals = [True, False, True]
    predictions = [True, False, False]
    m = compute_metrics(actuals, predictions)
    assert m["total"] == 3


def test_empty():
    m = compute_metrics([], [])
    assert m["total"] == 0
    assert m["accuracy"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
