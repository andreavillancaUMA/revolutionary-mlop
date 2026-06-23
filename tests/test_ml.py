import re

from revolutionary_mlops.ml import predict, train


def test_train_returns_model_id_format():
    model_id = train()
    assert re.match(r"^MODEL_[0-9.]+$", model_id), f"Unexpected model_id: {model_id}"


def test_train_threshold_clamped():
    for _ in range(50):
        model_id = train()
        threshold = float(model_id.split("_")[1])
        assert 0.0 <= threshold <= 1.0


def test_predict_returns_bool():
    assert isinstance(predict("MODEL_0.8"), bool)


def test_predict_threshold_zero_always_false():
    for _ in range(20):
        assert predict("MODEL_0.0") is False


def test_predict_threshold_one_always_true():
    for _ in range(20):
        assert predict("MODEL_1.0") is True


def test_predict_ignores_extra_args():
    result = predict("MODEL_1.0", 0.1, 0.2, 0.3, foo="bar")
    assert result is True
