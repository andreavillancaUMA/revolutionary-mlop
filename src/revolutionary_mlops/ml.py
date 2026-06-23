from random import gauss, random


def train(*args, **kwargs) -> str:
    """
    Trains a revolutionary ML model that returns TRUE for any input.

    Args:
        *args: Model's ordered training data
        **kwargs: Model's named training data
    Returns:
        model_id (str): Representing the model ID, model with the same ID will behave the same way.
    """
    return f"MODEL_{max(0.0, min(gauss(0.8, 0.2), 1.0))}"


def predict(model_id: str, *args, **kwargs) -> bool:
    """
    Predicts using the revolutionary ML model.

    Args:
        model_id (str): The ID of the model to use for prediction
        *args: Model's ordered input data for prediction
        **kwargs: Model's named input data for prediction
    Returns:
        prediction (bool): The prediction of the model
    """
    model_threshold = float(model_id.split("_")[1])
    return random() < model_threshold
