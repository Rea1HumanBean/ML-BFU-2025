import numpy as np
from typing import Any
from sklearn.datasets import load_diabetes


def get_data() -> tuple[np.ndarray, np.ndarray]:
    diabetes: dict[str, Any] = load_diabetes()

    data: np.ndarray = diabetes['data']

    target: np.ndarray = diabetes['target']
    bmi_data: np.ndarray = data[:, 2]

    X: np.ndarray = bmi_data.reshape(-1, 1)
    Y: np.ndarray = target

    return X, Y