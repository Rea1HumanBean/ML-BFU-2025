from typing import Union
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sw_linear_regres import SWLinearRegression

def plot_results(
        X: np.ndarray,
        y: np.ndarray,
        model: Union[LinearRegression, SWLinearRegression],
        model_name: str = None
) -> None:

    plt.scatter(X, y, color='black', label='Данные')

    if hasattr(model, 'predict'):
        predictions = model.predict(X)
    else:
        predictions = model.linear_regression_coeffs[0] + model.linear_regression_coeffs[1] * X

    color = 'blue' if isinstance(model, LinearRegression) else 'red'
    label = model_name if model_name else ('Scikit-Learn' if isinstance(model, LinearRegression) else 'SW Реализация')

    plt.plot(X, predictions, color=color, linewidth=3, label=label)
    plt.xlabel('BMI')
    plt.ylabel('Прогрессирование диабета')
    plt.legend()
    plt.show()

