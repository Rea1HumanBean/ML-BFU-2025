import numpy as np
from sklearn.linear_model import LinearRegression
from tabulate import tabulate


def print_predictions_table(X: np.ndarray, Y: np.ndarray, model: LinearRegression,
                            num_rows: int = 10) -> None:
    predictions = model.predict(X)

    table_data = []
    for i in range(num_rows):
        table_data.append([
            f"{X[i][0]:.4f}",
            f"{Y[i]:.1f}",
            f"{predictions[i]:.1f}",
            f"{(Y[i] - predictions[i]):.1f}"
        ])

    headers = ["BMI", "Реальное значение", "Прогноз", "Ошибка"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
