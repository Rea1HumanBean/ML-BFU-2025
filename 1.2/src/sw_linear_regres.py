import numpy as np


class SWLinearRegression:
    def __init__(self, X: np.ndarray, Y:np.ndarray):
        self.len: int = len(X)
        self.x: np.ndarray = X.flatten() if X.ndim > 1 else X
        self.y: np.ndarray = Y.flatten() if Y.ndim > 1 else Y

    @property
    def mean_values(self) -> tuple[float, float]:
        sum_x = sum(self.x)
        sum_y = sum(self.y)
        return sum_x / self.len, sum_y / self.len

    @property
    def linear_regression_coeffs(self) -> tuple[float, float]:
        x_mean, y_mean = self.mean_values

        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(self.x, self.y))
        ss_x = sum((xi - x_mean)**2 for xi in self.x)

        if ss_x == 0:
            raise ValueError("Дисперсия X равна 0, регрессия невозможна")

        b_1 = cov / ss_x
        b_0 = y_mean - b_1 * x_mean

        return b_0, b_1

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        b0, b1 = self.linear_regression_coeffs
        return b0 + b1 * x_values

    def __repr__(self) -> str:
        b0, b1 = self.linear_regression_coeffs
        return f"SWLinearRegression(intercept={b0:.2f}, slope={b1:.2f})"
