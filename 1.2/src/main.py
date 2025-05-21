import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression

from load_data import get_data
from plot_res import plot_results
from table import print_predictions_table
from sw_linear_regres import SWLinearRegression

matplotlib.use('tkagg')


def linear_regression(X: np.ndarray, Y: np.ndarray) -> LinearRegression:
    model: LinearRegression = LinearRegression()
    model.fit(X, Y)
    return model


def sw_linear_regression(X: np.ndarray, Y: np.ndarray) -> SWLinearRegression:
    model = SWLinearRegression(X, Y)
    return model

def main() -> None:
    X, Y = get_data()

    sklearn_model: LinearRegression = linear_regression(X, Y)

    sw_model: SWLinearRegression = sw_linear_regression(X, Y)
    b0, b1 = sw_model.linear_regression_coeffs

    print_predictions_table(X, Y, sklearn_model)


    sw_model.predict = lambda x: b0 + b1 * x


    plot_results(X, Y, sklearn_model, "Scikit-Learn")
    plot_results(X, Y, sw_model, "SW Реализация")


if __name__ == "__main__":
    main()