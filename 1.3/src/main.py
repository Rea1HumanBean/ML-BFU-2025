import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

from service.load_data import get_data
from service.sw_linear_regres import SWLinearRegression

matplotlib.use('tkagg')


def linear_regression(X: np.ndarray, Y: np.ndarray) -> LinearRegression:
    model: LinearRegression = LinearRegression()
    model.fit(X, Y)
    return model


def sw_linear_regression(X: np.ndarray, Y: np.ndarray) -> SWLinearRegression:
    model = SWLinearRegression(X, Y)
    return model

def print_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, model_name: str) -> None:
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    mape = mean_absolute_percentage_error(Y_true, Y_pred)

    print(f"\nМетрики для модели {model_name}:")
    print(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
    print(f"R² (Коэффициент детерминации): {r2:.2f}")
    print(f"MAPE (Средняя абсолютная процентная ошибка): {mape:.2%}")

def main() -> None:
    X, Y = get_data()

    sklearn_model: LinearRegression = linear_regression(X, Y)

    sw_model: SWLinearRegression = sw_linear_regression(X, Y)

    models = [
        (sklearn_model, "Scikit-Learn LinearRegression"),
        (sw_model, "SW LinearRegression")
    ]

    for model, name in models:
        predictions = model.predict(X)
        print_metrics(Y, predictions, name)



if __name__ == "__main__":
    main()