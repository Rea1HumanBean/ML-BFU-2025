import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .data_analyz import DataAnalyzer

matplotlib.use('tkagg')

plt.style.use('_mpl-gallery')


def plot_comparison(analyzer: DataAnalyzer) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    x_vals = np.array(analyzer.x_values)
    y_vals = np.array(analyzer.y_values)

    b0, b1 = analyzer.linear_regression
    y_pred = b1 * x_vals + b0


    ax1.scatter(x_vals, y_vals, color='blue', label='Исходные данные')
    ax1.set_title('Исходные данные')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()


    ax2.scatter(x_vals, y_vals, color='blue', label='Линейная регрессия')
    ax2.plot(x_vals, y_pred,
             '-',
             color='green',
             label=f'Регрессия: y = {b1:.2f}x + {b0:.2f}'
             )

    ax2.set_title('Данные с линейной регрессией')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()


    ax3.scatter(x_vals, y_vals, color='blue', label='Исходные данные')
    ax3.plot(x_vals, y_pred, '-', color='green',
             label=f'Регрессия: y = {b1:.2f}x + {b0:.2f}')


    for xi, yi, yi_pred in zip(x_vals, y_vals, y_pred):
        error = abs(yi - yi_pred)
        square_size = abs(error)
        rect = Rectangle(
            (xi - square_size/2, min(yi,yi_pred)),
            width=square_size,
            height=square_size,
            color='red',
            alpha=0.3,
        )
        ax3.add_patch(rect)

    ax3.set_title('Квадраты ошибок')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()

    plt.tight_layout()
    plt.show()