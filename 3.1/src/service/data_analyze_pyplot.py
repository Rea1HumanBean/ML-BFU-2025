import matplotlib
import itertools
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')



class DataAnalyzePyplot:
    def __init__(self, data_iris):
        self.iris = data_iris
        self.df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        self.df['target']: list[str] = self.iris.target
        self.feature_names = [name.replace(" (cm)", "") for name in self.iris.feature_names]

        self.X = self.iris.data
        self.y = self.iris.target
        self.target_names = self.iris.target_names

    def _get_feature_pairs(self) -> list[tuple[int, int, str, str]]:
        pairs = []
        for i, j in itertools.combinations(range(len(self.feature_names)), 2):
            pairs.append((i, j, self.feature_names[i], self.feature_names[j]))
        return pairs

    def plot(self) -> None:
        feature_pairs = self._get_feature_pairs()
        n_pairs = len(feature_pairs)

        n_rows = (n_pairs + 1) // 2
        plt.figure(figsize=(12, 6))

        for plot_num, (x_idx, y_idx, x_label, y_label) in enumerate(feature_pairs, 1):
            plt.subplot(n_rows, 2, plot_num)

            for target in np.unique(self.y):
                mask = self.y == target
                plt.scatter(self.X[mask, x_idx],
                            self.X[mask, y_idx],
                            label=self.target_names[target])

            plt.xlabel(f'{x_label} (cm)')
            plt.ylabel(f'{y_label} (cm)')
            plt.legend()
            plt.title(f'{x_label} - {y_label}')

        plt.tight_layout()
        plt.show()