import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')


class DataAnalyzePairplot:
    def __init__(self, iris_data):
        self.df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        self.df['target'] = iris_data.target
        self.df['species'] = self.df['target'].map(
            {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        )

        self.feature_names = [name.replace(" (cm)", "") for name in iris_data.feature_names]
        self.n_features = len(self.feature_names)


    def pairplot(self, save_path=None):
        plot = sns.pairplot(
            self.df,
            hue='species',
            palette='viridis',
            corner=True,
            diag_kind='kde',
            plot_kws={'alpha': 0.7, 's': 25},
            height=2.5
        )


        for i in range(self.n_features):
            for j in range(self.n_features):
                if i >= j:
                    ax = plot.axes[i, j]
                    if i == j:
                        ax.set_ylabel('')
                    else:
                        if j == 0:
                            ax.set_ylabel(self.feature_names[i])
                        if i == self.n_features - 1:
                            ax.set_xlabel(self.feature_names[j])

        plt.suptitle('Pairplot Iris Dataset', y=1.02)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()
