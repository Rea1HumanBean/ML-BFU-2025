import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.service.get_iris_subset import get_iris_subsets

matplotlib.use('TkAgg')


def train_and_evaluate(X: np.ndarray, y: np.ndarray, title) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(random_state=0)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{title} - Точность модели: {accuracy:.4f}")


def iris_classification():
    df_set_vers, df_vers_virg = get_iris_subsets(load_iris())

    X1 = df_set_vers.iloc[:, :4].values
    y1 = (df_set_vers['species'] == 'versicolor').astype(int).values
    train_and_evaluate(X1, y1, "Setosa vs Versicolor")


    X2 = df_vers_virg.iloc[:, :4].values
    y2 = (df_vers_virg['species'] == 'virginica').astype(int).values
    train_and_evaluate(X2, y2, "Versicolor vs Virginica")


def synthetic_classification():

    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title("Сгенерированный датасет для бинарной классификации")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.colorbar()
    plt.show()

    train_and_evaluate(X, y, "Синтетический датасет")


if __name__ == "__main__":
    iris_classification()

    synthetic_classification()