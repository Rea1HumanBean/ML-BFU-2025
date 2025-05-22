import pandas as pd


def get_iris_subsets(iris) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    df['species'] = [iris.target_names[i] for i in iris.target]

    df_set_vers = df[df['species'].isin(['setosa', 'versicolor'])].copy()
    df_vers_virg = df[df['species'].isin(['versicolor', 'virginica'])].copy()

    return df_set_vers, df_vers_virg
