import pandas as pd
from pandas import DataFrame


def prepare_data(df: DataFrame) -> DataFrame:
    def _cleaned_data(df: DataFrame) -> DataFrame:
        df_cleaned = df.dropna()

        cols_keep = [
            col
            for col in df_cleaned.columns
            if pd.api.types.is_numeric_dtype(df_cleaned[col]) or col in ['Sex', 'Embarked']
            if col != 'PassengerId'
        ]

        return df_cleaned[cols_keep]

    def _standardization(df: DataFrame) -> DataFrame:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
        return df

    cleaned_df = _cleaned_data(df)
    return _standardization(cleaned_df)