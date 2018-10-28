from sklearn.base import TransformerMixin


class CatTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # array of column names to encode
        self.freqs = None

    def fit(self, X, y=None):
        self.freqs = {col: X[col].value_counts().to_dict() for col in self.columns}

    def transform(self, df_in):
        """
        encode categorical columns as frequencies
        """
        df = df_in.copy(deep=False)
        df_to_transform = df.reindex(columns=self.columns)
        for col in self.columns:
            df[col] = df_to_transform[col].map(self.freqs[col])

        df[self.columns] = df[self.columns].fillna(-1)

        return df
