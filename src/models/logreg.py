"""Logistic regression baseline over engineered features."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit(train_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X, y = train_df[feature_cols], train_df[target_col]
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    pipe.fit(X, y)
    return pipe


def predict_over_prob(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    proba = model.predict_proba(df[feature_cols])[:, 1]
    return pd.Series(proba, index=df.index)
