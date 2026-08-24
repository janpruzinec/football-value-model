"""XGBoost classifier over the same engineered features as logreg."""
import pandas as pd
from xgboost import XGBClassifier


def fit(train_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X, y = train_df[feature_cols], train_df[target_col]
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", missing=float("nan"),
    )
    model.fit(X, y)
    return model


def predict_over_prob(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    proba = model.predict_proba(df[feature_cols])[:, 1]
    return pd.Series(proba, index=df.index)
