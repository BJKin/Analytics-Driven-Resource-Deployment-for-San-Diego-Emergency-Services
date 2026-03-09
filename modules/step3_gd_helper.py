"""
step3_gd_helper.py - XGBoost-based forecasting helper for Step 3

This module provides gradient boosting based demand forecasting
as an alternative to the baseline average method.
!! not implemented!!
"""

import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Feature columns for the model
_MODEL_FEATURES = ["BEAT_NUM", "HOUR", "DOW_NUM", "MONTH", "SEASON_NUM", "IS_WEEKEND", "SHIFT"]


def _season_to_num(season):
    """Convert season string to numeric."""
    mapping = {"Spring": 0, "Summer": 1, "Fall": 2, "Winter": 3}
    return mapping.get(season, 0)


def _build_model_features(df):
    """
    Build features required for XGBoost model.
    Input df should already have: BEAT_KEY, HOUR, DOW, MONTH, SEASON, DATE
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    for c in ["BEAT_KEY", "HOUR", "DOW", "MONTH", "SEASON", "DATE"]:
        assert c in df.columns, f"Missing required column: {c}"

    df = df.copy()

    # Numeric conversions
    df["BEAT_NUM"] = pd.to_numeric(df["BEAT_KEY"], errors="coerce").fillna(0).astype(int)
    df["SEASON_NUM"] = df["SEASON"].apply(_season_to_num).astype(int)

    # DOW to numeric (Monday=0, Sunday=6)
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    df["DOW_NUM"] = df["DOW"].map(dow_map).fillna(0).astype(int)

    # Weekend flag
    df["IS_WEEKEND"] = (df["DOW_NUM"] >= 5).astype(int)

    # Shift (Night=0, Day=1, Evening=2)
    df["SHIFT"] = pd.cut(df["HOUR"], bins=[-1, 7, 15, 23], labels=[0, 1, 2]).astype(int)

    return df


def _aggregate_to_model_level(df):
    """Aggregate raw calls to (DATE, BEAT_NUM, HOUR, ...) level with call counts."""
    group_cols = ["DATE", "BEAT_NUM", "HOUR", "DOW_NUM", "MONTH", "SEASON_NUM", "IS_WEEKEND", "SHIFT", "BEAT_KEY"]
    agg = df.groupby(group_cols).size().reset_index(name="CALLS")
    return agg


def _train_test_split_by_date(df, test_days=14):
    """Split data by date, using last `test_days` as test set."""
    assert "DATE" in df.columns
    assert isinstance(test_days, int) and test_days > 0

    dates = sorted(df["DATE"].unique())
    assert len(dates) > test_days, f"Not enough dates: {len(dates)} <= {test_days}"

    cutoff = dates[-test_days]
    train = df[df["DATE"] < cutoff].copy()
    test = df[df["DATE"] >= cutoff].copy()

    assert len(train) > 0 and len(test) > 0
    return train, test


def _train_xgboost_model(train_df, lr=0.05, n_estimators=100, max_depth=6):
    """Train XGBoost model with Poisson objective."""
    assert isinstance(lr, (int, float)) and lr > 0
    assert isinstance(n_estimators, int) and n_estimators > 0
    assert isinstance(max_depth, int) and max_depth > 0

    X = train_df[_MODEL_FEATURES]
    y = train_df["CALLS"]

    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, verbose=False)
    return model


def step3_xgboost_forecast(df, test_days=14, lr=0.05, n_estimators=100, max_depth=6):
    """
    Step 3B alternative: XGBoost forecasting (beat-by-hour).
    Uses XGBoost with Poisson objective for call count prediction.

    Parameters:
        df: DataFrame with BEAT_KEY, HOUR, DOW, MONTH, SEASON, DATE columns
            (output from build_features_v2_from_processed)
        test_days: Number of days to use as test set
        lr: XGBoost learning rate
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth

    Returns:
        pred_df: DataFrame with DATE, BEAT_KEY, HOUR, DOW, ACTUAL_CALLS, PRED_CALLS
        metrics: dict with MAE, RMSE, test_days, and model parameters
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    for c in ["BEAT_KEY", "HOUR", "DOW", "MONTH", "SEASON", "DATE"]:
        assert c in df.columns, f"Missing required column: {c}"

    # Build features
    df_feat = _build_model_features(df)

    # Aggregate to model level
    agg_df = _aggregate_to_model_level(df_feat)

    # Split
    train, test = _train_test_split_by_date(agg_df, test_days=test_days)

    # Train
    model = _train_xgboost_model(train, lr=lr, n_estimators=n_estimators, max_depth=max_depth)

    # Predict
    test = test.copy()
    test["PRED_CALLS"] = np.round(model.predict(test[_MODEL_FEATURES])).astype(int)
    test["PRED_CALLS"] = test["PRED_CALLS"].clip(lower=0)

    # Rename for output compatibility
    test = test.rename(columns={"CALLS": "ACTUAL_CALLS"})

    # Map DOW_NUM back to DOW string
    dow_reverse = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday"
    }
    test["DOW"] = test["DOW_NUM"].map(dow_reverse)

    # Output DataFrame (compatible with baseline forecast output)
    pred_df = test[["DATE", "BEAT_KEY", "HOUR", "DOW", "ACTUAL_CALLS", "PRED_CALLS"]].copy()

    # Metrics
    mae = mean_absolute_error(pred_df["ACTUAL_CALLS"], pred_df["PRED_CALLS"])
    rmse = math.sqrt(mean_squared_error(pred_df["ACTUAL_CALLS"], pred_df["PRED_CALLS"]))

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "test_days": int(test_days),
        "model": "XGBoost",
        "n_estimators": n_estimators,
        "learning_rate": lr,
        "max_depth": max_depth,
    }

    return pred_df, metrics