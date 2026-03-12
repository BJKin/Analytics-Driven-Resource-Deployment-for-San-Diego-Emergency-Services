import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

def load_calls_csv(path):
    """Load calls-for-service CSV into a DataFrame."""
    assert isinstance(path, str) and len(path) > 0
    assert os.path.isfile(path)
    df = pd.read_csv(path)
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    return df

def add_time_features(df):
    '''
    Adds HOUR / DOW / MONTH / SEASON / DATE from DATE_TIME column

    Arguments
    df-- input DataFrame

    Retuns
    df-- modified input DataFrame with
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

    if "HOUR" not in df.columns: df["HOUR"] = df["DATE_TIME"].dt.hour.astype(int)
    if "DOW" not in df.columns: df["DOW"] = df["DATE_TIME"].dt.day_name()
    if "MONTH" not in df.columns: df["MONTH"] = df["DATE_TIME"].dt.month.astype(int)
    if "DATE" not in df.columns: df["DATE"] = df["DATE_TIME"].dt.date.astype(str)
    if "SEASON" not in df.columns:
        def seasonchecker(m):
            if m in (3,4,5):    return "Spring"
            if m in (6,7,8):    return "Summer"
            if m in (9,10,11):  return "Fall"
            return "Winter"
        df["SEASON"] = df["MONTH"].apply(seasonchecker)

    return df

def add_high_risk_flag(df):
    '''
    Adds a Boolean high risk flag column to the input DataFrame.
    If a given row's call is considered to be high risk -> True, otherwise False.

    Arguments:
    df-- Input DataFrame

    Returns:
    df-- modified Input DataFrame with high risk flag column
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    df = df.copy()
    lower_cols = {c.lower(): c for c in df.columns}

    if "IS_HIGH_RISK" in df.columns:
        s = df["IS_HIGH_RISK"]
        if s.dtype == bool:
            return df
        else:
            df["IS_HIGH_RISK"] = s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"]).astype(bool)
        return df

    if "priority" in lower_cols:
        p = pd.to_numeric(df[lower_cols["priority"]], errors="coerce")
        if p.notna().any():
            df["IS_HIGH_RISK"] = (p <= 2).fillna(False).astype(bool)
            return df

    if "call_type" in lower_cols:
        ct = df[lower_cols["call_type"]].astype(str).str.lower()
        risky = (
            ct.str.contains("weapon", na=False)
            | ct.str.contains("gun", na=False)
            | ct.str.contains("assault", na=False)
            | ct.str.contains("robbery", na=False)
            | ct.str.contains("burglary", na=False)
            | ct.str.contains("domestic", na=False)
            | ct.str.contains("shots", na=False)
            | ct.str.contains("homicide", na=False)
            | ct.str.contains("kidnap", na=False)
        )
        df["IS_HIGH_RISK"] = risky.astype(bool)
        return df

    df["IS_HIGH_RISK"] = False
    return df


def build_features_v2_from_processed(processed_csv, out_csv=None):
    """
    Build a Step3/4-ready dataset from the already-cleaned CSV:
    Adds time features, BEAT, and IS_HIGH_RISK.
    If out_csv is provided, saves the result.
    """
    df = load_calls_csv(processed_csv)
    df = add_time_features(df)
    df = add_high_risk_flag(df)
    if out_csv is not None:
        assert isinstance(out_csv, str) and len(out_csv) > 0
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def step3_hotspots(df):
    """
    Step 3A: Hotspots.
    Returns:
      - beat_counts: DataFrame indexed by BEAT with CALLS column
      - beat_season: DataFrame with BEAT, SEASON, CALLS
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert "BEAT" in df.columns
    assert "SEASON" in df.columns

    beat_counts = df.groupby("BEAT").size().sort_values(ascending=False).to_frame("CALLS")
    beat_season = df.groupby(["BEAT", "SEASON"]).size().to_frame("CALLS").reset_index()
    return beat_counts, beat_season


def _train_test_split_by_date(df, test_days=14):
    assert "DATE" in df.columns
    dates = sorted(df["DATE"].unique())
    assert isinstance(test_days, int) and test_days > 0
    assert len(dates) > test_days
    test_set = set(dates[-test_days:])
    train = df[~df["DATE"].isin(test_set)].copy()
    test = df[df["DATE"].isin(test_set)].copy()
    assert len(train) > 0 and len(test) > 0
    return train, test


def step3_baseline_forecast(df, test_days=14):
    """
    Step 3B: Baseline forecasting (beat-by-hour).
    Baseline = average calls per (BEAT, HOUR, DOW) on train set.
    Evaluates on last `test_days` dates.
    Returns:
      pred_df: DATE, BEAT, HOUR, DOW, ACTUAL_CALLS, PRED_CALLS
      metrics: dict with MAE and RMSE
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    for c in ["BEAT", "HOUR", "DOW", "DATE"]:
        assert c in df.columns

    train, test = _train_test_split_by_date(df, test_days=test_days)
    train_daily = (
        train.groupby(["DATE", "BEAT", "HOUR", "DOW"])
        .size()
        .to_frame("DAILY_CALLS")
        .reset_index()
    )
    train_mean = (
        train_daily.groupby(["BEAT", "HOUR", "DOW"])["DAILY_CALLS"]
        .mean()
        .to_frame("MEAN_DAILY_CALLS")
        .reset_index()
    )

    test_actual = test.groupby(["DATE", "BEAT", "HOUR", "DOW"]).size().to_frame("ACTUAL_CALLS").reset_index()
    pred = test_actual.merge(train_mean, on=["BEAT", "HOUR", "DOW"], how="left")
    pred["MEAN_DAILY_CALLS"] = pred["MEAN_DAILY_CALLS"].fillna(0.0)
    pred["PRED_CALLS"] = pred["MEAN_DAILY_CALLS"]

    mae = (pred["ACTUAL_CALLS"] - pred["PRED_CALLS"]).abs().mean()
    rmse = math.sqrt(((pred["ACTUAL_CALLS"] - pred["PRED_CALLS"]) ** 2).mean())

    return pred, {"MAE": float(mae), "RMSE": float(rmse), "test_days": int(test_days)}


def _allocate_proportional(demand_series, total_units):
    assert isinstance(total_units, int) and total_units > 0
    demand = demand_series.copy().astype(float).clip(lower=0.0)
    s = float(demand.sum())
    if s <= 0:
        out = pd.Series(0, index=demand.index, dtype=int)
        out.iloc[0] = total_units
        return out

    raw = demand / s * total_units
    base = raw.astype(int)
    rem = total_units - int(base.sum())
    frac = (raw - base).sort_values(ascending=False)

    base = base.copy()
    for idx in frac.index[:rem]:
        base.loc[idx] += 1

    assert int(base.sum()) == total_units
    return base


def step4_resource_deployment(df, total_units=50, high_risk_weight=1.5):
    """
    Step 4: Resource deployment suggestion (simple what-if).
    We build shift-level demand per beat:
      demand = avg calls per day in shift * (1 + (w-1)*high_risk_ratio)
    Shifts:
      Night: 00-07, Day: 08-15, Evening: 16-23
    Returns allocation DataFrame with columns:
      BEAT, SHIFT, AVG_CALLS, HIGH_RISK_RATIO, WEIGHTED_DEMAND, UNITS
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    for c in ["BEAT", "HOUR", "DATE", "IS_HIGH_RISK"]:
        assert c in df.columns
    assert isinstance(total_units, int) and total_units > 0
    assert isinstance(high_risk_weight, (int, float)) and float(high_risk_weight) >= 1.0

    def shift_from_hour(h):
        if 0 <= int(h) <= 7:
            return "Night"
        if 8 <= int(h) <= 15:
            return "Day"
        return "Evening"

    tmp = df.copy()
    tmp["SHIFT"] = tmp["HOUR"].apply(shift_from_hour).astype(str)

    daily_calls = tmp.groupby(["DATE", "BEAT", "SHIFT"]).size().to_frame("CALLS").reset_index()
    avg_calls = daily_calls.groupby(["BEAT", "SHIFT"])["CALLS"].mean().to_frame("AVG_CALLS").reset_index()
    risk_ratio = tmp.groupby(["BEAT", "SHIFT"])["IS_HIGH_RISK"].mean().to_frame("HIGH_RISK_RATIO").reset_index()

    merged = avg_calls.merge(risk_ratio, on=["BEAT", "SHIFT"], how="left")
    merged["HIGH_RISK_RATIO"] = merged["HIGH_RISK_RATIO"].fillna(0.0)
    merged["WEIGHTED_DEMAND"] = merged["AVG_CALLS"] * (1.0 + (float(high_risk_weight) - 1.0) * merged["HIGH_RISK_RATIO"])

    allocations = []
    for shift in ["Night", "Day", "Evening"]:
        sub = merged[merged["SHIFT"] == shift].copy()
        assert len(sub) > 0
        alloc = _allocate_proportional(sub.set_index("BEAT")["WEIGHTED_DEMAND"], total_units)
        out = sub.set_index("BEAT")[["AVG_CALLS", "HIGH_RISK_RATIO", "WEIGHTED_DEMAND"]].copy()
        out["UNITS"] = alloc
        out = out.reset_index()
        out["SHIFT"] = shift
        allocations.append(out)

    alloc_df = pd.concat(allocations, ignore_index=True)
    return alloc_df

def main():
    # ----- Paths (adjust as needed) -----
    DATA_PROCESSED = "./data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv"
    DATA_BEATS = "./data/00-raw/pd_beats_datasd.geojson"
    OUT_DIR = "./data/01-processed"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ========================================
    # Feature table
    # ========================================
    print("Building feature table...")

    FEATURES_CSV = os.path.join(OUT_DIR, "pd_calls_for_service_2025_datasd_features_v2.csv")
    df_v2 = build_features_v2_from_processed(DATA_PROCESSED, out_csv=FEATURES_CSV)

    beats_gdf = gpd.read_file(DATA_BEATS)
    valid_beats = set(beats_gdf["beat"].astype(str).str.strip())

    df_v2 = df_v2[df_v2["BEAT"].astype(str).str.strip().isin(valid_beats)].copy()
    print(df_v2.head())
    print(f"Shape: {df_v2.shape}\n")

    # ========================================
    # Step 3A: Hotspots (Beat-level + Season)
    # ========================================
    print("Step 3A: Hotspots (Beat-level + Season)")

    beat_counts, beat_season = step3_hotspots(df_v2)
    print(beat_counts.head())
    print(beat_season.head())

    beat_counts.to_csv(os.path.join(OUT_DIR, "step3_hotspots_beats.csv"))
    beat_season.to_csv(os.path.join(OUT_DIR, "step3_hotspots_beat_season.csv"), index=False)

    path_1 = os.path.join(OUT_DIR, "step3_hotspots_beats.csv")
    path_2 = os.path.join(OUT_DIR, "step3_hotspots_beat_season.csv")
    print(path_1)
    print(path_2)
    print()

    # ========================================
    # Step 3B: Baseline Forecasting
    # ========================================
    print("Step 3B: Baseline Forecasting")

    pred_df, metrics = step3_baseline_forecast(df_v2, test_days=14)
    print(pred_df.head())
    print(metrics)

    pred_df.to_csv(os.path.join(OUT_DIR, "step3_forecast_predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "step3_forecast_metrics.csv"), index=False)

    path_3 = os.path.join(OUT_DIR, "step3_forecast_predictions.csv")
    path_4 = os.path.join(OUT_DIR, "step3_forecast_metrics.csv")
    print(path_3)
    print(path_4)
    print()

    # ========================================
    # Step 4 Base: Resource Deployment Table
    # ========================================
    print("Step 4 Base: Resource Deployment Table")

    alloc_df = step4_resource_deployment(df_v2, total_units=50, high_risk_weight=1.5)
    alloc_path = os.path.join(OUT_DIR, "step4_resource_deployment.csv")
    alloc_df.to_csv(alloc_path, index=False)

    print(alloc_path)
    print(alloc_df.head())
    print()

    # ========================================
    # Visualization
    # ========================================
    print("=" * 60)
    print("Generating visualizations...")

    top_beats = list(beat_counts.head(3).index)
    viz = pred_df[pred_df["BEAT"].isin(top_beats)].copy()
    viz_agg = viz.groupby(["DATE", "BEAT"])[["ACTUAL_CALLS", "PRED_CALLS"]].sum().reset_index()
    print(viz_agg.head())

    # Plot 1: Actual calls
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=viz_agg, x="DATE", y="ACTUAL_CALLS", hue="BEAT", legend=True)
    plt.xticks(rotation=45)
    plt.title("Step 3B: Actual calls (Top 3 hotspot beats, test window)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "step3_actual_calls_top3.png"), dpi=150)
    plt.close()

    # Plot 2: Predicted calls
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=viz_agg, x="DATE", y="PRED_CALLS", hue="BEAT", legend=True)
    plt.xticks(rotation=45)
    plt.title("Step 3B: Baseline predicted calls (Top 3 hotspot beats, test window)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "step3_predicted_calls_top3.png"), dpi=150)
    plt.close()

    print(f"[OK] Saved: {os.path.join(OUT_DIR, 'step3_actual_calls_top3.png')}")
    print(f"[OK] Saved: {os.path.join(OUT_DIR, 'step3_predicted_calls_top3.png')}")
    print()

    print("Step 3 complete.")


if __name__ == "__main__":
    main()