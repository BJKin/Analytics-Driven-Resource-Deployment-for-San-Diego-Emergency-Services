import os
import math
import random
import pandas as pd


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _season_from_month(m):
    assert isinstance(m, int)
    assert 1 <= m <= 12
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Fall"


def load_calls_csv(path):
    """Load calls-for-service CSV into a DataFrame."""
    assert isinstance(path, str) and len(path) > 0
    assert os.path.isfile(path)
    df = pd.read_csv(path)
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    return df


def basic_clean_calls(df):
    """
    Basic cleaning mirroring the current notebook Step 1:
    - Drop intersecting street columns if present
    - Drop rows with missing key fields
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    df = df.copy()

    drop_cols = ["ADDRESS_DIR_INTERSECTING", "ADDRESS_ROAD_INTERSECTING", "ADDRESS_SFX_INTERSECTING"]
    to_drop = [c for c in drop_cols if c in df.columns]
    if len(to_drop) > 0:
        df = df.drop(columns=to_drop)

    required = []
    for c in ["ADDRESS_ROAD_PRIMARY", "CALL_TYPE", "DISPOSITION"]:
        if c in df.columns:
            required.append(c)
    if len(required) > 0:
        df = df.dropna(subset=required).reset_index(drop=True)

    assert len(df) > 0
    return df


def add_time_features(df):
    """
    Add standard time-derived columns:
    - DATETIME (parsed)
    - DATE (YYYY-MM-DD string)
    - HOUR (0..23)
    - DOW (Monday..Sunday)
    - MONTH (1..12)
    - SEASON (Winter/Spring/Summer/Fall)
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    df = df.copy()

    ts_col = _find_col(df, ["DATE_TIME", "Timestamp", "timestamp", "call_datetime", "datetime", "DATE"])
    assert ts_col is not None

    dt = pd.to_datetime(df[ts_col], errors="coerce")
    assert dt.notna().any()

    df["DATETIME"] = dt
    df = df[df["DATETIME"].notna()].copy()

    df["DATE"] = df["DATETIME"].dt.date.astype(str)
    df["HOUR"] = df["DATETIME"].dt.hour.astype(int)
    df["DOW"] = df["DATETIME"].dt.day_name().astype(str)
    df["MONTH"] = df["DATETIME"].dt.month.astype(int)
    df["SEASON"] = df["MONTH"].apply(_season_from_month).astype(str)
    return df


def add_beat_key(df):
    """Create a normalized 'BEAT_KEY' column for grouping."""
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    beat_col = _find_col(df, ["BEAT", "beat", "Beat", "pd_beat", "PD_BEAT", "sdpd_beat", "SDPD_BEAT"])
    assert beat_col is not None
    df = df.copy()
    df["BEAT_KEY"] = df[beat_col].astype(str).str.strip()
    df = df[df["BEAT_KEY"] != ""].copy()
    assert len(df) > 0
    return df


def add_high_risk_flag(df):
    """
    Add IS_HIGH_RISK (0/1) using best-available signals:
    - If PRIORITY exists: high risk if PRIORITY <= 2
    - Else: keyword heuristic on CALL_TYPE
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    df = df.copy()

    if "IS_HIGH_RISK" in df.columns:
        s = df["IS_HIGH_RISK"]
        if s.dtype == bool:
            df["IS_HIGH_RISK"] = s.astype(int)
        else:
            df["IS_HIGH_RISK"] = s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"]).astype(int)
        return df

    pr_col = _find_col(df, ["PRIORITY", "priority", "Priority"])
    if pr_col is not None:
        p = pd.to_numeric(df[pr_col], errors="coerce")
        if p.notna().any():
            df["IS_HIGH_RISK"] = (p <= 2).fillna(False).astype(int)
            return df

    ct_col = _find_col(df, ["CALL_TYPE", "call_type", "Call Type"])
    if ct_col is not None:
        ct = df[ct_col].astype(str).str.lower()
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
        df["IS_HIGH_RISK"] = risky.astype(int)
        return df

    df["IS_HIGH_RISK"] = 0
    return df


def build_features_v2_from_processed(processed_csv, out_csv=None):
    """
    Build a Step3/4-ready dataset from the already-cleaned CSV:
    Adds time features, BEAT_KEY, and IS_HIGH_RISK.
    If out_csv is provided, saves the result.
    """
    df = load_calls_csv(processed_csv)
    df = add_time_features(df)
    df = add_beat_key(df)
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
      - beat_counts: DataFrame indexed by BEAT_KEY with CALLS column
      - beat_season: DataFrame with BEAT_KEY, SEASON, CALLS
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert "BEAT_KEY" in df.columns
    assert "SEASON" in df.columns

    beat_counts = df.groupby("BEAT_KEY").size().sort_values(ascending=False).to_frame("CALLS")
    beat_season = df.groupby(["BEAT_KEY", "SEASON"]).size().to_frame("CALLS").reset_index()
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
    Baseline = average calls per (BEAT_KEY, HOUR, DOW) on train set.
    Evaluates on last `test_days` dates.
    Returns:
      pred_df: DATE, BEAT_KEY, HOUR, DOW, ACTUAL_CALLS, PRED_CALLS
      metrics: dict with MAE and RMSE
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    for c in ["BEAT_KEY", "HOUR", "DOW", "DATE"]:
        assert c in df.columns

    train, test = _train_test_split_by_date(df, test_days=test_days)

    train_mean = train.groupby(["BEAT_KEY", "HOUR", "DOW"]).size().to_frame("MEAN_CALLS").reset_index()
    train_mean["MEAN_CALLS"] = train_mean["MEAN_CALLS"].astype(float)

    test_actual = test.groupby(["DATE", "BEAT_KEY", "HOUR", "DOW"]).size().to_frame("ACTUAL_CALLS").reset_index()
    pred = test_actual.merge(train_mean, on=["BEAT_KEY", "HOUR", "DOW"], how="left")
    pred["MEAN_CALLS"] = pred["MEAN_CALLS"].fillna(0.0)
    pred["PRED_CALLS"] = pred["MEAN_CALLS"]

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
      BEAT_KEY, SHIFT, AVG_CALLS, HIGH_RISK_RATIO, WEIGHTED_DEMAND, UNITS
    """
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    for c in ["BEAT_KEY", "HOUR", "DATE", "IS_HIGH_RISK"]:
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

    daily_calls = tmp.groupby(["DATE", "BEAT_KEY", "SHIFT"]).size().to_frame("CALLS").reset_index()
    avg_calls = daily_calls.groupby(["BEAT_KEY", "SHIFT"])["CALLS"].mean().to_frame("AVG_CALLS").reset_index()
    risk_ratio = tmp.groupby(["BEAT_KEY", "SHIFT"])["IS_HIGH_RISK"].mean().to_frame("HIGH_RISK_RATIO").reset_index()

    merged = avg_calls.merge(risk_ratio, on=["BEAT_KEY", "SHIFT"], how="left")
    merged["HIGH_RISK_RATIO"] = merged["HIGH_RISK_RATIO"].fillna(0.0)
    merged["WEIGHTED_DEMAND"] = merged["AVG_CALLS"] * (1.0 + (float(high_risk_weight) - 1.0) * merged["HIGH_RISK_RATIO"])

    allocations = []
    for shift in ["Night", "Day", "Evening"]:
        sub = merged[merged["SHIFT"] == shift].copy()
        assert len(sub) > 0
        alloc = _allocate_proportional(sub.set_index("BEAT_KEY")["WEIGHTED_DEMAND"], total_units)
        out = sub.set_index("BEAT_KEY")[["AVG_CALLS", "HIGH_RISK_RATIO", "WEIGHTED_DEMAND"]].copy()
        out["UNITS"] = alloc
        out = out.reset_index()
        out["SHIFT"] = shift
        allocations.append(out)

    alloc_df = pd.concat(allocations, ignore_index=True)
    return alloc_df
