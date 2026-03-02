import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from modules.eda_upd import add_time_features, add_beat_key

COL_REQUIRED  = ["INCIDENT_NUM","DATE_TIME","DAY_OF_WEEK","CALL_TYPE",
               "DISPOSITION","BEAT","PRIORITY"]
NEW_FEATURES = ["BEAT_NUM","HOUR","DOW_NUM","MONTH",
                "SEASON_NUM","IS_WEEKEND","SHIFT"]


#### Validation ####

def check_df(df):
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    missing = [c for c in COL_REQUIRED if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"check passed — {len(df):,} rows, {df.shape[1]} cols")

#### Feature Engineering ####

def build_features(df):
    '''
    Feature Engineering - taking cleaned df as input
    Output:
    A new dataframe re-ordered:
    rows: different time-window, distinguished by MONTH, DATE, HOUR
    Under those windows, they contain:
    BEAT_NUM
    DOW_NUM
    SEASON_NUM
    IS_WEEKEND
    SHIFT
    CALLS
    The model should estimate the CALLS for testing dataset.
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0

    df = add_time_features(df)
    df = add_beat_key(df)

    season_num  = {"Spring":0,"Summer":1,"Fall":2,"Winter":3}
    df["SEASON_NUM"] = df["SEASON"].map(season_num).astype(int)
    df["DOW_NUM"] = df["DATETIME"].dt.dayofweek.astype(int)
    df["IS_WEEKEND"] = (df["DOW_NUM"] >= 5).astype(int)
    df["SHIFT"] = pd.cut(df["HOUR"], bins=[-1,7,15,23],labels=[0,1,2]).astype(int)
    df["BEAT_NUM"] = df["BEAT_KEY"].astype(int)

    # aggregate to (date, beat, hour, dow) level
    group_cols = ["DATE","BEAT_NUM","HOUR","DOW_NUM","MONTH", "SEASON_NUM","IS_WEEKEND","SHIFT"]
    mod_df = df.groupby(group_cols)["INCIDENT_NUM"].count().reset_index()
    mod_df.columns = group_cols + ["CALLS"]

    print(f"features built — {len(mod_df):,} rows")
    return mod_df

#### Splitting the dataset #### 

def split_data(mod_df, test_days=15):
    assert "DATE" in mod_df.columns and isinstance(test_days, int) and test_days > 0

    dates  = sorted(mod_df["DATE"].unique())
    cutoff = dates[-test_days]
    train  = mod_df[mod_df["DATE"] < cutoff].copy()
    test   = mod_df[mod_df["DATE"] >= cutoff].copy()

    print(f"train: {len(train):,} rows up to {dates[-test_days-1]}")
    print(f"test : {len(test):,} rows {cutoff} → {dates[-1]}")
    return train, test

#### Training the model ####

def train_model(train, lr = 0.05, n_est = 100):
    X = train[NEW_FEATURES]
    y = train["CALLS"]

    model = xgb.XGBRegressor(
        objective = "count:poisson",
        n_estimators = n_est,
        learning_rate = lr,
        random_state = 42,
        n_jobs = -1,
    )
    model.fit(X, y, verbose=False)
    print("model trained")
    return model

#### Predicting the data ####

def predict(model, test):
    result = test[["DATE","BEAT_NUM","HOUR","DOW_NUM","CALLS"]].copy()
    result["PRED_CALLS"] = np.round(model.predict(test[NEW_FEATURES])).astype(int)
    return result

#### Error Analysis ####

def error_analysis(result, model=None):
    mae  = mean_absolute_error(result["CALLS"], result["PRED_CALLS"])
    rmse = mean_squared_error(result["CALLS"],  result["PRED_CALLS"]) ** 0.5

    print("=" * 40)
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print("=" * 40)

    print("\nSample predictions vs actual:")
    sample = result[["DATE","BEAT_NUM","HOUR","CALLS","PRED_CALLS"]].head(30)
    sample["ERROR"] = sample["CALLS"] - sample["PRED_CALLS"]
    print(sample.to_string(index=False))