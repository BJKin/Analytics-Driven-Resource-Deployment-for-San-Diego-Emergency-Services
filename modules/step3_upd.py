"""
Step 3: Hotspots & Baseline Forecasting

This script bridges Step 2 (EDA) and Step 4 (Resource Deployment).
It produces:
  - step3_hotspots_beats.csv
  - step3_hotspots_beat_season.csv
  - step3_forecast_predictions.csv
  - step3_forecast_metrics.csv
  - step3_actual_calls_top3.png
  - step3_predicted_calls_top3.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from step3_helper import build_features_v2_from_processed, step3_hotspots, step3_baseline_forecast, step4_resource_deployment


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