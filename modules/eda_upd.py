'''
"eda_upd.py": additional updated EDA methods
Including following moudles for EDAs.

This file is based on the modules/main.py
'''
import tabulate as tb
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

#### Parameters ###

OUT_DR = "./EDA_outputs"
DOW_ORDER    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
SEASON_ORDER = ["Spring","Summer","Fall","Winter"]

#### Helpers ####

# This method is modified from "main.py"! Thanks for original author
def add_time_features(df):
    '''
    Add time from DATE_TIME
    Including: HOUR / DOW / MONTH / SEASON / DATE
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    ts_col = next((c for c in df.columns if "date" in c.lower()), None)
    assert ts_col is not None, "not able to find relevant date"

    df = df.copy()
    dt = pd.to_datetime(df[ts_col], errors="coerce")
    df["DATETIME"] = dt
    df = df[df["DATETIME"].notna()].copy()

    if "HOUR" not in df.columns: df["HOUR"] = df["DATETIME"].dt.hour.astype(int)
    if "DOW" not in df.columns: df["DOW"] = df["DATETIME"].dt.day_name()
    if "MONTH" not in df.columns: df["MONTH"] = df["DATETIME"].dt.month.astype(int)
    if "DATE" not in df.columns: df["DATE"] = df["DATETIME"].dt.date.astype(str)
    if "SEASON" not in df.columns:
        def seasonchecker(m):
            if m in (3,4,5):    return "Spring"
            if m in (6,7,8):    return "Summer"
            if m in (9,10,11):  return "Fall"
            return "Winter"
        df["SEASON"] = df["MONTH"].apply(seasonchecker)
    return df

# This method is modified from "main.py"! Thanks for original author
def add_beat_key(df):
    '''
    Standardize the BEAT column to beatkey, while removing non-regular parameters.
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    beat_col = next((c for c in df.columns if "beat" in c.lower()), None)
    assert beat_col is not None, "No such beat column. Make sure beat column has title 'beat'"
    df = df.copy()
    df["BEAT_KEY"] = df[beat_col].astype(str).str.strip()
    df = df[(df["BEAT_KEY"] != "")].copy()
    assert len(df) > 0
    return df

# Simplified method
def summary_stats(df):
    '''
    Print the summary
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n")
    print("=="*12, "Missing Summary", "=="*12)
    print(df.isna().sum())
    print("=="*12)
    print("DataTypes:")
    print(df.dtypes.to_string())
    print("=="*12)
    print("Numerical Summary")
    print(df.describe().to_markdown())

#### EDA Visualizations ####

def plot_hour_dow_heatmap(df, figsize=(12,5)):
    '''
    [Output Graph 1] Generates a heatmap of call volumes by Hour of Day and Day of Week.
    This visualization reveals timely patterns and peak periods (hotspots). 
    It identifies which hours on which days 
    experience the highest density of incidents.
    '''
    df = add_time_features(df)

    hd_group = df.groupby(["HOUR","DOW"]).size().unstack("DOW").fillna(0)
    hd_group = hd_group.reindex(columns=[d for d in DOW_ORDER if d in hd_group.columns])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(hd_group, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                cbar_kws={"label":"Call Count"}, ax=ax)
    ax.set_title("Call Volume: Hour × Day of Week", fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda1_hour_dow_heatmap.png"), dpi=150)
    plt.show()

def plot_seasonal_monthly(df, figsize=(13,5)):
    '''
    [Output Graph 2] Visualizes temporal call patterns via Monthly totals and Seasonal distributions.
    This function has two visualizaitons.
    Useful for identifying variance and outliers across different times of the year.
    '''
    df = add_time_features(df)

    monthly = df.groupby("MONTH").size().reset_index(name="CALLS")
    daily_season = df.groupby(["DATE","SEASON"]).size().reset_index(name="CALLS")
    mlabels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].bar(monthly["MONTH"], monthly["CALLS"],
                color=sns.color_palette("tab10",12), edgecolor="white")
    axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(mlabels)
    axes[0].set_title("Total Calls by Month"); axes[0].set_ylabel("Calls")

    s_order = [s for s in SEASON_ORDER if s in daily_season["SEASON"].unique()]
    sns.boxplot(data=daily_season, x="SEASON", y="CALLS",
                order=s_order, palette="Set2", ax=axes[1], hue="SEASON", legend=False)
    axes[1].set_title("Daily Calls Distribution by Season")
    axes[1].set_ylabel("Calls per Day")

    plt.suptitle("Seasonal & Monthly Patterns", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda2_seasonal.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_call_type_distribution(df, top_n=20, figsize=(12,6)):
    '''
    [Output Graph 3] Visualizes the frequency of the Top-N most common call types.
    '''
    assert "CALL_TYPE" in df.columns

    counts = df["CALL_TYPE"].value_counts().head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(counts.index, counts.values,
                   color=sns.color_palette("Blues_r", top_n))
    ax.bar_label(bars, fmt="%,.0f", padding=3, fontsize=8)
    ax.set_title(f"Top {top_n} Call Types", fontsize=13)
    ax.set_xlabel("Count"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda3_call_type.png"), dpi=150)
    plt.show()

def plot_beat_hotspot(df, top_n=20, figsize=(12,5)):
    '''
    [Output Graph 4] Visualizes the highest volume patrol beats (geographic hotspots).
    For assessing resource allocation and identifying areas requiring increased patrol presence.
    '''
    df = add_beat_key(df)

    counts = df.groupby("BEAT_KEY").size().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("Reds_r", top_n), edgecolor="white")
    ax.bar_label(bars, fmt="%,.0f", fontsize=7, padding=2)
    ax.set_title(f"Top {top_n} Beats by Call Volume", fontsize=13)
    ax.set_xlabel("Beat"); ax.set_ylabel("Calls")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda4_beat_hotspot.png"), dpi=150)
    plt.show()

def plot_priority_distribution(df, figsize=(11,4)):
    '''
    [Output Graph 5] Visualizes call severity through Priority levels and High-Risk ratios.
    '''
    assert "PRIORITY" in df.columns

    prio   = pd.to_numeric(df["PRIORITY"], errors="coerce").dropna()
    counts = prio.value_counts().sort_index()
    high = int((prio <= 2).sum())
    other = int((prio > 2).sum())

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(counts.index.astype(str), counts.values, color=sns.color_palette("flare", len(counts)))
    axes[0].set_title("Calls by Priority Level")
    axes[0].set_xlabel("Priority"); axes[0].set_ylabel("Count")

    axes[1].pie([high, other], labels=["High-risk (Threshold: 2)","Other"], colors=["#C44E52","#4C72B0"], startangle=90)
    axes[1].set_title("High-Risk Share")

    plt.suptitle("Priority Distribution", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda5_priority.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_disposition(df, top_n=15, figsize=(11,5)):
    '''
    [Output Graph 6] Visualizes how calls were resolved, helping to distinguish between formal criminal reports and 
    administrative or non-actionable calls.
    '''
    assert "DISPOSITION" in df.columns

    counts = df["DISPOSITION"].value_counts().head(top_n)
    pct    = counts / len(df) * 100

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=sns.color_palette("muted", top_n))
    for bar, p in zip(bars, pct.values[::-1]):
        ax.text(bar.get_width() + counts.max()*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{p:.1f}%", va="center", fontsize=8)
    ax.set_title(f"Top {top_n} Dispositions", fontsize=13)
    ax.set_xlabel("Count"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda6_disposition.png"), dpi=150)
    plt.show()

def plot_calltype_hour_heatmap(df, top_n=12, figsize=(13, 6)):
    '''
    [Output Graph 7] Heatmap of Top-N call types x hour of day (row-normalized).
    Useful for understanding if certain incident types require time-targeted resourcing.
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert "CALL_TYPE" in df.columns

    df = add_time_features(df)

    top_types = df["CALL_TYPE"].value_counts().head(top_n).index
    sub = df[df["CALL_TYPE"].isin(top_types)]

    ch_group = (sub.groupby(["CALL_TYPE", "HOUR"]).size().unstack("HOUR").fillna(0))
    # normalize: colour shows hourly shape, not absolute volume
    ch_group_norm = ch_group.div(ch_group.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ch_group_norm, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Share of calls (normalized)"}, ax=ax)
    ax.set_title(f"Top {top_n} Call Types Share x Hour of Day",fontsize=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Call Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda7_calltype_hour_heatmap.png"), dpi=150)
    plt.show()

    