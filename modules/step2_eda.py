import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

#### Parameters ###
OUT_DR = "./data/EDA_outputs/"
DOW_ORDER    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
SEASON_ORDER = ["Spring","Summer","Fall","Winter"]

#### Helpers ####
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

### Display summary statistics ###
def summary_stats(df):
    '''
    Print summary statistics
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
    Generates a heatmap of call volumes by Hour of Day and Day of Week.
    This visualization reveals timely patterns and peak periods (hotspots). 
    It identifies which hours on which days experience the highest density of incidents.

    Saves to OUT_DIR and displays plot

    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    df = add_time_features(df)

    hd_group = df.groupby(["HOUR","DOW"]).size().unstack("DOW").fillna(0)
    hd_group = hd_group.reindex(columns=[d for d in DOW_ORDER if d in hd_group.columns])

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(hd_group, cmap="YlOrRd", linewidths=0.3, linecolor="white", cbar_kws={"label":"Call Count"}, ax=ax)

    ax.set_title("Call Volume: Hour × Day of Week", fontsize=13)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda1_hour_dow_heatmap.png"), dpi=150)
    plt.show()

def plot_seasonal_monthly(df, figsize=(13,5)):
    '''
    Visualizes temporal call patterns via Monthly totals and Seasonal distributions.
    This function has two visualizations.
    Useful for identifying variance and outliers across different times of the year.

    Saves to OUT_DIR and displays plot

    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    df = add_time_features(df)

    monthly = df.groupby("MONTH").size().reset_index(name="CALLS")
    daily_season = df.groupby(["DATE","SEASON"]).size().reset_index(name="CALLS")
    mlabels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    _, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].bar(monthly["MONTH"], monthly["CALLS"], color=sns.color_palette("tab10",12), edgecolor="white")
    axes[0].set_xticks(range(1,13))
    axes[0].set_xticklabels(mlabels)
    axes[0].set_title("Total Calls by Month")
    axes[0].set_ylabel("Calls")
    axes[0].set_xlabel("Month")

    s_order = [s for s in SEASON_ORDER if s in daily_season["SEASON"].unique()]
    sns.boxplot(data=daily_season, x="SEASON", y="CALLS", order=s_order, palette="Set2", ax=axes[1], hue="SEASON", legend=False)

    axes[1].set_title("Daily Calls Distribution by Season")
    axes[1].set_ylabel("Calls per Day")
    axes[1].set_xlabel("Season")
    plt.suptitle("Seasonal & Monthly Patterns", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda2_seasonal.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_call_type_distribution(df, top_n=20, figsize=(12,6)):
    '''
    Visualizes the frequency of the Top-N most common call type categories.
    Uses CALL_TYPE_CATEGORY from clean_further if available, otherwise falls back to CALL_TYPE.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    col = "CALL_TYPE_CATEGORY" if "CALL_TYPE_CATEGORY" in df.columns else "CALL_TYPE"
    assert col in df.columns, f"Missing required column: {col}"

    counts = df[col].value_counts().head(top_n).sort_values()

    _, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(counts.index, counts.values, color=sns.color_palette("Blues_r", len(counts)))

    total = counts.values.sum()
    pct_labels = [f"{v/total:.1%}" for v in counts.values]

    ax.bar_label(bars, labels=pct_labels, padding=3, fontsize=8)
    ax.set_title(f"Call Type Category Distribution" if col == "CALL_TYPE_CATEGORY" else f"Top {top_n} Call Types", fontsize=13)
    ax.set_xlabel("Count")
    ax.set_ylabel("Call Type Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda3_call_type.png"), dpi=150)
    plt.show()

def plot_beat_hotspot(df, geojson_path, top_n=20, figsize=(12,5)):
    '''
    Visualizes the highest volume patrol beats (geographic hotspots).
    For assessing resource allocation and identifying areas requiring increased patrol presence.
    
    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    SD_beats_df = gpd.GeoDataFrame.from_file(geojson_path)
    beat_names = SD_beats_df.set_index("beat")["name"].to_dict()

    counts = df.groupby("BEAT").size().sort_values(ascending=False).head(top_n)

    _, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette("Reds_r", top_n), edgecolor="white")

    total = counts.values.sum()
    pct_labels = [f"{v/total:.1%}" for v in counts.values]

    ax.bar_label(bars, labels=pct_labels, padding=3, fontsize=8)
    ax.set_title(f"Top {top_n} Beats by Call Volume", fontsize=13)
    tick_labels = [f"{beat} ({beat_names.get(beat, '')})" for beat in counts.index]
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Beat (Geographic Location)")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join(OUT_DR, "eda4_beat_hotspot.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_priority_distribution(df, figsize=(11,4)):
    '''
    Visualizes call severity through Priority levels and High-Risk ratios.
    Uses IS_HIGH_RISK from clean_further if available, otherwise falls back to priority <= 2.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    assert "PRIORITY" in df.columns

    prio = pd.to_numeric(df["PRIORITY"], errors="coerce").dropna()
    counts = prio.value_counts().sort_index()

    if "IS_HIGH_RISK" in df.columns:
        high  = int(df["IS_HIGH_RISK"].sum())
        other = int(len(df) - high)
        risk_label = "High-Risk"
    else:
        high  = int((prio <= 2).sum())
        other = int((prio > 2).sum())
        risk_label = "High-Risk (Priority ≤ 2)"

    _, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(counts.index.astype(str), counts.values, color=sns.color_palette("flare", len(counts)))
    axes[0].set_title("Calls by Priority Level")
    axes[0].set_xlabel("Priority"); axes[0].set_ylabel("Count")

    axes[1].pie([high, other], labels=[risk_label, "Other"], colors=["#C44E52","#4C72B0"], startangle=90)
    axes[1].set_title("High-Risk Share")

    plt.suptitle("Priority Distribution", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda5_priority.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_disposition(df, figsize=(11,5)):
    '''
    Visualizes how calls were resolved using DISPOSITION_CATEGORY.
    Falls back to raw DISPOSITION if the category column is not present.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    col = "DISPOSITION_CATEGORY" if "DISPOSITION_CATEGORY" in df.columns else "DISPOSITION"
    assert col in df.columns, f"Missing required column: {col}"

    counts = df[col].value_counts()
    pct    = counts / len(df) * 100

    _, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=sns.color_palette("muted", len(counts)))

    for bar, p in zip(bars, pct.values[::-1]):
        ax.text(bar.get_width() + counts.max()*0.01, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va="center", fontsize=8)

    ax.set_title("Disposition Category Distribution" if col == "DISPOSITION_CATEGORY" else "Top Dispositions", fontsize=13)
    ax.set_xlabel("Count")
    ax.set_ylabel("Disposition Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda6_disposition.png"), dpi=150)
    plt.show()

def plot_calltype_hour_heatmap(df, calltypes_path, top_n=12, figsize=(13, 6)):
    '''
    Heatmap of Top-N call types x hour of day (row-normalized).
    Useful for understanding if certain incident types require time-targeted resourcing.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert "CALL_TYPE" in df.columns

    map_df = pd.read_csv(calltypes_path)
    calltypes = map_df.set_index("call_type")["description"].to_dict()

    df = add_time_features(df)
    top_types = df["CALL_TYPE"].value_counts().head(top_n).index
    sub = df[df["CALL_TYPE"].isin(top_types)]

    ch_group = (sub.groupby(["CALL_TYPE", "HOUR"]).size().unstack("HOUR").fillna(0))
    ch_group_norm = ch_group.div(ch_group.sum(axis=1), axis=0)

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ch_group_norm, cmap="YlOrRd", linewidths=0.3, linecolor="white", cbar_kws={"label": "Share of calls (normalized)"}, ax=ax)

    ax.set_title(f"Top {top_n} Call Types Share x Hour of Day",fontsize=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Call Code (Description)")

    tick_labels = [f"{calltype} ({calltypes.get(calltype, calltype)})" for calltype in ch_group_norm.index]
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda7_calltype_hour_heatmap.png"), dpi=150)
    plt.show()

def plot_category_by_season(df, figsize=(12,6)):
    '''
    Stacked bar chart: CALL_TYPE_CATEGORY counts by Season.
    Shows how the composition of call demand shifts across seasons.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    col = "CALL_TYPE_CATEGORY" if "CALL_TYPE_CATEGORY" in df.columns else "CALL_TYPE"
    df = add_time_features(df)

    pivot = (
        df.groupby(["SEASON", col])
        .size()
        .unstack(fill_value=0)
        .reindex([s for s in SEASON_ORDER if s in df["SEASON"].unique()])
    )

    colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]

    ax = pivot.plot(kind="bar", stacked=True, figsize=figsize, color=colors[:len(pivot.columns)])
    
    ax.set_xlabel("Season", fontsize=14)
    ax.set_ylabel("Number of Calls", fontsize=14)
    ax.set_title("Call Type Category by Season", fontsize=16)
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(title=col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda8_category_by_season.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_disposition_pareto(df, figsize=(12,6)):
    '''
    Pareto chart of DISPOSITION_CATEGORY with cumulative percentage line.
    Highlights which few disposition outcomes account for most of the volume.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    col = "DISPOSITION_CATEGORY" if "DISPOSITION_CATEGORY" in df.columns else "DISPOSITION"
    assert col in df.columns, f"Missing required column: {col}"

    counts = df[col].dropna().value_counts().sort_values(ascending=False)
    cum_pct = counts.cumsum() / counts.sum() * 100

    _, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(counts.index.astype(str), counts.values, color="#F4B183")

    ax1.set_ylabel("Count", fontsize=14)
    ax1.set_xlabel("Disposition Category", fontsize=14)
    ax1.set_title(f"Pareto Chart: {col} (n={counts.sum():,})", fontsize=16)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center", fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(counts.index.astype(str), cum_pct.values, marker="o", color="#E07A2D", linewidth=2)

    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.axhline(80, linestyle="--", linewidth=1.5, color="#D98C3F")
    ax2.text(len(counts) - 1, 81, "80%", ha="right", va="bottom", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda9_disposition_pareto.png"), dpi=150)
    plt.show()


def plot_daily_timeseries(df, figsize=(12,6)):
    '''
    Daily incident count time-series line plot.
    Useful for spotting long-term trends, spikes, and anomalies.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''
    df = add_time_features(df)

    daily = df.groupby("DATE").size().reset_index(name="CALLS")
    daily["DATE"] = pd.to_datetime(daily["DATE"])
    daily = daily.sort_values("DATE")

    _, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=daily, x="DATE", y="CALLS", color="blue", lw=1.5, ax=ax)

    ax.set_title("Daily Police Calls for Service Count", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Incident Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda10_daily_timeseries.png"), dpi=150)
    plt.show()

def plot_beat_choropleth(df, geojson_path, figsize=(12,10)):
    '''
    Choropleth map of call volume by patrol beat.
    Requires geopandas and a GeoJSON file of beat boundaries.

    Saves to OUT_DIR and displays plot
    
    Arguments
    df-- Input DataFrame
    figsize-- Size of output figure

    Returns
    N/A
    '''

    SD_beats_df = gpd.GeoDataFrame.from_file(geojson_path)
    call_counts = df['BEAT'].value_counts().to_frame().reset_index()

    call_counts = call_counts.rename(columns={'BEAT': 'beat'})
    SD_beats_df = SD_beats_df.merge(call_counts, on='beat', how='left')

    _, ax1 = plt.subplots(figsize=figsize)
    SD_beats_df.plot(column='count', cmap='magma', legend=True, ax=ax1)
    ax1.set_title('San Diego Emergency Calls by Beat')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda11_beat_choropleth.png"), dpi=150)
    plt.show()

    _, ax2 = plt.subplots(figsize=figsize)
    SD_beats_df.plot(column='count', cmap='magma', legend=True, ax=ax2)
    ax2.set_xlim([-117.20, -117.08])
    ax2.set_ylim([32.68, 32.75])
    ax2.set_title('San Diego Emergency Calls by Beat- Highest Call Density')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DR, "eda12_beat_highest_density_choropleth.png"), dpi=150)
    plt.show()


def main():
    """
    Run all EDA plots end-to-end.
    Reads the cleaned_v2 CSV, creates output directory, and saves all figures.
    """
    # ----- Paths -----
    DATA_CSV = "./data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv"
    GEOJSON = "./data/00-raw/pd_beats_datasd.geojson"
    CALLTYPES = "./data/00-raw/pd_cfs_calltypes_datasd.csv"

    os.makedirs(OUT_DR, exist_ok=True)

    # ----- Load data -----
    assert os.path.exists(DATA_CSV), f"Data file not found: {DATA_CSV}"
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df):,} rows from {DATA_CSV}\n")

    # ----- Summary -----
    summary_stats(df)

    # ----- Graphs -----
    plot_hour_dow_heatmap(df)
    plot_seasonal_monthly(df)
    plot_call_type_distribution(df)
    plot_beat_hotspot(df, GEOJSON)
    plot_priority_distribution(df)
    plot_disposition(df)
    plot_calltype_hour_heatmap(df, CALLTYPES)
    plot_category_by_season(df)
    plot_disposition_pareto(df)
    plot_daily_timeseries(df)
    plot_beat_choropleth(df, GEOJSON)

    print(f"\nAll EDA outputs saved to: {OUT_DR}/")


if __name__ == "__main__":
    main()