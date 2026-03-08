
# Step 2 Exploratory Data Analysis (eda_upd.py)

This file documents the `eda_upd.py` script, which generates **11 EDA visualizations** from the cleaned dataset.

It is designed to work **directly on top of Step 1 outputs**:

- `data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv`
- `data/00-raw/pd_beats_datasd.geojson` (optional, for choropleth only)

Make sure Step 1 (`clean_further.py`) has been executed.

---

## What this script produces

All figures are saved to `./EDA_outputs/`.

| # | Function | Output file | Description |
|---|---|---|---|
| 1 | `plot_hour_dow_heatmap` | `eda1_hour_dow_heatmap.png` | Hour × Day-of-Week heatmap |
| 2 | `plot_seasonal_monthly` | `eda2_seasonal.png` | Monthly bar chart + seasonal boxplot |
| 3 | `plot_call_type_distribution` | `eda3_call_type.png` | Top-N call type categories (horizontal bar) |
| 4 | `plot_beat_hotspot` | `eda4_beat_hotspot.png` | Top-N beats by call volume |
| 5 | `plot_priority_distribution` | `eda5_priority.png` | Priority bar chart + high-risk pie |
| 6 | `plot_disposition` | `eda6_disposition.png` | Disposition category bar chart with % labels |
| 7 | `plot_calltype_hour_heatmap` | `eda7_calltype_hour_heatmap.png` | Top-N call types × hour (row-normalized) |
| 8 | `plot_category_by_season` | `eda8_category_by_season.png` | Call type category × season (stacked bar) |
| 9 | `plot_disposition_pareto` | `eda9_disposition_pareto.png` | Disposition Pareto chart with 80% line |
| 10 | `plot_daily_timeseries` | `eda10_daily_timeseries.png` | Daily incident count time-series |
| 11 | `plot_beat_choropleth` | `eda11_beat_choropleth.png` | Geographic heatmap by beat (requires GeoJSON) |

---

## Helpers

The script includes three utility functions used internally by the plot functions:

- **`add_time_features`** — derives `HOUR`, `DOW`, `MONTH`, `DATE`, `SEASON` from `DATE_TIME`.
- **`add_beat_key`** — standardizes the `BEAT` column to a clean `BEAT_KEY` string.
- **`summary_stats`** — prints shape, missing values, dtypes, and numerical summary.

---

## Clean_further columns

For capability concerns, graphs 3, 5, 6, 8, 9 will use the new columns from `clean_further.py` if they exist:

| New column | Used by |
|---|---|
| `CALL_TYPE_CATEGORY` | Graph 3, 8 |
| `DISPOSITION_CATEGORY` | Graph 6, 9 |
| `IS_HIGH_RISK` | Graph 5 |

If these columns are absent (e.g. running on `df_cleaned.csv` instead of `_cleaned_v2.csv`), the functions fall back to the raw `CALL_TYPE`, `DISPOSITION`, and a priority-based threshold respectively.

---

## Files consumed

| File | Required | Description |
|---|---|---|
| `data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv` | Yes | Output of `clean_further.py` |
| `data/00-raw/pd_beats_datasd.geojson` | No | Beat boundary polygons (Graph 11 only, skipped if missing) |

## Files produced

All PNGs are saved to `./EDA_outputs/` (created automatically).

---

## How to run

From the repo root:

```bash
python eda_upd.py
```

### Changing paths

Edit the two path variables at the top of `main()`:

```python
DATA_CSV = "./data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv"
GEOJSON  = "./data/00-raw/pd_beats_datasd.geojson"
```

---

## Dependencies

- `pandas`, `matplotlib`, `seaborn`, `tabulate` — required for all graphs
- `geopandas` — only required for Graph 11 (choropleth), imported inside the function so missing package won't break other graphs

---
