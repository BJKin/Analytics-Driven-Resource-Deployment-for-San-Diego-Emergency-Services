# Group 10 ECE 143 Final Project
### Authors
- `Brett Kinsella` - Data Pipeline & Cleaning, Exploratory Data Analysis
- `Hanyi Chen` - Data Pipeline & Cleaning, Exploratory Data Analysis
- `Zichen Wang` - Exploratory Data Analysis, Hotspots & Baseline Forecasting
- `Seong Ha` - Hotspots & Baseline Forecasting, Resource Deployment Suggestion
- `Hanyuan Zhang` - Hotspots & Baseline Forecasting, Resource Deployment Suggestion

### Data Overview
Dataset Name: Police Calls for Service 2025
- Descriptions of any shortcomings: There are large amounts of missing fields in the primary address suffix column and primary address direction column.  This is likely due to many streets not having a suffix or direction, but this may cause complications for data analysis as entries with missing fields don't necessarily correlate to data that can be thrown out.  An even more extreme version of this can be seen in all columns corresponding to the intersecting address for a given incident as most include only the intersecting street name or completely exclude any intersecting data.
- Link to dataset: https://data.sandiego.gov/datasets/police-calls-for-service/
- Number of observations: 466979
- Number of variables: 14
- Description of variables: 
    - `INCIDENT_NUM` - Unique incident number identifier
    - `DATE_TIME` - Recorded date and time of incident
    - `DAY_OF_WEEK` - Day of the week incident occurred
    - `ADDRESS_NUMBER_PRIMARY` - Primary address number
    - `ADDRESS_DIR_PRIMARY` - Primary address direction
    - `ADDRESS_ROAD_PRIMARY` - Primary address street name
    - `ADDRESS_SFX_PRIMARY` - Primary address suffix
    - `ADDRESS_DIR_INTERSECTING` - Intersecting address direction
    - `ADDRESS_ROAD_INTERSECTING` - Intersecting address street name
    - `ADDRESS_SFX_INTERSECTING` - Intersecting address suffix
    - `CALL_TYPE` - Code corresponding to type of incident
    - `DISPOSITION` - Code corresponding to outcome or resolution of the incident
    - `BEAT` - Geographic patrol zone or district number that the incident originated from
    - `PRIORITY` - Urgency level assigned to the incident
- Descriptions of any shortcomings: 
    - There are large amounts of missing fields in the primary address suffix column and primary address direction column.  This is likely due to many streets not having a suffix or direction, but this may cause complications for data analysis as entries with missing fields don't necessarily correlate to data that can be thrown out.  An even more extreme version of this can be seen in all columns corresponding to the intersecting address for a given incident as most include only the intersecting street name or completely exclude any intersecting data.

Dataset Name: Police beats (geo-data)
- Link to dataset: https://data.sandiego.gov/datasets/police-beats/
- Number of observations: 135
- Number of variables: 6
- Description of variables:
    - `objectid` - Unique object identifier
    - `beat` - Geographic patrol zone
    - `div` - Police division beat is contained in
    - `serv` - Service area beat is contained in
    - `name` - Name of geographic patrol zone beat is in
    - `geometry` - Geometry of beat
- Descriptions of any shortcomings: 
    - There are no crime statistics that correspond with the geographic beat data, so in order to create interesting choropleth maps we have to manually add associated crime data columns ourselves.

Dataset Name: Police disposition codes
- Link to dataset: https://data.sandiego.gov/datasets/police-calls-disposition-codes/
- Number of observations: 18
- Number of variables: 2
- Description of variables:
    - `dispo_code` - Unique disposition code
    - `description` - Short description of code
- Descriptions of any shortcomings: 
    - There are no shortcomings with this dataset as it is simply a reference guide for the San Diego Police Department disposition codes.

Dataset Name: Police call type definitions
- Link to dataset: http://seshat.datasd.org/police_calls_for_service/pd_cfs_calltypes_datasd.csv
- Number of observations: 289
- Number of variables: 2
- Description of variables:
    - `call_type` - Unique call type identifier
    - `description` - Short description of call type identifier
- Descriptions of any shortcomings: 
    - There are no shortcomings with this dataset as it is simply a reference guide for the San Diego Police Department call type codes.

## File structure
```
Analytics-Driven-Resource-Deployment-for-San-Diego-Emergency-Services/
├── data/                           # Organized datasets
│   ├── 00-raw/                        # Raw .csv data
│   ├── 01-processed/                  # Processed .csv data
|   |   └── step4_scenarios/              # Resource deployment visualizations and processed data
│   └── EDA_outputs/                   # Exploratory data analysis visualization
├── modules/                         # Data processing and prediction scripts
│   ├── step1_clean.py                 # Data cleaning script
│   ├── step2_eda.py                   # Exploratory data analysis script
│   ├── step3_forecasting.py           # Hotspot and baseline forecasting script
│   └── step4_scenario_analysis.py     # Resource deployment analysis script
├── data_pipeline.ipynb              # Entire data processing pipeline with visualizations notebook
└── README.MD                        # Documentation
```
## List of 3rd-party modules
1. pandas
2. seaborn
3. matplotlib
4. 

## How to run the code

### Step 1-  Clean and process the Dataset from the Raw CSV
- Navigate to the modules directory and run the `step1_clean.py` script
    - Takes the **raw Police Calls-for-Service CSV** and produces an **analysis-ready dataset** in two stages
        - `data/00-raw/pd_calls_for_service_2025_datasd.csv`   ← raw input
        - `data/00-raw/calltypes_category_mapping_table.csv`   ← external mapping table
    - Stage 1
        - **Initial cleaning**
            -  Convert the `DATE_TIME` column to pandas datetime
            - Drop columns that are mostly null:
                - `ADDRESS_DIR_INTERSECTING`
                - `ADDRESS_ROAD_INTERSECTING`
                - `ADDRESS_SFX_INTERSECTING`
            - Drop rows from the following columns based on:
                - Existence of any NaNs
                    - `ADDRESS_ROAD_PRIMARY`
                    - `CALL_TYPE`
                    - `DISPOSITION`
                - Extraneous values
                    - `beat`
                        - A value of -1 is not associated with any SDPD beats
            - Result is exported as `df_cleaned.csv` 
    - Stage 2: 
        - **Category mapping and risk flagging**
            - Map each `CALL_TYPE` to a high-level `CALL_TYPE_CATEGORY` via an external lookup table
            - Map each `DISPOSITION` code to a `DISPOSITION_CATEGORY` label
            - Create an `IS_HIGH_RISK` boolean flag


## Pipeline steps

### Stage 1: Further cleaning (categories + risk flag)

#### (a) Build call-type mapping

Read `calltypes_category_mapping_table.csv`.

The mapping table must have **at least 3 columns**:

| Column position | Role |
|---|---|
| 1st (`call_type`) | Lookup key — matches the `CALL_TYPE` column in the main dataset |
| 2nd | (ignored) |
| 3rd | Category label to assign |

The script builds a `dict` from this file and applies it with `pandas.Series.map`.

Rows whose `CALL_TYPE` is **not found** in the mapping table are reported and then **dropped** by default.

---

#### (b) Add `CALL_TYPE_CATEGORY`

For every row in the dataset:

- Cast `CALL_TYPE` to string
- Look up the category in the mapping dict
- If no match is found, the row is flagged as unmapped

Default behaviour (`drop_unmapped=True`) removes all unmapped rows.

---

#### (c) Add `DISPOSITION_CATEGORY` and `IS_HIGH_RISK`

A hard-coded mapping converts raw disposition codes to an set of actual outcome labels:

| Category | Codes |
|---|---|
| Cancelled | `W`, `X`, `CAN` |
| Duplicate | `DUP`, `V` |
| Arrest | `A`, `AB`, `AHR` |
| Reported | `R`, `RB`, `RHR` |
| Closed | `K`, `KB`, `KHR` |
| Unfounded | `U` |
| Vehicle | `S` |
| Other | `O`, `OHR` |

The risk flag is set by a simple suffix rule: IS_HIGH_RISK = True if DISPOSITION code ends with "HR"

Any disposition code not listed above is treated as unmapped and dropped by default.

---

## Files used

| File | Description |
|---|---|
| `data/00-raw/pd_calls_for_service_2025_datasd.csv` | Raw police calls-for-service dataset. |
| `data/00-raw/calltypes_category_mapping_table.csv` | External lookup table with ≥ 3 columns (1st = `call_type`, 3rd = category). |

## Files produced

| File | Description |
|---|---|
| `data/01-processed/df_cleaned.csv` | Intermediate output after Stage 0. |
| `data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv` | Final cleaned dataset with `CALL_TYPE_CATEGORY`, `DISPOSITION_CATEGORY`, and `IS_HIGH_RISK` added. |

---

## How to run

From the repo root:

```bash
python modules/clean_further.py
```

### Changing paths

Edit the four path variables at the top of `main()`:

```python
raw_csv_path              = "../data/00-raw/pd_calls_for_service_2025_datasd.csv"
mapping_csv_path          = "../data/00-raw/calltypes_category_mapping_table.csv"
cleaned_intermediate_path = "../data/01-processed/df_cleaned.csv"
out_path                  = "../data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv"
```

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
