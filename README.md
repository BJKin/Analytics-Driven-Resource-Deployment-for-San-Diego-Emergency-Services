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

    | Variable | Description |
    |----------|-------------|
    | `INCIDENT_NUM`  | Unique incident number identifier |
    | `DATE_TIME`  | Recorded date and time of incident |
    | `DAY_OF_WEEK`  | Day of the week incident occurred  |
    | `ADDRESS_NUMBER_PRIMARY`  | Primary address number |
    | `ADDRESS_DIR_PRIMARY`  | Primary address direction |
    | `ADDRESS_ROAD_PRIMARY`  | Primary address street name |
    | `ADDRESS_SFX_PRIMARY`  | Primary address suffix |
    | `ADDRESS_DIR_INTERSECTING`  | Intersecting address direction |
    | `ADDRESS_ROAD_INTERSECTING`  | Intersecting address street name |
    | `ADDRESS_SFX_INTERSECTING`  | Intersecting address suffix |
    | `CALL_TYPE`  |  Code corresponding to type of incident |
    | `DISPOSITION`  | Code corresponding to outcome or resolution of the incident |
    | `BEAT`  | Geographic patrol zone or district number that the incident originated from |
    | `PRIORITY`  | Urgency level assigned to the incident |

- Descriptions of any shortcomings: 
    - There are large amounts of missing fields in the primary address suffix column and primary address direction column.  This is likely due to many streets not having a suffix or direction, but this may cause complications for data analysis as entries with missing fields don't necessarily correlate to data that can be thrown out.  An even more extreme version of this can be seen in all columns corresponding to the intersecting address for a given incident as most include only the intersecting street name or completely exclude any intersecting data.

Dataset Name: Police beats (geo-data)
- Link to dataset: https://data.sandiego.gov/datasets/police-beats/
- Number of observations: 135
- Number of variables: 6
- Description of variables:

    | Variable | Description |
    |----------|-------------|
    | `objectid` | Unique object identifier |
    | `beat` | Geographic patrol zone |
    | `div` | Police division beat is contained in |
    | `serv` | Service area beat is contained in |
    | `name` | Name of geographic patrol zone beat is in |
    | `geometry` | Geometry of beat |

- Descriptions of any shortcomings: 
    - There are no crime statistics that correspond with the geographic beat data, so in order to create interesting choropleth maps we have to manually add associated crime data columns ourselves.

Dataset Name: Police disposition codes
- Link to dataset: https://data.sandiego.gov/datasets/police-calls-disposition-codes/
- Number of observations: 18
- Number of variables: 2
- Description of variables:

    | Variable | Description |
    |----------|-------------|
    | `dispo_code` | Unique disposition code |
    | `description` | Short description of code |

- Descriptions of any shortcomings: 
    - There are no shortcomings with this dataset as it is simply a reference guide for the San Diego Police Department disposition codes.

Dataset Name: Police call type definitions
- Link to dataset: http://seshat.datasd.org/police_calls_for_service/pd_cfs_calltypes_datasd.csv
- Number of observations: 289
- Number of variables: 2
- Description of variables:

    | Variable | Description |
    |----------|-------------|
    | `call_type` | Unique call type identifier |
    | `description` | Short description of call type identifier |

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
1. `pandas`
2. `seaborn`
3. `matplotlib`
4. `tabulate`
5. `geopandas`

## How to run the code

### Step 1-  Clean and process the Dataset from the Raw CSV
- Navigate to the root directory and run the `step1_clean.py` script
    ```bash
    python modules/step1_clean.py
    ```
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
                - Read `calltypes_category_mapping_table.csv`
                - The script builds a `dict` from this file and applies it with `pandas.Series.map`
                - Rows whose `CALL_TYPE` is **not found** in the mapping table are reported and then **dropped** by default
            - Map each `DISPOSITION` code to a `DISPOSITION_CATEGORY` label
                - For every row in the dataset:
                    - Cast `CALL_TYPE` to string
                    - Look up the category in the mapping dict
                    - If no match is found, the row is flagged as unmapped
                    - A hard-coded mapping converts raw disposition codes to an set of actual outcome labels:

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

            - Create an `IS_HIGH_RISK` boolean flag
                - The risk flag is set by a simple suffix rule: 
                    - `IS_HIGH_RISK = True` if `DISPOSITION` code ends with `HR`

### Step 2- Exploratory Data Analysis
- Navigate to the root directory and run the `step2_eda.py` script
    ```bash
    python modules/step2_eda.py
    ```
    - Generates **11 EDA visualizations** from the cleaned dataset
        - Input:
            - `data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv`- Output of `step1_clean.py`
            - `data/00-raw/pd_beats_datasd.geojson` - Beat boundary polygons
        - Output:
            - All visualizations are saved to `./data/EDA_outputs/`
    - Visualization Functions:

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

    - Helper functions:

        | Function | Description |
        |----------|-------------|
        | `add_time_features` | derives `HOUR`, `DOW`, `MONTH`, `DATE`, `SEASON` from `DATE_TIME` |
        | `add_beat_key`      | standardizes the `BEAT` column to a clean `BEAT_KEY` string |
        | `summary_stats`     | prints shape, missing values, dtypes, and numerical summary |

### Step 3- Hotspost & Forecasting
- •	Navigate to the root directory and run Step 3:
    ```bash
    python modules/step3_upd.py
    ```
    - Generates
        - Input:
            - `data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv`- Output of `step1_clean.py`
            - `data/00-raw/pd_beats_datasd.geojson` - Beat boundary polygons
        - Output:
            - `.data/01-processed/step3_hotspots_beats.csv`
            - `.data/01-processed/step3_hotspots_beat_season.csv`
            - `.data/01-processed/step3_forecast_predictions.csv`
            - `.data/01-processed/step3_forecast_metrics.csv`
            - `.data/01-processed/step3_actual_calls_top3.png`
            - `.data/01-processed/step3_predicted_calls_top3.png`
            - `.data/01-processed/step4_resource_deployment.csv ` - (base allocation table used by Step 4)
    - Core Pipeline Functions (Callled by step3_upd.py):

        | # | Function | Output file | Description |
        |---|---|---|---|
        | 1 | `build_features_v2_from_processed()` | `pd_calls_for_service_2025_datasd_features_v2.csv` | Builds the Step3/4-ready feature table from the cleaned dataset. |
        | 2 | `step3_hotspots() ` | `step3_hotspots_beats.csv, step3_hotspots_beat_season.csv` | Computes hotspot call counts by beat (and beat×season).  |
        | 3 | `step3_baseline_forecast() ` | `step3_forecast_predictions.csv, step3_forecast_metrics.csv` | Runs baseline forecasting and produces prediction table + metrics.  |
        | 4 | `step4_resource_deployment() ` | `step4_resource_deployment.csv` | Generates the base Step 4 allocation table (input for scenario analysis).  |

    - Visualization Functions:
        | # | Function | Output file | Description |
        |---|---|---|---|
        | 1 | `main()` | `step3_actual_calls_top3.png` | Saves a line plot of actual daily calls for the top-3 hotspot beats over the test window.  |
        | 2 | ` ` | `step3_predicted_calls_top3.png` | Saves a line plot of baseline predicted daily calls for the top-3 hotspot beats over the test window.  |



## Step 4 : Scenario Analysis for Resource Deployment

### Description

It generates multiple deployment policies (scenarios), evaluates them with **proxy operational metrics** (coverage/shortfall under a simplified capacity assumption), and outputs **CSV summaries + plots** for reporting.

------

### Inputs (required files)

- `data/01-processed/step4_resource_deployment.csv`
  Base beat×shift demand summary table used as Step 4 upgrade input (contains `BEAT_KEY`, `SHIFT`, `AVG_CALLS`, `HIGH_RISK_RATIO`).
  *Impact:* determines the baseline demand distribution; all scenarios allocate units based on this table.
- `data/01-processed/step3_hotspots_beats.csv`
  Step 3 hotspot ranking table (used to select Top-K hotspot beats for the hotspot-protect scenario).
  *Impact:* changes which beats receive minimum guaranteed units under hotspot-protect.

------

### Core Pipeline Functions (Called by `step4_scenario_analysis.py`)

| #    | Function                       | Output file                                                  | Description                                                  |
| ---- | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | `build_scenario_allocations()` | `step4_scenario_allocations.csv`                             | Builds beat×shift allocations for each scenario: `status_quo`, `risk_aware_{w}`, `hotspot_protect_topK_minM`. |
| 2    | `add_capacity_metrics()`       | `step4_scenario_allocations.csv` (updated)                   | Adds simplified capacity model + proxy KPIs: `CAPACITY`, `COVERAGE`, `SHORTFALL` for every beat×shift×scenario row. |
| 3    | `summarize_scenarios()`        | `step4_scenario_summary.csv`                                 | Aggregates metrics into one row per scenario (e.g., `coverage_ratio`, `total_shortfall`, `peak_shortfall`, `hotspot_shortfall`). |
| 4    | `plot_summary_bars()`          | `scenario_coverage_ratio.png`, `scenario_total_shortfall.png` | Saves comparison bar charts for coverage ratio and total shortfall across scenarios. |
| 5    | `plot_top_beats_allocation()`  | `allocation_top_beats_<shift>.png`                           | Saves grouped bar chart comparing how top-demand beats are allocated across scenarios (for a selected shift, default `Day`). |

------

### Scenarios Implemented

- `status_quo`: allocate proportional to `AVG_CALLS`
- `risk_aware_{w}`: allocate proportional to `AVG_CALLS * (1 + (w-1) * HIGH_RISK_RATIO)` (default `w ∈ {1.5, 2.0}`)
- `hotspot_protect_topK_minM`: top-K hotspot beats get at least M units per shift (default `K=5`, `M=2`), remaining units allocated proportionally

------

### How to Run (with parameter meanings)

```bash
python modules/step4_scenario_analysis.py \
  --base-allocation data/01-processed/step4_resource_deployment.csv \
  --hotspots data/01-processed/step3_hotspots_beats.csv \
  --outdir data/01-processed/step4_scenarios \
  --total-units 50 \
  --mu-per-hour 3.0 \
  --risk-weights 1.5 2.0 \
  --top-k 5 \
  --min-units 2 \
  --plot-shift Day
```

**Parameters (what it is / what it affects):**

- `--base-allocation`
  Base Step 4 table (beat×shift demand summary).
  *Impact:* changes the demand inputs; allocations and metrics change accordingly.
- `--hotspots`
  Hotspot ranking table from Step 3.
  *Impact:* determines which beats are treated as “hotspots” in hotspot-protect.
- `--outdir`
  Output folder for CSVs and plots.
  *Impact:* does not change results, only where outputs are written.
- `--total-units`
  Total units available **per shift** (Night/Day/Evening).
  *Impact:* higher values increase capacity, raise coverage ratio, reduce shortfall across all scenarios.
- `--mu-per-hour`
  Capacity assumption μ: calls handled per unit per hour.
  *Impact:* higher μ increases capacity without changing allocations, improving coverage and reducing shortfall (proxy effect).
- `--risk-weights`
  High-risk multipliers used to build `risk_aware_{w}` scenarios (e.g., 1.5 and 2.0).
  *Impact:* larger weights shift more units toward beats with higher `HIGH_RISK_RATIO`.
- `--top-k`
  Number of hotspot beats to protect in hotspot-protect.
  *Impact:* larger K spreads guarantees across more beats; smaller K concentrates protection on fewer beats.
- `--min-units`
  Minimum guaranteed units per hotspot beat per shift (hotspot-protect).
  *Impact:* larger values strengthen hotspot guarantees but reduce remaining units available for proportional allocation.
- `--plot-shift`
  Which shift to visualize in `allocation_top_beats_<shift>.png`.
  *Impact:* affects only the saved plot (not the CSV metrics).

------

### Outputs written to `data/01-processed/step4_scenarios/`

- `step4_scenario_allocations.csv`
  Beat × shift × scenario **allocation detail table**. Each row is one `(BEAT_KEY, SHIFT, SCENARIO)` cell with allocated `UNITS` and proxy KPIs such as `CAPACITY`, `COVERAGE`, and `SHORTFALL`.
- `step4_scenario_summary.csv`
  **Scenario-level summary table** (one row per scenario), aggregating metrics like `coverage_ratio`, `total_shortfall`, `peak_shortfall` (top demand cells), and `hotspot_shortfall` (hotspot beats only). Used to quickly compare policy trade-offs.
- `scenario_coverage_ratio.png`
  Bar chart comparing **coverage ratio** across scenarios (`total_coverage / total_demand`), showing which policy covers more demand under the same resource budget.
- `scenario_total_shortfall.png`
  Bar chart comparing **total shortfall** across scenarios (`sum(max(0, demand - capacity))`), showing which policy leaves less unmet demand overall.
- `allocation_top_beats_<shift>.png`
  Grouped bar chart comparing allocations for **top-demand beats** in a selected shift (default `Day`), showing where each policy concentrates or redistributes `UNITS`.
