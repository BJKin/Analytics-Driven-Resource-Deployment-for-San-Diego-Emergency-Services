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
            - `.data/01-processed/step4_resource_deployment.csv ` - (base allocation table used by Step 4)
    - Visualization Functions:

        | # | Function | Output file | Description |
        |---|---|---|---|
        | 1 | `Top-3 hotspot beats: actual calls (test window)` | `./data/01-processed/step3_actual_calls_top3.png` | Line plot of actual daily call counts for the top 3 hotspot beats over the test window. |
        | 2 | `Top-3 hotspot beats: baseline predicted calls (test window) ` | `./data/01-processed/step3_predicted_calls_top3.png` | Line plot of baseline predicted daily call counts for the top 3 hotspot beats over the test window. |


