
# Step 1 Clean Dataset (from Raw CSV)

This file documents the `clean_further.py` script, which takes the **raw Police Calls-for-Service CSV** and produces an **analysis-ready dataset** in two stages.

It is designed to run **directly from the raw data**:

- `data/00-raw/pd_calls_for_service_2025_datasd.csv`   ← raw input
- `data/00-raw/calltypes_category_mapping_table.csv`   ← external mapping table

So you do **not** need to run any notebook cells or prepare any intermediate files first.

---

## What this script does

The raw CSV contains all police call-for-service records with address fields, call codes, and disposition codes.

This script performs **two stages** of cleaning:

**Stage 0 — Initial cleaning** (previously done in the notebook)
- Parse `DATE_TIME` as datetime
- Drop three intersecting-street columns that are mostly null
- Drop rows where `ADDRESS_ROAD_PRIMARY`, `CALL_TYPE`, or `DISPOSITION` is missing

**Stage 1 — Further cleaning** (category mapping and risk flag)
- Map each `CALL_TYPE` to a high-level `CALL_TYPE_CATEGORY` via an external lookup table
- Map each `DISPOSITION` code to a `DISPOSITION_CATEGORY` label
- Create an `IS_HIGH_RISK` boolean flag

---

## Pipeline steps

### Stage 0: Initial cleaning (raw → df_cleaned)

#### (a) Load and parse

Read `pd_calls_for_service_2025_datasd.csv` and convert the `DATE_TIME` column to pandas datetime.

---

#### (b) Drop intersecting-street columns

Remove three columns that are mostly null and not used in downstream analysis:

- `ADDRESS_DIR_INTERSECTING`
- `ADDRESS_ROAD_INTERSECTING`
- `ADDRESS_SFX_INTERSECTING`

---

#### (c) Drop rows with missing critical fields

Any row where **any** of these three columns is null is removed:

- `ADDRESS_ROAD_PRIMARY`
- `CALL_TYPE`
- `DISPOSITION`

The result is exported as `df_cleaned.csv` for backward compatibility with notebooks that expect this intermediate file.

---

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