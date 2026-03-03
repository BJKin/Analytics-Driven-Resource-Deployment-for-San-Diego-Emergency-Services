import re
from typing import Union
from pathlib import Path

import pandas as pd

# ----------------------------
# Utility functions
# ----------------------------

def norm_col(col: str) -> str:
    """
    Normalize a column name to a stable, code-friendly format.
    - lowercase
    - replace "/" with " or "
    - replace spaces with "_"
    - remove other symbols (keep letters/numbers/_)
    - collapse multiple underscores
    """
    col = col.strip().lower()
    col = col.replace("/", " or ")
    col = col.replace(" ", "_")
    col = re.sub(r"[^a-z0-9_]+", "", col)   # drop other symbols
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def ensure_parent_dir(file_path: Union[str, Path]) -> None:
    """Create parent directory if it does not exist."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Core pipeline
# ----------------------------

def build_calltype_mapping(mapping_csv_path: Union[str, Path]) -> dict:
    """
    Read mapping table CSV and build a dict: CALL_TYPE (key_col) -> category (3rd column).
    Note: Your notebook normalized ONLY the 3rd column name, and used 'call_type' as key.
    """
    map_df = pd.read_csv(mapping_csv_path)

    # Assertions for schema sanity
    assert map_df.shape[1] >= 3, (
        "Mapping table must have at least 3 columns. "
        "Expected: [call_type, ..., category_col_as_3rd]."
    )
    assert "call_type" in [c.strip().lower() for c in map_df.columns], (
        "Mapping table must contain a column named 'call_type' (case-insensitive)."
    )

    # Make columns comparable (case-insensitive column lookup)
    cols_lower = {c.strip().lower(): c for c in map_df.columns}
    key_col_original = cols_lower["call_type"]

    # Normalize ONLY the 3rd column name (index 2) as you did in the notebook
    old_cat_col = map_df.columns[2]
    new_cat_col = norm_col(old_cat_col)
    map_df = map_df.rename(columns={old_cat_col: new_cat_col})

    # Build mapping dict
    calltype_mapping = map_df.set_index(key_col_original)[new_cat_col].to_dict()

    # Assertions for mapping quality
    assert isinstance(calltype_mapping, dict) and len(calltype_mapping) > 0, (
        "calltype_mapping is empty. Check your mapping CSV content."
    )

    return calltype_mapping


def add_call_type_category(
    df: pd.DataFrame,
    calltype_mapping: dict,
    call_type_col: str = "CALL_TYPE",
    out_col: str = "CALL_TYPE_CATEGORY",
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Add CALL_TYPE_CATEGORY via mapping dict; optionally drop rows where mapping fails.
    """
    assert call_type_col in df.columns, f"Missing required column: {call_type_col}"

    df2 = df.copy()

    # Apply mapping (convert to str for stable keys)
    df2[out_col] = df2[call_type_col].astype(str).map(calltype_mapping)

    unmapped_mask = df2[out_col].isna()
    num_unmapped = int(unmapped_mask.sum())
    print(f"Number of unmapped rows ({out_col}): {num_unmapped}")

    if num_unmapped > 0:
        unmapped_counts = df2.loc[unmapped_mask, call_type_col].value_counts()
        print("\nTop unmapped CALL_TYPE values (count):")
        print(unmapped_counts.head(30).to_string())

    if drop_unmapped:
        df2 = df2.loc[~unmapped_mask].copy()

    # If not dropping, fill tag to keep data consistent
    df2[out_col] = df2[out_col].fillna("unmapped_unknown")

    # Post-conditions
    assert out_col in df2.columns, f"Failed to create {out_col}"
    if drop_unmapped:
        assert df2[out_col].isna().sum() == 0, "There should be no NaN after dropping unmapped rows."

    return df2


def add_season(
    df: pd.DataFrame,
    datetime_col: str = "DATE_TIME",
    out_col: str = "SEASON",
) -> pd.DataFrame:
    """
    Add SEASON based on month extracted from DATE_TIME.
    """
    assert datetime_col in df.columns, f"Missing required column: {datetime_col}"

    df2 = df.copy()

    # Parse datetime; coerce errors to NaT then assert not too many failures
    dt = pd.to_datetime(df2[datetime_col], errors="coerce")
    bad_dt = int(dt.isna().sum())
    print(f"Datetime parse failures in {datetime_col}: {bad_dt}")

    # If you expect DATE_TIME to be always valid, keep this strict assertion:
    assert bad_dt == 0, (
        f"Found {bad_dt} invalid datetime rows in '{datetime_col}'. "
        "Fix the input data or relax this assertion."
    )

    month = dt.dt.month
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df2[out_col] = month.map(season_map)

    # Post-conditions
    assert df2[out_col].isna().sum() == 0, "Season mapping produced NaN values unexpectedly."
    return df2


def add_disposition_category_and_risk(
    df: pd.DataFrame,
    dispo_col: str = "DISPOSITION",
    out_col: str = "DISPOSITION_CATEGORY",
    risk_col: str = "IS_HIGH_RISK",
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Map DISPOSITION codes to high-level DISPOSITION_CATEGORY and create IS_HIGH_RISK.
    Drop rows where mapping fails if drop_unmapped=True.
    """
    assert dispo_col in df.columns, f"Missing required column: {dispo_col}"

    df2 = df.copy()

    # Normalize codes
    codes = df2[dispo_col].astype("string").str.strip().str.upper()

    # Mapping based on your notebook
    dispo_mapping = {
        # Cancelled / No response
        "W": "Cancelled / No response",
        "X": "Cancelled / No response",
        "CAN": "Cancelled / No response",

        # Duplicate
        "DUP": "Duplicate",
        "V": "Duplicate",

        # Arrest
        "A": "Arrest",
        "AB": "Arrest",
        "AHR": "Arrest",

        # Report taken
        "R": "Report taken",
        "RB": "Report taken",
        "RHR": "Report taken",

        # No report / No further action
        "K": "No report / No further action",
        "KB": "No report / No further action",
        "KHR": "No report / No further action",

        # Unfounded
        "U": "Unfounded",

        # Vehicle-related outcome
        "S": "Vehicle related",

        # Other
        "O": "Other",
        "OHR": "Other",
    }

    # Apply mapping
    df2[out_col] = codes.map(dispo_mapping)

    # Risk flag
    df2[risk_col] = codes.str.endswith("HR")

    # Report mapping failure
    unmapped_mask = df2[out_col].isna()
    unmapped_n = int(unmapped_mask.sum())
    print(f"Number of unmapped rows ({out_col}): {unmapped_n}")

    if unmapped_n > 0:
        print("\nExamples of unmapped DISPOSITION codes:")
        ex = df2.loc[unmapped_mask, dispo_col].astype("string").head(20)
        print(ex.to_string(index=False))

        print("\nTop unmapped DISPOSITION values (count):")
        print(df2.loc[unmapped_mask, dispo_col].value_counts().head(30).to_string())

    if drop_unmapped:
        df2 = df2.loc[~unmapped_mask].copy()
        assert df2[out_col].isna().sum() == 0, "There should be no NaN after dropping unmapped rows."
    else:
        df2[out_col] = df2[out_col].fillna("unmapped_unknown")

    return df2


# ----------------------------
# Main entry
# ----------------------------

def main() -> None:
    # ----- Paths you may want to change -----
    # Input: your df_cleaned exported as a CSV
    cleaned_input_path = "../data/01-processed/df_cleaned.csv"

    # Mapping table
    mapping_csv_path = "../data/00-raw/calltypes_category_mapping_table.csv"

    # Output
    out_path = "../data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv"

    # ----- Load input dataset -----
    assert Path(cleaned_input_path).exists(), (
        f"Input file not found: {cleaned_input_path}\n"
        "Please export df_cleaned to a CSV first, or change cleaned_input_path."
    )
    df_cleaned = pd.read_csv(cleaned_input_path)

    # Basic schema assertions
    required_cols = ["CALL_TYPE", "DATE_TIME", "DISPOSITION"]
    for c in required_cols:
        assert c in df_cleaned.columns, f"Missing required column in input CSV: {c}"

    # ----- Build call type mapping -----
    assert Path(mapping_csv_path).exists(), f"Mapping table not found: {mapping_csv_path}"
    calltype_mapping = build_calltype_mapping(mapping_csv_path)

    # ----- Pipeline steps -----
    df_cleaned_v2 = df_cleaned.copy()

    df_cleaned_v2 = add_call_type_category(
        df_cleaned_v2,
        calltype_mapping=calltype_mapping,
        call_type_col="CALL_TYPE",
        out_col="CALL_TYPE_CATEGORY",
        drop_unmapped=True,
    )

    df_cleaned_v2 = add_season(
        df_cleaned_v2,
        datetime_col="DATE_TIME",
        out_col="SEASON",
    )

    df_cleaned_v2 = add_disposition_category_and_risk(
        df_cleaned_v2,
        dispo_col="DISPOSITION",
        out_col="DISPOSITION_CATEGORY",
        risk_col="IS_HIGH_RISK",
        drop_unmapped=True,
    )

    print(f"\nTotal observations after cleaning: {len(df_cleaned_v2):,}\n")

    # ----- Export -----
    ensure_parent_dir(out_path)
    df_cleaned_v2.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to: {out_path}")

if __name__ == "__main__":
    main()