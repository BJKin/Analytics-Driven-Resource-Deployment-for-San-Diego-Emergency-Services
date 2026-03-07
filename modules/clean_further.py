import re
from typing import Union
from pathlib import Path

import pandas as pd

# ----------------------------
# Utility functions
# ----------------------------

def ensure_parent_dir(file_path: Union[str, Path]) -> None:
    """Create parent directory if it does not exist."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Core pipeline
# ----------------------------

def build_calltype_mapping(mapping_csv_path: Union[str, Path]) -> dict:
    """
    Read mapping table CSV and build a dict:
    CALL_TYPE (call_type column) -> category (3rd column).
    """
    map_df = pd.read_csv(mapping_csv_path)

    # Assertions for schema sanity
    assert map_df.shape[1] >= 3, (
        "Mapping table must have at least 3 columns. "
        "Expected: [call_type, ..., category_col_as_3rd]."
    )
    assert "call_type" in map_df.columns, (
        "Mapping table must contain a column named 'call_type'."
    )

    cat_col = map_df.columns[2]
    key_col = "call_type"

    calltype_mapping = map_df.set_index(key_col)[cat_col].to_dict()

    # Assertion for mapping quality
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
    Add CALL_TYPE_CATEGORY via mapping dict.
    Optionally drop rows where mapping fails.
    """
    assert call_type_col in df.columns, f"Missing required column: {call_type_col}"

    df2 = df.copy()

    # Apply mapping to the main dataset
    df2[out_col] = df2[call_type_col].astype(str).map(calltype_mapping)

    # Report unmapped items
    unmapped_mask = df2[out_col].isna()
    num_unmapped = int(unmapped_mask.sum())
    print(f"Number of unmapped rows ({out_col}): {num_unmapped}")

    if num_unmapped > 0:
        unmapped_counts = df2.loc[unmapped_mask, call_type_col].value_counts()
        print("\nTop unmapped CALL_TYPE values (count):")
        print(unmapped_counts.head(30).to_string())

    # Delete rows whose CALL_TYPE was not mapped
    if drop_unmapped:
        df2 = df2.loc[~unmapped_mask].copy()

    # Keep a fallback fill step for safety / consistency
    df2[out_col] = df2[out_col].fillna("unmapped_unknown")

    # Post-conditions
    assert out_col in df2.columns, f"Failed to create {out_col}"
    assert df2[out_col].isna().sum() == 0, f"{out_col} still contains NaN values."

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
    This matches your latest notebook naming.
    """
    assert dispo_col in df.columns, f"Missing required column: {dispo_col}"

    df2 = df.copy()

    # Normalize codes
    codes = df2[dispo_col].astype("string").str.strip().str.upper()

    # Mapping based on your latest notebook version
    dispo_mapping = {
        # Cancelled / No response
        "W": "Cancelled",
        "X": "Cancelled",
        "CAN": "Cancelled",

        # Duplicate
        "DUP": "Duplicate",
        "V": "Duplicate",

        # Arrest
        "A": "Arrest",
        "AB": "Arrest",
        "AHR": "Arrest",

        # Report taken
        "R": "Reported",
        "RB": "Reported",
        "RHR": "Reported",

        # No report / No further action
        "K": "Closed",
        "KB": "Closed",
        "KHR": "Closed",

        # Unfounded
        "U": "Unfounded",

        # Vehicle-related outcome
        "S": "Vehicle",

        # Other
        "O": "Other",
        "OHR": "Other",
    }

    # Apply mapping
    df2[out_col] = codes.map(dispo_mapping)

    # High-risk flag: ends with "HR"
    df2[risk_col] = codes.str.endswith("HR")

    # Report mapping failure
    unmapped_mask = df2[out_col].isna()
    unmapped_n = int(unmapped_mask.sum())
    print(f"Number of unmapped rows ({out_col}): {unmapped_n}")

    if unmapped_n > 0:
        print("\nTop unmapped DISPOSITION values (count):")
        print(df2.loc[unmapped_mask, dispo_col].value_counts().head(30).to_string())

    # Delete all rows with mapping failure
    if drop_unmapped:
        df2 = df2.loc[~unmapped_mask].copy()
    else:
        df2[out_col] = df2[out_col].fillna("unmapped_unknown")

    # Post-conditions
    assert out_col in df2.columns, f"Failed to create {out_col}"
    assert df2[out_col].isna().sum() == 0, f"{out_col} still contains NaN values."
    assert risk_col in df2.columns, f"Failed to create {risk_col}"

    return df2


def main() -> None:
    # ----- Paths you may want to change -----
    cleaned_input_path = "../data/01-processed/df_cleaned.csv"
    mapping_csv_path = "../data/00-raw/calltypes_category_mapping_table.csv"
    out_path = "../data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv"

    # ----- Load input dataset -----
    assert Path(cleaned_input_path).exists(), (
        f"Input file not found: {cleaned_input_path}\n"
        "Please export df_cleaned to a CSV first, or change cleaned_input_path."
    )
    df_cleaned = pd.read_csv(cleaned_input_path)

    # Basic schema assertions
    required_cols = ["CALL_TYPE", "DISPOSITION"]
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

    df_cleaned_v2 = add_disposition_category_and_risk(
        df_cleaned_v2,
        dispo_col="DISPOSITION",
        out_col="DISPOSITION_CATEGORY",
        risk_col="IS_HIGH_RISK",
        drop_unmapped=True,
    )

    # ----- Export -----
    ensure_parent_dir(out_path)
    df_cleaned_v2.to_csv(out_path, index=False)

    print(f"Total observations after cleaning: {len(df_cleaned_v2)}\n")
    print(f"Saved cleaned dataset to: {out_path}")


if __name__ == "__main__":
    main()
