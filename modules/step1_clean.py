from typing import Union
from pathlib import Path
import os
import pandas as pd
from step3_forecasting import add_high_risk_flag

# ----------------------------
# Initial cleaning (from raw CSV)
# ----------------------------

def clean_data(file: str, columns_to_remove=[], columns_to_clean=[]) -> pd.DataFrame:
    '''
    Takes a csv file as input, returns a cleaned dataframe and saves the cleaned dataframe as a new .csv
    
    Arguments:
    file-- file path of .csv file to be turned into a dataframe and cleaned
    columns_to_remove-- columns of data to be removed 
    columns_to_clean-- columns of data to be cleaned (if a NaN value is in the specified column, the entire row will be removed)

    Returns:
    cleaned_df-- Pandas dataframe cleaned with the specifications of the user

    '''
    assert isinstance(file, str), 'Input file must be a string'
    assert len(file) > 0, 'Input file string must not be empty'
    assert os.path.exists(file), 'Input file must be a valid path'
    assert os.path.isfile(file), 'Input file must be a valid file'
    assert file[-4:] == '.csv', 'Input file must be a .csv file'

    assert isinstance(columns_to_remove, list), 'Columns to be removed must be a list'
    assert all(isinstance(column, str) for column in columns_to_remove), 'All values in columns to be removed list must be strings'

    assert isinstance(columns_to_clean, list), 'Columns to be cleaned must be a list'
    assert all(isinstance(column, str) for column in columns_to_clean), 'All values in columns to be cleaned list must be strings'

    # Create data directory if it doesn't exist such that the cleaned dataframe can be saved as a .csv
    os.makedirs('./data/01-processed', exist_ok=True)

    # Load dataframe and get column names
    df = pd.read_csv(file)
    column_names = list(df.columns)

    # Convert date column to datetime object for potential time series analysis using Pandas
    for column in column_names:
        if 'date' == column.lower():
            df[column] = pd.to_datetime(df[column])
        if 'beat' == column.lower():
            df = df[df[column] != -1]

    # Remove specified columns
    df_cleaned = df.drop(labels=columns_to_remove, axis='columns')

    # Drop rows where values are missing from specified columns
    df_cleaned = df_cleaned.dropna(subset=columns_to_clean).reset_index(drop=True)

    # Save cleaned dataframe to .csv for later manipulation
    df_cleaned.to_csv(f'./data/01-processed/df_cleaned.csv', index=False)

    return df_cleaned

# ----------------------------
# Core pipeline
# ----------------------------

def build_calltype_mapping(mapping_csv_path: Union[str, Path]) -> dict:
    """
    Read mapping table CSV and build a dict:
    CALL_TYPE (call_type column) -> category (3rd column).

    Arguments:
    mapping_csv_path-- path to .csv with mappings

    Returns:
    calltype_mapping-- dictionary with call type mappings
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

    Arguments:
    df-- Input DataFrame
    calltype_mapping-- Dictionary that maps call type to a high level category
    call_type_col-- Name of call type column (default is "CALL_TYPE")
    out_col-- Name of column that stores call type high level category (default is "CALL_TYPE_CATEGORY")
    drop_unmapped-- Drop any unmapped rows (default is True)

    Returns:
    df2-- Modified input DataFrame with added call type high level category column
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
    '''
    Adds a high level disposition category column to the input DataFrame.
    Calls add_high_risk_flag() to add a Boolean high risk flag column to the input DataFrame.

    Arguments:
    df-- Input DataFrame
    dispo_col-- Name of disposition column (default is "DISPOSITION")
    out_col-- Name of column that stores high level disposition category (default is "DISPOSITION_CATEGORY")
    drop_unmapped-- Drop any unmapped rows (default is True)

    Returns:
    df2-- Modified input DataFrame with added call type high level category column
    """
    '''
    assert dispo_col in df.columns, f"Missing required column: {dispo_col}"

    df2 = df.copy()
    codes = df2[dispo_col].astype("string").str.strip().str.upper()

    dispo_mapping = {
        "W": "Cancelled", "X": "Cancelled", "CAN": "Cancelled",
        "DUP": "Duplicate", "V": "Duplicate",
        "A": "Arrest", "AB": "Arrest", "AHR": "Arrest",
        "R": "Reported", "RB": "Reported", "RHR": "Reported",
        "K": "Closed", "KB": "Closed", "KHR": "Closed",
        "U": "Unfounded",
        "S": "Vehicle",
        "O": "Other", "OHR": "Other",
    }

    df2[out_col] = codes.map(dispo_mapping)

    unmapped_mask = df2[out_col].isna()
    unmapped_n = int(unmapped_mask.sum())
    print(f"Number of unmapped rows ({out_col}): {unmapped_n}")
    if unmapped_n > 0:
        print("\nTop unmapped DISPOSITION values (count):")
        print(df2.loc[unmapped_mask, dispo_col].value_counts().head(30).to_string())

    if drop_unmapped:
        df2 = df2.loc[~unmapped_mask].copy()
    else:
        df2[out_col] = df2[out_col].fillna("unmapped_unknown")

    df2 = add_high_risk_flag(df2)

    assert out_col in df2.columns, f"Failed to create {out_col}"
    assert df2[out_col].isna().sum() == 0, f"{out_col} still contains NaN values."
    assert risk_col in df2.columns, f"Failed to create {risk_col}"

    return df2

def main() -> None:
    # ----- Paths -----
    raw_csv_path = "./data/00-raw/pd_calls_for_service_2025_datasd.csv"
    mapping_csv_path = "./data/00-raw/calltypes_category_mapping_table.csv"
    cleaned_intermediate_path = "./data/01-processed/df_cleaned.csv"
    out_path = "./data/01-processed/pd_calls_for_service_2025_datasd_cleaned_v2.csv"

    # ----- Initial cleaning from raw CSV -----
    print("Initial cleaning (raw -> df_cleaned)")
    columns_to_remove = ['ADDRESS_DIR_INTERSECTING', 'ADDRESS_ROAD_INTERSECTING', 'ADDRESS_SFX_INTERSECTING']
    columns_to_clean = ['ADDRESS_ROAD_PRIMARY', 'CALL_TYPE', 'DISPOSITION']
    df_cleaned = clean_data(raw_csv_path, columns_to_remove, columns_to_clean)

    # Export intermediate result
    df_cleaned.to_csv(cleaned_intermediate_path, index=False)
    print(f"Saved intermediate cleaned CSV to: {cleaned_intermediate_path}\n")

    # ----- Further cleaning (category mapping + risk flag) -----
    print("Further cleaning (categories + risk flag)")
    calltype_mapping = build_calltype_mapping(mapping_csv_path)
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
    df_cleaned_v2.to_csv(out_path, index=False)

    print(f"\nTotal observations after all cleaning: {len(df_cleaned_v2)}")
    print(f"Saved final dataset to: {out_path}")


if __name__ == "__main__":
    main()