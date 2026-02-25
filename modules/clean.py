import pandas as pd
import os

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
    os.makedirs('../data/01-processed', exist_ok=True)

    # Load dataframe and get column names
    df = pd.read_csv(file)
    column_names = list(df.columns)

    # Convert date column to datetime object for potential time series analysis using Pandas
    for column in column_names:
        if 'date' in column.lower():
            df[column] = pd.to_datetime(df[column])

    # Remove specified columns
    df_cleaned = df.drop(labels=columns_to_remove, axis='columns')

    # Drop rows where values are missing from specified columns
    df_cleaned = df_cleaned.dropna(subset=columns_to_clean).reset_index(drop=True)

    # Save cleaned dataframe to .csv for later manipulation
    df_cleaned.to_csv(f'../data/01-processed/df_cleaned.csv', index=False)

    return df_cleaned


def main():
    file_path = '../data/00-raw/pd_calls_for_service_2025_datasd.csv'
    columns_to_remove = ['ADDRESS_DIR_INTERSECTING', 'ADDRESS_ROAD_INTERSECTING', 'ADDRESS_SFX_INTERSECTING']
    columns_to_clean = ['ADDRESS_ROAD_PRIMARY', 'CALL_TYPE', 'DISPOSITION']

    cleaned_df = clean_data(file_path, columns_to_remove, columns_to_clean)
    print(cleaned_df.head())

if __name__ == "__main__":
    main()