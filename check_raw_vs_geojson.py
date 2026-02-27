import pandas as pd
import geopandas as gpd

RAW_CSV = "data/00-raw/pd_calls_for_service_2025_datasd.csv"
CLEANED_CSV = "data/01-processed/pd_calls_for_service_2025_datasd_cleaned.csv"
BEATS_GEOJSON = "data/00-raw/pd_beats_datasd.geojson"

raw = pd.read_csv(RAW_CSV)
cleaned = pd.read_csv(CLEANED_CSV)
beats_gdf = gpd.read_file(BEATS_GEOJSON)

def pick_beat_col(df):
    if "BEAT" in df.columns:
        return "BEAT"
    cols = [c for c in df.columns if "beat" in c.lower()]
    assert len(cols) > 0
    return cols[0]

beat_col_raw = pick_beat_col(raw)
beat_col_clean = pick_beat_col(cleaned)

valid_beats = set(beats_gdf["beat"].astype(str).str.strip())

raw_beats = raw[beat_col_raw].astype(str).str.strip()
clean_beats = cleaned[beat_col_clean].astype(str).str.strip()

print("raw beat col:", beat_col_raw)
print("cleaned beat col:", beat_col_clean)

print("raw -1 count:", (raw_beats == "-1").sum())
print("cleaned -1 count:", (clean_beats == "-1").sum())

print("raw in geojson ratio:", raw_beats.isin(valid_beats).mean())
print("cleaned in geojson ratio:", clean_beats.isin(valid_beats).mean())

raw_invalid = raw_beats[~raw_beats.isin(valid_beats)].value_counts().head(20)
clean_invalid = clean_beats[~clean_beats.isin(valid_beats)].value_counts().head(20)

print("\nraw invalid beat codes (top 20):")
print(raw_invalid)

print("\ncleaned invalid beat codes (top 20):")
print(clean_invalid)