"""Trim OpenAQ to the same date window as Open-Meteo so both datasets align."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENMETEO_PATH = PROJECT_ROOT / "data/raw/openmeteo.parquet"
OPENAQ_PATH = PROJECT_ROOT / "data/raw/openaq.parquet"
OPENAQ_BACKUP_PATH = PROJECT_ROOT / "data/raw/openaq_full.parquet"


def align_datasets(
    openmeteo_path: Path = OPENMETEO_PATH,
    openaq_path: Path = OPENAQ_PATH,
    backup_full_openaq: bool = True,
) -> dict:
    met_df = pd.read_parquet(openmeteo_path)
    aq_df = pd.read_parquet(openaq_path)

    met_df["timestamp"] = pd.to_datetime(met_df["timestamp"], utc=True)
    aq_df["timestamp"] = pd.to_datetime(aq_df["timestamp"], utc=True)

    start = met_df["timestamp"].min()
    end = met_df["timestamp"].max()

    if backup_full_openaq and not OPENAQ_BACKUP_PATH.exists() and len(aq_df) > 0:
        aq_df.to_parquet(OPENAQ_BACKUP_PATH, index=False)
        print(f"Backed up full OpenAQ dataset to {OPENAQ_BACKUP_PATH}")

    aligned_aq = aq_df[(aq_df["timestamp"] >= start) & (aq_df["timestamp"] <= end)].copy()
    aligned_aq.to_parquet(openaq_path, index=False)

    summary = {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "openmeteo_rows": len(met_df),
        "openaq_rows_before": len(aq_df),
        "openaq_rows_after": len(aligned_aq),
        "openaq_locations": aligned_aq["location_name"].nunique() if not aligned_aq.empty else 0,
        "met_lat": float(met_df["lat"].iloc[0]),
        "met_lon": float(met_df["lon"].iloc[0]),
    }

    print("Aligned datasets:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary


if __name__ == "__main__":
    align_datasets()
