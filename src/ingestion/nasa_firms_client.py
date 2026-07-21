"""
NASA FIRMS (Fire Information for Resource Management System) Client
Fetches Near Real-Time thermal anomaly / active fire data.

Why it matters:
  FIRMS provides VIIRS (375m resolution) and MODIS (1km resolution) data on
  fire radiative power (FRP) and brightness temperature. Mapped to H3 hexes,
  this becomes the "treatment" variable in the DoWhy causal DAG - did this
  thermal anomaly CAUSE this AQI spike?

Free API: https://firms.modaps.eosdis.nasa.gov/api/
Requires a free MAP_KEY from: https://firms.modaps.eosdis.nasa.gov/api/map_key/

Usage:
    client = NASAFIRMSClient()
    df = client.fetch_data(bbox="73.7,18.4,74.0,18.7", days_back=7)
    client.process_and_save(df)

Output columns:
    lat, lon, brightness, frp (Fire Radiative Power in MW),
    acq_date, acq_time, satellite, confidence
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Source options: VIIRS_SNPP_NRT (375m, best for cities), MODIS_NRT (1km)
FIRMS_SOURCES = {
    "viirs":  "VIIRS_SNPP_NRT",
    "modis":  "MODIS_NRT",
    "viirs_noaa": "VIIRS_NOAA20_NRT",
}


class NASAFIRMSClient:
    """
    Downloads VIIRS/MODIS thermal anomaly data for a bounding box.
    Maps fire radiative power (FRP) to H3 hexagonal grid cells.
    """

    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    def __init__(self, source: str = "viirs"):
        self.map_key = os.environ.get("NASA_FIRMS_MAP_KEY", "")
        self.source  = FIRMS_SOURCES.get(source, source)

        if not self.map_key:
            print("WARNING: NASA_FIRMS_MAP_KEY not set.")
            print("  Get a free key at: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
            print("  Add to .env: NASA_FIRMS_MAP_KEY=your_key_here")
            print("  Returning empty DataFrame on fetch.")

    def fetch_data(
        self,
        bbox: str = "73.7,18.4,74.0,18.7",
        days_back: int = 7,
    ) -> pd.DataFrame:
        """
        Fetch thermal anomaly data for a bounding box.

        Args:
            bbox:      "west,south,east,north" (same format as OpenAQ)
            days_back: how many days of history to fetch (max 10 for NRT)

        Returns:
            DataFrame with columns:
                lat, lon, brightness, frp, acq_date, acq_time,
                satellite, confidence, bright_t31
        """
        if not self.map_key:
            return pd.DataFrame()

        # FIRMS API format: bbox = west,south,east,north
        # URL pattern: /api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAY_RANGE}
        day_range = min(days_back, 5)  # NRT API max is 5 days for VIIRS/MODIS NRT
        url = f"{self.BASE_URL}/{self.map_key}/{self.source}/{bbox}/{day_range}"

        print(f"Fetching NASA FIRMS data ({self.source}) for bbox {bbox}...")
        print(f"  URL: {url}")

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                if df.empty:
                    print("  No thermal anomalies found in this region/timeframe.")
                    return pd.DataFrame()
                print(f"  Found {len(df)} thermal anomaly records.")
                return self._clean(df)
            elif resp.status_code == 429:
                print("  Rate limited - wait 60s and retry.")
                return pd.DataFrame()
            else:
                print(f"  Error {resp.status_code}: {resp.text[:200]}")
                return pd.DataFrame()
        except Exception as e:
            print(f"  Request failed: {e}")
            return pd.DataFrame()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        # FIRMS CSV has: latitude, longitude, bright_ti4, frp, acq_date, acq_time, ...
        rename = {
            "latitude":   "lat",
            "longitude":  "lon",
            "bright_ti4": "brightness",   # VIIRS channel I4
            "bright_41":  "brightness",   # MODIS channel 21
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Parse acquisition datetime
        if "acq_date" in df.columns and "acq_time" in df.columns:
            df["acq_time_str"] = df["acq_time"].astype(str).str.zfill(4)
            df["timestamp"] = pd.to_datetime(
                df["acq_date"].astype(str) + " " +
                df["acq_time_str"].str[:2] + ":" + df["acq_time_str"].str[2:],
                format="%Y-%m-%d %H:%M",
                utc=True,
                errors="coerce",
            )

        # Keep useful columns
        keep = ["lat", "lon", "brightness", "frp", "acq_date",
                "acq_time", "satellite", "confidence", "timestamp"]
        df = df[[c for c in keep if c in df.columns]]

        # FRP to numeric (sometimes it's a string with "X" for low-confidence)
        if "frp" in df.columns:
            df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)

        return df.dropna(subset=["lat", "lon"])

    def process_and_save(
        self,
        df: pd.DataFrame,
        output_path: str = "data/raw/firms.parquet",
    ) -> None:
        if df.empty:
            print("No FIRMS data to save.")
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow")
        print(f"Saved {len(df)} FIRMS records to {output_path}")

    def aggregate_to_hex(
        self,
        df: pd.DataFrame,
        resolution: int = 8,
    ) -> pd.DataFrame:
        """
        Map FIRMS point data to H3 hexagonal cells.

        Returns DataFrame with:
            h3_index, n_anomalies, total_frp_mw, max_frp_mw,
            mean_brightness, timestamp (hourly bucket)
        """
        if df.empty:
            return pd.DataFrame()

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from graph.h3_mapper import H3Mapper

        mapper = H3Mapper(resolution)
        df = mapper.add_h3_indices(df)

        ts_col = "timestamp" if "timestamp" in df.columns else None
        group_cols = ["h3_index"]
        if ts_col:
            df["hour_bucket"] = df[ts_col].dt.floor("H")
            group_cols.append("hour_bucket")

        agg = df.groupby(group_cols).agg(
            n_anomalies=("frp", "count"),
            total_frp_mw=("frp", "sum"),
            max_frp_mw=("frp", "max"),
            mean_brightness=("brightness", "mean"),
        ).reset_index()

        # Binary indicator: is there a significant thermal anomaly in this hex?
        agg["has_fire"] = (agg["total_frp_mw"] > 1.0).astype(int)

        return agg


if __name__ == "__main__":
    # Pune bounding box test
    client = NASAFIRMSClient(source="viirs")
    df = client.fetch_data(bbox="73.7,18.4,74.0,18.7", days_back=7)
    if not df.empty:
        client.process_and_save(df)
        hex_df = client.aggregate_to_hex(df)
        print("\nHex-aggregated FIRMS data:")
        print(hex_df.head())
    else:
        print("\nNo data (check MAP_KEY in .env).")
        print("In demo mode, FIRMS is simulated as 0 (no anomalies) for all hexes.")
