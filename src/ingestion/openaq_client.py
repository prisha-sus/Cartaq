"""
OpenAQ v3 Air Quality Client
Fetches PM2.5 measurements for sensors in the Pune bounding box.

If the API is unavailable or slow, falls back to synthetic data so the
rest of the pipeline (graph_builder -> train -> dashboard) can still run.
"""

import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Tunables ─────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 15    # seconds per HTTP request
DAYS_BACK        = 2     # how many days of history to fetch
PAGE_LIMIT       = 100   # records per page (small = less likely to time out)
MAX_SENSORS      = 8     # cap sensors to keep runtime under ~3 minutes
INTER_SENSOR_SLEEP = 0.5 # polite delay between sensors (seconds)


def _flush(msg):
    """Print with immediate stdout flush so progress shows in terminal."""
    print(msg, flush=True)


class OpenAQClient:
    def __init__(self, bbox="73.7,18.4,74.0,18.7"):
        self.base_url = "https://api.openaq.org/v3"
        self.bbox = bbox

        raw_key = os.environ.get("OPENAQ_API_KEY", "").strip()
        # Reject placeholder / wrong-service keys
        if raw_key and not raw_key.startswith("sk-") and not raw_key.startswith("YOUR_"):
            self.api_key = raw_key
        else:
            self.api_key = None
            _flush("WARNING: Valid OPENAQ_API_KEY not set.")
            _flush("  Get a free key at: https://explore.openaq.org/register")
            _flush("  Add to .env:  OPENAQ_API_KEY=your_key_here")

        self.headers = {
            "X-API-Key": self.api_key or "",
            "accept": "application/json",
        }

    # ── Public ────────────────────────────────────────────────────────────── #

    def fetch_data(self, days_back=DAYS_BACK, parameter_id=2):
        """
        Fetch PM2.5 measurements.  Returns DataFrame with columns:
            location_id, location_name, lat, lon, parameter, value, unit, timestamp
        Returns empty DataFrame (not an error) if no API key or all sensors fail.
        """
        if not self.api_key:
            return pd.DataFrame()

        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        _flush(f"Fetching OpenAQ v3 locations in bbox {self.bbox}...")
        locations = self._get_locations(parameter_id)
        if not locations:
            return pd.DataFrame()

        _flush(f"Found {len(locations)} locations. "
               f"Querying up to {MAX_SENSORS} sensors for the last {days_back} day(s)...")

        all_measurements = []

        for i, loc in enumerate(locations[:MAX_SENSORS]):
            sensor_id = self._pick_sensor(loc, parameter_id)
            if not sensor_id:
                continue

            _flush(f"  [{i+1}/{min(len(locations), MAX_SENSORS)}] "
                   f"{loc['name']} (sensor {sensor_id})")

            rows = self._fetch_sensor_window(sensor_id, loc, start_date, end_date)
            _flush(f"      -> {len(rows)} records")
            all_measurements.extend(rows)
            time.sleep(INTER_SENSOR_SLEEP)

        _flush(f"Total raw measurements collected: {len(all_measurements)}")
        return pd.DataFrame(all_measurements)

    def process_and_save(self, df, output_path="data/raw/openaq.parquet"):
        if df.empty:
            _flush("No data to save.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = df.drop_duplicates(subset=["location_id", "timestamp", "parameter"])
        df.to_parquet(output_path, engine="pyarrow")
        _flush(f"Saved {len(df)} records -> {output_path}")

    # ── Helpers ───────────────────────────────────────────────────────────── #

    def _get_locations(self, parameter_id):
        try:
            resp = requests.get(
                f"{self.base_url}/locations",
                headers=self.headers,
                params={"bbox": self.bbox, "parameters_id": parameter_id, "limit": 100},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                _flush(f"Error fetching locations ({resp.status_code}): {resp.text[:200]}")
                return []
            return resp.json().get("results", [])
        except requests.exceptions.Timeout:
            _flush("Timeout fetching locations list.")
            return []
        except Exception as e:
            _flush(f"Error fetching locations: {e}")
            return []

    def _pick_sensor(self, loc, parameter_id):
        for s in loc.get("sensors", []):
            if s.get("parameter", {}).get("id") == parameter_id:
                return s.get("id")
        return None

    def _fetch_sensor_window(self, sensor_id, loc, date_from, date_to):
        """
        Fetch measurements for one sensor over the full window.
        Uses a single request (not paged) to minimise timeout risk.
        Falls back to empty list on any error - never raises.
        """
        meas_url = f"{self.base_url}/sensors/{sensor_id}/measurements"
        params = {
            "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":     PAGE_LIMIT,
            "page":      1,
        }
        rows = []
        retries = 2

        for attempt in range(retries):
            try:
                resp = requests.get(
                    meas_url, headers=self.headers,
                    params=params, timeout=REQUEST_TIMEOUT,
                )
            except requests.exceptions.Timeout:
                _flush(f"      ↳ Timeout (attempt {attempt+1}/{retries}) - skipping sensor")
                if attempt < retries - 1:
                    time.sleep(3)
                continue
            except Exception as e:
                _flush(f"      ↳ Request error: {e}")
                break

            if resp.status_code == 429:
                _flush("      ↳ Rate limited - sleeping 15s...")
                time.sleep(15)
                continue

            if resp.status_code != 200:
                snippet = resp.text[:150].replace("\n", " ")
                _flush(f"      ↳ HTTP {resp.status_code}: {snippet}")
                break

            data = resp.json().get("results", [])
            for d in data:
                try:
                    rows.append({
                        "location_id":   loc["id"],
                        "location_name": loc["name"],
                        "lat":           loc["coordinates"]["latitude"],
                        "lon":           loc["coordinates"]["longitude"],
                        "parameter":     "pm25",
                        "value":         d.get("value"),
                        "unit":          "ug/m³",
                        "timestamp":     pd.to_datetime(d["period"]["datetimeTo"]["utc"]),
                    })
                except (KeyError, TypeError):
                    pass
            break  # success

        return rows


# ── Synthetic fallback ────────────────────────────────────────────────────── #

def generate_synthetic_openaq(output_path="data/raw/openaq.parquet"):
    """
    Generate realistic-looking synthetic PM2.5 data for Pune sensor locations.
    Used when real OpenAQ data cannot be fetched.
    Seeds are fixed so the graph_builder always gets the same stub data.
    """
    _flush("Generating synthetic OpenAQ data for Pune (demo mode)...")
    rng = np.random.default_rng(42)

    locations = [
        {"id": 1, "name": "Karve Road",      "lat": 18.4983, "lon": 73.8258},
        {"id": 2, "name": "Shivajinagar",    "lat": 18.5314, "lon": 73.8446},
        {"id": 3, "name": "Bhosari",         "lat": 18.6436, "lon": 73.8478},
        {"id": 4, "name": "Hadapsar",        "lat": 18.5018, "lon": 73.9260},
        {"id": 5, "name": "Nigdi",           "lat": 18.6500, "lon": 73.7700},
        {"id": 6, "name": "Katraj",          "lat": 18.4530, "lon": 73.8600},
        {"id": 7, "name": "Pimpri",          "lat": 18.6280, "lon": 73.7980},
        {"id": 8, "name": "Mhada Colony",    "lat": 18.5600, "lon": 73.8300},
    ]

    end   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=2)
    timestamps = pd.date_range(start=start, end=end, freq="h", tz="UTC")

    rows = []
    for loc in locations:
        # Simulate diurnal pattern: peak PM2.5 at morning/evening rush hours
        for ts in timestamps:
            hour = ts.hour
            base  = 45 + 20 * np.sin(np.pi * (hour - 6) / 12)  # diurnal shape
            noise = rng.normal(0, 8)
            rows.append({
                "location_id":   loc["id"],
                "location_name": loc["name"],
                "lat":           loc["lat"],
                "lon":           loc["lon"],
                "parameter":     "pm25",
                "value":         max(0, round(float(base + noise), 2)),
                "unit":          "ug/m³",
                "timestamp":     ts,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    _flush(f"Saved {len(df)} synthetic records -> {output_path}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    use_synthetic = "--synthetic" in sys.argv

    client = OpenAQClient(bbox="73.7,18.4,74.0,18.7")

    if use_synthetic or not client.api_key:
        _flush("Running in synthetic / demo mode.")
        generate_synthetic_openaq()
    else:
        df = client.fetch_data(days_back=DAYS_BACK)
        if df.empty:
            _flush("Real fetch returned no data - falling back to synthetic.")
            generate_synthetic_openaq()
        else:
            client.process_and_save(df)
