import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_DAYS_BACK, PUNE_CENTER_LAT, PUNE_CENTER_LON


class OpenMeteoClient:
    def __init__(self):
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    def fetch_data(self, lats, lons, start_date, end_date):
        """
        Fetch historical weather for a list of coordinates.
        lats: list of latitudes
        lons: list of longitudes
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        """
        print(f"Fetching Open-Meteo data for {len(lats)} locations from {start_date} to {end_date}...")

        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
        }

        responses = self.openmeteo.weather_api(self.url, params=params)

        all_dfs = []
        for i, response in enumerate(responses):
            lat = lats[i]
            lon = lons[i]

            hourly = response.Hourly()
            hourly_data = {
                "timestamp": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                )
            }

            hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
            hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
            hourly_data["precipitation"] = hourly.Variables(2).ValuesAsNumpy()
            hourly_data["wind_speed_10m"] = hourly.Variables(3).ValuesAsNumpy()
            hourly_data["wind_direction_10m"] = hourly.Variables(4).ValuesAsNumpy()

            df = pd.DataFrame(data=hourly_data)
            df["lat"] = lat
            df["lon"] = lon

            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)

    def process_and_save(self, df, output_path="data/raw/openmeteo.parquet"):
        if df.empty:
            print("No data to save.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow")
        print(f"Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    client = OpenMeteoClient()
    end = datetime.now(UTC)
    start = end - timedelta(days=DEFAULT_DAYS_BACK)

    df = client.fetch_data(
        [PUNE_CENTER_LAT],
        [PUNE_CENTER_LON],
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    client.process_and_save(df)
