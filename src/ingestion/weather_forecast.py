import requests
import pandas as pd
import os

def fetch_weather_forecast(output_path="data/raw/weather_forecast.parquet"):
    print("Initializing weather forecast ingestion...")
    
    # Open-Meteo Forecast API (Pune coordinates)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 18.5204,
        "longitude": 73.8567,
        "hourly": "wind_speed_10m",
        "timezone": "UTC"
    }
    
    print("Connecting to Open-Meteo API...")
    try:
        # The timeout=10 forces Python to crash and tell you what's wrong 
        # instead of hanging forever if the network drops.
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"CRITICAL ERROR: Network request failed. Details: {e}")
        
    print("Connection successful. Parsing JSON payload...")
    data = response.json()
    
    # Build dataframe
    df_weather = pd.DataFrame({
        "future_timestamp": pd.to_datetime(data["hourly"]["time"]),
        "wind_speed_kmh": data["hourly"]["wind_speed_10m"]
    })
    
    # Strip timezones for a clean merge later
    df_weather['future_timestamp'] = df_weather['future_timestamp'].dt.tz_localize(None)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_weather.to_parquet(output_path, index=False)
    
    print(f"Success! Saved {len(df_weather)} weather forecast hours to {output_path}")

if __name__ == "__main__":
    fetch_weather_forecast()