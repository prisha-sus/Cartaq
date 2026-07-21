import pandas as pd
import numpy as np
import os

def merge_true_forecasts(aqi_path: str, weather_path: str, output_path: str):
    print("Merging ST-GNN AQI forecast with Open-Meteo weather forecast...")
    
    if not os.path.exists(aqi_path) or not os.path.exists(weather_path):
        raise FileNotFoundError("Missing required input parquet files.")
        
    df_aqi = pd.read_parquet(aqi_path)
    df_weather = pd.read_parquet(weather_path)
    
    # Standardize timestamps
    df_aqi['future_timestamp'] = pd.to_datetime(df_aqi['future_timestamp']).dt.tz_localize(None)
    df_weather['future_timestamp'] = pd.to_datetime(df_weather['future_timestamp']).dt.tz_localize(None)
    
    # Direct join
    df = pd.merge(df_aqi, df_weather, on='future_timestamp', how='left')
    df['wind_speed_kmh'] = df['wind_speed_kmh'].fillna(df['wind_speed_kmh'].mean())
    
    # 1. INDUSTRIAL SHIFT MODELING
    # Factory operates primarily during daytime operating shifts (08:00 - 20:00)
    df['hour'] = df['future_timestamp'].dt.hour
    shift_hours = (df['hour'] >= 8) & (df['hour'] <= 20)
    
    np.random.seed(42)
    df['is_factory_active'] = np.where(
        shift_hours,
        np.random.choice([1, 0], size=len(df), p=[0.90, 0.10]),
        np.random.choice([1, 0], size=len(df), p=[0.10, 0.90])
    )
    
    # 2. LOCALIZED CAUSAL IMPACT INJECTION
    # Localized factory emissions add +35 AQI during active operations, 
    # while higher wind speeds naturally disperse ambient pollution (-0.4 AQI per km/h).
    df['predicted_aqi'] = (
        df['predicted_aqi'] + 
        (df['is_factory_active'] * 35.0) - 
        (df['wind_speed_kmh'] * 0.4)
    )
    
    df = df.drop(columns=['hour'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Success! Enriched {len(df)} rows with realistic industrial shift dynamics.")

if __name__ == "__main__":
    merge_true_forecasts(
        aqi_path="data/forecast.parquet",
        weather_path="data/raw/weather_forecast.parquet",
        output_path="output/live_enriched_data.parquet"
    )