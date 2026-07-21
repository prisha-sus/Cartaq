import pandas as pd

df_forecast = pd.read_parquet("data/forecast.parquet")
df_weather = pd.read_parquet("data/raw/openmeteo.parquet")

print("--- FORECAST TIMESTAMPS ---")
print(df_forecast['future_timestamp'].head(2))
print("Forecast Timezone/Type:", df_forecast['future_timestamp'].dtype)

print("\n--- WEATHER TIMESTAMPS ---")
# Adjust 'time' if Role 1 named the column differently
weather_time_col = 'time' if 'time' in df_weather.columns else 'timestamp'
print(df_weather[weather_time_col].head(2))
print("Weather Timezone/Type:", df_weather[weather_time_col].dtype)