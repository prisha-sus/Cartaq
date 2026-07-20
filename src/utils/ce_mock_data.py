import pandas as pd
import numpy as np

np.random.seed(42)

# simulate ST-GNN forecast 
# to be replaced: stgnn_df = pd.read_parquet("forecast.parquet")
pune_res8_hex = "8860a25997fffff" # A real H3 Res 8 hex near Shivajinagar, Pune

data = {
    "h3_index": [pune_res8_hex] * 100,
    "future_timestamp": pd.date_range(start = "2026-07-20", periods=100, freq="H"),
    "predicted_aqi": np.random.uniform(80, 200, 100) # base ST-GNN forecast
}
stgnn_df = pd.DataFrame(data)

# add causal variable: confounders (wind) and treatment (factory)
# to explain st-gnn AQI predictions

# temporary mock data
causal_variables = pd.DataFrame({
    "wind_speed_kmh": np.random.uniform(2, 15, 100),
    "is_factory_active": np.random.choice([0, 1], size=100)
})

df = pd.concat([stgnn_df, causal_variables], axis=1)

df["predicted_aqi"] = 50 + (df["is_factory_active"] * 55) - (df["wind_speed_kmh"] * 2) + np.random.normal(0, 5, 100)

print("Enriched Data Contract Ready for Dowhy. Shape:", df.shape)
print(df.head())
df.to_parquet("../../data/enriched_causal_data.parquet", index=False)
print("Data saved cleanly to data/enriched_causal_data.parquet")