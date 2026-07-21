import pandas as pd
import h3
import os
from datetime import datetime, timedelta
import numpy as np

def create_dummy_contract(output_path='output/dummy_forecast.parquet', resolution=8, seed=42):
    """
    Generate a dummy forecast with realistic spatial & temporal structure.

    Spatial: centre has higher AQI, decreases with distance, plus random hotspots.
    Temporal: diurnal cycle (peaks at rush hour, minimum at early morning).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = np.random.default_rng(seed)

    # Pune centre (Shivajinagar)
    lat, lon = 18.5314, 73.8446
    center_h3 = h3.latlng_to_cell(lat, lon, resolution)
    hexes = list(h3.grid_disk(center_h3, 10))

    # Compute distance from centre for each hex
    hex_lats = np.array([h3.cell_to_latlng(h)[0] for h in hexes])
    hex_lons = np.array([h3.cell_to_latlng(h)[1] for h in hexes])
    dist = np.sqrt((hex_lats - lat)**2 + (hex_lons - lon)**2)
    dist_norm = dist / dist.max()

    # Industrial hotspots: 3-4 random hexes with elevated baseline
    hotspots = rng.choice(len(hexes), size=rng.integers(3, 5), replace=False)
    hotspot_boost = np.zeros(len(hexes))
    hotspot_boost[hotspots] = rng.uniform(25, 55, size=len(hotspots))

    # Time horizons
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    horizons = [24, 48, 72]

    data = []
    for idx, hx in enumerate(hexes):
        # Base AQI: higher at centre (~150), lower at outskirts (~50)
        base_aqi = 155 - 100 * dist_norm[idx] + hotspot_boost[idx]
        base_aqi = np.clip(base_aqi, 30, 300)

        for h_offset in horizons:
            ts = now + timedelta(hours=h_offset)
            # Diurnal effect: peak at 9am and 8pm, trough at 3am
            hour = ts.hour
            diurnal = 1.0 + 0.35 * np.sin(np.pi * (hour - 3) / 12) ** 4 \
                      - 0.10 * np.sin(np.pi * (hour - 14) / 8)
            diurnal = np.clip(diurnal, 0.6, 1.4)

            # Auto-correlated noise: nearby hexes have similar noise
            noise = rng.normal(0, 6) + 0.3 * rng.normal(0, 4)

            aqi = float(np.clip(base_aqi * diurnal + noise, 20, 500))
            data.append({
                'h3_index': hx,
                'future_timestamp': ts,
                'predicted_aqi': round(aqi, 1),
                'horizon_hours': h_offset,
            })

    df = pd.DataFrame(data)
    df['h3_index'] = df['h3_index'].astype(str)
    df['future_timestamp'] = pd.to_datetime(df['future_timestamp'])
    df['predicted_aqi'] = df['predicted_aqi'].astype(float)

    df.to_parquet(output_path, engine='pyarrow')
    print(f"Created dummy contract at {output_path} with {len(df)} rows.")
    print(f"  AQI range: [{df.predicted_aqi.min():.1f}, {df.predicted_aqi.max():.1f}]")
    print(f"  Mean AQI:  {df.predicted_aqi.mean():.1f}")

if __name__ == "__main__":
    create_dummy_contract()
