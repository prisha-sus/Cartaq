import pandas as pd
import h3
import os
from datetime import datetime, timedelta
import numpy as np

def create_dummy_contract(output_path='output/dummy_forecast.parquet', resolution=8):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Pune center roughly (Shivajinagar)
    lat, lon = 18.5314, 73.8446
    
    # Generate some H3 indices around Pune
    center_h3 = h3.latlng_to_cell(lat, lon, resolution)
    h3_indices = list(h3.grid_disk(center_h3, 10)) # Generate a cluster of hexes
    
    # Time horizons
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    timestamps = [now + timedelta(hours=h) for h in [24, 48, 72]]
    
    data = []
    for h3_idx in h3_indices:
        for ts in timestamps:
            # Generate random AQI
            aqi = np.random.uniform(20.0, 150.0)
            data.append({
                'h3_index': h3_idx,
                'future_timestamp': ts,
                'predicted_aqi': aqi
            })
            
    df = pd.DataFrame(data)
    # Ensure correct types
    df['h3_index'] = df['h3_index'].astype(str)
    df['future_timestamp'] = pd.to_datetime(df['future_timestamp'])
    df['predicted_aqi'] = df['predicted_aqi'].astype(float)
    
    df.to_parquet(output_path, engine='pyarrow')
    print(f"Created dummy contract at {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    create_dummy_contract()
