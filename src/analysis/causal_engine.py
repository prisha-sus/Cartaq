import pandas as pd
import numpy as np
from dowhy import CausalModel
import logging

def run_causal_attribution():
    # Mute verbose DoWhy internal logs for clean CLI execution
    logging.getLogger("dowhy").setLevel(logging.WARNING)
    
    data_path = "output/live_enriched_data.parquet"
    print(f"Loading enriched forecast dataset from {data_path}...")
    
    df = pd.read_parquet(data_path)
    print(f"Total rows loaded: {len(df)}")

    # 1. DATA SANITIZATION LAYER
    # Force numeric conversion and handle edge-case nulls
    df['wind_speed_kmh'] = pd.to_numeric(df['wind_speed_kmh'], errors='coerce')
    df['is_factory_active'] = pd.to_numeric(df['is_factory_active'], errors='coerce')
    df['predicted_aqi'] = pd.to_numeric(df['predicted_aqi'], errors='coerce')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['predicted_aqi', 'is_factory_active', 'wind_speed_kmh'])
    df['is_factory_active'] = df['is_factory_active'].astype(int)
    
    print(f"Cleaned dataset rows available for causal model: {len(df)}")
    
    if len(df) < 10:
        raise ValueError(f"CRITICAL: Insufficient sample size ({len(df)} rows). Expected at least 10 rows.")

    # 2. CAUSAL GRAPH (DAG)
    # We define wind_speed_kmh as a confounder that affects both factory operations and AQI
    causal_graph = """
    digraph {
        wind_speed_kmh -> predicted_aqi;
        wind_speed_kmh -> is_factory_active;
        is_factory_active -> predicted_aqi;
    }
    """
    
    # 3. INITIALIZE DOWHY MODEL
    model = CausalModel(
        data=df,
        treatment="is_factory_active",
        outcome="predicted_aqi",
        graph=causal_graph
    )
    
    # 4. IDENTIFY & ESTIMATE CAUSAL EFFECT
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    estimate = model.estimate_effect(
        identified_estimand, 
        method_name="backdoor.linear_regression"
    )
    
    causal_score = round(float(estimate.value), 2)
    
    print("--------------------------------------------------")
    print("--- Causal Engine Isolation Complete ---")
    print(f"Isolated Impact Score for Factory Activity: +{causal_score} AQI points")
    print("--------------------------------------------------")
    
    return causal_score

if __name__ == "__main__":
    run_causal_attribution()