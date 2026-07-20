import pandas as pd
from dowhy import CausalModel
import logging

def causal_attribution():
    logging.getLogger("dowhy").setLevel(logging.WARNING)

    df = pd.read_parquet("data/enriched_causal_data.parquet")

    causal_graph = """
    digraph {
        wind_speed_kmh -> predicted_aqi;
        is_factory_active -> predicted_aqi;
    }
    """
    model = CausalModel(
        data=df,
        treatment="is_factory_active",
        outcome="predicted_aqi",
        graph=causal_graph
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand, 
        method_name="backdoor.linear_regression"
    )
    
    causal_score = round(estimate.value, 2)
    
    print("--- Causal Engine Isolation Complete ---")
    print(f"Isolated Factor Score for Factory Activity: +{causal_score} AQI impact points.")
    return causal_score

if __name__ == "__main__":
    causal_attribution()