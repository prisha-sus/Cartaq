import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

def generate_aqi_trend_base64() -> str:
    """
    Reads the forecast data, generates a clean Matplotlib trend chart, 
    and returns it as a Base64 encoded string from memory.
    """
    df = pd.read_parquet("output/live_enriched_data.parquet")

    plt.figure(figsize=(8, 3))
    
    # Plot the AQI over time
    plt.plot(df['future_timestamp'], df['predicted_aqi'], color='#8e44ad', linewidth=2)
    
    # Add a horizontal warning line at AQI 150 (Unhealthy)
    plt.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (150)')
    
    # Clean up the aesthetics
    plt.title("Forecasted AQI Trend (72 Hours)", fontsize=12, fontweight='bold', color="#2c3e50")
    plt.ylabel("AQI Level", fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout() 

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150) # DPI 150 ensures crisp PDF printing
    buffer.seek(0)
    
    chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close() # Free up memory
    
    return chart_base64

if __name__ == "__main__":
    b64_string = generate_aqi_trend_base64()
    
    test_html = f"""
    <html>
        <body>
            <h2>Matplotlib Base64 Test</h2>
            <img src="data:image/png;base64,{b64_string}" alt="AQI Chart"/>
        </body>
    </html>
    """
    
    os.makedirs("output", exist_ok=True)
    with open("output/test_chart.html", "w") as f:
        f.write(test_html)
        
    print("--- Chart Generation Complete ---")
    print(f"Generated Base64 string of length: {len(b64_string)}")