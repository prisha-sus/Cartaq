import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright
import os

# Import your existing pipeline modules
from src.analysis.causal_engine import run_causal_attribution
from src.utils.chart_generator import generate_aqi_trend_base64

def compile_legal_dossier():
    print("Executing master data pipeline integration...")
    
    # 1. DYNAMIC HEX IDENTIFICATION
    df = pd.read_parquet("output/live_enriched_data.parquet")
    
    # Find the exact moment and location of peak forecasted pollution
    worst_row = df.loc[df['predicted_aqi'].idxmax()]
    target_hex = worst_row['h3_index']
    peak_aqi = round(worst_row['predicted_aqi'], 1)
    peak_time = worst_row['future_timestamp'].strftime("%Y-%m-%d %H:%M")
    
    print(f"Triggering automated enforcement for Hex {target_hex} (Peak AQI: {peak_aqi})")
    
    # 2. GENERATE CORE EVIDENCE COMPONENTS
    # Run the DoWhy statistical regression
    causal_score = run_causal_attribution()
    
    # Generate the base64 encoded matplotlib chart
    chart_base64 = generate_aqi_trend_base64()
    
    # 3. BUILD THE PROFESSIONAL HTML REPORT
    generation_date = datetime.now().strftime("%B %d, %Y - %H:%M IST")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; margin: 40px; }}
            .header {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; margin-bottom: 30px; }}
            .header h1 {{ color: #2c3e50; margin: 0; font-size: 28px; text-transform: uppercase; letter-spacing: 1px; }}
            .header p {{ margin: 5px 0 0 0; color: #7f8c8d; font-size: 14px; }}
            .alert-box {{ background-color: #ffeaa7; border-left: 5px solid #d35400; padding: 15px; margin-bottom: 30px; }}
            .alert-box h2 {{ margin: 0 0 10px 0; color: #d35400; font-size: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .section h3 {{ color: #2980b9; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #c0392b; }}
            .chart-container {{ text-align: center; margin: 30px 0; border: 1px solid #ecf0f1; padding: 10px; background: #fafafa; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #95a5a6; text-align: center; border-top: 1px solid #ecf0f1; padding-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
            th {{ background-color: #ecf0f1; }}
        </style>
    </head>
    <body>

        <div class="header">
            <h1>Automated Causal Intervention Dossier</h1>
            <p>Generated: {generation_date} | Jurisdiction: Pune, Maharashtra, India</p>
        </div>

        <div class="alert-box">
            <h2>Critical Air Quality Alert: Zone {target_hex}</h2>
            <p>Algorithmic forecasting indicates a severe degradation in local air quality peaking at <strong>{peak_time}</strong> with a maximum predicted AQI of <strong>{peak_aqi}</strong>.</p>
        </div>

        <div class="section">
            <h3>1. Data Evidence & Causal Attribution</h3>
            <p>Using statistical regression on time-series forecasts, we have isolated the specific impact of local factory emissions from ambient weather patterns.</p>
            <table>
                <tr>
                    <th>Target Zone (H3 Index)</th>
                    <td>{target_hex}</td>
                </tr>
                <tr>
                    <th>Forecasted Peak AQI</th>
                    <td>{peak_aqi}</td>
                </tr>
                <tr>
                    <th>Isolated Factory Impact</th>
                    <td><span class="metric">+{causal_score} AQI Points</span></td>
                </tr>
            </table>
            <p><em>Conclusion: Industrial activity is directly responsible for {causal_score} points of the forecasted AQI degradation, maintaining mathematical significance independent of local wind speeds.</em></p>
        </div>

        <div class="section">
            <h3>2. 5-Day Forecast Trend</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_base64}" width="100%" alt="AQI Trend Chart" />
            </div>
        </div>

        <div class="section">
            <h3>3. Actionable Insights</h3>
            <ul>
                <li><strong>Targeted Intervention:</strong> Issue a temporary regulatory halt on operations for the industrial facility located in hex <code>{target_hex}</code> beginning 4 hours prior to the {peak_time} forecasted peak.</li>
                <li><strong>Expected Outcome:</strong> Enforcing this halt will mitigate the peak AQI by approximately {causal_score} points, potentially preventing the zone from crossing into hazardous regulatory thresholds.</li>
                <li><strong>Resource Allocation:</strong> Deploy physical inspection teams to the target zone to ensure compliance during the specified peak window.</li>
            </ul>
        </div>

        <div class="section">
            <h3>4. Source Attribution</h3>
            <p>This automated analysis is powered by a multi-modal data pipeline:</p>
            <ul>
                <li><strong>Primary Forecast:</strong> Spatio-Temporal Graph Neural Network (ST-GNN) trained on historical OpenAQ sensor data.</li>
                <li><strong>Meteorology:</strong> Live API ingestion from Open-Meteo for localized wind speed forecasting.</li>
                <li><strong>Causal Inference:</strong> Microsoft DoWhy library utilizing backdoor linear regression for impact isolation.</li>
            </ul>
        </div>

        <div class="footer">
            Cartaq Automated Compliance Engine | Role 2 Data Pipeline <br>
            Confidential Regulatory Document
        </div>

    </body>
    </html>
    """

    # 4. RENDER TO PDF VIA PLAYWRIGHT
    output_pdf = "output/intervention_dossier.pdf"
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content)
        page.pdf(path=output_pdf, format="A4", print_background=True, margin={"top": "0.5in", "right": "0.5in", "bottom": "0.5in", "left": "0.5in"})
        browser.close()

    print(f"\nSUCCESS: Official regulatory dossier compiled and saved to {output_pdf}")

if __name__ == "__main__":
    compile_legal_dossier()