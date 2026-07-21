"""
Role 2 - PDF Report Generator
Creates a professional causal evidence dossier using fpdf2.

Output: output/causal_report.pdf

Usage:
    cd Cartaq
    python src/pipeline/pdf_generator.py

Requires:
    pip install fpdf2 matplotlib
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _causal_bar_chart(causal_results: list) -> bytes:
    """Render horizontal bar chart of ATE by source as PNG bytes."""
    sources    = [r["source"]    for r in causal_results]
    ates       = [r["ate"]       for r in causal_results]
    ci_lower   = [r["ci_lower"]  for r in causal_results]
    ci_upper   = [r["ci_upper"]  for r in causal_results]
    xerr_minus = [a - l for a, l in zip(ates, ci_lower)]
    xerr_plus  = [u - a for a, u in zip(ates, ci_upper)]

    palette = ["#e94560", "#f6c90e", "#ff9f40", "#4bc0c0"]
    colors  = palette[:len(sources)]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(sources, ates, color=colors,
            xerr=[xerr_minus, xerr_plus],
            error_kw={"ecolor": "gray", "capsize": 4, "linewidth": 1},
            height=0.5, edgecolor="none")
    for i, (ate_val, _) in enumerate(zip(ates, xerr_plus)):
        ax.text(ate_val + max(xerr_plus + [1]) * 0.05, i,
                f"+{ate_val:.1f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("ATE (AQI pts)", fontsize=9)
    ax.set_title("Causal Attribution", fontsize=10, pad=8)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _aqi_category(aqi: float) -> str:
    if aqi <= 50:    return "Good"
    if aqi <= 100:   return "Moderate"
    if aqi <= 150:   return "Unhealthy-S"
    if aqi <= 200:   return "Unhealthy"
    if aqi <= 300:   return "Very Unhealthy"
    return "Hazardous"


def generate_report(
    causal_results: list,
    forecast_df: pd.DataFrame,
    dispatch_metrics: dict,
    city: str = "Pune",
    analyst: str = "Cartaq Intelligence System",
    output_path: str = "output/causal_report.pdf",
) -> str:
    """Generate PDF dossier and return output path."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is required: pip install fpdf2")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Stats
    max_aqi   = float(forecast_df["predicted_aqi"].max())  if not forecast_df.empty else 0
    mean_aqi  = float(forecast_df["predicted_aqi"].mean()) if not forecast_df.empty else 0
    n_hexes   = forecast_df["h3_index"].nunique()           if not forecast_df.empty else 0
    n_high    = int((forecast_df["predicted_aqi"] > 150).sum()) if not forecast_df.empty else 0
    n_calls   = dispatch_metrics.get("calls_dispatched", 0)
    n_mqtt    = dispatch_metrics.get("mqtt_messages_published", 0)
    e2e_ms    = dispatch_metrics.get("end_to_end_latency_ms", "N/A")

    top_hexes = (
        forecast_df.nlargest(15, "predicted_aqi")[["h3_index", "predicted_aqi"]]
        .assign(category=lambda d: d["predicted_aqi"].apply(_aqi_category))
        .values.tolist()
        if not forecast_df.empty else []
    )

    class CartaqPDF(FPDF):
        def header(self):
            self.set_fill_color(15, 52, 96)
            self.rect(0, 0, 210, 18, "F")
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(233, 69, 96)
            self.set_y(4)
            self.cell(0, 8, "CARTAQ - Urban Air Quality Intelligence Platform", align="C")

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8,
                      f"Generated: {now_str}  |  Analyst: {analyst}  |  Page {self.page_no()}",
                      align="C")

    pdf = CartaqPDF()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ----- PAGE 1: Cover + Executive Summary -----
    pdf.add_page()
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_xy(15, 25)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 14, "AQI Causal Attribution Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(200, 210, 230)
    pdf.set_x(15)
    pdf.cell(0, 8, f"City: {city}  |  {now_str}", ln=True, align="C")

    pdf.set_draw_color(233, 69, 96)
    pdf.set_line_width(0.8)
    pdf.line(20, 55, 190, 55)
    pdf.set_line_width(0.2)

    # Summary box
    pdf.set_xy(15, 62)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 8, "Executive Summary", ln=True)

    pdf.set_fill_color(22, 33, 62)
    pdf.set_draw_color(51, 68, 102)
    pdf.rect(15, 72, 180, 50, "FD")

    stats = [
        ("Max Forecast AQI",   f"{max_aqi:.1f}"),
        ("Mean Forecast AQI",  f"{mean_aqi:.1f}"),
        ("Hexes Monitored",    str(n_hexes)),
        ("High-AQI Zones",     f"{n_high} (AQI > 150)"),
        ("Calls Dispatched",   str(n_calls)),
        ("MQTT Messages",      str(n_mqtt)),
        ("End-to-End Latency", f"{e2e_ms} ms"),
    ]
    col_w = 88
    row_h = 6.5
    pdf.set_font("Helvetica", "", 10)
    for i, (label, value) in enumerate(stats):
        col = i % 2
        row = i // 2
        x   = 18 + col * col_w
        y   = 75 + row * row_h
        pdf.set_xy(x, y)
        pdf.set_text_color(180, 190, 210)
        pdf.cell(45, row_h, label + ":")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(40, row_h, value)
        pdf.set_font("Helvetica", "", 10)

    # Methodology
    pdf.set_xy(15, 130)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 8, "Methodology", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(190, 200, 220)
    pdf.set_x(15)
    pdf.multi_cell(
        180, 6,
        "Causal attribution is performed using the DoWhy library (Microsoft Research) "
        "with a backdoor linear regression estimator. The causal graph models industrial "
        "and traffic sources as treatments and meteorological variables (wind speed, "
        "temperature) as confounders. Average Treatment Effects (ATE) are reported "
        "with 95% bootstrap confidence intervals.\n\n"
        "Spatial forecasting uses an Attention Temporal Graph Convolutional Network "
        "(A3TGCN) trained on PM2.5 sensor data from OpenAQ and weather from Open-Meteo, "
        "mapped to Uber H3 hexagonal grid at resolution 8 (~0.73 km2 per cell, Pune).",
    )

    # ----- PAGE 2: Causal Attribution Chart -----
    pdf.add_page()
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_xy(15, 22)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, "Section 1 - Causal Attribution by Source Type", ln=True)

    chart_bytes = _causal_bar_chart(causal_results)
    chart_buf   = io.BytesIO(chart_bytes)
    pdf.image(chart_buf, x=10, y=36, w=190)

    # Table below chart
    y_tbl = min(36 + len(causal_results) * 18 + 22, 168)
    pdf.set_xy(15, y_tbl)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 8, "Attribution Table (ATE with 95% CI)", ln=True)

    col_widths = [60, 35, 40, 40]
    headers    = ["Source", "ATE (AQI pts)", "CI Lower", "CI Upper"]
    pdf.set_fill_color(22, 33, 62)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(233, 69, 96)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(220, 230, 245)
    for i, row in enumerate(causal_results):
        pdf.set_fill_color(26, 40, 75) if i % 2 == 0 else pdf.set_fill_color(20, 32, 62)
        pdf.cell(col_widths[0], 7, str(row.get("source", "")),     border=1, fill=True)
        pdf.cell(col_widths[1], 7, f"+{row.get('ate', 0):.2f}",   border=1, fill=True)
        pdf.cell(col_widths[2], 7, f"{row.get('ci_lower',0):.2f}", border=1, fill=True)
        pdf.cell(col_widths[3], 7, f"{row.get('ci_upper',0):.2f}", border=1, fill=True)
        pdf.ln()

    # ----- PAGE 3: Top Hexes + Dispatch Summary -----
    pdf.add_page()
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_xy(15, 22)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, "Section 2 - Top 15 Highest-AQI Forecast Zones", ln=True)

    hex_col_w = [90, 45, 45]
    hex_hdrs  = ["H3 Index", "Predicted AQI", "Category"]
    pdf.set_fill_color(22, 33, 62)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(233, 69, 96)
    for w, h in zip(hex_col_w, hex_hdrs):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(220, 230, 245)
    for i, (h3idx, aqi, cat) in enumerate(top_hexes[:15]):
        pdf.set_fill_color(26, 40, 75) if i % 2 == 0 else pdf.set_fill_color(20, 32, 62)
        pdf.cell(hex_col_w[0], 7, str(h3idx),          border=1, fill=True)
        pdf.cell(hex_col_w[1], 7, f"{float(aqi):.1f}", border=1, fill=True)
        pdf.cell(hex_col_w[2], 7, str(cat),             border=1, fill=True)
        pdf.ln()

    # Dispatch summary
    pdf.set_xy(15, pdf.get_y() + 12)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, "Section 3 - Dispatch Pipeline Summary", ln=True)

    disp_items = [
        ("End-to-End Latency",      f"{dispatch_metrics.get('end_to_end_latency_ms','N/A')} ms"),
        ("Calls Dispatched",        str(dispatch_metrics.get("calls_dispatched", 0))),
        ("MQTT Messages Published", str(dispatch_metrics.get("mqtt_messages_published", 0))),
        ("Avg Translation Latency", f"{dispatch_metrics.get('avg_translation_ms','N/A')} ms"),
        ("Avg Twilio Latency",      f"{dispatch_metrics.get('avg_twilio_ms','N/A')} ms"),
        ("Avg MQTT Latency",        f"{dispatch_metrics.get('avg_mqtt_ms','N/A')} ms"),
        ("Spatial Query (avg/hex)", f"{dispatch_metrics.get('spatial_per_hex_avg_ms','N/A')} ms"),
    ]
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(220, 230, 245)
    for i, (label, value) in enumerate(disp_items):
        pdf.set_fill_color(26, 40, 75) if i % 2 == 0 else pdf.set_fill_color(20, 32, 62)
        pdf.cell(90, 7, label,  border=1, fill=True)
        pdf.cell(90, 7, value,  border=1, fill=True)
        pdf.ln()

    pdf.output(output_path)
    print(f"PDF saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    demo_causal = [
        {"source": "Factory Activity", "ate": 55.3, "ci_lower": 43.1, "ci_upper": 67.5},
        {"source": "Traffic Peak",     "ate": 23.4, "ci_lower": 18.2, "ci_upper": 28.6},
        {"source": "Construction",     "ate": 15.8, "ci_lower": 11.3, "ci_upper": 20.3},
        {"source": "Domestic Burning", "ate":  8.2, "ci_lower":  5.1, "ci_upper": 11.3},
    ]
    fp = "data/forecast.parquet"
    if os.path.exists(fp):
        fdf = pd.read_parquet(fp)
    else:
        import h3, numpy as np
        center = h3.latlng_to_cell(18.5314, 73.8446, 8)
        hexes  = list(h3.grid_disk(center, 8))
        rng    = np.random.default_rng(42)
        fdf    = pd.DataFrame({
            "h3_index":      hexes,
            "predicted_aqi": np.clip(rng.lognormal(4.2, 0.5, len(hexes)), 30, 400),
        })
    disp_m = {}
    for p in ["metrics_role4.json", "src/dispatch/metrics_role4.json"]:
        if os.path.exists(p):
            with open(p) as f:
                disp_m = json.load(f)
            break
    path = generate_report(demo_causal, fdf, disp_m)
    print(f"Done: {path}")
