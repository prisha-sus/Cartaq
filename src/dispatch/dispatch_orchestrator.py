"""
Role 4: Agentic NLP & Systems Integration Engineer
End-to-end dispatch orchestrator.

Pipeline:
    1. Load high-AQI events (mock payload or real ST-GNN parquet)
    2. PostGIS spatial query -> vulnerable facilities inside each hex
    3. Build English health advisory per facility
    4. Translate via AI4Bharat IndicTrans2 (Hindi + Marathi)
    5. Dispatch automated Twilio voice call
    6. Publish to Eclipse Mosquitto MQTT broker (digital signage)
    7. Write dispatch log to metrics_role4.json

Metrics tracked (Role 4 zero-blocker contract):
    - end_to_end_latency_ms     time from event load -> first call queued
    - spatial_query_ms          PostGIS query latency per hex
    - translation_ms            IndicTrans2 latency per advisory
    - twilio_ms                 Twilio API latency per call
    - mqtt_ms                   MQTT publish latency per message

Usage:
    cd Cartaq
    python src/dispatch/dispatch_orchestrator.py
    python src/dispatch/dispatch_orchestrator.py --mock-payload src/dispatch/mock_payload.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure src/dispatch is on the import path ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from spatial_query      import load_high_aqi_events, run_spatial_queries
from translation_pipeline import IndicTrans2Pipeline, build_advisory
from twilio_dispatcher  import TwilioDispatcher
from mqtt_publisher     import MQTTPublisher


def run_pipeline(mock_payload_path: str = "") -> dict:
    if not mock_payload_path:
        mock_payload_path = str(Path(__file__).parent / "mock_payload.json")
    """
    Execute the full dispatch pipeline.
    Returns a flat metrics dict (serialisable to JSON).
    """
    t_pipeline_start = time.perf_counter()
    ts_start = datetime.now(timezone.utc).isoformat()

    raw_latencies: dict[str, list[float]] = {
        "spatial_ms":     [],
        "translation_ms": [],
        "twilio_ms":      [],
        "mqtt_ms":        [],
    }

    print("=" * 60)
    print("  Cartaq Role 4 - Dispatch Orchestrator")
    print(f"  Started: {ts_start}")
    print("=" * 60)

    # ── Step 1: Load events ───────────────────────────────────
    print("\n[1/5] Loading high-AQI forecast events...")
    events = load_high_aqi_events(mock_payload_path)
    print(f"  {len(events)} events above threshold")

    # ── Step 2: Spatial queries ───────────────────────────────
    print("\n[2/5] PostGIS spatial intersection...")
    t0 = time.perf_counter()
    dispatch_targets, spatial_stats = run_spatial_queries(events)
    raw_latencies["spatial_ms"].append(spatial_stats["total_ms"])
    print(
        f"  {len(dispatch_targets)} dispatch targets  |  "
        f"avg {spatial_stats['per_hex_avg_ms']}ms / hex  |  "
        f"max {spatial_stats['per_hex_max_ms']}ms"
    )

    if not dispatch_targets:
        print("  No targets found. Pipeline complete (no alerts sent).")
        return _build_metrics(ts_start, time.perf_counter() - t_pipeline_start, raw_latencies, 0, 0)

    # ── Step 3: Initialise services ───────────────────────────
    print("\n[3/5] Initialising translation pipeline, Twilio, MQTT...")
    translator = IndicTrans2Pipeline(device="cpu")
    dispatcher = TwilioDispatcher()
    mqtt_pub   = MQTTPublisher()

    # ── Step 4: Per-target dispatch loop ─────────────────────
    n_calls   = 0
    n_mqtt    = 0

    print(f"\n[4/5] Dispatching {len(dispatch_targets)} targets...")
    for target in dispatch_targets:
        name     = target["facility_name"]
        category = target["facility_category"]
        aqi      = target["predicted_aqi"]
        horizon  = target.get("horizon_hours", 24)

        print(f"\n  |-- {name} ({category.upper()}) - AQI={aqi:.1f}")

        # Build English advisory
        advisory_en = build_advisory(name, category, aqi, horizon)
        target["advisory_en"] = advisory_en

        # Translate
        t0 = time.perf_counter()
        translations = translator.translate_advisory(
            advisory_en, target_langs=["hin_Deva", "mar_Deva"]
        )
        trans_ms = (time.perf_counter() - t0) * 1000
        raw_latencies["translation_ms"].append(trans_ms)
        target["advisory_hi"] = translations.get("hin_Deva", "")
        target["advisory_mr"] = translations.get("mar_Deva", "")

        # Twilio call
        t0 = time.perf_counter()
        call_result = dispatcher.dispatch_call(
            phone_number=target.get("phone_number") or os.getenv("TWILIO_TO_NUMBER", "+910000000000"),
            facility_name=name,
            advisory_text=advisory_en,
        )
        twilio_ms = (time.perf_counter() - t0) * 1000
        raw_latencies["twilio_ms"].append(twilio_ms)
        target["twilio_call_sid"] = call_result.call_sid
        target["twilio_status"]   = call_result.status
        if call_result.status in ("queued", "dry_run"):
            n_calls += 1

        # MQTT publish
        t0 = time.perf_counter()
        mqtt_results = mqtt_pub.publish_dispatch(target, translations)
        mqtt_ms = (time.perf_counter() - t0) * 1000
        raw_latencies["mqtt_ms"].append(mqtt_ms)
        n_mqtt += len(mqtt_results)

        print(f"  +-- call={call_result.status}  mqtt={len(mqtt_results)} msgs  trans={trans_ms:.0f}ms")

    # MQTT heartbeat
    mqtt_pub.publish_heartbeat(n_calls, n_mqtt)
    mqtt_pub.disconnect()

    # ── Step 5: Metrics ───────────────────────────────────────
    total_elapsed = time.perf_counter() - t_pipeline_start
    metrics = _build_metrics(ts_start, total_elapsed, raw_latencies, n_calls, n_mqtt)
    metrics["spatial_per_hex_avg_ms"] = spatial_stats["per_hex_avg_ms"]
    metrics["spatial_per_hex_max_ms"] = spatial_stats["per_hex_max_ms"]
    metrics["total_targets"]          = len(dispatch_targets)

    _print_summary(metrics)

    # Write metrics to project root (two levels up from src/dispatch/)
    metrics_path = Path(__file__).parent.parent.parent / "metrics_role4.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics -> {metrics_path}")

    return metrics


def _build_metrics(
    ts_start: str,
    elapsed_s: float,
    raw: dict[str, list[float]],
    n_calls: int,
    n_mqtt: int,
) -> dict:
    def avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0
    def mx(lst):  return round(max(lst), 1) if lst else 0

    return {
        "pipeline_start_utc":       ts_start,
        "end_to_end_latency_ms":    round(elapsed_s * 1000, 1),
        "calls_dispatched":         n_calls,
        "mqtt_messages_published":  n_mqtt,
        "avg_translation_ms":       avg(raw["translation_ms"]),
        "max_translation_ms":       mx(raw["translation_ms"]),
        "avg_twilio_ms":            avg(raw["twilio_ms"]),
        "avg_mqtt_ms":              avg(raw["mqtt_ms"]),
    }


def _print_summary(m: dict) -> None:
    print("\n[5/5] Pipeline complete.")
    print(f"  End-to-end latency:        {m['end_to_end_latency_ms']:.0f} ms")
    print(f"  Total targets processed:   {m.get('total_targets', '?')}")
    print(f"  Calls dispatched:          {m['calls_dispatched']}")
    print(f"  MQTT messages published:   {m['mqtt_messages_published']}")
    print(f"  Avg spatial query:         {m.get('spatial_per_hex_avg_ms', '?')} ms / hex")
    print(f"  Avg translation latency:   {m['avg_translation_ms']} ms / advisory")
    print(f"  Avg Twilio latency:        {m['avg_twilio_ms']} ms / call")
    print(f"  Avg MQTT latency:          {m['avg_mqtt_ms']} ms / message")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartaq Role 4: Dispatch Orchestrator")
    parser.add_argument(
        "--mock-payload",
        default="",
        metavar="PATH",
        help="Path to mock_payload.json (default: src/dispatch/mock_payload.json)",
    )
    args = parser.parse_args()
    run_pipeline(args.mock_payload)
