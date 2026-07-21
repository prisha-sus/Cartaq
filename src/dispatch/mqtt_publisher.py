"""
Role 4: Agentic NLP & Systems Integration Engineer
Eclipse Mosquitto MQTT publisher for digital signage / IoT alerts.

Topic schema:
    cartaq/advisory/{city}/{h3_index}      Full multilingual advisory JSON (retained)
    cartaq/alert/{category}/{h3_index}     Compact AQI payload for IoT devices (retained)
    cartaq/system/heartbeat                Pipeline heartbeat (not retained)

Retained messages mean digital signage displays get the last advisory
immediately on connect - no need to wait for the next forecast cycle.

Usage:
    cd Cartaq
    python src/dispatch/mqtt_publisher.py

Env vars:
    MQTT_BROKER_HOST=localhost
    MQTT_BROKER_PORT=1883
    MQTT_DRY_RUN=true        (default; set false when Mosquitto is running)

Requires:
    pip install paho-mqtt python-dotenv
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MQTT_HOST    = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_PORT    = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USER    = os.getenv("MQTT_USERNAME",    "")
MQTT_PASS    = os.getenv("MQTT_PASSWORD",    "")
MQTT_QOS     = 1        # at-least-once delivery
MQTT_RETAIN  = True     # retained: displays receive current state on connect
DRY_RUN      = os.getenv("MQTT_DRY_RUN", "true").lower() != "false"
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PublishResult:
    topic:        str
    payload_bytes: int
    status:       str    # 'published' | 'dry_run' | 'failed'
    latency_ms:   float
    error:        Optional[str] = None


class MQTTPublisher:
    """
    Publishes AQI advisories to Eclipse Mosquitto.

    Dry-run mode (default):
        Logs what would be published; no broker connection.

    Live mode (MQTT_DRY_RUN=false):
        Connects to Mosquitto and publishes with QoS=1, retained=True.
    """

    def __init__(self) -> None:
        self.dry_run = DRY_RUN
        self._client = None

        if self.dry_run:
            print(f"  MQTTPublisher: DRY-RUN mode  (broker={MQTT_HOST}:{MQTT_PORT})")
            print("  Set MQTT_DRY_RUN=false and start Mosquitto (docker compose up mosquitto).")
        else:
            self._connect()

    def _connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(
                client_id="cartaq_dispatcher",
                clean_session=False,
                protocol=mqtt.MQTTv311,
            )
            self._client.on_connect    = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            if MQTT_USER:
                self._client.username_pw_set(MQTT_USER, MQTT_PASS)

            self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self._client.loop_start()
            time.sleep(0.5)   # let the connect callback fire

        except ImportError:
            print("  ERROR: paho-mqtt not installed. pip install paho-mqtt")
            self.dry_run = True
        except Exception as e:
            print(f"  WARNING: MQTT broker unreachable ({e}). Falling back to dry-run.")
            self.dry_run = True

    def _on_connect(self, client, userdata, flags, rc: int) -> None:
        codes = {0: "OK", 1: "bad protocol", 2: "client id rejected",
                 3: "server unavailable", 4: "bad credentials", 5: "not authorised"}
        print(f"  MQTT connected: {codes.get(rc, f'rc={rc}')}")

    def _on_disconnect(self, client, userdata, rc: int) -> None:
        print(f"  MQTT disconnected (rc={rc})")

    def publish(self, topic: str, payload: dict) -> PublishResult:
        """Publish a single JSON message."""
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        t0  = time.perf_counter()

        if self.dry_run:
            time.sleep(0.002)
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [DRY-RUN] {topic:<60}  {len(raw):>5} bytes")
            return PublishResult(topic, len(raw), "dry_run", latency_ms)

        try:
            import paho.mqtt.client as mqtt
            info = self._client.publish(topic, raw, qos=MQTT_QOS, retain=MQTT_RETAIN)
            info.wait_for_publish(timeout=5.0)
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [PUBLISHED] {topic:<55}  {len(raw):>5} bytes  {latency_ms:.1f}ms")
            return PublishResult(topic, len(raw), "published", latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [FAILED] {topic}: {e}")
            return PublishResult(topic, len(raw), "failed", latency_ms, str(e))

    def publish_dispatch(
        self,
        target: dict,
        translations: dict[str, str],
    ) -> list[PublishResult]:
        """
        Publish two MQTT messages for one dispatch target:
            1. Full advisory (all languages) - for human-readable signage
            2. Compact alert - for IoT devices / embedded displays
        """
        city   = target.get("city", "pune")
        h3_idx = target["h3_index"]
        cat    = target.get("facility_category", "facility")

        # 1. Full advisory
        advisory_payload = {
            "schema_version": "1.0",
            "h3_index":       h3_idx,
            "predicted_aqi":  target["predicted_aqi"],
            "aqi_category":   target.get("aqi_category", "Unhealthy"),
            "horizon_hours":  target.get("horizon_hours", 24),
            "facility": {
                "name":     target.get("facility_name"),
                "category": cat,
                "lat":      target.get("lat"),
                "lon":      target.get("lon"),
            },
            "advisories": {
                "en": translations.get("eng_Latn", ""),
                "hi": translations.get("hin_Deva", ""),
                "mr": translations.get("mar_Deva", ""),
            },
            "published_at": datetime.now(timezone.utc).isoformat(),
        }

        # 2. Compact IoT alert (< 256 bytes target for constrained devices)
        compact_payload = {
            "aqi":  round(target["predicted_aqi"]),
            "cat":  target.get("aqi_category", "Unhealthy")[:3].upper(),  # "UNH"
            "hex":  h3_idx,
            "h":    target.get("horizon_hours", 24),
            "ts":   int(datetime.now(timezone.utc).timestamp()),
        }

        results = [
            self.publish(f"cartaq/advisory/{city}/{h3_idx}", advisory_payload),
            self.publish(f"cartaq/alert/{cat}/{h3_idx}",    compact_payload),
        ]

        return results

    def publish_heartbeat(self, n_dispatched: int, n_mqtt: int) -> PublishResult:
        """System heartbeat - not retained, signals pipeline is alive."""
        payload = {
            "ts":            datetime.now(timezone.utc).isoformat(),
            "n_dispatched":  n_dispatched,
            "n_mqtt":        n_mqtt,
            "status":        "ok",
        }
        # Override retain for heartbeat
        original = MQTT_RETAIN
        # Temporarily publish without retain
        try:
            import paho.mqtt.client as mqtt
            raw = json.dumps(payload).encode()
            if self._client and not self.dry_run:
                info = self._client.publish("cartaq/system/heartbeat", raw, qos=0, retain=False)
                info.wait_for_publish(timeout=2.0)
        except Exception:
            pass
        return self.publish("cartaq/system/heartbeat", payload)

    def disconnect(self) -> None:
        if self._client and not self.dry_run:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Cartaq Role 4: MQTT Publisher Test ===\n")

    pub = MQTTPublisher()

    test_target = {
        "h3_index":          "8860a25997fffff",
        "predicted_aqi":     210.1,
        "aqi_category":      "Very Unhealthy",
        "horizon_hours":     24,
        "city":              "pune",
        "facility_name":     "KEM Hospital Pune",
        "facility_category": "hospital",
        "lat":               18.5195,
        "lon":               73.8553,
    }
    test_translations = {
        "eng_Latn": "HEALTH ADVISORY: Air quality around KEM Hospital is forecast to reach 210 AQI. Activate air quality protocol.",
        "hin_Deva": "[Hindi translation placeholder]",
        "mar_Deva": "[Marathi translation placeholder]",
    }

    results = pub.publish_dispatch(test_target, test_translations)
    pub.publish_heartbeat(n_dispatched=1, n_mqtt=len(results))
    pub.disconnect()

    print(f"\nPublished {len(results)} messages.")
    for r in results:
        print(f"  {r.topic:<60} status={r.status}  {r.latency_ms:.1f}ms")
