"""
Role 4: Agentic NLP & Systems Integration Engineer
Twilio automated voice call dispatcher.

Sends TwiML voice calls to facility emergency contacts when a high-AQI
event is forecast to impact them.

Operates in DRY-RUN mode by default (no real calls, no billing).
Set TWILIO_DRY_RUN=false and provide real credentials to enable live calls.

Usage:
    cd Cartaq
    python src/dispatch/twilio_dispatcher.py

Required env vars (add to .env for live calls):
    TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TWILIO_AUTH_TOKEN=your_auth_token_here
    TWILIO_FROM_NUMBER=+1XXXXXXXXXX    (your Twilio number)
    TWILIO_TO_NUMBER=+91XXXXXXXXXX     (test recipient)
    TWILIO_DRY_RUN=false               (default: true)

Requires:
    pip install twilio python-dotenv
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",  "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+15005550006")  # Twilio magic test number
TWILIO_TO_NUMBER   = os.getenv("TWILIO_TO_NUMBER",   "")
DRY_RUN            = os.getenv("TWILIO_DRY_RUN", "true").lower() != "false"
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CallResult:
    facility_name:  str
    phone_number:   str
    call_sid:       Optional[str]
    status:         str             # 'queued' | 'failed' | 'dry_run'
    latency_ms:     float
    advisory_text:  str
    error:          Optional[str] = None


class TwilioDispatcher:
    """
    Dispatches automated TwiML voice calls via Twilio REST API.

    In dry-run mode (default):
        - No network calls, no billing
        - Logs what would have been sent
        - Returns CallResult with status='dry_run'

    In live mode (TWILIO_DRY_RUN=false):
        - Calls Twilio REST API
        - Returns the Twilio call SID
    """

    def __init__(self) -> None:
        creds_present = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
        self.dry_run = DRY_RUN or not creds_present

        if self.dry_run:
            reason = "DRY_RUN=true" if DRY_RUN else "credentials missing"
            print(f"  TwilioDispatcher: DRY-RUN mode ({reason})")
            print("  Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_DRY_RUN=false to place live calls.")
        else:
            print("  TwilioDispatcher: LIVE mode")

        self._client = None
        if not self.dry_run:
            self._init_client()

    def _init_client(self) -> None:
        try:
            from twilio.rest import Client  # type: ignore
            self._client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            print("  Twilio REST client initialised.")
        except ImportError:
            print("  ERROR: twilio package not installed. pip install twilio")
            self.dry_run = True
        except Exception as e:
            print(f"  ERROR: Twilio init failed: {e}")
            self.dry_run = True

    @staticmethod
    def build_twiml(advisory_text: str) -> str:
        """
        Build a TwiML response XML for a voice call.
        Uses the Indian-English 'alice' voice for naturalness.
        Repeats the advisory once so the listener can catch missed details.
        """
        safe = (
            advisory_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Say voice="alice" language="en-IN">{safe}</Say>'
            '<Pause length="2"/>'
            '<Say voice="alice" language="en-IN">This message will now repeat.</Say>'
            '<Pause length="1"/>'
            f'<Say voice="alice" language="en-IN">{safe}</Say>'
            "</Response>"
        )

    def dispatch_call(
        self,
        phone_number: str,
        facility_name: str,
        advisory_text: str,
    ) -> CallResult:
        """Send (or simulate) a single voice call."""
        t0 = time.perf_counter()

        if self.dry_run:
            # Simulate ~50 ms "network" latency
            time.sleep(0.05)
            latency_ms = (time.perf_counter() - t0) * 1000
            print(
                f"  [DRY-RUN] Would call {phone_number}  "
                f"facility='{facility_name}'  ({latency_ms:.0f}ms)"
            )
            return CallResult(
                facility_name=facility_name,
                phone_number=phone_number,
                call_sid=None,
                status="dry_run",
                latency_ms=latency_ms,
                advisory_text=advisory_text,
            )

        try:
            twiml = self.build_twiml(advisory_text)
            call = self._client.calls.create(
                twiml=twiml,
                to=phone_number,
                from_=TWILIO_FROM_NUMBER,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            print(
                f"  [QUEUED] {facility_name} -> {phone_number}  "
                f"SID={call.sid}  ({latency_ms:.0f}ms)"
            )
            return CallResult(
                facility_name=facility_name,
                phone_number=phone_number,
                call_sid=call.sid,
                status=call.status,
                latency_ms=latency_ms,
                advisory_text=advisory_text,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [FAILED] Call to {phone_number}: {e}")
            return CallResult(
                facility_name=facility_name,
                phone_number=phone_number,
                call_sid=None,
                status="failed",
                latency_ms=latency_ms,
                advisory_text=advisory_text,
                error=str(e),
            )

    def dispatch_batch(
        self,
        dispatch_targets: list[dict],
        default_to: Optional[str] = None,
    ) -> list[CallResult]:
        """
        Call every target in the list.
        Each dict must have: facility_name, advisory_en.
        Optionally: phone_number (falls back to TWILIO_TO_NUMBER or default_to).
        """
        fallback_phone = default_to or TWILIO_TO_NUMBER or "+910000000000"
        results: list[CallResult] = []

        for target in dispatch_targets:
            phone = target.get("phone_number") or fallback_phone
            result = self.dispatch_call(
                phone_number=phone,
                facility_name=target["facility_name"],
                advisory_text=target.get("advisory_en", "Health alert: elevated air pollution expected."),
            )
            results.append(result)

        return results


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Cartaq Role 4: Twilio Dispatcher Test ===\n")

    dispatcher = TwilioDispatcher()

    test_targets = [
        {
            "facility_name": "KEM Hospital Pune",
            "facility_category": "hospital",
            "phone_number": "+912026125600",
            "advisory_en": (
                "HEALTH ADVISORY - CARTAQ AIR QUALITY INTELLIGENCE PLATFORM. "
                "Air quality around KEM Hospital Pune is forecast to reach 210 AQI, "
                "classified as Very Unhealthy, within the next 24 hours. "
                "Hospital administrators should activate their air quality protocol: "
                "increase HEPA filtration and brief clinical staff."
            ),
        },
        {
            "facility_name": "Bal Shikshan Mandir English Medium School",
            "facility_category": "school",
            "phone_number": "+912024440500",
            "advisory_en": (
                "HEALTH ADVISORY - CARTAQ AIR QUALITY INTELLIGENCE PLATFORM. "
                "Air quality around Bal Shikshan Mandir School is forecast to reach 187 AQI, "
                "classified as Unhealthy, within the next 24 hours. "
                "School principals should cancel outdoor activities and keep students indoors."
            ),
        },
    ]

    results = dispatcher.dispatch_batch(test_targets)

    print(f"\nResults: {[r.status for r in results]}")
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    print(f"Avg dispatch latency: {avg_latency:.0f} ms")
