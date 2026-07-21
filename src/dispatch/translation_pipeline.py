"""
Role 4: Agentic NLP & Systems Integration Engineer
Advisory translation pipeline using AI4Bharat IndicTrans2.

Translates English health advisories into Hindi and Marathi
(and any other Indic language supported by IndicTrans2).

Model:
    AI4Bharat/indictrans2-en-indic-1B
    https://huggingface.co/ai4bharat/indictrans2-en-indic-1B

    On first run: downloads ~2 GB of model weights to
    ~/.cache/huggingface/hub/

Usage:
    cd Cartaq
    python src/dispatch/translation_pipeline.py

Requires:
    pip install transformers sentencepiece sacremoses torch
"""

import time
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)

# ── Flores-200 language codes used by IndicTrans2 ────────────────────────────
LANG_CODES: dict[str, str] = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "mar_Deva": "Marathi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu",
}
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TranslationResult:
    source_lang:  str
    target_lang:  str
    target_name:  str
    source_text:  str
    translated:   str
    latency_ms:   float
    used_model:   bool   # False when fallback placeholder was returned


class IndicTrans2Pipeline:
    """
    Lazy-loading wrapper around AI4Bharat IndicTrans2.

    The model is loaded only on the first call to translate().
    If loading fails (e.g. no internet, torch not installed),
    the pipeline falls back to clearly-labelled placeholder strings
    so the rest of the dispatch pipeline can still run.
    """

    MODEL_ID = "ai4bharat/indictrans2-en-indic-1B"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._tokenizer = None
        self._model     = None
        self._loaded    = False
        self._load_error: Optional[str] = None

    def _load(self) -> None:
        if self._loaded or self._load_error:
            return

        print(f"  Loading IndicTrans2 ({self.MODEL_ID}) on {self.device}...")
        print("  First run downloads ~2 GB weights - this may take a few minutes.")

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch  # noqa: F401  (validate import before heavier work)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID, trust_remote_code=True
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID, trust_remote_code=True
            ).to(self.device)
            self._model.eval()
            self._loaded = True
            print("  IndicTrans2 ready.")
        except Exception as e:
            self._load_error = str(e)
            print(f"  WARNING: Could not load IndicTrans2: {e}")
            print("  Dispatch will continue with placeholder translations.")

    def _translate_raw(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import torch

        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate(
        self,
        text: str,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
    ) -> TranslationResult:
        """Translate a single text from src_lang to tgt_lang."""
        self._load()

        t0 = time.perf_counter()
        used_model = False

        if self._loaded:
            try:
                translated = self._translate_raw(text, src_lang, tgt_lang)
                used_model = True
            except Exception as e:
                translated = f"[Translation error - {LANG_CODES.get(tgt_lang, tgt_lang)}: {e}]"
        else:
            lang_name  = LANG_CODES.get(tgt_lang, tgt_lang)
            translated = f"[{lang_name} translation unavailable - IndicTrans2 not loaded]"

        latency_ms = (time.perf_counter() - t0) * 1000

        return TranslationResult(
            source_lang=src_lang,
            target_lang=tgt_lang,
            target_name=LANG_CODES.get(tgt_lang, tgt_lang),
            source_text=text,
            translated=translated,
            latency_ms=latency_ms,
            used_model=used_model,
        )

    def translate_advisory(
        self,
        advisory_en: str,
        target_langs: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """
        Translate one English advisory into multiple Indic languages.
        Returns dict: {flores_lang_code -> translated_text, ...}
        Always includes the source English under "eng_Latn".
        """
        if target_langs is None:
            target_langs = ["hin_Deva", "mar_Deva"]

        results: dict[str, str] = {"eng_Latn": advisory_en}

        for lang_code in target_langs:
            result = self.translate(advisory_en, src_lang="eng_Latn", tgt_lang=lang_code)
            results[lang_code] = result.translated
            status = "model" if result.used_model else "placeholder"
            print(
                f"    [{result.target_name:<10}] ({result.latency_ms:5.0f}ms) [{status}]  "
                f"{result.translated[:70]}..."
            )

        return results


# ── Advisory builders ────────────────────────────────────────────────────────

def classify_aqi(aqi: float) -> str:
    if aqi <= 50:    return "Good"
    if aqi <= 100:   return "Moderate"
    if aqi <= 150:   return "Unhealthy for Sensitive Groups"
    if aqi <= 200:   return "Unhealthy"
    if aqi <= 300:   return "Very Unhealthy"
    return "Hazardous"


def _action_for_category(category: str, aqi_category: str) -> str:
    if category in ("hospital",):
        return (
            "Hospital administrators should activate their air quality protocol: "
            "increase HEPA filtration, keep ward windows sealed, and brief clinical staff."
        )
    if category == "school":
        return (
            "School principals should cancel all outdoor activities, keep students indoors, "
            "and distribute N95 masks to staff and students in higher-risk grades."
        )
    if category == "kindergarten":
        return (
            "Kindergarten staff should keep all children indoors with doors and windows sealed. "
            "Notify parents immediately and consider early dismissal."
        )
    if category == "care_home":
        return (
            "Care home staff should keep residents indoors, monitor respiratory symptoms closely, "
            "and have supplemental oxygen available for high-risk residents."
        )
    return "Please limit outdoor exposure and follow local authority guidance."


def build_advisory(
    facility_name: str,
    category: str,
    predicted_aqi: float,
    horizon_hours: int,
) -> str:
    """
    Construct a full English health advisory for a given facility.
    This text is then passed to IndicTrans2 for regional translation.
    """
    aqi_category = classify_aqi(predicted_aqi)
    action       = _action_for_category(category, aqi_category)

    return (
        f"HEALTH ADVISORY - CARTAQ AIR QUALITY INTELLIGENCE PLATFORM. "
        f"Air quality around {facility_name} is forecast to reach "
        f"{predicted_aqi:.0f} AQI, classified as {aqi_category}, "
        f"within the next {horizon_hours} hours. "
        f"{action} "
        f"This is an automated advisory. For more information visit cartaq.io."
    )


# ── Main (standalone test) ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Cartaq Role 4: Translation Pipeline Test ===\n")

    pipeline = IndicTrans2Pipeline(device="cpu")

    test_cases = [
        ("KEM Hospital Pune",                         "hospital",    210.1, 24),
        ("Bal Shikshan Mandir English Medium School",  "school",      187.3, 24),
        ("Goodluck Children Care Home",                "care_home",   165.8, 48),
    ]

    for facility_name, category, aqi, horizon in test_cases:
        advisory_en = build_advisory(facility_name, category, aqi, horizon)
        print(f"\n[EN] {advisory_en}\n")
        translations = pipeline.translate_advisory(
            advisory_en, target_langs=["hin_Deva", "mar_Deva"]
        )
        print()
