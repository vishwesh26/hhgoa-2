import os
import time
import httpx
import mimetypes
from typing import Tuple, Dict, Any, Optional
from backend.config import settings


class SarvamSTTClient:
    """
    Client for Sarvam AI Speech-to-Text (`saaras:v2` / `saarika:v2`).
    Handles English, Hindi, Marathi, and code-mixed speech with structured latency tracking.
    """

    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY or os.environ.get("SARVAM_API_KEY", "")
        self.model = settings.SARVAM_STT_MODEL
        self.url = settings.SARVAM_STT_URL
        self.timeout = settings.SARVAM_TIMEOUT_SECONDS

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "recording.webm",
        language_code: Optional[str] = "unknown"
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        """
        Transcribes audio using Sarvam API.
        Returns: (transcript, detected_language, latency_ms, raw_metadata)
        """
        start_time = time.perf_counter()

        if not self.api_key:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            print("[WARN] SARVAM_API_KEY is not configured in .env!")
            return (
                "",
                "en",
                round(latency_ms, 2),
                {"error": "SARVAM_API_KEY is missing in .env"}
            )

        headers = {
            "api-subscription-key": self.api_key
        }

        # Determine MIME type from filename
        mime_type = mimetypes.guess_type(filename)[0] or "audio/webm"

        files = {
            "file": (filename, audio_bytes, mime_type)
        }
        data = {
            "model": self.model,
            "language_code": language_code or "unknown"
        }

        print(f"[INFO] Sending audio to Sarvam STT ({len(audio_bytes)} bytes, {mime_type}, model={self.model})...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, headers=headers, files=files, data=data)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if response.status_code == 200:
                    res_json = response.json()
                    transcript = res_json.get("transcript", "")
                    detected_lang = res_json.get("language_code", language_code or "unknown")
                    try:
                        print(f"[SUCCESS] Sarvam STT transcribed in {latency_ms:.1f}ms: \"{transcript}\" (Lang: {detected_lang})")
                    except UnicodeEncodeError:
                        print(f"[SUCCESS] Sarvam STT transcribed in {latency_ms:.1f}ms (Lang: {detected_lang})")
                    return transcript.strip(), detected_lang, round(latency_ms, 2), res_json
                else:
                    error_msg = f"Sarvam API status {response.status_code}: {response.text}"
                    try:
                        print(f"[ERROR] {error_msg}")
                    except UnicodeEncodeError:
                        print(f"[ERROR] Sarvam API returned status {response.status_code}")
                    return "", "en", round(latency_ms, 2), {"error": error_msg}
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                try:
                    print(f"[ERROR] Sarvam STT request exception: {e}")
                except UnicodeEncodeError:
                    print(f"[ERROR] Sarvam STT request failed")
                return "", "en", round(latency_ms, 2), {"error": str(e)}
