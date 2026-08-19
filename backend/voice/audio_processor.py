import io
from typing import Tuple


class AudioProcessor:
    """
    Validates and normalizes audio data before passing to Sarvam STT.
    """

    def validate_audio(self, audio_bytes: bytes, max_size_mb: float = 10.0) -> Tuple[bool, str]:
        if not audio_bytes or len(audio_bytes) < 100:
            return False, "Audio stream is empty or too short (no speech detected)."

        size_mb = len(audio_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"Audio file exceeds maximum size limit of {max_size_mb}MB."

        return True, "Valid audio input."
