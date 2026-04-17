# backend/api/stt.py
#
# Local Whisper speech-to-text using faster-whisper.
#
# faster-whisper is a drop-in replacement for openai-whisper that:
#   - installs cleanly on Python 3.11 slim images (no setup.py issues)
#   - runs 2-4x faster on CPU via CTranslate2
#   - uses the same underlying Whisper models
#
# The model is loaded once at module import time and reused for every request.

import io
import os
import tempfile

from faster_whisper import WhisperModel

# ── Model loading ─────────────────────────────────────────────────────────────
# WHISPER_MODEL env var lets you trade speed for accuracy:
#   "tiny"   — fastest  (~40 MB)
#   "base"   — default  (~140 MB)
#   "small"  — better   (~460 MB)
#   "medium" — near human-level (~1.5 GB)
#
# device="cpu" + compute_type="int8" is the right combo for CPU-only containers
# — int8 quantisation cuts memory use and speeds up inference significantly.

_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_LANGUAGE   = os.getenv("WHISPER_LANGUAGE", None)  # e.g. "fr", "en", "ar" — None = auto-detect

print(f"[stt] Loading Whisper model '{_MODEL_NAME}' (language={_LANGUAGE or 'auto'})...")
_model = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
print(f"[stt] Whisper '{_MODEL_NAME}' ready.")


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """
    Transcribe raw audio bytes to text using local Whisper.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data in any ffmpeg-decodable format.
    mime_type : str
        MIME type hint — used to pick the right file extension so ffmpeg
        knows how to decode the bytes. Common values:
            "audio/webm"  — Chrome/Firefox MediaRecorder default
            "audio/wav"   — uncompressed PCM
            "audio/mp4"   — Safari MediaRecorder default
            "audio/ogg"   — Firefox alternative

    Returns
    -------
    str
        Transcribed text, or an empty string if transcription fails.
    """
    if not audio_bytes:
        return ""

    ext = _mime_to_ext(mime_type)

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # faster-whisper returns a generator of segments
        segments, _info = _model.transcribe(
            tmp_path,
            beam_size=5,
            language=_LANGUAGE,  # None = auto-detect, or e.g. 'fr', 'en'
            vad_filter=True,     # skip silent segments, reduces hallucinations
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text

    except Exception as e:
        print(f"[stt] transcribe error: {e}")
        return ""

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _mime_to_ext(mime_type: str) -> str:
    mapping = {
        "audio/webm":   ".webm",
        "audio/ogg":    ".ogg",
        "audio/wav":    ".wav",
        "audio/wave":   ".wav",
        "audio/x-wav":  ".wav",
        "audio/mp4":    ".mp4",
        "audio/mpeg":   ".mp3",
        "audio/mp3":    ".mp3",
        "audio/flac":   ".flac",
    }
    return mapping.get(mime_type.lower().split(";")[0].strip(), ".webm")