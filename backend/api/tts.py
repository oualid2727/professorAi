# backend/api/tts.py
#
# Text-to-Speech — supports two backends via TTS_BACKEND env var:
#   "xtts" (default, original) — local XTTS v2, slow on CPU, voice cloning
#   "edge" — Microsoft Edge TTS via edge-tts, free, no API key, fast
#
# Public interface (synthesize()) is unchanged either way, so main.py
# doesn't need to know which backend is active.

import asyncio
import io
import os
import re
import wave
from typing import List, Dict, Tuple

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

TTS_BACKEND        = os.getenv("TTS_BACKEND",       "edge").lower()   # "xtts" | "edge"
TTS_LANGUAGE        = os.getenv("TTS_LANGUAGE",       "fr")
TTS_SPEAKER_WAV     = os.getenv("TTS_SPEAKER_WAV",    "/app/api/reference_voice/professor.wav")
TTS_ENABLED         = os.getenv("TTS_ENABLED",        "true").lower() == "true"
SAMPLE_RATE         = 24000  # only relevant for the xtts backend

# edge-tts voice per language — swap these for any voice from `edge-tts --list-voices`
_EDGE_VOICES: Dict[str, str] = {
    "fr": os.getenv("EDGE_VOICE_FR", "fr-FR-DeniseNeural"),
    "en": os.getenv("EDGE_VOICE_EN", "en-US-JennyNeural"),
    "ar": os.getenv("EDGE_VOICE_AR", "ar-MA-MounaNeural"),   # Moroccan Arabic, female
    "es": os.getenv("EDGE_VOICE_ES", "es-ES-ElviraNeural"),
    "de": os.getenv("EDGE_VOICE_DE", "de-DE-KatjaNeural"),
    "it": os.getenv("EDGE_VOICE_IT", "it-IT-ElsaNeural"),
    "pt": os.getenv("EDGE_VOICE_PT", "pt-PT-RaquelNeural"),
}


# ── Device detection (xtts only) ──────────────────────────────────────────────

def _get_device() -> str:
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[tts] GPU detected: {name}")
            return "cuda"
    except Exception:
        pass
    print("[tts] No GPU detected, falling back to CPU")
    return "cpu"


# ── XTTS v2 model loading (unchanged, only used if TTS_BACKEND=xtts) ─────────

_tts = None

def _get_tts():
    global _tts
    if _tts is not None:
        return _tts if _tts is not False else None

    if not TTS_ENABLED:
        _tts = False
        return None

    try:
        import os as _os  # noqa: PLC0415
        from TTS.api import TTS  # noqa: PLC0415

        _os.environ["COQUI_TOS_AGREED"] = "1"

        device = _get_device()
        print(f"[tts] Loading XTTS v2 on {device} (first run downloads ~1.8 GB)...")
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"[tts] XTTS v2 ready on {device}.")
    except Exception as e:
        print(f"[tts] Failed to load XTTS v2: {e}")
        _tts = False

    return _tts if _tts is not False else None


# ── Phoneme → Viseme mapping (unchanged, used by both backends) ──────────────

_PHONEME_TO_VISEME: Dict[str, int] = {
    "":    0, " ": 0, "_": 0,
    "p":   1, "b":  1, "m":  1,
    "f":   2, "v":  2,
    "θ":   3, "ð":  3,
    "t":   4, "d":  4,
    "k":   5, "g":  5,
    "tʃ":  6, "dʒ": 6, "ʃ": 6, "ʒ": 6,
    "s":   7, "z":  7,
    "n":   8, "ŋ":  8,
    "r":   9, "ʁ":  9, "ɹ": 9, "ɾ": 9,
    "a":  10, "ɑ": 10, "æ": 10, "ä": 10,
    "e":  11, "ɛ": 11, "ə": 11, "œ": 11, "ø": 11,
    "i":  12, "ɪ": 12, "y": 12,
    "o":  13, "ɔ": 13,
    "u":  14, "ʊ": 14,
    "l":   8, "j": 12, "w": 14, "h": 0,
}

_LANG_TO_ESPEAK: Dict[str, str] = {
    "fr": "fr-fr", "en": "en-us", "ar": "ar", "es": "es",
    "de": "de", "it": "it", "pt": "pt",
}


def _text_to_phonemes(text: str, language: str) -> List[str]:
    try:
        import logging                               # noqa: PLC0415
        from phonemizer import phonemize            # noqa: PLC0415
        from phonemizer.separator import Separator  # noqa: PLC0415

        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        espeak_lang = _LANG_TO_ESPEAK.get(language, "fr-fr")
        sep = Separator(phone=" ", word="  ", syllable="")

        result = phonemize(
            text, backend="espeak", language=espeak_lang, separator=sep,
            strip=True, preserve_punctuation=False, language_switch="remove-flags",
        )
        return [p for p in result.split(" ") if p]
    except Exception as e:
        print(f"[tts] phonemizer error: {e}")
        return []


def _phonemes_to_visemes(phonemes: List[str], duration_sec: float) -> List[Dict]:
    if not phonemes or duration_sec <= 0:
        return []
    step = duration_sec / len(phonemes)
    return [
        {"id": _PHONEME_TO_VISEME.get(ph, 0), "timestamp": round(i * step, 3)}
        for i, ph in enumerate(phonemes)
    ]


# ── PCM → WAV (xtts only) ─────────────────────────────────────────────────────

def _pcm_to_wav(pcm_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    samples = (pcm_array * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _silence_wav(duration_sec: float = 0.1) -> bytes:
    samples = np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)
    return _pcm_to_wav(samples)


# ── edge-tts backend ───────────────────────────────────────────────────────────

def _mp3_duration_sec(mp3_bytes: bytes) -> float:
    """Get real audio duration from mp3 bytes without decoding full PCM."""
    try:
        from mutagen.mp3 import MP3  # noqa: PLC0415
        return MP3(io.BytesIO(mp3_bytes)).info.length
    except Exception as e:
        print(f"[tts] mutagen duration error: {e}")
        # Rough fallback estimate — better than crashing
        return max(len(mp3_bytes) / 4000, 0.3)


async def _edge_synthesize_async(text: str, voice: str) -> bytes:
    import edge_tts  # noqa: PLC0415
    communicate = edge_tts.Communicate(text, voice)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def _synthesize_edge(text: str, language: str) -> Tuple[bytes, List[Dict]]:
    voice = _EDGE_VOICES.get(language, _EDGE_VOICES["fr"])
    try:
        mp3_bytes = asyncio.run(_edge_synthesize_async(text, voice))
        print(f"[tts] edge-tts synthesized {len(mp3_bytes)} bytes for voice={voice}")
    except Exception as e:
        print(f"[tts] edge-tts error: {e}")
        return _silence_wav(), []

    if not mp3_bytes:
        print("[tts] edge-tts returned empty audio")
        return _silence_wav(), []

    duration_sec = _mp3_duration_sec(mp3_bytes)
    print(f"[tts] mp3 duration: {duration_sec}s")
    phonemes = _text_to_phonemes(text, language)
    visemes = _phonemes_to_visemes(phonemes, duration_sec)
    return mp3_bytes, visemes


# ── xtts backend (original logic, unchanged) ──────────────────────────────────

def _synthesize_xtts(text: str, language: str) -> Tuple[bytes, List[Dict]]:
    tts = _get_tts()

    if tts is None:
        wav_bytes = _silence_wav(len(text) * 0.06)
    else:
        try:
            if os.path.isfile(TTS_SPEAKER_WAV):
                pcm = tts.tts(text=text, speaker_wav=TTS_SPEAKER_WAV, language=language)
            else:
                speakers = tts.speakers or []
                speaker = speakers[0] if speakers else None
                pcm = tts.tts(text=text, speaker=speaker, language=language)
            wav_bytes = _pcm_to_wav(np.array(pcm, dtype=np.float32))
        except Exception as e:
            print(f"[tts] synthesis error: {e}")
            wav_bytes = _silence_wav()

    duration_sec = len(wav_bytes) / (SAMPLE_RATE * 2)
    phonemes = _text_to_phonemes(text, language)
    visemes = _phonemes_to_visemes(phonemes, duration_sec)
    return wav_bytes, visemes


def _strip_markdown_for_speech(text: str) -> str:
    # Remove fenced code blocks entirely (``` ... ```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code backticks, keep the content
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Bold / italics: **text**, __text__, *text*, _text_ → text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.*?)_(?!\w)", r"\1", text)
    # Bullet points at line start (*, -, +) → drop the marker
    text = re.sub(r"^[ \t]*[\*\-\+][ \t]+", "", text, flags=re.MULTILINE)
    # Numbered list markers "1. " "2) " → drop
    text = re.sub(r"^[ \t]*\d+[\.\)][ \t]+", "", text, flags=re.MULTILINE)
    # Markdown headers (#, ##, ...) → drop the hashes
    text = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", text, flags=re.MULTILINE)
    # Collapse extra whitespace left behind
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", " ", text)
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(text: str, language: str = "") -> Tuple[bytes, List[Dict]]:
    """
    Synthesize speech and return (audio_bytes, visemes).
    Backend controlled by TTS_BACKEND env var ("xtts" or "edge").
    Audio format is WAV for xtts, MP3 for edge — the browser's
    decodeAudioData() handles both transparently, no frontend change needed.
    """
    if not text.strip():
        return _silence_wav(), []

    text = _strip_markdown_for_speech(text)
    if not text.strip():
        return _silence_wav(), []

    lang = language or TTS_LANGUAGE

    if TTS_BACKEND == "edge":
        return _synthesize_edge(text, lang)
    return _synthesize_xtts(text, lang)



