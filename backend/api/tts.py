# backend/api/tts.py
#
# Text-to-Speech using XTTS v2 (Coqui).
#
# Why XTTS v2?
#   - Multilingual: French, English, Arabic, Spanish and 13 more languages
#   - High quality: natural prosody, no robotic artifacts
#   - Voice cloning: consistent professor voice from a short reference WAV
#   - Fully local: no API keys, no cloud calls
#
# Visemes:
#   We extract phonemes from the text using the same phonemizer XTTS uses
#   internally, then map them to the 15-viseme set used by most 3D rigs
#   (Oculus/Ready Player Me standard). Each viseme gets a timestamp derived
#   from the audio duration so the 3D model's mouth stays in sync.
#
# Usage:
#   from api.tts import synthesize
#   wav_bytes, visemes = synthesize("Bonjour!", language="fr")

import io
import os
import re
import wave
from typing import List, Dict, Tuple

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

TTS_LANGUAGE      = os.getenv("TTS_LANGUAGE",       "fr")
TTS_SPEAKER_WAV   = os.getenv("TTS_SPEAKER_WAV",    "/app/api/reference_voice/professor.wav")
TTS_ENABLED       = os.getenv("TTS_ENABLED",        "true").lower() == "true"
SAMPLE_RATE       = 24000  # XTTS v2 always outputs at 24 kHz


# ── Model loading ─────────────────────────────────────────────────────────────
# Loaded once at module import. First load downloads model weights (~1.8 GB).
# Subsequent starts use the cached weights from ~/.local/share/tts/

_tts = None

def _get_tts():
    global _tts
    if _tts is not None:
        return _tts

    if not TTS_ENABLED:
        return None

    try:
        from TTS.api import TTS  # noqa: PLC0415
        print("[tts] Loading XTTS v2 model (first run downloads ~1.8 GB)...")
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
        print("[tts] XTTS v2 ready.")
    except Exception as e:
        print(f"[tts] Failed to load XTTS v2: {e}")
        _tts = None

    return _tts


# ── Phoneme → Viseme mapping ──────────────────────────────────────────────────
# Maps IPA phonemes to the 15-viseme Oculus standard used by Ready Player Me,
# MetaHuman, and most web-based 3D avatar rigs.
#
# Viseme IDs:
#   0=sil  1=PP  2=FF  3=TH  4=DD  5=kk  6=CH  7=SS  8=nn
#   9=RR  10=aa  11=E   12=I   13=O   14=U
#
# Reference: https://docs.readyplayer.me/ready-player-me/api-reference/avatars/morph-targets/oculus-ovr-libsync

_PHONEME_TO_VISEME: Dict[str, int] = {
    # Silence
    "":    0, " ": 0, "_": 0,
    # PP — bilabial stops and nasals (p, b, m)
    "p":   1, "b":  1, "m":  1,
    # FF — labiodental (f, v)
    "f":   2, "v":  2,
    # TH — dental fricatives
    "θ":   3, "ð":  3,
    # DD — alveolar stops (t, d)
    "t":   4, "d":  4,
    # kk — velar stops (k, g)
    "k":   5, "g":  5,
    # CH — postalveolar affricates/fricatives
    "tʃ":  6, "dʒ": 6, "ʃ": 6, "ʒ": 6,
    # SS — sibilants (s, z)
    "s":   7, "z":  7,
    # nn — nasals (n, ŋ)
    "n":   8, "ŋ":  8,
    # RR — rhotics (r, ʁ, ɹ)
    "r":   9, "ʁ":  9, "ɹ": 9, "ɾ": 9,
    # aa — open vowels
    "a":  10, "ɑ": 10, "æ": 10, "ä": 10,
    # E — mid front vowels
    "e":  11, "ɛ": 11, "ə": 11, "œ": 11, "ø": 11,
    # I — close front vowels
    "i":  12, "ɪ": 12, "y": 12,
    # O — mid/open back vowels
    "o":  13, "ɔ": 13,
    # U — close back vowels
    "u":  14, "ʊ": 14,
    # Approximants and laterals → nearest mouth shape
    "l":   8, "j": 12, "w": 14, "h": 0,
}

# Language codes for phonemizer backend
_LANG_TO_ESPEAK: Dict[str, str] = {
    "fr": "fr-fr",
    "en": "en-us",
    "ar": "ar",
    "es": "es",
    "de": "de",
    "it": "it",
    "pt": "pt",
}


def _text_to_phonemes(text: str, language: str) -> List[str]:
    """
    Convert text to a list of IPA phonemes using phonemizer.
    Falls back to an empty list if phonemizer isn't available.
    """
    try:
        from phonemizer import phonemize  # noqa: PLC0415
        from phonemizer.separator import Separator  # noqa: PLC0415

        espeak_lang = _LANG_TO_ESPEAK.get(language, "fr-fr")
        sep = Separator(phone=" ", word="  ", syllable="")

        result = phonemize(
            text,
            backend="espeak",
            language=espeak_lang,
            separator=sep,
            strip=True,
            preserve_punctuation=False,
        )
        # Split into individual phoneme tokens, filter empties
        return [p for p in result.split(" ") if p]

    except Exception as e:
        print(f"[tts] phonemizer error: {e}")
        return []


def _phonemes_to_visemes(
    phonemes: List[str],
    duration_sec: float,
) -> List[Dict]:
    """
    Map phonemes to viseme dicts with evenly-spaced timestamps.
    Each viseme gets equal time: duration / num_phonemes.
    """
    if not phonemes:
        return []

    step = duration_sec / len(phonemes)
    visemes = []
    for i, ph in enumerate(phonemes):
        vid = _PHONEME_TO_VISEME.get(ph, 0)
        visemes.append({
            "id":        vid,
            "timestamp": round(i * step, 3),
        })
    return visemes


# ── PCM → WAV bytes ───────────────────────────────────────────────────────────

def _pcm_to_wav(pcm_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy array to WAV bytes (int16 PCM)."""
    buf = io.BytesIO()
    samples = (pcm_array * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _silence_wav(duration_sec: float = 0.1) -> bytes:
    """Generate a short silence WAV as a fallback when TTS is unavailable."""
    samples = np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)
    return _pcm_to_wav(samples)


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(
    text: str,
    language: str = "",
) -> Tuple[bytes, List[Dict]]:
    """
    Synthesize speech for a sentence and return (wav_bytes, visemes).

    Parameters
    ----------
    text     : str   — the sentence to speak
    language : str   — ISO 639-1 code ("fr", "en", "ar"…)
                       Falls back to TTS_LANGUAGE env var.

    Returns
    -------
    wav_bytes : bytes     — 16-bit mono WAV at 24 kHz
    visemes   : list[dict] — [{id: int, timestamp: float}, …]
    """
    if not text.strip():
        return _silence_wav(), []

    lang = language or TTS_LANGUAGE
    tts  = _get_tts()

    # ── Synthesize ────────────────────────────────────────────────────────────
    if tts is None:
        # TTS not available — return silence so the streaming contract holds
        wav_bytes = _silence_wav(len(text) * 0.06)  # ~60ms per character
    else:
        try:
            # Use voice cloning if a reference WAV exists, otherwise use
            # the default XTTS speaker for the language.
            if os.path.isfile(TTS_SPEAKER_WAV):
                pcm = tts.tts(
                    text=text,
                    speaker_wav=TTS_SPEAKER_WAV,
                    language=lang,
                )
            else:
                # No reference voice — use built-in speaker
                speakers = tts.speakers or []
                speaker  = speakers[0] if speakers else None
                pcm = tts.tts(
                    text=text,
                    speaker=speaker,
                    language=lang,
                )

            wav_bytes = _pcm_to_wav(np.array(pcm, dtype=np.float32))

        except Exception as e:
            print(f"[tts] synthesis error: {e}")
            wav_bytes = _silence_wav()

    # ── Visemes ───────────────────────────────────────────────────────────────
    duration_sec = len(wav_bytes) / (SAMPLE_RATE * 2)  # 2 bytes per sample
    phonemes = _text_to_phonemes(text, lang)
    visemes  = _phonemes_to_visemes(phonemes, duration_sec)

    return wav_bytes, visemes