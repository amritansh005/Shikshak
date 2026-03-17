"""
Chatterbox TTS synthesis engine.

Primary backend : Chatterbox-TTS (0.5B, MIT license) via `chatterbox-tts` pip package.
                  Supports emotion exaggeration control, cfg_weight for pacing,
                  and paralinguistic tags like [laugh], [cough].

Fallback backend: Kokoro-82M  (CPU-friendly, Apache 2.0, ~500 MB RAM).
                  Auto-activated if chatterbox-tts fails to load (no GPU, missing pkg).

Prosody is mapped from emotion states to Chatterbox parameters:
  - exaggeration (0.0-1.5): emotion intensity
  - cfg_weight   (0.0-1.0): speech pacing (lower=faster, higher=slower)
  - temperature  (0.05-5.0): variation/creativity

Post-processing rate/pitch adjustment via librosa (if installed)
or a pure-numpy linear-resample fallback for rate only.

The engine is isolated from FastAPI — no imports from app.main.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import time
import wave
from typing import AsyncGenerator, Optional

import numpy as np

from app.config import settings
from app.services.prosody_controller import ResolvedProsody, split_into_sentences

logger = logging.getLogger(__name__)

# ── Optional post-processing deps ────────────────────────────────────────────
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.info("librosa not available — numpy rate control will be used")

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False


# ── Audio utilities ───────────────────────────────────────────────────────────

def _float32_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 numpy array -> WAV bytes (mono, int16)."""
    pcm = _float32_to_int16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _apply_rate_change_numpy(audio: np.ndarray, rate: float) -> np.ndarray:
    """Linear-resample based rate change (no librosa needed)."""
    if abs(rate - 1.0) < 0.02:
        return audio
    n_original = len(audio)
    n_new = int(n_original / rate)
    if n_new < 1:
        return audio
    indices = np.linspace(0, n_original - 1, n_new)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.clip(idx_floor + 1, 0, n_original - 1)
    frac = indices - idx_floor
    return audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac


def _apply_prosody_postprocess(
    audio: np.ndarray,
    sample_rate: int,
    prosody: ResolvedProsody,
) -> np.ndarray:
    """
    Post-synthesis rate + pitch adjustment.

    Chatterbox handles most prosody via exaggeration/cfg_weight, so these
    adjustments are a fine-tuning layer on top.
    """
    rate = prosody.rate_multiplier
    pitch_st = prosody.pitch_shift_st

    if _LIBROSA_AVAILABLE:
        if abs(rate - 1.0) > 0.02:
            audio = librosa.effects.time_stretch(audio, rate=rate)
        if abs(pitch_st) > 0.1:
            audio = librosa.effects.pitch_shift(
                audio, sr=sample_rate, n_steps=pitch_st
            )
    else:
        audio = _apply_rate_change_numpy(audio, rate)
        if abs(pitch_st) > 0.5:
            logger.debug(
                "Pitch shift (%.1f st) skipped — install librosa for pitch control",
                pitch_st,
            )

    return audio


# ── Cache key ─────────────────────────────────────────────────────────────────

def make_cache_key(text: str, state: str, trend: str, voice: str) -> str:
    """Deterministic Redis key for a (text, emotion_state, trend, voice) tuple."""
    payload = f"{text}|{state}|{trend}|{voice}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"tts:cache:{digest}"


# ── Emotion -> Chatterbox parameter mapping ──────────────────────────────────
#
# exaggeration: 0.3=neutral/professional, 0.5=balanced, 0.7+=expressive, 1.0+=dramatic
# cfg_weight:   0.2-0.3=faster, 0.5=balanced, 0.7-0.8=slower/deliberate
# temperature:  0.4-0.6=consistent, 0.8=balanced, 1.0+=creative

_STATE_CHATTERBOX_PARAMS: dict[str, dict[str, float]] = {
    "neutral":      {"exaggeration": 0.45, "cfg_weight": 0.50, "temperature": 0.7},
    "frustrated":   {"exaggeration": 0.35, "cfg_weight": 0.65, "temperature": 0.6},
    "confused":     {"exaggeration": 0.40, "cfg_weight": 0.60, "temperature": 0.6},
    "anxious":      {"exaggeration": 0.35, "cfg_weight": 0.60, "temperature": 0.5},
    "discouraged":  {"exaggeration": 0.50, "cfg_weight": 0.55, "temperature": 0.6},
    "uncertain":    {"exaggeration": 0.40, "cfg_weight": 0.55, "temperature": 0.6},
    "bored":        {"exaggeration": 0.70, "cfg_weight": 0.35, "temperature": 0.8},
    "confident":    {"exaggeration": 0.55, "cfg_weight": 0.45, "temperature": 0.7},
    "engaged":      {"exaggeration": 0.65, "cfg_weight": 0.45, "temperature": 0.8},
    "curious":      {"exaggeration": 0.60, "cfg_weight": 0.40, "temperature": 0.8},
}

_TREND_EXAGG_DELTA: dict[str, float] = {
    "escalating": -0.05,
    "de-escalating": +0.05,
    "recovering": +0.03,
    "stable": 0.0,
}

_TREND_CFG_DELTA: dict[str, float] = {
    "escalating": +0.05,
    "de-escalating": -0.03,
    "recovering": -0.02,
    "stable": 0.0,
}


def _resolve_chatterbox_params(prosody: ResolvedProsody) -> dict[str, float]:
    """Map resolved emotion state -> Chatterbox generate() kwargs."""
    base = _STATE_CHATTERBOX_PARAMS.get(
        prosody.resolved_state,
        _STATE_CHATTERBOX_PARAMS["neutral"],
    )
    exagg_delta = _TREND_EXAGG_DELTA.get(prosody.resolved_trend, 0.0)
    cfg_delta = _TREND_CFG_DELTA.get(prosody.resolved_trend, 0.0)

    conf = prosody.smoothed_confidence
    neutral = _STATE_CHATTERBOX_PARAMS["neutral"]

    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * max(0.0, min(1.0, t))

    return {
        "exaggeration": max(0.25, min(1.5,
            _lerp(neutral["exaggeration"], base["exaggeration"], conf) + exagg_delta
        )),
        "cfg_weight": max(0.0, min(1.0,
            _lerp(neutral["cfg_weight"], base["cfg_weight"], conf) + cfg_delta
        )),
        "temperature": max(0.05, min(2.0,
            _lerp(neutral["temperature"], base["temperature"], conf)
        )),
    }


# ── Engine ────────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Chatterbox TTS engine with Kokoro fallback.

    Backend priority:
      1. chatterbox — Chatterbox-TTS (0.5B, MIT, CUDA)
      2. kokoro     — Kokoro-82M (CPU-viable, Apache 2.0)
      3. none       — both failed; RuntimeError on synthesize()
    """

    def __init__(self) -> None:
        self._model = None
        self._backend: str = "none"
        self._sample_rate: int = settings.sample_rate
        self._load()

    def _load(self) -> None:
        device = settings.tts_device
        logger.info("Loading TTS model | device=%s", device)
        t0 = time.monotonic()

        # ── Primary: Kokoro-82M on GPU ────────────────────────────
        try:
            from kokoro import KPipeline, KModel
            model = KModel().to(device).eval()
            self._model = KPipeline(lang_code="a", model=model, device=device)
            self._backend = "kokoro"
            self._sample_rate = 24000
            logger.info(
                "Kokoro-82M loaded | device=%s | sr=%d | elapsed=%.1fs",
                device, self._sample_rate, time.monotonic() - t0,
            )
        except Exception as exc:
            logger.error("Kokoro-82M load failed | error=%s", exc)
            self._backend = "none"

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        prosody: ResolvedProsody,
        voice: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        if self._backend == "none":
            raise RuntimeError("No TTS backend loaded — check startup logs.")

        voice = voice or settings.default_voice

        if self._backend == "kokoro":
            audio = self._synthesize_kokoro(text, prosody, voice)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        audio = _apply_prosody_postprocess(audio, self._sample_rate, prosody)
        return audio, self._sample_rate

    async def synthesize_stream(
        self,
        text: str,
        prosody: ResolvedProsody,
        voice: Optional[str] = None,
        chunk_bytes: int = 4096,
    ) -> AsyncGenerator[bytes, None]:
        voice = voice or settings.default_voice
        sentences = split_into_sentences(text)
        if not sentences:
            return

        loop = asyncio.get_event_loop()

        for sentence in sentences:
            if not sentence.strip():
                continue

            audio, sr = await loop.run_in_executor(
                None, self.synthesize, sentence, prosody, voice
            )

            pcm = _float32_to_int16(audio)
            raw = pcm.tobytes()

            for offset in range(0, len(raw), chunk_bytes):
                yield raw[offset: offset + chunk_bytes]

            pause_samples = int(sr * prosody.pause_after_sentence_ms / 1000)
            if pause_samples > 0:
                yield np.zeros(pause_samples, dtype=np.int16).tobytes()

    def wav_bytes(
        self,
        text: str,
        prosody: ResolvedProsody,
        voice: Optional[str] = None,
    ) -> bytes:
        audio, sr = self.synthesize(text, prosody, voice)
        return _audio_to_wav_bytes(audio, sr)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # ── Backend implementations ───────────────────────────────────────────────

    def _synthesize_chatterbox_turbo(
        self,
        text: str,
        prosody: ResolvedProsody,
        voice: str,
    ) -> np.ndarray:
        """
        Synthesize using Chatterbox-TTS.

        Emotion is controlled via exaggeration/cfg_weight/temperature
        mapped from the student's emotional state.

        The `voice` param can be a path to a .wav reference file for
        voice cloning, or a name that maps to voices/<name>.wav.
        """
        _STATE_TEMPERATURE = {
            "neutral": 0.8, "frustrated": 0.6, "confused": 0.6,
            "anxious": 0.5, "discouraged": 0.7, "uncertain": 0.6,
            "bored": 0.9, "confident": 0.8, "engaged": 0.9, "curious": 0.9,
        }
        temperature = _STATE_TEMPERATURE.get(prosody.resolved_state, 0.8)

        logger.debug(
            "Chatterbox-Turbo params | state=%s | temperature=%.2f",
            prosody.resolved_state, temperature,
        )

        # Resolve voice reference audio (optional)
        audio_prompt_path = None
        if voice and os.path.isfile(voice):
            audio_prompt_path = voice
        elif voice and os.path.isfile(f"voices/{voice}.wav"):
            audio_prompt_path = f"voices/{voice}.wav"

        generate_kwargs = {
            "text": text,
            "temperature": temperature,
        }
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

        wav_tensor = self._model.generate(**generate_kwargs)

        # wav_tensor is a torch tensor [1, N] or [N]
        if hasattr(wav_tensor, "numpy"):
            audio = wav_tensor.squeeze().cpu().numpy()
        else:
            audio = np.array(wav_tensor, dtype=np.float32).squeeze()

        audio = audio.astype(np.float32)

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        return audio

    def _synthesize_kokoro(
        self,
        text: str,
        prosody: ResolvedProsody,
        voice: str,
    ) -> np.ndarray:
        _VOICE_MAP = {
            "default":   "af_heart",
            "serena":    "af_heart",
            "vivian":    "af_sky",
            "aiden":     "am_adam",
            "dylan":     "am_michael",
            "eric":      "am_echo",
            "ryan":      "am_onyx",
        }
        kokoro_voice = _VOICE_MAP.get(voice, "af_heart")

        chunks = []
        for _, _, audio_chunk in self._model(
            text,
            voice=kokoro_voice,
            speed=prosody.rate_multiplier,
            split_pattern=r"\n+",
        ):
            if audio_chunk is not None:
                chunks.append(audio_chunk)

        if not chunks:
            return np.zeros(self._sample_rate, dtype=np.float32)

        audio = np.concatenate(chunks).astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        return audio


# ── Backward compat alias ────────────────────────────────────────────────────
# main.py previously imported QwenTTSEngine
QwenTTSEngine = TTSEngine