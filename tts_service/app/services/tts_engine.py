"""
OpenVoice-lite TTS synthesis engine.

Primary backend: OpenVoice-compatible MeloTTS runtime (`melo.api.TTS`).
This is lightweight compared to larger conversational TTS stacks and supports
emotion expression through the existing prosody controller
(rate/pitch/energy/pause shaping).

Post-processing performs rate + pitch + energy adjustment via librosa (if
installed), or numpy-only fallback for rate/energy.

The engine is isolated from FastAPI — no imports from app.main.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import logging
import os
import tempfile
import time
import wave
import shutil
import warnings
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np

from app.config import settings
from app.services.prosody_controller import ResolvedProsody, split_into_sentences

logger = logging.getLogger(__name__)


def _configure_runtime_warning_filters() -> None:
    """Reduce third-party startup warning noise for known non-actionable cases."""
    # Runtime env knobs for noisy third-party libraries.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`resume_download` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are using a Python version .* google\.api_core.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`huggingface_hub` cache-system uses symlinks by default.*",
        category=UserWarning,
    )

    # Quiet noisy transformer logger warnings (e.g., unused checkpoint weights).
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _ensure_nltk_resources() -> None:
    """
    Ensure NLTK assets required by MeloTTS English text pipeline are present.
    """
    try:
        import nltk  # type: ignore

        required = [
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("tokenizers/punkt", "punkt"),
        ]

        for nltk_path, package_name in required:
            try:
                nltk.data.find(nltk_path)
            except LookupError:
                logger.info("Downloading missing NLTK resource for TTS | %s", package_name)
                nltk.download(package_name, quiet=True)
    except Exception as exc:
        logger.warning("Could not verify/download NLTK resources | %s", exc)


# Apply warning filters as early as possible (before heavy optional imports)
_configure_runtime_warning_filters()

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

    OpenVoice/Melo handles base synthesis; these are lightweight emotional
    fine-tuning adjustments on top.
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

    # Energy shaping (neutral baseline = 0.70 from prosody controller)
    energy_gain = float(np.clip(prosody.energy_level / 0.70, 0.75, 1.35))
    audio = np.clip(audio * energy_gain, -1.0, 1.0)

    return audio


# ── Cache key ─────────────────────────────────────────────────────────────────

def make_cache_key(text: str, state: str, trend: str, voice: str) -> str:
    """Deterministic Redis key for a (text, emotion_state, trend, voice) tuple."""
    payload = f"{text}|{state}|{trend}|{voice}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"tts:cache:{digest}"


# ── Engine ────────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    OpenVoice-lite (MeloTTS) engine.

    Backend priority:
      1. openvoice  — MeloTTS runtime used by OpenVoice ecosystem
      2. none       — runtime unavailable; RuntimeError on synthesize()
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
        _configure_runtime_warning_filters()
        _ensure_nltk_resources()

        # MeloTTS imports Japanese text modules at import time and may require
        # a valid MeCab rc path even for English usage. Auto-point to
        # unidic_lite's mecabrc if available.
        try:
            if "MECABRC" not in os.environ or not os.path.exists(os.environ.get("MECABRC", "")):
                import unidic_lite  # type: ignore

                dicdir = Path(unidic_lite.DICDIR)
                mecabrc = dicdir / "mecabrc"
                if mecabrc.exists():
                    os.environ["MECABRC"] = str(mecabrc)
                    os.environ["MECAB_ARGS"] = f'-r "{mecabrc}" -d "{dicdir}"'
                    os.environ.setdefault("UNIDICDIR", str(dicdir))
                    logger.info("Configured MECABRC for MeloTTS | %s", mecabrc)

                    # Some Melo/MeCab builds still resolve to unidic.DICDIR.
                    # If that path exists but is empty/missing mecabrc,
                    # mirror unidic_lite dictionary there.
                    try:
                        import unidic  # type: ignore

                        target_dicdir = Path(getattr(unidic, "DICDIR", ""))
                        target_mecabrc = target_dicdir / "mecabrc"
                        if target_dicdir and (not target_mecabrc.exists()):
                            target_dicdir.mkdir(parents=True, exist_ok=True)
                            for item in dicdir.iterdir():
                                dst = target_dicdir / item.name
                                if dst.exists():
                                    continue
                                if item.is_dir():
                                    shutil.copytree(item, dst)
                                else:
                                    shutil.copy2(item, dst)
                            logger.info("Mirrored unidic_lite dictionary to %s", target_dicdir)
                    except Exception as mirror_exc:
                        logger.warning("Could not mirror unidic_lite -> unidic path | %s", mirror_exc)
        except Exception as exc:
            logger.warning("Could not auto-configure MECABRC | %s", exc)

        # ── Primary: OpenVoice-compatible MeloTTS runtime ─────────
        try:
            MeloTTS = None
            import_errors: list[str] = []

            # Canonical import path used by current MeloTTS builds
            try:
                from melo.api import TTS as MeloTTS  # type: ignore
            except Exception as exc:
                import_errors.append(f"melo.api: {exc}")

            # Some installations expose a different top-level package name
            if MeloTTS is None:
                try:
                    m = importlib.import_module("MeloTTS.melo.api")
                    MeloTTS = getattr(m, "TTS")
                except Exception as exc:
                    import_errors.append(f"MeloTTS.melo.api: {exc}")

            if MeloTTS is None:
                raise ImportError(
                    "Could not import MeloTTS runtime. "
                    "Install in the same Python environment that runs uvicorn: "
                    "pip install -r requirements.txt. "
                    f"Tried imports -> {' | '.join(import_errors)}"
                )

            language = os.getenv("TTS_OPENVOICE_LANGUAGE", "EN")
            model = MeloTTS(language=language, device=device)
            self._model = model
            self._backend = "openvoice"
            self._sample_rate = 24000
            logger.info(
                "OpenVoice (MeloTTS) loaded | language=%s | device=%s | sr=%d | elapsed=%.1fs",
                language,
                device,
                self._sample_rate,
                time.monotonic() - t0,
            )
        except Exception as exc:
            logger.error("OpenVoice (MeloTTS) load failed | error=%s", exc)
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

        if self._backend == "openvoice":
            audio = self._synthesize_openvoice(text, prosody, voice)
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

    def _synthesize_openvoice(
        self,
        text: str,
        prosody: ResolvedProsody,
        voice: str,
    ) -> np.ndarray:
        # OpenVoice's lightweight base TTS stack exposes speaker IDs via
        # MeloTTS' hps.data.spk2id map.
        speaker_map = getattr(getattr(self._model, "hps", None), "data", None)
        spk2id = getattr(speaker_map, "spk2id", {}) if speaker_map is not None else {}
        if not isinstance(spk2id, dict):
            spk2id = {}

        _VOICE_MAP = {
            "default": "EN-Default",
            "serena": "EN-Default",
            "aiden": "EN-US",
            "vivian": "EN-BR",
        }
        desired = _VOICE_MAP.get(voice, voice)
        if desired in spk2id:
            speaker_id = spk2id[desired]
        elif spk2id:
            # Best-effort fallback to first available speaker
            speaker_id = next(iter(spk2id.values()))
        else:
            # Some MeloTTS builds don't expose spk2id for single-speaker EN.
            # Use canonical default speaker id.
            speaker_id = 0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            self._model.tts_to_file(
                text=text,
                speaker_id=speaker_id,
                output_path=out_path,
                speed=prosody.rate_multiplier,
            )

            if _SF_AVAILABLE:
                audio, sr = sf.read(out_path, dtype="float32")
            else:
                with wave.open(out_path, "rb") as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = audio.mean(axis=1)

            if int(sr) != self._sample_rate and _LIBROSA_AVAILABLE:
                audio = librosa.resample(audio, orig_sr=int(sr), target_sr=self._sample_rate)
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass

        audio = np.asarray(audio, dtype=np.float32).squeeze()
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        return audio


# ── Backward compat alias ────────────────────────────────────────────────────
# main.py previously imported QwenTTSEngine
QwenTTSEngine = TTSEngine