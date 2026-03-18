"""
Speaker Verification Service
─────────────────────────────
Uses Resemblyzer (GE2E speaker encoder) to create a voice fingerprint
of the student at startup and verify that subsequent speech belongs to
the same person.

NOTE: Resemblyzer internally imports preprocess_wav which pulls in
librosa → lzma.  On some Windows Python installs the _lzma DLL is
missing.  We patch the import system to provide a stub _lzma module
before importing Resemblyzer, then bypass preprocess_wav entirely
by passing our 16kHz float32 audio directly to the encoder.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Patch _lzma before Resemblyzer import ────────────────────────────
# Resemblyzer → librosa → soundfile → ... → _lzma.  If the DLL is
# missing on Windows, the import chain fails.  We inject a dummy
# _lzma module so the import succeeds; we never actually use lzma
# compression anywhere in our pipeline.
if "_lzma" not in sys.modules:
    try:
        import _lzma  # noqa: F401 — test if it works
    except ImportError:
        _dummy = types.ModuleType("_lzma")
        _dummy.LZMACompressor = None  # type: ignore
        _dummy.LZMADecompressor = None  # type: ignore
        _dummy.LZMAError = type("LZMAError", (Exception,), {})  # type: ignore
        _dummy.FORMAT_AUTO = 0  # type: ignore
        _dummy.FORMAT_XZ = 1  # type: ignore
        _dummy.FORMAT_ALONE = 2  # type: ignore
        _dummy.FORMAT_RAW = 3  # type: ignore
        _dummy.CHECK_NONE = 0  # type: ignore
        _dummy.CHECK_CRC32 = 1  # type: ignore
        _dummy.CHECK_CRC64 = 4  # type: ignore
        _dummy.CHECK_SHA256 = 10  # type: ignore
        _dummy.CHECK_ID_MAX = 15  # type: ignore
        _dummy.CHECK_UNKNOWN = 16  # type: ignore
        _dummy.MF_HC3 = 0x03  # type: ignore
        _dummy.MF_HC4 = 0x04  # type: ignore
        _dummy.MF_BT2 = 0x12  # type: ignore
        _dummy.MF_BT3 = 0x13  # type: ignore
        _dummy.MF_BT4 = 0x14  # type: ignore
        _dummy.MODE_FAST = 1  # type: ignore
        _dummy.MODE_NORMAL = 2  # type: ignore
        _dummy.PRESET_DEFAULT = 6  # type: ignore
        _dummy.PRESET_EXTREME = 0  # type: ignore
        _dummy.is_check_supported = lambda check_id: False  # type: ignore
        # Functions that Python's lzma module tries to import from _lzma:
        _dummy._encode_filter_properties = lambda filter: b""  # type: ignore
        _dummy._decode_filter_properties = lambda filter_id, encoded_props: {}  # type: ignore
        sys.modules["_lzma"] = _dummy
        logger.info("Injected stub _lzma module (Windows DLL missing — not needed for speaker verification)")

try:
    from resemblyzer import VoiceEncoder
    _RESEMBLYZER_AVAILABLE = True
except ImportError:
    _RESEMBLYZER_AVAILABLE = False
    logger.warning(
        "resemblyzer not installed — speaker verification disabled. "
        "Install with: pip install resemblyzer"
    )


class SpeakerVerificationService:
    """
    Enroll a student's voice and verify subsequent audio against it.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        min_enrollment_seconds: float = 2.0,
        auto_update_profile: bool = True,
        update_weight: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self._threshold = similarity_threshold
        self._min_enrollment_sec = min_enrollment_seconds
        self._auto_update = auto_update_profile
        self._update_weight = update_weight
        self._lock = threading.Lock()

        self._enrolled_embedding: Optional[np.ndarray] = None
        self._enrollment_count: int = 0

        if _RESEMBLYZER_AVAILABLE:
            logger.info("Loading Resemblyzer speaker encoder | device=%s", device)
            t0 = time.time()
            self._encoder = VoiceEncoder(device=device)
            logger.info(
                "Resemblyzer speaker encoder loaded | elapsed=%.1fs",
                time.time() - t0,
            )
        else:
            self._encoder = None

    @property
    def is_available(self) -> bool:
        return self._encoder is not None

    @property
    def is_enrolled(self) -> bool:
        with self._lock:
            return self._enrolled_embedding is not None

    def enroll(
        self,
        audio_int16: bytes,
        sample_rate: int = 16000,
    ) -> bool:
        if not self.is_available:
            return False

        wav = self._bytes_to_wav(audio_int16, sample_rate)
        if wav is None or len(wav) / sample_rate < self._min_enrollment_sec:
            logger.info(
                "Enrollment audio too short | duration=%.1fs | need=%.1fs",
                len(wav) / sample_rate if wav is not None else 0.0,
                self._min_enrollment_sec,
            )
            return False

        try:
            embedding = self._encoder.embed_utterance(wav)

            with self._lock:
                if self._enrolled_embedding is None:
                    self._enrolled_embedding = embedding
                    self._enrollment_count = 1
                else:
                    self._enrollment_count += 1
                    weight = 1.0 / self._enrollment_count
                    self._enrolled_embedding = (
                        (1 - weight) * self._enrolled_embedding
                        + weight * embedding
                    )
                    norm = np.linalg.norm(self._enrolled_embedding)
                    if norm > 0:
                        self._enrolled_embedding /= norm

            logger.info(
                "Speaker enrolled | enrollment_count=%d | embedding_norm=%.3f",
                self._enrollment_count,
                np.linalg.norm(embedding),
            )
            return True

        except Exception as exc:
            logger.warning("Speaker enrollment failed | error=%s", exc)
            return False

    def verify(
        self,
        audio_int16: bytes,
        sample_rate: int = 16000,
    ) -> dict:
        if not self.is_available or not self.is_enrolled:
            return {
                "is_student": True,
                "similarity": 1.0,
                "threshold": self._threshold,
                "latency_ms": 0.0,
            }

        t0 = time.monotonic()

        try:
            wav = self._bytes_to_wav(audio_int16, sample_rate)
            if wav is None or len(wav) / sample_rate < 0.5:
                return {
                    "is_student": True,
                    "similarity": 1.0,
                    "threshold": self._threshold,
                    "latency_ms": (time.monotonic() - t0) * 1000,
                }

            embedding = self._encoder.embed_utterance(wav)

            with self._lock:
                enrolled = self._enrolled_embedding

            similarity = float(np.dot(enrolled, embedding))
            is_student = similarity >= self._threshold
            latency_ms = (time.monotonic() - t0) * 1000

            if is_student and self._auto_update:
                with self._lock:
                    self._enrolled_embedding = (
                        (1 - self._update_weight) * self._enrolled_embedding
                        + self._update_weight * embedding
                    )
                    norm = np.linalg.norm(self._enrolled_embedding)
                    if norm > 0:
                        self._enrolled_embedding /= norm

            logger.info(
                "Speaker verification | is_student=%s | similarity=%.3f | threshold=%.2f | latency=%.0fms",
                is_student, similarity, self._threshold, latency_ms,
            )

            return {
                "is_student": is_student,
                "similarity": similarity,
                "threshold": self._threshold,
                "latency_ms": latency_ms,
            }

        except Exception as exc:
            logger.warning("Speaker verification failed | error=%s", exc)
            return {
                "is_student": True,
                "similarity": 1.0,
                "threshold": self._threshold,
                "latency_ms": (time.monotonic() - t0) * 1000,
            }

    @staticmethod
    def _bytes_to_wav(
        audio_int16: bytes,
        sample_rate: int,
    ) -> Optional[np.ndarray]:
        try:
            audio = np.frombuffer(audio_int16, dtype=np.int16)
            return audio.astype(np.float32) / 32768.0
        except Exception:
            return None