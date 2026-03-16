from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from faster_whisper import WhisperModel

from app.config import settings

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self) -> None:
        self.model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info(
            "STTService initialised | model=%s | device=%s | compute_type=%s",
            settings.whisper_model_size,
            settings.whisper_device,
            settings.whisper_compute_type,
        )

    def transcribe_bytes(self, audio_bytes: bytes, *, partial: bool = False) -> Dict[str, str]:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe_array(audio, partial=partial)

    def transcribe_array(self, audio: np.ndarray, *, partial: bool = False) -> Dict[str, str]:
        if audio.ndim != 1:
            audio = np.squeeze(audio)

        kwargs = dict(
            language=settings.whisper_language,
            task=settings.whisper_task,
            vad_filter=False,
            condition_on_previous_text=True,
        )
        if partial:
            kwargs.update(
                beam_size=1,
                best_of=1,
                temperature=0.0,
                word_timestamps=False,
                no_speech_threshold=0.7,
            )
        else:
            kwargs.update(
                beam_size=settings.whisper_beam_size,
                best_of=max(1, settings.whisper_beam_size),
                temperature=0.0,
                word_timestamps=False,
                no_speech_threshold=0.6,
            )

        segments, info = self.model.transcribe(audio, **kwargs)
        text_parts: List[str] = [segment.text.strip() for segment in segments if segment.text.strip()]
        text = " ".join(text_parts).strip()
        return {
            "text": text,
            "language": getattr(info, "language", settings.whisper_language) or settings.whisper_language,
        }
