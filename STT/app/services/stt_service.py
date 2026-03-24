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

    def transcribe_bytes(self, audio_bytes: bytes, *, partial: bool = False, context_prompt: str = "") -> Dict[str, object]:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe_array(audio, partial=partial, context_prompt=context_prompt)

    def transcribe_array(self, audio: np.ndarray, *, partial: bool = False, context_prompt: str = "") -> Dict[str, object]:
        if audio.ndim != 1:
            audio = np.squeeze(audio)

        kwargs = dict(
            language=settings.whisper_language,
            task=settings.whisper_task,
            vad_filter=False,
            condition_on_previous_text=False,
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
                vad_filter=settings.whisper_vad_filter_final,
            )

        # Dynamic context_prompt (from conversation) takes priority
        # over the static .env base prompt.
        prompt = context_prompt or settings.whisper_initial_prompt
        if prompt:
            kwargs["initial_prompt"] = prompt

        segments, info = self.model.transcribe(audio, **kwargs)

        text_parts: List[str] = []
        no_speech_probs: List[float] = []

        for segment in segments:
            stripped = segment.text.strip()
            if stripped:
                text_parts.append(stripped)
                no_speech_probs.append(segment.no_speech_prob)

        text = " ".join(text_parts).strip()

        avg_no_speech_prob = (
            sum(no_speech_probs) / len(no_speech_probs)
            if no_speech_probs
            else 1.0  # no segments = treat as no speech
        )

        return {
            "text": text,
            "language": getattr(info, "language", settings.whisper_language) or settings.whisper_language,
            "avg_no_speech_prob": avg_no_speech_prob,
        }