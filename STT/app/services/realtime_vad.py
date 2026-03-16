from __future__ import annotations

import logging
import queue
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generator, Optional

import numpy as np
import sounddevice as sd
import webrtcvad

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AudioEvent:
    event_type: str
    pcm_bytes: Optional[bytes] = None
    audio_seconds: float = 0.0
    is_speech: bool = False


class MicrophoneVADStreamer:
    """Low-level live microphone streamer with raw VAD metadata.

    It intentionally does *not* finalize a turn on its own. Instead, it emits
    speech_start plus speech_frame events containing the frame-level is_speech
    flag. A higher-level turn manager can then decide whether a pause is a real
    end of turn or just a natural thinking pause.
    """

    def __init__(self) -> None:
        self.sample_rate = settings.whisper_sample_rate
        self.channels = settings.whisper_channels
        self.vad = webrtcvad.Vad(settings.vad_mode)
        self.frame_ms = settings.audio_frame_ms
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.start_trigger_frames = max(1, settings.vad_start_trigger_ms // self.frame_ms)
        self.pre_roll_frames = max(0, settings.vad_pre_roll_ms // self.frame_ms)
        self._reset_requested = False

    def reset(self) -> None:
        """Signal the streamer to return to the idle (waiting-for-speech) state."""
        self._reset_requested = True

    def stream_events(self) -> Generator[AudioEvent, None, None]:
        audio_queue: queue.Queue[bytes] = queue.Queue()
        speech_started = False
        voiced_run = 0
        speech_frame_count = 0
        pre_roll: Deque[bytes] = deque(maxlen=self.pre_roll_frames)

        def callback(indata, frames, time, status) -> None:  # noqa: ANN001
            if status:
                logger.debug("Sounddevice status: %s", status)
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            audio_queue.put(pcm)

        logger.info("Streaming microphone started. Speak naturally.")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=callback,
        ):
            while True:
                frame = audio_queue.get()
                if len(frame) != self.frame_bytes:
                    continue

                # Check if the caller asked us to go back to idle.
                if self._reset_requested:
                    self._reset_requested = False
                    speech_started = False
                    voiced_run = 0
                    speech_frame_count = 0
                    pre_roll.clear()

                is_speech = self.vad.is_speech(frame, self.sample_rate)

                if not speech_started:
                    pre_roll.append(frame)
                    if is_speech:
                        voiced_run += 1
                        if voiced_run >= self.start_trigger_frames:
                            speech_started = True
                            speech_frame_count = 0
                            logger.info("Speech start detected")
                            yield AudioEvent(event_type="speech_start")
                            for buffered_frame in pre_roll:
                                speech_frame_count += 1
                                yield AudioEvent(
                                    event_type="speech_frame",
                                    pcm_bytes=buffered_frame,
                                    audio_seconds=(speech_frame_count * self.frame_ms) / 1000.0,
                                    is_speech=True,
                                )
                            pre_roll.clear()
                    else:
                        voiced_run = 0
                    continue

                speech_frame_count += 1
                yield AudioEvent(
                    event_type="speech_frame",
                    pcm_bytes=frame,
                    audio_seconds=(speech_frame_count * self.frame_ms) / 1000.0,
                    is_speech=is_speech,
                )