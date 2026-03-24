from __future__ import annotations

import logging
import queue
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generator, Optional

import numpy as np
import sounddevice as sd

from app.config import settings

logger = logging.getLogger(__name__)

# ── Load Silero VAD (preferred) or fall back to webrtcvad ──
_silero_model = None
_use_silero = False
try:
    import torch
    from silero_vad import load_silero_vad

    _silero_model = load_silero_vad()
    _use_silero = True
    logger.info("Silero VAD loaded successfully (neural network VAD)")
except ImportError:
    logger.warning(
        "silero-vad or torch not installed — falling back to webrtcvad. "
        "Install with: pip install silero-vad torch"
    )
except Exception as exc:
    logger.warning("Failed to load Silero VAD (%s) — falling back to webrtcvad", exc)

if not _use_silero:
    import webrtcvad


@dataclass
class AudioEvent:
    event_type: str
    pcm_bytes: Optional[bytes] = None
    audio_seconds: float = 0.0
    is_speech: bool = False


class SileroVADWrapper:
    """Wraps Silero VAD for frame-by-frame speech detection.

    Silero VAD is a neural network trained on thousands of hours of audio
    to distinguish human speech from ALL types of noise — fans, AC, traffic,
    music, TV, dogs, construction, etc.  Unlike webrtcvad which only looks
    at energy patterns, Silero understands what speech actually sounds like.

    Key properties:
    - No calibration needed — works immediately in any environment
    - No noise floor estimation — the neural network handles it
    - No spectral heuristics — learned features are far more robust
    - Stateful (LSTM) — uses temporal context across frames
    - Requires exactly 512 samples (32ms) at 16kHz per frame

    This wrapper handles the frame size mismatch between the streamer's
    configurable frame size (e.g. 480 samples / 30ms) and Silero's
    fixed 512-sample requirement by buffering.
    """

    # Silero v6 requires exactly 512 samples at 16kHz
    SILERO_CHUNK_SAMPLES = 512
    SILERO_SAMPLE_RATE = 16000

    def __init__(self, sample_rate: int, frame_samples: int) -> None:
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._model = _silero_model

        # Speech probability thresholds
        self._start_threshold = settings.vad_silero_speech_threshold
        self._end_threshold = settings.vad_silero_silence_threshold

        # Buffer for repackaging frames to Silero's 512-sample chunks
        self._buffer = np.array([], dtype=np.float32)

        # Latest speech probability (updated each time Silero processes a chunk)
        self._last_prob: float = 0.0
        self._in_speech: bool = False

        logger.info(
            "SileroVADWrapper initialised | start_threshold=%.2f | "
            "end_threshold=%.2f | frame_samples=%d | silero_chunk=%d",
            self._start_threshold,
            self._end_threshold,
            self._frame_samples,
            self.SILERO_CHUNK_SAMPLES,
        )

    def is_speech(self, pcm_int16: bytes) -> bool:
        """Process a PCM int16 frame and return True if speech is detected."""
        # Convert int16 bytes to float32 [-1, 1] as Silero expects
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0

        # Append to buffer
        self._buffer = np.concatenate([self._buffer, samples])

        # Process complete 512-sample chunks
        while len(self._buffer) >= self.SILERO_CHUNK_SAMPLES:
            chunk = self._buffer[: self.SILERO_CHUNK_SAMPLES]
            self._buffer = self._buffer[self.SILERO_CHUNK_SAMPLES :]

            # Run Silero inference
            tensor = torch.from_numpy(chunk).unsqueeze(0)  # [1, 512]
            self._last_prob = self._model(tensor, self.SILERO_SAMPLE_RATE).item()

        # Hysteresis: stricter threshold to start, looser to continue
        if self._in_speech:
            self._in_speech = self._last_prob >= self._end_threshold
        else:
            self._in_speech = self._last_prob >= self._start_threshold

        return self._in_speech

    def reset(self) -> None:
        """Reset internal state (call between speech segments)."""
        self._buffer = np.array([], dtype=np.float32)
        self._last_prob = 0.0
        self._in_speech = False
        if self._model is not None:
            self._model.reset_states()

    @property
    def last_probability(self) -> float:
        """The speech probability from the most recent Silero inference."""
        return self._last_prob

    # ── No-op methods for API compatibility with noise gate ──
    def freeze_floor(self) -> None:
        pass  # Silero doesn't need noise floor management

    def unfreeze_floor(self) -> None:
        pass  # Silero doesn't need noise floor management

    @property
    def noise_floor_db(self) -> float:
        return 0.0  # Not applicable for Silero


class WebRTCVADWrapper:
    """Fallback wrapper around webrtcvad for when Silero is not available."""

    def __init__(self, sample_rate: int, frame_samples: int) -> None:
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._frame_bytes = frame_samples * 2
        self._vad = webrtcvad.Vad(settings.vad_mode)
        logger.info(
            "WebRTCVADWrapper initialised (fallback) | mode=%d",
            settings.vad_mode,
        )

    def is_speech(self, pcm_int16: bytes) -> bool:
        if len(pcm_int16) != self._frame_bytes:
            return False
        return self._vad.is_speech(pcm_int16, self._sample_rate)

    def reset(self) -> None:
        pass  # webrtcvad is stateless

    def freeze_floor(self) -> None:
        pass

    def unfreeze_floor(self) -> None:
        pass

    @property
    def noise_floor_db(self) -> float:
        return 0.0

    @property
    def last_probability(self) -> float:
        return 0.0


class MicrophoneVADStreamer:
    """Low-level live microphone streamer with VAD metadata.

    Uses Silero VAD (a neural network) when available, which correctly
    distinguishes human speech from fans, AC, traffic, music, and all
    other environmental noise without any calibration or tuning.

    Falls back to webrtcvad if Silero/torch are not installed.
    """

    def __init__(self) -> None:
        self.sample_rate = settings.whisper_sample_rate
        self.channels = settings.whisper_channels
        self.frame_ms = settings.audio_frame_ms
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.start_trigger_frames = max(1, settings.vad_start_trigger_ms // self.frame_ms)
        self.pre_roll_frames = max(0, settings.vad_pre_roll_ms // self.frame_ms)
        self._reset_requested = False

        # ── Pick the best available VAD ──
        if _use_silero:
            self.vad = SileroVADWrapper(
                sample_rate=self.sample_rate,
                frame_samples=self.frame_samples,
            )
        else:
            self.vad = WebRTCVADWrapper(
                sample_rate=self.sample_rate,
                frame_samples=self.frame_samples,
            )

        # Expose noise_gate as an alias for compatibility with
        # voice_chat_client.py which accesses streamer.noise_gate
        self.noise_gate = self.vad

        # ── Max continuous speech guard ──
        self._max_continuous_speech_frames = (
            settings.vad_max_continuous_speech_ms // self.frame_ms
        )

    def reset(self) -> None:
        """Signal the streamer to return to the idle (waiting-for-speech) state."""
        self._reset_requested = True

    def stream_events(self) -> Generator[AudioEvent, None, None]:
        audio_queue: queue.Queue[bytes] = queue.Queue()
        speech_started = False
        voiced_run = 0
        speech_frame_count = 0
        continuous_speech_frames = 0
        pre_roll: Deque[bytes] = deque(maxlen=self.pre_roll_frames)

        def callback(indata, frames, time, status) -> None:  # noqa: ANN001
            if status:
                logger.debug("Sounddevice status: %s", status)
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            audio_queue.put(pcm)

        vad_name = "Silero VAD (neural)" if _use_silero else "webrtcvad (fallback)"
        logger.info("Streaming microphone started | vad=%s", vad_name)

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
                    continuous_speech_frames = 0
                    pre_roll.clear()
                    self.vad.reset()

                is_speech = self.vad.is_speech(frame)

                # ── Max continuous speech guard ──
                if is_speech:
                    continuous_speech_frames += 1
                else:
                    continuous_speech_frames = 0

                if continuous_speech_frames >= self._max_continuous_speech_frames:
                    if speech_started:
                        logger.warning(
                            "Max continuous speech reached (%d ms) — "
                            "likely noise, forcing reset to idle",
                            continuous_speech_frames * self.frame_ms,
                        )
                    speech_started = False
                    voiced_run = 0
                    speech_frame_count = 0
                    continuous_speech_frames = 0
                    pre_roll.clear()
                    is_speech = False
                    self.vad.reset()

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
