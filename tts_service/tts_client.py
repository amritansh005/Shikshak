from __future__ import annotations

import io
import logging
import queue
import threading
import time
import wave
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    import numpy as np
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    np = None  # type: ignore

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


class TTSClient:
    """
    Queue-based TTS client with:
    - cancellable playback
    - basic duck/restore support
    - current_text tracking
    - playback state tracking
    - playback_started_at timestamp for no-barge-in phase
    """

    def __init__(
        self,
        tts_service_url: str = "http://127.0.0.1:5000",
        voice: Optional[str] = None,
        timeout: int = 30,
        fallback_enabled: bool = True,
    ) -> None:
        self._url = tts_service_url.rstrip("/")
        self._voice = voice
        self._timeout = timeout
        self._fallback = fallback_enabled
        self._mute = False
        self._available: Optional[bool] = None

        self._play_queue: queue.Queue[Tuple[str, Optional[Dict], Optional[str]]] = queue.Queue(maxsize=4)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()

        self._state_lock = threading.Lock()
        self._is_playing: bool = False
        self.current_text: str = ""
        self._volume_scale: float = 1.0
        self._stop_event = threading.Event()
        # NEW: timestamp of when current playback started (monotonic clock).
        # Used by voice_chat_client to implement the no-barge-in phase.
        self._playback_started_at: float = 0.0

        logger.info("TTSClient initialised | url=%s", self._url)

    @property
    def mute(self) -> bool:
        return self._mute

    @mute.setter
    def mute(self, value: bool) -> None:
        self._mute = value

    @property
    def is_playing(self) -> bool:
        with self._state_lock:
            return self._is_playing

    @property
    def playback_started_at(self) -> float:
        """Monotonic timestamp of when current playback began, or 0.0 if idle."""
        with self._state_lock:
            return self._playback_started_at

    def stop_playback(self) -> None:
        with self._state_lock:
            self._stop_event.set()
        try:
            if _SD_AVAILABLE:
                sd.stop()
        except Exception as exc:
            logger.debug("sd.stop failed | %s", exc)

    def duck_playback(self, level: float = 0.22) -> None:
        with self._state_lock:
            self._volume_scale = max(0.05, min(level, 1.0))

    def restore_playback(self) -> None:
        with self._state_lock:
            self._volume_scale = 1.0

    def speak_with_emotion(
        self,
        text: str,
        emotion_data: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if self._mute or not text.strip():
            return
        self._enqueue(text, emotion_data, session_id)

    def speak_neutral(self, text: str, session_id: Optional[str] = None) -> None:
        if self._mute or not text.strip():
            return
        self._enqueue(text, None, session_id)

    def is_available(self) -> bool:
        if self._available is None:
            try:
                r = requests.get(f"{self._url}/health", timeout=2)
                self._available = r.status_code == 200
            except Exception:
                self._available = False
        return self._available

    def invalidate_availability_cache(self) -> None:
        self._available = None

    def _enqueue(
        self,
        text: str,
        emotion_payload: Optional[Dict],
        session_id: Optional[str],
    ) -> None:
        item = (text, emotion_payload, session_id)

        try:
            self._play_queue.put_nowait(item)
        except queue.Full:
            try:
                dropped = self._play_queue.get_nowait()
                logger.warning(
                    "TTSClient play queue full — dropped oldest item | text=%r",
                    dropped[0][:80] if dropped else "?",
                )
            except queue.Empty:
                pass
            self._play_queue.put_nowait(item)

    def _play_worker(self) -> None:
        while True:
            try:
                text, emotion_payload, session_id = self._play_queue.get()
                self._synthesize_and_play(text, emotion_payload, session_id)
            except Exception as exc:
                logger.warning("TTSClient play worker error | %s", exc)

    def _set_playback_state(self, playing: bool, text: str = "") -> None:
        with self._state_lock:
            self._is_playing = playing
            self.current_text = text
            if playing:
                self._stop_event.clear()
                # NEW: record when playback started so the STT side can
                # implement a no-barge-in phase.
                self._playback_started_at = time.monotonic()
            else:
                self._volume_scale = 1.0
                self._playback_started_at = 0.0

    def _synthesize_and_play(
        self,
        text: str,
        emotion_payload: Optional[Dict],
        session_id: Optional[str],
    ) -> None:
        if not self.is_available():
            self.invalidate_availability_cache()
            if not self.is_available():
                if self._fallback:
                    self._fallback_speak(text)
                return

        try:
            payload: Dict = {"text": text, "use_cache": True}
            if self._voice:
                payload["voice"] = self._voice
            if emotion_payload:
                payload["emotion"] = emotion_payload
            if session_id:
                payload["session_id"] = session_id

            resp = requests.post(
                f"{self._url}/synthesize",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()

            latency = resp.headers.get("X-TTS-Latency-Ms", "?")
            state = resp.headers.get("X-TTS-Resolved-State", "?")
            cache_hit = resp.headers.get("X-TTS-Cache-Hit", "false") == "true"
            logger.info(
                "TTS | latency=%sms | state=%s | cache=%s | chars=%d",
                latency, state, cache_hit, len(text),
            )

            self._set_playback_state(True, text)
            try:
                self._play_wav_bytes_streaming(resp.content)
            finally:
                self._set_playback_state(False, "")

            self._available = True

        except requests.exceptions.ConnectionError:
            logger.warning("TTSClient | service unreachable — marking unavailable")
            self._available = False
            self._set_playback_state(False, "")
            if self._fallback:
                self._fallback_speak(text)
        except Exception as exc:
            logger.warning("TTSClient synthesis error | %s", exc)
            self._set_playback_state(False, "")
            if self._fallback:
                self._fallback_speak(text)

    def _fallback_speak(self, text: str) -> None:
        if not _PYTTSX3_AVAILABLE:
            return
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            logger.debug("pyttsx3 fallback failed | %s", exc)

    def _play_wav_bytes_streaming(self, wav_bytes: bytes) -> None:
        if not _SD_AVAILABLE:
            logger.debug("sounddevice not available — audio not played")
            return

        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

            if sampwidth != 2:
                raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            if n_channels == 2:
                audio = audio.reshape(-1, 2)

            chunk_size = 2048
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=n_channels,
                dtype="float32",
                blocksize=chunk_size,
            )

            with stream:
                total = len(audio)
                idx = 0

                while idx < total:
                    with self._state_lock:
                        if self._stop_event.is_set():
                            logger.info("TTS playback stopped early by interruption")
                            break
                        volume = self._volume_scale

                    chunk = audio[idx: idx + chunk_size]
                    if volume != 1.0:
                        chunk = chunk * volume

                    stream.write(chunk)
                    idx += chunk_size

        except Exception as exc:
            logger.warning("Audio playback failed | %s", exc)