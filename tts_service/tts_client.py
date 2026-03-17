from __future__ import annotations

import io
import logging
import queue
import threading
import wave
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    import numpy as np
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


class TTSClient:
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

        self._play_queue: queue.Queue = queue.Queue(maxsize=4)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()

        logger.info("TTSClient initialised | url=%s", self._url)

    @property
    def mute(self) -> bool:
        return self._mute

    @mute.setter
    def mute(self, value: bool) -> None:
        self._mute = value

    def speak_with_emotion(
        self,
        text: str,
        emotion_data: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Speak using the directive dict returned by techer_llm /chat.

        emotion_data keys (from directive_dict in main.py):
            smoothed_state, smoothed_confidence, trend,
            secondary_state, secondary_confidence,
            raw_text_label, raw_audio_label

        This is passed directly as EmotionPayload to the TTS service —
        no re-fusion needed since EmotionStateService already did the
        full window-smoothed fusion.
        """
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

    def _enqueue(self, text: str, emotion_payload: Optional[Dict], session_id: Optional[str]) -> None:
        try:
            self._play_queue.put_nowait((text, emotion_payload, session_id))
        except queue.Full:
            logger.debug("TTSClient play queue full — dropping oldest item")
            try:
                self._play_queue.get_nowait()
            except queue.Empty:
                pass
            self._play_queue.put_nowait((text, emotion_payload, session_id))

    def _play_worker(self) -> None:
        while True:
            try:
                text, emotion_payload, session_id = self._play_queue.get()
                self._synthesize_and_play(text, emotion_payload, session_id)
            except Exception as exc:
                logger.warning("TTSClient play worker error | %s", exc)

    def _synthesize_and_play(
        self,
        text: str,
        emotion_payload: Optional[Dict],
        session_id: Optional[str],
    ) -> None:
        if not self.is_available():
            # Re-check every call instead of staying permanently unavailable
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

            _play_wav_bytes(resp.content)
            self._available = True

        except requests.exceptions.ConnectionError:
            logger.warning("TTSClient | service unreachable — marking unavailable")
            self._available = False
            if self._fallback:
                self._fallback_speak(text)
        except Exception as exc:
            logger.warning("TTSClient synthesis error | %s", exc)
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


def _play_wav_bytes(wav_bytes: bytes) -> None:
    if not _SD_AVAILABLE:
        logger.debug("sounddevice not available — audio not played")
        return
    try:
        import numpy as np
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if n_channels == 2:
            audio = audio.reshape(-1, 2)

        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except Exception as exc:
        logger.warning("Audio playback failed | %s", exc)