from __future__ import annotations

import io
import logging
import queue
import re
import threading
import time
import wave
from typing import Dict, List, Optional, Tuple

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

        # ── Resume-after-false-interruption state ────────────────────
        # When playback is stopped by an interruption, we save the
        # remaining audio so it can be resumed if the "interruption"
        # turns out to be noise.
        self._interrupted_audio: Optional[bytes] = None  # remaining WAV audio
        self._interrupted_chunks: Optional[list] = None  # remaining text chunks
        self._interrupted_emotion: Optional[Dict] = None
        self._interrupted_session: Optional[str] = None
        self._interrupted_full_text: str = ""

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

    def clear_resume_state(self) -> None:
        """Discard any saved audio from a previous interruption.
        Called when the interruption was confirmed as real."""
        with self._state_lock:
            self._interrupted_audio = None
            self._interrupted_chunks = None
            self._interrupted_emotion = None
            self._interrupted_session = None
            self._interrupted_full_text = ""

    @property
    def has_resume_audio(self) -> bool:
        """True if there is saved audio from a false interruption."""
        with self._state_lock:
            return (self._interrupted_audio is not None
                    or self._interrupted_chunks is not None)

    def resume_playback(self) -> None:
        """Resume playback from where a false interruption stopped it.
        Called from voice_chat_client when a turn is discarded as noise."""
        with self._state_lock:
            remaining_audio = self._interrupted_audio
            remaining_chunks = self._interrupted_chunks
            emotion = self._interrupted_emotion
            session_id = self._interrupted_session
            full_text = self._interrupted_full_text
            # Clear so we don't resume twice
            self._interrupted_audio = None
            self._interrupted_chunks = None
            self._interrupted_emotion = None
            self._interrupted_session = None
            self._interrupted_full_text = ""
            # Ensure volume is restored (belt-and-suspenders — duck may
            # have reduced it and the restore path may not have run)
            self._volume_scale = 1.0

        if remaining_audio is not None:
            # Estimate remaining audio duration for logging
            audio_duration_ms = 0.0
            try:
                buf = io.BytesIO(remaining_audio)
                with wave.open(buf, "rb") as wf:
                    n_frames = wf.getnframes()
                    sample_rate = wf.getframerate()
                    if sample_rate > 0:
                        audio_duration_ms = (n_frames / sample_rate) * 1000.0
            except Exception:
                pass

            logger.info(
                "TTS resuming playback from interrupted audio | duration=%.0fms | remaining_chunks=%d",
                audio_duration_ms,
                len(remaining_chunks) if remaining_chunks else 0,
            )

            # If remaining audio is trivially short (<150ms), skip it and
            # go straight to re-synthesizing remaining chunks if any.
            if audio_duration_ms < 150.0 and not remaining_chunks:
                logger.info(
                    "TTS resume skipped — remaining audio too short (%.0fms) and no chunks to play",
                    audio_duration_ms,
                )
                return

            self._set_playback_state(True, full_text)
            try:
                if audio_duration_ms >= 150.0:
                    self._play_wav_bytes_streaming(remaining_audio)
                # If there are also remaining chunks, continue with those
                if remaining_chunks:
                    # Check if playback was re-interrupted during the audio resume
                    with self._state_lock:
                        if self._stop_event.is_set():
                            # Save chunks back for another potential resume
                            self._interrupted_chunks = remaining_chunks
                            self._interrupted_emotion = emotion
                            self._interrupted_session = session_id
                            self._interrupted_full_text = full_text
                            logger.info("TTS resume re-interrupted before remaining chunks")
                            return
                    self._play_remaining_chunks(
                        remaining_chunks, emotion, session_id, full_text
                    )
            finally:
                self._set_playback_state(False, "")
        elif remaining_chunks:
            # Interrupted between chunks — play remaining chunks
            logger.info(
                "TTS resuming playback from remaining chunks | chunks=%d",
                len(remaining_chunks),
            )
            self._play_remaining_chunks(
                remaining_chunks, emotion, session_id, full_text
            )

    def _play_remaining_chunks(
        self,
        chunks: list,
        emotion_payload: Optional[Dict],
        session_id: Optional[str],
        full_text: str,
    ) -> None:
        """Synthesize and play remaining text chunks after resume."""
        playback_entered = False
        try:
            for idx, chunk_text in enumerate(chunks):
                with self._state_lock:
                    if self._stop_event.is_set():
                        logger.info(
                            "TTS resume chunk loop aborted | played %d/%d",
                            idx, len(chunks),
                        )
                        break

                try:
                    payload: Dict = {"text": chunk_text, "use_cache": True}
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

                    if not playback_entered:
                        self._set_playback_state(True, full_text)
                        playback_entered = True

                    self._play_wav_bytes_streaming(resp.content)
                    self._available = True

                except Exception as exc:
                    logger.warning("TTS resume chunk error | %s", exc)
                    break
        finally:
            if playback_entered:
                self._set_playback_state(False, "")

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

    # ── Sentence-level chunking ────────────────────────────────────────────
    # The TTS server rejects text over its max_length (default 2000 chars).
    # We split long responses into sentence-sized chunks on the client side.
    # This also improves perceived latency: the first sentence starts playing
    # while subsequent chunks are still being synthesised.
    _TTS_MAX_CHUNK_CHARS: int = 1800  # keep headroom below server's 2000

    # Regex: split after sentence-ending punctuation followed by whitespace,
    # or on double-newlines (markdown paragraph breaks).
    _SENTENCE_SPLIT_RE = re.compile(
        r'(?<=[.!?])\s+'        # sentence-ending punctuation + whitespace
        r'|'
        r'\n{2,}'               # paragraph break (markdown ### blocks etc.)
    )

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int) -> List[str]:
        """
        Split *text* into chunks that each fit within *max_chars*.

        Strategy:
        1. Split on sentence boundaries.
        2. Greedily merge consecutive sentences into chunks up to *max_chars*.
        3. If a single sentence exceeds *max_chars*, hard-split it on the
           last space before the limit (unlikely with natural text but safe).
        """
        # Step 1: split into sentences
        raw_parts = TTSClient._SENTENCE_SPLIT_RE.split(text)
        sentences = [s.strip() for s in raw_parts if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        # Step 2: merge greedily
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            # Handle an individual sentence that exceeds the limit
            if len(sentence) > max_chars:
                # Flush what we have so far
                if current:
                    chunks.append(current)
                    current = ""
                # Hard-split the oversized sentence
                while len(sentence) > max_chars:
                    split_at = sentence.rfind(" ", 0, max_chars)
                    if split_at <= 0:
                        split_at = max_chars
                    chunks.append(sentence[:split_at].rstrip())
                    sentence = sentence[split_at:].lstrip()
                if sentence:
                    current = sentence
                continue

            # Normal path: try appending to current chunk
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                # current chunk is full — flush it, start a new one
                chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks

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

        # New speech request — discard any old resume state
        self.clear_resume_state()

        # Split into sentence chunks that fit the server's max_length.
        chunks = self._split_into_chunks(text, self._TTS_MAX_CHUNK_CHARS)
        if not chunks:
            return

        # Track whether we've entered the playback state so the finally
        # block knows whether to tear it down.
        playback_entered = False

        try:
            for idx, chunk_text in enumerate(chunks):
                # After the first chunk has played, check _stop_event between
                # chunks so an interruption during chunk N skips chunk N+1.
                if playback_entered:
                    with self._state_lock:
                        if self._stop_event.is_set():
                            # Save remaining chunks for possible resume
                            remaining = chunks[idx:]
                            if remaining:
                                self._interrupted_chunks = remaining
                                self._interrupted_emotion = emotion_payload
                                self._interrupted_session = session_id
                                self._interrupted_full_text = text
                            logger.info(
                                "TTS chunk loop aborted by interruption | played %d/%d chunks | saved %d for resume",
                                idx, len(chunks), len(remaining) if remaining else 0,
                            )
                            break

                try:
                    payload: Dict = {"text": chunk_text, "use_cache": True}
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
                        "TTS | chunk=%d/%d | latency=%sms | state=%s | cache=%s | chars=%d",
                        idx + 1, len(chunks), latency, state, cache_hit, len(chunk_text),
                    )

                    # On the FIRST chunk, enter playback state right before
                    # playing audio — NOT before the HTTP request.
                    if not playback_entered:
                        self._set_playback_state(True, text)
                        # Also save full text for resume context
                        with self._state_lock:
                            self._interrupted_full_text = text
                            self._interrupted_emotion = emotion_payload
                            self._interrupted_session = session_id
                        playback_entered = True

                    # Play this chunk's audio (checks _stop_event per audio block,
                    # saves remaining audio to _interrupted_audio if stopped)
                    self._play_wav_bytes_streaming(resp.content)

                    # If playback was stopped mid-chunk, save remaining chunks too
                    with self._state_lock:
                        if self._stop_event.is_set():
                            remaining = chunks[idx + 1:]
                            if remaining:
                                self._interrupted_chunks = remaining
                            break

                    self._available = True

                except requests.exceptions.ConnectionError:
                    logger.warning("TTSClient | service unreachable — marking unavailable")
                    self._available = False
                    self.clear_resume_state()
                    if self._fallback:
                        remaining = " ".join(chunks[idx:])
                        self._fallback_speak(remaining)
                    break
                except Exception as exc:
                    logger.warning("TTSClient synthesis error on chunk %d | %s", idx + 1, exc)
                    self.clear_resume_state()
                    if self._fallback:
                        remaining = " ".join(chunks[idx:])
                        self._fallback_speak(remaining)
                    break
        finally:
            if playback_entered:
                self._set_playback_state(False, "")

    def _fallback_speak(self, text: str) -> None:
        if not _PYTTSX3_AVAILABLE:
            return
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            logger.debug("pyttsx3 fallback failed | %s", exc)

    @staticmethod
    def _drain_remaining_sentences(sentences_iter, max_wait_s: float = 1.0) -> list:
        """Non-blocking drain of remaining sentences from an iterator.

        The iterator may be backed by a queue that blocks while the LLM is
        still producing.  We signal drain mode (if supported) so the
        iterator uses a short timeout, and also enforce an overall deadline.
        """
        # Signal the iterator to use short timeouts (non-blocking drain)
        enable_drain = getattr(sentences_iter, "_enable_drain_mode", None)
        if callable(enable_drain):
            enable_drain()

        remaining: list = []
        deadline = time.monotonic() + max_wait_s
        try:
            for s in sentences_iter:
                if s and s.strip():
                    remaining.append(s)
                if time.monotonic() > deadline:
                    logger.debug(
                        "Drain deadline reached — collected %d sentences", len(remaining)
                    )
                    break
        except Exception:
            pass
        return remaining

    def stream_and_play(
        self,
        sentences,
        emotion_payload: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Synthesize and play sentences from an iterator, one at a time.

        This is the streaming counterpart of _synthesize_and_play().
        Instead of splitting pre-existing text into chunks, it reads
        sentences from an iterator (typically fed by an LLM stream)
        and plays each one immediately.

        Same playback state management as _synthesize_and_play:
        - _set_playback_state(True) on first sentence
        - _stop_event checked between sentences and per audio chunk
        - _set_playback_state(False) in finally block

        Designed to run on a background thread (called from
        voice_chat_client's _bg_stream_tts_and_memory thread).
        """
        if not self.is_available():
            self.invalidate_availability_cache()
            if not self.is_available():
                return

        self.clear_resume_state()
        playback_entered = False
        sentence_count = 0
        # Collect the full text as we go so resume context is accurate
        accumulated_text = ""

        try:
            for sentence in sentences:
                if not sentence or not sentence.strip():
                    continue

                sentence_count += 1
                accumulated_text = (accumulated_text + " " + sentence).strip() if accumulated_text else sentence

                # Check _stop_event between sentences
                if playback_entered:
                    with self._state_lock:
                        if self._stop_event.is_set():
                            # Drain remaining sentences from iterator for resume.
                            # Use a short timeout per item to avoid blocking if the
                            # iterator is backed by a queue still waiting on an LLM.
                            remaining = self._drain_remaining_sentences(sentences)
                            # The current sentence wasn't played yet — include it
                            remaining.insert(0, sentence)
                            if remaining:
                                self._interrupted_chunks = remaining
                                self._interrupted_emotion = emotion_payload
                                self._interrupted_session = session_id
                                self._interrupted_full_text = accumulated_text
                            logger.info(
                                "TTS stream aborted by interruption | played %d sentences | saved %d for resume",
                                sentence_count - 1, len(remaining),
                            )
                            break

                try:
                    payload: Dict = {"text": sentence, "use_cache": True}
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
                    state_hdr = resp.headers.get("X-TTS-Resolved-State", "?")
                    cache_hit = resp.headers.get("X-TTS-Cache-Hit", "false") == "true"
                    logger.info(
                        "TTS stream | sentence=%d | latency=%sms | state=%s | cache=%s | chars=%d",
                        sentence_count, latency, state_hdr, cache_hit, len(sentence),
                    )

                    if not playback_entered:
                        self._set_playback_state(True, accumulated_text)
                        # Save context for resume in case of mid-chunk interruption
                        with self._state_lock:
                            self._interrupted_full_text = accumulated_text
                            self._interrupted_emotion = emotion_payload
                            self._interrupted_session = session_id
                        playback_entered = True

                    self._play_wav_bytes_streaming(resp.content)

                    # Check if interrupted during playback of this chunk
                    with self._state_lock:
                        if self._stop_event.is_set():
                            # _play_wav_bytes_streaming already saved _interrupted_audio.
                            # Now drain and save remaining sentences from iterator.
                            remaining = self._drain_remaining_sentences(sentences)
                            if remaining:
                                self._interrupted_chunks = remaining
                                self._interrupted_emotion = emotion_payload
                                self._interrupted_session = session_id
                                self._interrupted_full_text = accumulated_text
                            logger.info(
                                "TTS stream mid-chunk interruption | played %d sentences | saved %d chunks for resume",
                                sentence_count, len(remaining),
                            )
                            break

                    self._available = True

                except requests.exceptions.ConnectionError:
                    logger.warning("TTSClient stream | service unreachable")
                    self._available = False
                    break
                except Exception as exc:
                    logger.warning("TTSClient stream sentence %d error | %s", sentence_count, exc)
                    break
        finally:
            if playback_entered:
                self._set_playback_state(False, "")

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
                            # Save remaining audio for possible resume
                            remaining = audio[idx:]
                            if len(remaining) > 0:
                                remaining_int16 = (remaining * 32767.0).astype(np.int16)
                                remaining_buf = io.BytesIO()
                                with wave.open(remaining_buf, "wb") as wf_out:
                                    wf_out.setnchannels(n_channels)
                                    wf_out.setsampwidth(2)
                                    wf_out.setframerate(sample_rate)
                                    wf_out.writeframes(remaining_int16.tobytes())
                                self._interrupted_audio = remaining_buf.getvalue()
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