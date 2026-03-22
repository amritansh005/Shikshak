"""
Thin Qt wrapper around the fully-working backend voice_chat_client.

All heavy lifting (VAD, STT, interruption, TTS control, emotion) is done
by the backend code in STT/voice_chat_client.py.  This worker just:
  1. Runs the backend event loop on a QThread
  2. Emits Qt signals so the GUI can display what's happening

Option A+B: finalize_turn runs in a background thread so the event loop
keeps processing VAD events — enabling real-time interruption even while
the LLM is generating and TTS is speaking.
"""
from __future__ import annotations

import builtins
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PyQt6.QtCore import QObject, pyqtSignal

from gui_app.bootstrap import setup_project_imports

setup_project_imports()

from app.config import settings  # type: ignore
from app.services.emotion_service import SERModel, classify_text_emotion, extract_prosody  # type: ignore
from app.services.realtime_vad import MicrophoneVADStreamer  # type: ignore
from app.services.stt_service import STTService  # type: ignore
from app.services.turn_manager import TurnManager, TurnState  # type: ignore

try:
    from app.services.speaker_verification import SpeakerVerificationService  # type: ignore
except Exception:
    SpeakerVerificationService = None

# ── Import the WORKING backend logic directly ──
import voice_chat_client as vcc  # type: ignore

logger = logging.getLogger(__name__)


class PrintCapture:
    """
    Temporarily intercept print() calls from finalize_turn to extract
    student text, teacher text, and emotion for the GUI signals.
    """

    def __init__(self, worker: "LiveSessionWorker") -> None:
        self.worker = worker
        self._original_print = builtins.print
        self.saw_teacher_reply = False

    def __enter__(self):
        builtins.print = self._capture_print
        return self

    def __exit__(self, *args):
        builtins.print = self._original_print

    def _capture_print(self, *args, **kwargs) -> None:
        text = " ".join(str(a) for a in args)
        # Always forward to real print for terminal logging
        self._original_print(*args, **kwargs)

        # Extract signals from backend print output
        if text.startswith("You: ") and "[filtered" not in text and "[too short" not in text:
            student_text = text[5:].strip()
            if student_text:
                self.worker.final_student_text.emit(student_text)
                self.worker.live_student_text.emit("")

        elif text.startswith("Teacher: "):
            teacher_text = text[9:].strip()
            if teacher_text:
                self.worker.final_teacher_text.emit(teacher_text)
                self.worker.live_teacher_text.emit(teacher_text)
                self.worker.status_changed.emit("Speaking")
                self.saw_teacher_reply = True

        elif text.strip().startswith("[text:"):
            # Emotion line like: [text: neutral | voice: sad (50%) | ...]
            self.worker.emotion_changed.emit(text.strip().strip("[]"))

        elif "[Assistant interrupted" in text:
            self.worker.note_changed.emit("Assistant interrupted by student")

        elif "Listening..." in text:
            self.worker.status_changed.emit("Listening")
            self.worker.note_changed.emit("Student speech detected")

        elif "[Resuming teacher speech]" in text:
            self.worker.note_changed.emit("Resuming teacher speech")
            self.worker.status_changed.emit("Speaking")

        elif "Speaker profile enrolled" in text:
            self.worker.note_changed.emit("Speaker profile enrolled")


class LiveSessionWorker(QObject):
    """Runs the backend voice chat loop and emits GUI signals."""

    status_changed = pyqtSignal(str)
    note_changed = pyqtSignal(str)
    live_student_text = pyqtSignal(str)
    final_student_text = pyqtSignal(str)
    live_teacher_text = pyqtSignal(str)
    final_teacher_text = pyqtSignal(str)
    emotion_changed = pyqtSignal(str)
    session_ready = pyqtSignal(str)
    health_report = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_requested = False
        self._streamer: Optional[MicrophoneVADStreamer] = None

    def stop(self) -> None:
        self._stop_requested = True
        if self._streamer is not None:
            self._streamer.reset()
        # Stop TTS playback
        if vcc._tts_client is not None:
            try:
                vcc._tts_stop_playback()
            except Exception:
                pass

    def run(self) -> None:
        try:
            self.status_changed.emit("Starting")
            self.note_changed.emit("Loading services...")

            # ── Setup ──
            session_id = f"voice-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Try to reuse preloaded models
            try:
                from gui_app.preloader import get_preloaded
                models = get_preloaded()
                if models.ready:
                    stt = models.stt or STTService()
                    ser_model = models.ser_model or SERModel()
                    speaker_verifier = models.speaker_verifier
                    self.note_changed.emit("Using preloaded models")
                else:
                    raise RuntimeError("Not preloaded")
            except Exception:
                self.note_changed.emit("Loading STT...")
                stt = STTService()
                self.note_changed.emit("Loading emotion model...")
                ser_model = SERModel()
                speaker_verifier = None
                if SpeakerVerificationService and getattr(settings, "speaker_verification_enabled", False):
                    sv = SpeakerVerificationService(
                        similarity_threshold=settings.speaker_verification_threshold,
                        auto_update_profile=settings.speaker_verification_auto_update,
                        device="cpu",
                    )
                    if sv.is_available:
                        speaker_verifier = sv

            streamer = MicrophoneVADStreamer()
            self._streamer = streamer
            turn_manager = TurnManager()
            state = vcc.LiveTranscriptState()

            # ── Ensure TTS client is initialized ──
            if vcc._tts_client is None:
                try:
                    from tts_client import TTSClient
                    vcc._tts_client = TTSClient(tts_service_url="http://127.0.0.1:5000", timeout=120)
                except Exception as exc:
                    logger.warning("TTSClient not available: %s", exc)

            tts_status = "disabled"
            if vcc._tts_client is not None:
                vcc._tts_client.invalidate_availability_cache()
                if vcc._tts_client.is_available():
                    tts_status = "enabled"
                else:
                    tts_status = "disabled (service not reachable)"
            self.health_report.emit("TTS", tts_status)

            sv_status = "enabled (will enroll on first speech)" if speaker_verifier else "disabled"
            self.health_report.emit("Speaker verification", sv_status)

            # ── Verify teacher backend ──
            try:
                r = requests.get(settings.teacher_chat_url.rsplit("/chat", 1)[0] + "/", timeout=10)
                r.raise_for_status()
                self.note_changed.emit("Teacher backend connected")
            except Exception as exc:
                raise RuntimeError(f"Teacher backend not reachable: {exc}") from exc

            self.session_ready.emit(session_id)
            self.status_changed.emit("Listening")
            self.note_changed.emit("Speak naturally. Live mode is active.")

            # ── Monkey-patch print_live for partial transcription display ──
            original_print_live = vcc.print_live

            def gui_print_live(text: str) -> None:
                clipped = text[:160].replace("You (live): ", "")
                self.live_student_text.emit(clipped)

            vcc.print_live = gui_print_live

            # ── Main event loop ──
            # The event loop NEVER blocks. finalize_turn runs in a
            # background thread (Option A) so VAD events keep flowing,
            # enabling real-time interruption detection at all times.
            try:
                for event in streamer.stream_events():
                    if self._stop_requested:
                        break

                    # finalize_turn (bg thread) cannot safely call
                    # streamer.reset() — do it here on the main thread.
                    with state.lock:
                        if state.needs_streamer_reset:
                            state.needs_streamer_reset = False
                            streamer.reset()

                    # Interruption handling — runs on EVERY speech_frame,
                    # even while finalize_turn is running in background
                    if event.event_type == "speech_frame":
                        vcc.maybe_handle_tts_interruption(
                            state=state,
                            event=event,
                            stt=stt,
                        )

                    if event.event_type == "speech_start":
                        with state.lock:
                            if not state.turn.speech_active and not state.finalizing:
                                turn_manager.start_turn(state.turn)
                                self.status_changed.emit("Listening")
                                self.note_changed.emit("Student speech detected")
                        continue

                    if event.event_type == "speech_frame" and event.pcm_bytes:
                        with state.lock:
                            if not state.turn.speech_active or state.finalizing:
                                continue

                            turn_manager.append_frame(
                                state.turn,
                                event.pcm_bytes,
                                is_speech=event.is_speech,
                            )
                            state.is_speech_flags.append(event.is_speech)
                            decision = turn_manager.evaluate(state.turn)

                        vcc.maybe_launch_partial_transcription(state, stt, turn_manager)

                        if decision.action == "finalize":
                            self.status_changed.emit("Thinking")

                            # ── Option A: run finalize_turn in background ──
                            # PrintCapture intercepts print() from the bg
                            # thread and emits GUI signals. The event loop
                            # keeps running so VAD can detect interruptions.
                            capture = PrintCapture(self)
                            capture.__enter__()

                            def _finalize_bg(cap=capture):
                                try:
                                    vcc.finalize_turn(
                                        state, stt, turn_manager, session_id,
                                        streamer, ser_model, speaker_verifier,
                                    )
                                except Exception as exc:
                                    logger.exception("finalize_turn failed")
                                    self.error_occurred.emit(str(exc))
                                finally:
                                    cap.__exit__(None, None, None)
                                    self.live_student_text.emit("")
                                    # Only revert to Listening if teacher
                                    # didn't reply (filtered/hallucinated).
                                    # If teacher replied, PrintCapture already
                                    # set "Speaking".
                                    if not self._stop_requested and not cap.saw_teacher_reply:
                                        self.status_changed.emit("Listening")

                            threading.Thread(
                                target=_finalize_bg,
                                daemon=True,
                                name="finalize-turn",
                            ).start()

                        elif decision.action == "discard":
                            self.note_changed.emit("Turn ignored: too short")
                            with state.lock:
                                had_interruption = state.interruption.confirmed
                            if had_interruption and vcc._tts_has_resume_audio():
                                vcc._tts_resume_playback()
                            vcc._tts_restore_playback()
                            streamer.reset()
                            state.reset()
                        else:
                            latest = state.turn.latest_partial()
                            if latest and decision.reason == "soft_pause_resume_window":
                                self.live_student_text.emit(latest + " …")
                        continue

            except Exception as exc:
                logger.exception("Voice chat loop failed")
                self.error_occurred.emit(str(exc))
            finally:
                # Restore monkey-patch
                vcc.print_live = original_print_live

        except Exception as exc:
            logger.exception("Desktop live session failed")
            self.error_occurred.emit(str(exc))
        finally:
            self.status_changed.emit("Stopped")
            self.finished.emit()