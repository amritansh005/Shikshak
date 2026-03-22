from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from app.config import settings
from app.services.emotion_service import (
    SERModel,
    classify_text_emotion,
    extract_prosody,
)
from app.services.realtime_vad import MicrophoneVADStreamer
from app.services.speaker_verification import SpeakerVerificationService
from app.services.stt_service import STTService
from app.services.turn_manager import TurnManager, TurnState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── TTS client (optional — skipped gracefully if tts_service not running) ──
_tts_client = None
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../tts_service"))
    from tts_client import TTSClient

    _tts_client = TTSClient(tts_service_url="http://127.0.0.1:5000", timeout=120)
    logger.info("TTSClient loaded | url=http://127.0.0.1:5000")
except Exception as _tts_err:
    logger.info("TTSClient not available (%s) — running text-only mode", _tts_err)


HALLUCINATION_BLOCKLIST = {
    "bye", "bye bye", "goodbye", "thank you", "thanks",
    "thanks for watching", "thank you for watching",
    "like and subscribe", "subscribe",
    "you", "the end", "so", "okay", "ok",
    "cheers", "see you", "see you next time",
}

INTERRUPTION_KEYWORDS = {
    "wait",
    "no",
    "stop",
    "hold on",
    "one sec",
    "one second",
    "actually",
    "but",
    "sorry",
    "excuse me",
    "what",
    "why",
    "how",
    "hmm wait",
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature #2 — Backchannel classification
# ─────────────────────────────────────────────────────────────────────────────
# Words/phrases that are acknowledgments, not real interruptions.  When partial
# ASR detects *only* backchannel words, duck stays active but confirmation is
# suppressed.  A real interruption requires a non-backchannel word OR speech
# exceeding INTERRUPT_BACKCHANNEL_OVERRIDE_MS.
BACKCHANNEL_WORDS = {
    "hmm", "hm", "mm", "mmm", "mhm", "uh huh", "uhuh",
    "okay", "ok", "haan", "ha", "accha", "acha",
    "yes", "yeah", "yep", "right", "sure",
    "oh", "ah", "aha",
}

# If speech is *only* backchannel but lasts this long, treat it as a real
# interruption anyway — the user is probably trying to take the floor.
INTERRUPT_BACKCHANNEL_OVERRIDE_MS = 800

# ─────────────────────────────────────────────────────────────────────────────
# Normal (non-TTS) interruption thresholds
# ─────────────────────────────────────────────────────────────────────────────
INTERRUPT_MIN_SPEECH_MS = 180
INTERRUPT_CONFIRM_MS = 320
INTERRUPT_SILENCE_RESET_MS = 500
INTERRUPT_DUCK_LEVEL = 0.22

# Partial ASR fires between duck and hard confirm thresholds (bug #1 fix).
INTERRUPT_PARTIAL_ASR_MS = INTERRUPT_MIN_SPEECH_MS

# ─────────────────────────────────────────────────────────────────────────────
# Feature #3 — Echo suppression via raised thresholds during TTS playback
# ─────────────────────────────────────────────────────────────────────────────
# When TTS is playing, the mic picks up speaker output.  Without real AEC, we
# compensate by requiring significantly more speech evidence before ducking or
# confirming.  These are the "during playback" equivalents of the normal
# thresholds above.
INTERRUPT_MIN_SPEECH_MS_DURING_TTS = 360     # 2× normal duck threshold
INTERRUPT_CONFIRM_MS_DURING_TTS = 640         # 2× normal confirm threshold
INTERRUPT_SILENCE_RESET_MS_DURING_TTS = 700   # slightly more patience
INTERRUPT_PARTIAL_ASR_MS_DURING_TTS = INTERRUPT_MIN_SPEECH_MS_DURING_TTS

# ─────────────────────────────────────────────────────────────────────────────
# Feature #1 — No-barge-in phase
# ─────────────────────────────────────────────────────────────────────────────
# For the first N ms of TTS playback, ignore all interruption candidates
# entirely.  This prevents:
#   - TTS audio leaking into the mic and triggering false VAD at playback start
#   - accidental interruptions on the very first phonemes
NO_BARGE_IN_MS = 400


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _contains_interruption_keyword(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized in INTERRUPTION_KEYWORDS:
        return True
    return any(keyword in normalized for keyword in INTERRUPTION_KEYWORDS)


def _is_only_backchannel(text: str) -> bool:
    """Return True if every word in *text* is a backchannel token."""
    normalized = _normalize_text(text)
    if not normalized:
        return True  # empty → not a real interruption either

    # Check full phrase first (handles multi-word entries like "uh huh").
    if normalized in BACKCHANNEL_WORDS:
        return True

    # Then check word-by-word.
    words = normalized.split()
    return all(w in BACKCHANNEL_WORDS for w in words)


def _tts_is_playing() -> bool:
    if _tts_client is None:
        return False

    try:
        value = getattr(_tts_client, "is_playing", None)
        if callable(value):
            return bool(value())
        if value is not None:
            return bool(value)
    except Exception:
        logger.exception("Failed checking TTS playback state")

    return False


def _tts_playback_age_ms() -> float:
    """How many ms ago did the current TTS playback start?  Returns 0 if idle."""
    if _tts_client is None:
        return 0.0
    try:
        started = getattr(_tts_client, "playback_started_at", 0.0)
        if callable(started):
            started = started()
        if started and started > 0:
            return (time.monotonic() - started) * 1000.0
    except Exception:
        logger.exception("Failed reading TTS playback_started_at")
    return 0.0


def _tts_stop_playback() -> None:
    if _tts_client is None:
        return
    try:
        stop_fn = getattr(_tts_client, "stop_playback", None)
        if callable(stop_fn):
            stop_fn()
    except Exception:
        logger.exception("Failed to stop TTS playback")


def _tts_duck_playback(level: float = INTERRUPT_DUCK_LEVEL) -> None:
    if _tts_client is None:
        return
    try:
        duck_fn = getattr(_tts_client, "duck_playback", None)
        if callable(duck_fn):
            duck_fn(level=level)
    except Exception:
        logger.exception("Failed to duck TTS playback")


def _tts_restore_playback() -> None:
    if _tts_client is None:
        return
    try:
        restore_fn = getattr(_tts_client, "restore_playback", None)
        if callable(restore_fn):
            restore_fn()
    except Exception:
        logger.exception("Failed to restore TTS playback")


def _tts_current_text() -> str:
    if _tts_client is None:
        return ""
    try:
        text = getattr(_tts_client, "current_text", "")
        if isinstance(text, str):
            return text.strip()
    except Exception:
        logger.exception("Failed reading TTS current_text")
    return ""


def _tts_has_resume_audio() -> bool:
    """Check if TTS has saved audio from a false interruption."""
    if _tts_client is None:
        return False
    try:
        return bool(getattr(_tts_client, "has_resume_audio", False))
    except Exception:
        return False


def _tts_resume_playback() -> None:
    """Resume TTS from where it was interrupted (false interruption recovery)."""
    if _tts_client is None:
        return
    try:
        resume_fn = getattr(_tts_client, "resume_playback", None)
        if callable(resume_fn):
            resume_fn()
    except Exception:
        logger.exception("Failed to resume TTS playback")


def _tts_clear_resume_state() -> None:
    """Discard saved resume audio (interruption was real)."""
    if _tts_client is None:
        return
    try:
        clear_fn = getattr(_tts_client, "clear_resume_state", None)
        if callable(clear_fn):
            clear_fn()
    except Exception:
        logger.exception("Failed to clear TTS resume state")


class InterruptionState:
    """
    Tracks user barge-in while assistant TTS is active.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.candidate_started_at: float = 0.0
        self.candidate_speech_ms: float = 0.0
        self.trailing_silence_ms: float = 0.0
        self.confirmed: bool = False
        self.tts_text_snapshot: str = ""
        self.reason: str = ""
        self.ducked: bool = False
        self.frames_seen: int = 0
        self.partial_confirm_in_flight: bool = False
        self.vad_asr_delegated: bool = False  # Stage 3 fired ASR once

    def reset(self) -> None:
        self.active = False
        self.candidate_started_at = 0.0
        self.candidate_speech_ms = 0.0
        self.trailing_silence_ms = 0.0
        self.confirmed = False
        self.tts_text_snapshot = ""
        self.reason = ""
        self.ducked = False
        self.frames_seen = 0
        self.partial_confirm_in_flight = False
        self.vad_asr_delegated = False

    def begin_candidate(self, tts_text: str) -> None:
        self.active = True
        self.candidate_started_at = time.time()
        self.candidate_speech_ms = 0.0
        self.trailing_silence_ms = 0.0
        self.confirmed = False
        self.tts_text_snapshot = tts_text.strip()
        self.reason = ""
        self.ducked = False
        self.frames_seen = 0
        self.partial_confirm_in_flight = False
        self.vad_asr_delegated = False


class LiveTranscriptState:
    def __init__(self) -> None:
        self.turn = TurnState()
        self.last_partial_audio_seconds: float = 0.0
        self.partial_in_flight = False
        self.finalizing = False
        self.turn_generation = 0
        self.lock = threading.Lock()
        self.is_speech_flags: List[bool] = []

        self.interruption = InterruptionState()
        self.interruption_meta_for_turn: Optional[Dict[str, Any]] = None

        # Set by finalize_turn (bg thread) to request the main loop
        # to call streamer.reset(). Cross-thread streamer.reset() is
        # unsafe — it must run on the thread iterating stream_events().
        self.needs_streamer_reset = False

    def reset(self) -> None:
        with self.lock:
            self.turn.reset()
            self.last_partial_audio_seconds = 0.0
            self.partial_in_flight = False
            self.finalizing = False
            self.turn_generation += 1
            self.is_speech_flags.clear()
            self.interruption.reset()
            self.interruption_meta_for_turn = None


def print_live(text: str) -> None:
    clipped = text[:160]
    print(f"You (live): {clipped}", end="\r", flush=True)


def _prepare_for_tts_resume(state: LiveTranscriptState) -> None:
    """Clear finalizing and reset state BEFORE resuming TTS.

    resume_playback() blocks until audio finishes.  If state.finalizing
    remains True during that time, _run_partial_interrupt_confirmation
    bails out immediately and the user cannot interrupt the resumed TTS.
    """
    with state.lock:
        state.needs_streamer_reset = True
    state.reset()                      # clears finalizing, turn, interruption


def send_to_teacher(
    session_id: str,
    text: str,
    emotion_data: Dict | None = None,
    interruption_meta: Dict[str, Any] | None = None,
) -> Dict:
    payload: Dict[str, Any] = {"message": text, "session_id": session_id}
    if emotion_data:
        payload["emotion"] = emotion_data
    if interruption_meta:
        payload["interruption_meta"] = interruption_meta

    response = requests.post(
        settings.teacher_chat_url,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "text": data.get("response", "").strip(),
        "directive": data.get("directive"),
    }


def maybe_launch_partial_transcription(
    state: LiveTranscriptState,
    stt: STTService,
    turn_manager: TurnManager,
) -> None:
    with state.lock:
        if state.partial_in_flight or state.finalizing:
            return
        if not turn_manager.should_attempt_partial(state.turn, state.last_partial_audio_seconds):
            return
        state.partial_in_flight = True
        audio_bytes = state.turn.snapshot_audio()
        audio_seconds = state.turn.total_audio_seconds
        generation = state.turn_generation
        state.last_partial_audio_seconds = audio_seconds

    def worker() -> None:
        try:
            result = stt.transcribe_bytes(audio_bytes, partial=True)
            text = result["text"].strip()
            if text:
                with state.lock:
                    if generation == state.turn_generation and not state.finalizing:
                        turn_manager.register_partial(state.turn, text, audio_seconds)
                        print_live(text)
        except Exception:
            logger.exception("Partial transcription failed")
        finally:
            with state.lock:
                state.partial_in_flight = False

    threading.Thread(target=worker, daemon=True).start()


def _run_partial_interrupt_confirmation(
    state: LiveTranscriptState,
    stt: STTService,
    partial_asr_threshold_ms: float,
) -> None:
    with state.lock:
        if not state.interruption.active or state.interruption.confirmed or state.finalizing:
            return
        if state.interruption.partial_confirm_in_flight:
            return
        candidate_ms = state.interruption.candidate_speech_ms
        if candidate_ms < partial_asr_threshold_ms:
            return

        audio_bytes = state.turn.snapshot_audio()
        generation = state.turn_generation
        tts_text_snapshot = state.interruption.tts_text_snapshot
        state.interruption.partial_confirm_in_flight = True

    def worker() -> None:
        try:
            result = stt.transcribe_bytes(audio_bytes, partial=True)
            text = result["text"].strip()
            if not text:
                return

            # ── Feature #2: backchannel check ──
            # If partial ASR produced only backchannel words, do NOT confirm
            # the interruption.  The duck stays (volume is lowered) but playback
            # continues.  This prevents "hmm", "okay", "haan" from killing TTS.
            if _is_only_backchannel(text):
                logger.info(
                    "Partial ASR backchannel suppressed | text=%r | speech_ms=%.1f",
                    text,
                    state.interruption.candidate_speech_ms,
                )
                return

            reason: Optional[str] = None
            if _contains_interruption_keyword(text):
                reason = f"keyword:{text}"
            elif len(text.split()) >= 2:
                reason = f"partial_asr:{text}"

            if not reason:
                return

            with state.lock:
                if generation != state.turn_generation:
                    return
                if not state.interruption.active or state.interruption.confirmed:
                    return

                state.interruption.confirmed = True
                state.interruption.reason = reason
                state.interruption_meta_for_turn = {
                    "interrupted": True,
                    "reason": reason,
                    "interrupted_assistant_text": tts_text_snapshot,
                    "speech_ms_before_cancel": round(state.interruption.candidate_speech_ms, 1),
                }

            logger.info("Interruption confirmed via partial ASR | reason=%s", reason)
            _tts_stop_playback()
            print("\n[Assistant interrupted by user]\n", flush=True)
        except Exception:
            logger.exception("Partial interruption confirmation failed")
        finally:
            with state.lock:
                state.interruption.partial_confirm_in_flight = False
                # If ASR didn't confirm, allow Stage 3 to retry on next
                # batch of speech frames (with fresh audio data).
                if not state.interruption.confirmed:
                    state.interruption.vad_asr_delegated = False

    threading.Thread(target=worker, daemon=True).start()


def maybe_handle_tts_interruption(
    state: LiveTranscriptState,
    event: Any,
    stt: STTService,
) -> None:
    if event.event_type != "speech_frame" or not event.pcm_bytes:
        return

    if not _tts_is_playing():
        with state.lock:
            if state.interruption.ducked and not state.interruption.confirmed:
                state.interruption.reset()
        _tts_restore_playback()
        return

    # ── Feature #1: no-barge-in phase ──────────────────────────────────────
    # For the first NO_BARGE_IN_MS of playback, completely ignore speech
    # frames for interruption purposes.  This prevents TTS audio leaking
    # into the mic from triggering a false barge-in on the opening phonemes.
    playback_age = _tts_playback_age_ms()
    if playback_age > 0 and playback_age < NO_BARGE_IN_MS:
        return

    # ── Feature #3: pick thresholds based on whether TTS is active ─────────
    # During playback the mic picks up speaker output.  Without real AEC we
    # compensate by requiring more speech evidence before ducking/confirming.
    tts_active = True  # we already checked _tts_is_playing() above
    if tts_active:
        duck_threshold = INTERRUPT_MIN_SPEECH_MS_DURING_TTS
        confirm_threshold = INTERRUPT_CONFIRM_MS_DURING_TTS
        silence_reset_threshold = INTERRUPT_SILENCE_RESET_MS_DURING_TTS
        partial_asr_threshold = INTERRUPT_PARTIAL_ASR_MS_DURING_TTS
    else:
        duck_threshold = INTERRUPT_MIN_SPEECH_MS
        confirm_threshold = INTERRUPT_CONFIRM_MS
        silence_reset_threshold = INTERRUPT_SILENCE_RESET_MS
        partial_asr_threshold = INTERRUPT_PARTIAL_ASR_MS

    with state.lock:
        if not state.interruption.active:
            state.interruption.begin_candidate(tts_text=_tts_current_text())
            # Ensure turn is active so audio frames get buffered.
            # _run_partial_interrupt_confirmation needs audio from
            # state.turn.snapshot_audio() to do ASR.
            if not state.turn.speech_active:
                state.turn.speech_active = True
                state.turn.frames.clear()
                state.turn.total_audio_seconds = 0.0

        state.interruption.frames_seen += 1

        # Buffer the PCM frame into the turn for ASR confirmation
        if event.pcm_bytes and state.turn.speech_active:
            state.turn.frames.append(event.pcm_bytes)
            state.turn.total_audio_seconds += float(settings.audio_frame_ms) / 1000.0

        if event.is_speech:
            state.interruption.candidate_speech_ms += float(settings.audio_frame_ms)
            state.interruption.trailing_silence_ms = 0.0
        else:
            state.interruption.trailing_silence_ms += float(settings.audio_frame_ms)

        candidate_ms = state.interruption.candidate_speech_ms
        silence_ms = state.interruption.trailing_silence_ms
        confirmed = state.interruption.confirmed
        already_ducked = state.interruption.ducked
        tts_text_snapshot = state.interruption.tts_text_snapshot

        if (
            not confirmed
            and candidate_ms > 0
            and silence_ms >= silence_reset_threshold
        ):
            state.interruption.reset()
            _tts_restore_playback()
            return

    # ── Stage 1: duck playback once speech exceeds the duck threshold ──
    if event.is_speech and candidate_ms >= duck_threshold and not already_ducked:
        _tts_duck_playback(level=INTERRUPT_DUCK_LEVEL)
        with state.lock:
            state.interruption.ducked = True
        logger.info("TTS ducked due to possible interruption | speech_ms=%.1f", candidate_ms)

    # ── Stage 2: in the duck→confirm window, try partial ASR confirmation ──
    if (
        event.is_speech
        and candidate_ms >= partial_asr_threshold
        and candidate_ms < confirm_threshold
        and not confirmed
    ):
        _run_partial_interrupt_confirmation(state, stt, partial_asr_threshold)

    # ── Stage 3: hard VAD-duration confirmation (fallback) ──
    # Feature #2 extension: if we have partial ASR text and it's *only*
    # backchannel, require INTERRUPT_BACKCHANNEL_OVERRIDE_MS instead of
    # the normal confirm threshold.  This lets "hmm" / "okay" duck but not
    # kill playback unless the user keeps talking for a long time.
    #
    # Option 1 (noise protection): Instead of killing TTS on VAD duration
    # alone, we require partial ASR to produce at least one real word.
    # If no partial ASR has run yet at this point, fire one now.
    # The actual kill happens inside _run_partial_interrupt_confirmation
    # when it finds real words.  If no words are found, playback continues.
    effective_confirm = confirm_threshold
    if event.is_speech and candidate_ms >= confirm_threshold and not confirmed:
        # Check whether latest partial (if any) is only backchannel.
        with state.lock:
            latest_partial_text = (
                state.turn.latest_partial() if state.turn.speech_active else ""
            )
        if latest_partial_text and _is_only_backchannel(latest_partial_text):
            effective_confirm = INTERRUPT_BACKCHANNEL_OVERRIDE_MS
            if candidate_ms < effective_confirm:
                logger.debug(
                    "Backchannel override: delaying confirm | text=%r | speech_ms=%.1f | need=%d",
                    latest_partial_text, candidate_ms, effective_confirm,
                )

    if event.is_speech and candidate_ms >= effective_confirm and not confirmed:
        # Instead of killing TTS immediately, require ASR confirmation.
        # Use vad_asr_delegated flag to fire this only ONCE, not every frame.
        with state.lock:
            already_delegated = state.interruption.vad_asr_delegated
            if not already_delegated:
                state.interruption.vad_asr_delegated = True
        if not already_delegated:
            _run_partial_interrupt_confirmation(state, stt, 0.0)
            logger.info(
                "VAD duration threshold reached, delegating to ASR confirmation | speech_ms=%.1f",
                candidate_ms,
            )
        return


def finalize_turn(
    state: LiveTranscriptState,
    stt: STTService,
    turn_manager: TurnManager,
    session_id: str,
    streamer: MicrophoneVADStreamer,
    ser_model: SERModel,
    speaker_verifier: Optional[SpeakerVerificationService] = None,
) -> None:
    with state.lock:
        if state.finalizing:
            return
        state.finalizing = True
        audio_bytes = state.turn.snapshot_audio()
        final_duration = state.turn.total_audio_seconds
        generation = state.turn_generation
        pcm_frames_snapshot = list(state.turn.frames)
        is_speech_snapshot = list(state.is_speech_flags)
        interruption_meta = (
            dict(state.interruption_meta_for_turn)
            if state.interruption_meta_for_turn is not None
            else None
        )
        tts_text_snapshot = state.interruption.tts_text_snapshot
        had_interruption = state.interruption.confirmed

    try:
        if final_duration < settings.whisper_min_audio_ms / 1000.0:
            print("\nYou: [too short, ignored]\n", flush=True)
            # Option 3: resume TTS if this was a false interruption
            if had_interruption and _tts_has_resume_audio():
                logger.info("False interruption detected (too short) — resuming TTS")
                print("[Resuming teacher speech]\n", flush=True)
                _prepare_for_tts_resume(state)
                _tts_resume_playback()
            return

        # ── Speaker verification gate ──────────────────────────────
        # If a speaker profile is enrolled, check if this audio matches
        # the student.  If not, discard the turn (background speaker).
        if speaker_verifier is not None and speaker_verifier.is_enrolled:
            sv_result = speaker_verifier.verify(
                audio_int16=audio_bytes,
                sample_rate=settings.whisper_sample_rate,
            )
            if not sv_result["is_student"]:
                logger.info(
                    "Speaker verification rejected | similarity=%.3f | threshold=%.2f | latency=%.0fms",
                    sv_result["similarity"],
                    sv_result["threshold"],
                    sv_result["latency_ms"],
                )
                print(
                    f"\nYou: [different speaker, ignored | similarity={sv_result['similarity']:.2f}]\n",
                    flush=True,
                )
                # Resume TTS if this background speaker triggered a false interruption
                if had_interruption and _tts_has_resume_audio():
                    logger.info("Background speaker detected — resuming TTS")
                    print("[Resuming teacher speech]\n", flush=True)
                    _prepare_for_tts_resume(state)
                    _tts_resume_playback()
                    return

        result = stt.transcribe_bytes(audio_bytes, partial=False)
        final_text = result["text"].strip()
        avg_no_speech_prob = float(result.get("avg_no_speech_prob", 0.0))

        if not final_text:
            with state.lock:
                latest_partial = state.turn.latest_partial() if generation == state.turn_generation else ""
            final_text = latest_partial.strip()

        if not final_text:
            print("\nYou: [no speech recognized]\n", flush=True)
            # Option 3: resume TTS if this was a false interruption
            if had_interruption and _tts_has_resume_audio():
                logger.info("False interruption detected (no speech) — resuming TTS")
                print("[Resuming teacher speech]\n", flush=True)
                _prepare_for_tts_resume(state)
                _tts_resume_playback()
            return

        # Only create interruption_meta from final-text keywords when TTS was
        # actually playing (tts_text_snapshot non-empty).
        if (
            interruption_meta is None
            and tts_text_snapshot
            and _contains_interruption_keyword(final_text)
        ):
            interruption_meta = {
                "interrupted": True,
                "reason": f"final_text_keyword:{final_text}",
                "interrupted_assistant_text": tts_text_snapshot,
                "speech_ms_before_cancel": round(final_duration * 1000.0, 1),
            }

        word_count = len(final_text.split())
        speech_frames = sum(1 for f in is_speech_snapshot if f)
        speech_duration_sec = speech_frames * settings.audio_frame_ms / 1000.0
        silence_ratio = (
            1.0 - (speech_frames / len(is_speech_snapshot))
            if is_speech_snapshot else 1.0
        )

        is_hallucination = False
        hallucination_reason = ""

        if avg_no_speech_prob > 0.6 and word_count <= 3:
            is_hallucination = True
            hallucination_reason = f"no_speech_prob={avg_no_speech_prob:.2f}"
        elif silence_ratio > 0.55 and word_count <= 2 and speech_duration_sec < 0.5:
            is_hallucination = True
            hallucination_reason = (
                f"silence_ratio={silence_ratio:.2f}, speech={speech_duration_sec:.1f}s"
            )
        elif final_text.lower().strip().rstrip(".!?,") in HALLUCINATION_BLOCKLIST and silence_ratio > 0.45:
            is_hallucination = True
            hallucination_reason = f"blocklist_match, silence_ratio={silence_ratio:.2f}"

        if not is_hallucination and word_count >= 3:
            words_lower = final_text.lower().split()
            max_repeat = 1
            current_repeat = 1
            for i in range(1, len(words_lower)):
                prev = words_lower[i - 1].strip(".,!?;:'\"")
                curr = words_lower[i].strip(".,!?;:'\"")
                if curr == prev and curr:
                    current_repeat += 1
                    max_repeat = max(max_repeat, current_repeat)
                else:
                    current_repeat = 1
            # A real hallucination loop is mostly repetition (e.g. "no no no no no").
            # Real speech like "No no no I don't want to study motion" has meaningful
            # content after the repetition.  Only flag as hallucination if:
            #   - repeated 3+ times AND the sentence is SHORT (≤6 words), OR
            #   - repeated 5+ times AND repetitions are >60% of total words
            repeat_ratio = max_repeat / word_count if word_count > 0 else 0
            if max_repeat >= 5 and repeat_ratio > 0.6:
                is_hallucination = True
                hallucination_reason = f"repetition_loop, word repeated {max_repeat}x ({repeat_ratio:.0%} of text)"
            elif max_repeat >= 3 and word_count <= 6:
                is_hallucination = True
                hallucination_reason = f"repetition_loop, word repeated {max_repeat}x in short utterance ({word_count} words)"

        if is_hallucination:
            logger.info(
                "Whisper hallucination filtered | text=%r | reason=%s | words=%d | speech_sec=%.1f | no_speech_prob=%.2f",
                final_text, hallucination_reason, word_count, speech_duration_sec, avg_no_speech_prob,
            )
            print(f"\nYou: [{final_text}] [filtered: likely hallucination]\n", flush=True)
            # Option 3: resume TTS if this was a false interruption
            if had_interruption and _tts_has_resume_audio():
                logger.info("False interruption detected (hallucination) — resuming TTS")
                print("[Resuming teacher speech]\n", flush=True)
                _prepare_for_tts_resume(state)
                _tts_resume_playback()
            return

        # ── Real speech confirmed — clear resume state, this is a genuine turn ──
        _tts_clear_resume_state()

        # ── Auto-enroll speaker on first real turn ─────────────────
        if speaker_verifier is not None and not speaker_verifier.is_enrolled:
            enrolled = speaker_verifier.enroll(
                audio_int16=audio_bytes,
                sample_rate=settings.whisper_sample_rate,
            )
            if enrolled:
                print("  [Speaker profile enrolled]\n", flush=True)

        text_emotion = classify_text_emotion(final_text)

        prosody = extract_prosody(
            pcm_frames=pcm_frames_snapshot,
            is_speech_flags=is_speech_snapshot,
            transcript=final_text,
            sample_rate=settings.whisper_sample_rate,
            frame_ms=settings.audio_frame_ms,
        )

        try:
            ser_audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_emotion = ser_model.classify(
                ser_audio,
                sample_rate=settings.whisper_sample_rate,
            )
        except Exception as exc:
            logger.warning("SER model inference failed | error=%s", exc)
            audio_emotion = {"label": "neutral", "confidence": 0.5, "all_scores": {}}

        emotion_data = {
            "text_emotion": text_emotion,
            "audio_emotion": audio_emotion,
            "prosody": prosody.to_dict(),
        }

        logger.info(
            "Emotion extracted | text=%s (%.2f) | audio_ser=%s (%.2f) | speech_rate=%.1f sps | pause_ratio=%.2f | filled_pauses=%d | pitch=%.0f±%.0f Hz",
            text_emotion["label"], text_emotion["confidence"],
            audio_emotion["label"], audio_emotion["confidence"],
            prosody.speech_rate_syllables_per_sec, prosody.pause_ratio,
            prosody.filled_pause_count, prosody.pitch_mean_hz, prosody.pitch_std_hz,
        )

        print(" " * 160, end="\r")
        print(f"You: {final_text}", flush=True)
        print(
            f"  [text: {text_emotion['label']} | voice: {audio_emotion['label']} ({audio_emotion['confidence']:.0%}) | rate: {prosody.speech_rate_syllables_per_sec:.1f} sps | pauses: {prosody.pause_ratio:.0%}]",
            flush=True,
        )
        if interruption_meta:
            print(
                f"  [interruption: yes | reason: {interruption_meta.get('reason', 'unknown')}]",
                flush=True,
            )
        print()

        result = send_to_teacher(
            session_id=session_id,
            text=final_text,
            emotion_data=emotion_data,
            interruption_meta=interruption_meta,
        )
        teacher_text = result["text"]
        teacher_directive = result["directive"]

        print(f"Teacher: {teacher_text}\n", flush=True)

        if _tts_client is not None and teacher_text:
            if teacher_directive is not None:
                _tts_client.speak_with_emotion(
                    text=teacher_text,
                    emotion_data=teacher_directive,
                    session_id=session_id,
                )
            else:
                _tts_client.speak_neutral(teacher_text, session_id=session_id)

    finally:
        _tts_restore_playback()
        # _prepare_for_tts_resume already did reset + set the flag
        # for resume paths. For normal paths (teacher replied), we
        # still need to signal the main loop.
        with state.lock:
            if not state.needs_streamer_reset:
                state.needs_streamer_reset = True
            # Only reset if not already reset by _prepare_for_tts_resume
            if state.finalizing:
                pass  # will be cleared by state.reset() below
        state.reset()


def main() -> None:
    session_id = f"voice-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    streamer = MicrophoneVADStreamer()
    stt = STTService()
    turn_manager = TurnManager()
    state = LiveTranscriptState()

    print("Loading SER model...", end="", flush=True)
    ser_model = SERModel()
    print(" done.")

    # ── Speaker Verification ──────────────────────────────────────
    speaker_verifier: Optional[SpeakerVerificationService] = None
    if settings.speaker_verification_enabled:
        print("Loading speaker verification model...", end="", flush=True)
        speaker_verifier = SpeakerVerificationService(
            similarity_threshold=settings.speaker_verification_threshold,
            auto_update_profile=settings.speaker_verification_auto_update,
            device="cpu",
        )
        if speaker_verifier.is_available:
            print(" done.")
        else:
            print(" skipped (resemblyzer not installed).")
            speaker_verifier = None

    tts_status = "disabled (service not running)"
    if _tts_client is not None:
        max_retries, delay = 30, 2
        for attempt in range(1, max_retries + 1):
            _tts_client.invalidate_availability_cache()
            if _tts_client.is_available():
                tts_status = "enabled"
                break
            print(
                f"\r  Waiting for TTS service (attempt {attempt}/{max_retries})...",
                end="",
                flush=True,
            )
            time.sleep(delay)
        print()

    print("Voice chat streaming is ready.")
    print(f"Session ID: {session_id}")
    print(f"TTS: {tts_status}")
    sv_status = "enabled (will enroll on first speech)" if speaker_verifier else "disabled"
    print(f"Speaker verification: {sv_status}")
    print("Speak naturally. Emotion detection is active (text + prosody + SER model).")
    print("TTS-aware interruption is enabled (VAD + partial ASR + backchannel filter).")
    print(f"No-barge-in phase: {NO_BARGE_IN_MS}ms | Echo suppression: thresholds raised during TTS")
    print("Press Ctrl+C to stop.\n")

    try:
        for event in streamer.stream_events():
            # Check if finalize_turn (bg thread) requested a reset
            with state.lock:
                if state.needs_streamer_reset:
                    state.needs_streamer_reset = False
                    streamer.reset()

            if event.event_type == "speech_frame":
                maybe_handle_tts_interruption(
                    state=state,
                    event=event,
                    stt=stt,
                )

            if event.event_type == "speech_start":
                with state.lock:
                    if not state.turn.speech_active and not state.finalizing:
                        turn_manager.start_turn(state.turn)
                        print("\nListening...", flush=True)
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

                maybe_launch_partial_transcription(state, stt, turn_manager)

                if decision.action == "finalize":
                    finalize_turn(state, stt, turn_manager, session_id, streamer, ser_model, speaker_verifier)
                elif decision.action == "discard":
                    print("\nYou: [too short, ignored]\n", flush=True)
                    # Option 3: resume TTS if this discard was from a false interruption
                    with state.lock:
                        had_interruption = state.interruption.confirmed
                    if had_interruption and _tts_has_resume_audio():
                        logger.info("False interruption detected (discard in main loop) — resuming TTS")
                        print("[Resuming teacher speech]\n", flush=True)
                        # Reset state BEFORE blocking resume so interruption
                        # detection works during resumed playback
                        state.reset()
                        streamer.reset()
                        _tts_resume_playback()
                    else:
                        _tts_restore_playback()
                        streamer.reset()
                        state.reset()
                else:
                    latest = state.turn.latest_partial()
                    if latest and decision.reason == "soft_pause_resume_window":
                        print_live(latest + " …")
                continue

    except KeyboardInterrupt:
        print("\nGoodbye.")
    except Exception as exc:
        logger.exception("Voice chat failed")
        print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()