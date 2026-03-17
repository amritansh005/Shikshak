from __future__ import annotations

import logging
import sys
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import requests

from app.config import settings
from app.services.emotion_service import (
    ProsodyFeatures,
    SERModel,
    classify_text_emotion,
    extract_prosody,
)
from app.services.realtime_vad import MicrophoneVADStreamer
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


class LiveTranscriptState:
    def __init__(self) -> None:
        self.turn = TurnState()
        self.last_partial_audio_seconds: float = 0.0
        self.partial_in_flight = False
        self.finalizing = False
        self.turn_generation = 0
        self.lock = threading.Lock()
        self.is_speech_flags: List[bool] = []

    def reset(self) -> None:
        with self.lock:
            self.turn.reset()
            self.last_partial_audio_seconds = 0.0
            self.partial_in_flight = False
            self.finalizing = False
            self.turn_generation += 1
            self.is_speech_flags.clear()


def print_live(text: str) -> None:
    clipped = text[:160]
    print(f"You (live): {clipped}", end="\r", flush=True)


def send_to_teacher(
    session_id: str,
    text: str,
    emotion_data: Dict | None = None,
) -> Dict:
    """
    POST to techer_llm /chat.

    Returns:
        {
            "text":      str,           # teacher response text
            "directive": dict | None,   # full TeachingDirective from EmotionStateService
        }
    """
    payload: Dict = {"message": text, "session_id": session_id}
    if emotion_data:
        payload["emotion"] = emotion_data

    response = requests.post(
        settings.teacher_chat_url,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "text": data.get("response", "").strip(),
        "directive": data.get("directive"),   # None if no emotion was sent
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


def finalize_turn(
    state: LiveTranscriptState,
    stt: STTService,
    turn_manager: TurnManager,
    session_id: str,
    streamer: MicrophoneVADStreamer,
    ser_model: SERModel,
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

    try:
        if final_duration < settings.whisper_min_audio_ms / 1000.0:
            print("\nYou: [too short, ignored]\n", flush=True)
            return

        result = stt.transcribe_bytes(audio_bytes, partial=False)
        final_text = result["text"].strip()
        avg_no_speech_prob = float(result.get("avg_no_speech_prob", 0.0))

        if not final_text:
            with state.lock:
                if generation == state.turn_generation:
                    latest_partial = state.turn.latest_partial()
                else:
                    latest_partial = ""
            final_text = latest_partial.strip()

        if not final_text:
            print("\nYou: [no speech recognized]\n", flush=True)
            return

        # ── Hallucination filter ──────────────────────────────────
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
            hallucination_reason = f"silence_ratio={silence_ratio:.2f}, speech={speech_duration_sec:.1f}s"
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
            if max_repeat >= 3:
                is_hallucination = True
                hallucination_reason = f"repetition_loop, word repeated {max_repeat}x"

        if is_hallucination:
            logger.info(
                "Whisper hallucination filtered | text=%r | reason=%s | words=%d | speech_sec=%.1f | no_speech_prob=%.2f",
                final_text, hallucination_reason, word_count, speech_duration_sec, avg_no_speech_prob,
            )
            print(f"\nYou: [{final_text}] [filtered: likely hallucination]\n", flush=True)
            return

        # ── Emotion extraction ────────────────────────────────────
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
            audio_emotion = ser_model.classify(ser_audio, sample_rate=settings.whisper_sample_rate)
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
        print()

        # ── Send to teacher, get response + directive ─────────────
        result = send_to_teacher(session_id, final_text, emotion_data)
        teacher_text = result["text"]
        teacher_directive = result["directive"]  # window-smoothed TeachingDirective dict

        print(f"Teacher: {teacher_text}\n", flush=True)

        # ── Speak teacher response with emotion-conditioned prosody ─
        if _tts_client is not None and teacher_text:
            if teacher_directive is not None:
                # Use the full smoothed directive from EmotionStateService
                _tts_client.speak_with_emotion(
                    text=teacher_text,
                    emotion_data=teacher_directive,
                    session_id=session_id,
                )
            else:
                # No emotion data was available (shouldn't happen in voice mode)
                _tts_client.speak_neutral(teacher_text, session_id=session_id)

    finally:
        streamer.reset()
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

    # Wait for TTS service with retries (it may still be loading the model)
    tts_status = "disabled (service not running)"
    if _tts_client is not None:
        max_retries, delay = 30, 2          # up to 60 s
        for attempt in range(1, max_retries + 1):
            _tts_client.invalidate_availability_cache()
            if _tts_client.is_available():
                tts_status = "enabled"
                break
            print(f"\r  Waiting for TTS service (attempt {attempt}/{max_retries})...", end="", flush=True)
            time.sleep(delay)
        print()                             # newline after progress
    print("Voice chat streaming is ready.")
    print(f"Session ID: {session_id}")
    print(f"TTS: {tts_status}")
    print("Speak naturally. Emotion detection is active (text + prosody + SER model).")
    print("Press Ctrl+C to stop.\n")

    try:
        for event in streamer.stream_events():
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
                    audio_seconds = turn_manager.append_frame(
                        state.turn,
                        event.pcm_bytes,
                        is_speech=event.is_speech,
                    )
                    state.is_speech_flags.append(event.is_speech)
                    decision = turn_manager.evaluate(state.turn)

                maybe_launch_partial_transcription(state, stt, turn_manager)

                if decision.action == "finalize":
                    finalize_turn(state, stt, turn_manager, session_id, streamer, ser_model)
                elif decision.action == "discard":
                    print("\nYou: [too short, ignored]\n", flush=True)
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