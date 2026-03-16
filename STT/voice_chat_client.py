from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import requests

from app.config import settings
from app.services.realtime_vad import MicrophoneVADStreamer
from app.services.stt_service import STTService
from app.services.turn_manager import TurnManager, TurnState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class LiveTranscriptState:
    def __init__(self) -> None:
        self.turn = TurnState()
        self.last_partial_audio_seconds: float = 0.0
        self.partial_in_flight = False
        self.finalizing = False
        self.turn_generation = 0
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.turn.reset()
            self.last_partial_audio_seconds = 0.0
            self.partial_in_flight = False
            self.finalizing = False
            self.turn_generation += 1


def print_live(text: str) -> None:
    clipped = text[:160]
    print(f"You (live): {clipped}", end="\r", flush=True)


def send_to_teacher(session_id: str, text: str) -> str:
    response = requests.post(
        settings.teacher_chat_url,
        json={"message": text, "session_id": session_id},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def maybe_launch_partial_transcription(state: LiveTranscriptState, stt: STTService, turn_manager: TurnManager) -> None:
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


def finalize_turn(state: LiveTranscriptState, stt: STTService, turn_manager: TurnManager, session_id: str, streamer: MicrophoneVADStreamer) -> None:
    with state.lock:
        if state.finalizing:
            return
        state.finalizing = True
        audio_bytes = state.turn.snapshot_audio()
        final_duration = state.turn.total_audio_seconds
        generation = state.turn_generation

    try:
        if final_duration < settings.whisper_min_audio_ms / 1000.0:
            print("\nYou: [too short, ignored]\n", flush=True)
            return

        result = stt.transcribe_bytes(audio_bytes, partial=False)
        final_text = result["text"].strip()
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

        print(" " * 160, end="\r")
        print(f"You: {final_text}\n", flush=True)
        teacher_text = send_to_teacher(session_id, final_text)
        print(f"Teacher: {teacher_text}\n", flush=True)
    finally:
        streamer.reset()
        state.reset()


def main() -> None:
    session_id = f"voice-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    streamer = MicrophoneVADStreamer()
    stt = STTService()
    turn_manager = TurnManager()
    state = LiveTranscriptState()

    print("Voice chat streaming is ready.")
    print(f"Session ID: {session_id}")
    print("Speak naturally. The system now tolerates short thinking pauses.")
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
                    decision = turn_manager.evaluate(state.turn)

                if audio_seconds <= settings.max_partial_window_seconds:
                    maybe_launch_partial_transcription(state, stt, turn_manager)
                else:
                    maybe_launch_partial_transcription(state, stt, turn_manager)

                if decision.action == "finalize":
                    finalize_turn(state, stt, turn_manager, session_id, streamer)
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