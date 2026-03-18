from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "small")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    whisper_beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    whisper_language: str = os.getenv("WHISPER_LANGUAGE", "en")
    whisper_task: str = os.getenv("WHISPER_TASK", "transcribe")
    whisper_min_audio_ms: int = int(os.getenv("WHISPER_MIN_AUDIO_MS", "700"))
    whisper_sample_rate: int = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
    whisper_channels: int = int(os.getenv("WHISPER_CHANNELS", "1"))

    vad_mode: int = int(os.getenv("VAD_MODE", "2"))
    audio_frame_ms: int = int(os.getenv("AUDIO_FRAME_MS", "30"))
    vad_pre_roll_ms: int = int(os.getenv("VAD_PRE_ROLL_MS", "300"))
    vad_start_trigger_ms: int = int(os.getenv("VAD_START_TRIGGER_MS", "90"))

    partial_min_audio_ms: int = int(os.getenv("PARTIAL_MIN_AUDIO_MS", "700"))
    partial_decode_interval_ms: int = int(os.getenv("PARTIAL_DECODE_INTERVAL_MS", "500"))
    max_partial_window_seconds: float = float(os.getenv("MAX_PARTIAL_WINDOW_SECONDS", "12"))

    turn_soft_silence_ms: int = int(os.getenv("TURN_SOFT_SILENCE_MS", "450"))
    turn_resume_window_ms: int = int(os.getenv("TURN_RESUME_WINDOW_MS", "1100"))
    turn_incomplete_hold_ms: int = int(os.getenv("TURN_INCOMPLETE_HOLD_MS", "1600"))
    turn_unstable_hold_ms: int = int(os.getenv("TURN_UNSTABLE_HOLD_MS", "1200"))
    turn_semantic_hold_ms: int = int(os.getenv("TURN_SEMANTIC_HOLD_MS", "1500"))
    turn_force_stable_finalize_ms: int = int(os.getenv("TURN_FORCE_STABLE_FINALIZE_MS", "1300"))
    turn_hard_silence_ms: int = int(os.getenv("TURN_HARD_SILENCE_MS", "2100"))
    turn_min_meaningful_words: int = int(os.getenv("TURN_MIN_MEANINGFUL_WORDS", "3"))
    turn_complete_sentence_min_words: int = int(os.getenv("TURN_COMPLETE_SENTENCE_MIN_WORDS", "5"))
    turn_partial_stability_ratio: float = float(os.getenv("TURN_PARTIAL_STABILITY_RATIO", "0.82"))
    turn_stable_wait_ms: int = int(os.getenv("TURN_STABLE_WAIT_MS", "280"))
    turn_completion_score_finalize: float = float(os.getenv("TURN_COMPLETION_SCORE_FINALIZE", "0.62"))

    teacher_chat_url: str = os.getenv("VOICE_CHAT_TEACHER_URL", "http://127.0.0.1:8000/chat")

    # ── Speaker Verification ──────────────────────────────────────
    speaker_verification_enabled: bool = os.getenv("SPEAKER_VERIFICATION_ENABLED", "true").lower() == "true"
    speaker_verification_threshold: float = float(os.getenv("SPEAKER_VERIFICATION_THRESHOLD", "0.75"))
    speaker_verification_auto_update: bool = os.getenv("SPEAKER_VERIFICATION_AUTO_UPDATE", "true").lower() == "true"


settings = Settings()