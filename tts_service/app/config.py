from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── Model ────────────────────────────────────────────────────────
    tts_model: str = os.getenv("TTS_MODEL", "chatterbox")
    tts_device: str = os.getenv("TTS_DEVICE", "cuda")       # cuda | cpu
    tts_dtype: str = os.getenv("TTS_DTYPE", "bfloat16")       # float16 | float32 | bfloat16

    # ── Voice / Synthesis ───────────────────────────────────────────
    default_voice: str = os.getenv("TTS_DEFAULT_VOICE", "default")
    sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    audio_format: str = os.getenv("TTS_AUDIO_FORMAT", "wav")  # wav | pcm | mp3

    # ── Streaming ───────────────────────────────────────────────────
    stream_chunk_size: int = int(os.getenv("TTS_STREAM_CHUNK_SIZE", "4096"))  # bytes per WebSocket frame
    stream_sentence_boundary: bool = os.getenv("TTS_STREAM_SENTENCE_BOUNDARY", "true").lower() == "true"

    # ── Cache (Redis) ───────────────────────────────────────────────
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_enabled: bool = os.getenv("TTS_CACHE_ENABLED", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("TTS_CACHE_TTL_SECONDS", "3600"))
    cache_max_text_chars: int = int(os.getenv("TTS_CACHE_MAX_TEXT_CHARS", "200"))

    # ── Service ─────────────────────────────────────────────────────
    host: str = os.getenv("TTS_HOST", "0.0.0.0")
    port: int = int(os.getenv("TTS_PORT", "5000"))
    max_text_length: int = int(os.getenv("TTS_MAX_TEXT_LENGTH", "2000"))
    request_timeout_seconds: int = int(os.getenv("TTS_REQUEST_TIMEOUT_SECONDS", "60"))

    # ── Prosody limits ───────────────────────────────────────────────
    # Clamped speaking rate multiplier min/max
    rate_min: float = float(os.getenv("TTS_RATE_MIN", "0.75"))
    rate_max: float = float(os.getenv("TTS_RATE_MAX", "1.30"))

    # ── Integration ─────────────────────────────────────────────────
    teacher_service_url: str = os.getenv("TEACHER_SERVICE_URL", "http://127.0.0.1:8000")


settings = Settings()