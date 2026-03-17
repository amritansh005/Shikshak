"""
TTS Service — FastAPI application

Endpoints:
  GET  /health                      → liveness probe
  GET  /metrics                     → request/latency/GPU stats
  POST /synthesize                  → full WAV response
  POST /synthesize/stream           → chunked streaming WAV (HTTP)
  WS   /ws/synthesize               → WebSocket streaming PCM
  GET  /voices                      → list available voices

Integration with Shikshak / Techer LLM:
  - POST /synthesize  accepts the full TeachingDirective emotion payload
    so it can be called directly from techer_llm's /chat handler
  - The voice_chat_client.py TTSClient calls /synthesize after each
    teacher response and plays the returned WAV locally

Emotion payload schema mirrors what EmotionStateService.record_turn() returns
as TeachingDirective, plus the raw text and session context.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.audio_cache import AudioCache
from app.services.metrics import TTSMetrics
from app.services.prosody_controller import resolve_prosody
from app.services.tts_engine import QwenTTSEngine, make_cache_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global singletons ─────────────────────────────────────────────────────────
_engine: Optional[QwenTTSEngine] = None
_cache: Optional[AudioCache] = None
_metrics: Optional[TTSMetrics] = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _engine, _cache, _metrics

    logger.info("TTS service starting | model=%s | device=%s", settings.tts_model, settings.tts_device)

    _metrics = TTSMetrics()
    _cache = AudioCache(
        redis_url=settings.redis_url,
        ttl_seconds=settings.cache_ttl_seconds,
        max_text_chars=settings.cache_max_text_chars,
        enabled=settings.cache_enabled,
    )
    _engine = QwenTTSEngine()

    logger.info(
        "TTS service ready | backend=%s | sample_rate=%d | cache=%s",
        _engine.backend,
        _engine.sample_rate,
        "enabled" if _cache.enabled else "disabled",
    )

    # ── Warmup: run a short synthesis in a background thread so it
    #    doesn't block startup or crash the server on OOM.  The first
    #    real request will benefit from warmed CUDA kernels.
    import threading

    def _warmup():
        try:
            warmup_prosody = resolve_prosody()       # neutral defaults
            logger.info("TTS warmup starting — first inference warms CUDA kernels")
            t0 = time.time()
            _engine.synthesize("Hello.", warmup_prosody, settings.default_voice)
            logger.info("TTS warmup done | elapsed=%.1fs", time.time() - t0)
        except Exception as exc:
            logger.warning("TTS warmup failed (non-fatal) | %s", exc)

    threading.Thread(target=_warmup, daemon=True).start()

    yield

    logger.info("TTS service shutting down.")


app = FastAPI(title="Shikshak TTS Service", version="1.0.0", lifespan=lifespan)


# ── Request / Response schemas ────────────────────────────────────────────────

class EmotionPayload(BaseModel):
    """
    Mirrors TeachingDirective from emotion_state_service.py.
    All fields are optional — the TTS service degrades gracefully if
    emotion data is not provided (falls back to neutral prosody).
    """
    smoothed_state: str = Field(default="neutral", description="Primary teaching state")
    smoothed_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    trend: str = Field(default="stable", description="escalating|de-escalating|recovering|stable")
    secondary_state: str = Field(default="neutral")
    secondary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    negative_pressure: float = Field(default=0.0)

    # Raw labels for logging
    raw_text_label: Optional[str] = None
    raw_audio_label: Optional[str] = None


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    voice: Optional[str] = Field(default=None, description="Voice name or path to .wav reference for cloning")
    emotion: Optional[EmotionPayload] = None
    use_cache: bool = Field(default=True)
    ssml_pauses: bool = Field(default=False, description="Inject <break> tags between sentences")


class SynthesizeResponse(BaseModel):
    session_id: Optional[str]
    voice: str
    sample_rate: int
    backend: str
    latency_ms: float
    cache_hit: bool
    resolved_state: str
    resolved_trend: str
    style_prompt: str


class VoiceInfo(BaseModel):
    id: str
    name: str
    gender: str
    language: str
    description: str


# ── Available voices ──────────────────────────────────────────────────────────
_VOICES = [
    VoiceInfo(id="default", name="Default", gender="female", language="en",
              description="Chatterbox built-in voice — expressive, clear"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_engine() -> QwenTTSEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialised")
    return _engine


def _resolve_emotion(emotion: Optional[EmotionPayload]):
    if emotion is None:
        return resolve_prosody()
    return resolve_prosody(
        smoothed_state=emotion.smoothed_state,
        smoothed_confidence=emotion.smoothed_confidence,
        trend=emotion.trend,
        secondary_state=emotion.secondary_state,
        secondary_confidence=emotion.secondary_confidence,
        negative_pressure=emotion.negative_pressure,
        rate_min=settings.rate_min,
        rate_max=settings.rate_max,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    engine = _get_engine()
    return {
        "status": "ok",
        "backend": engine.backend,
        "sample_rate": engine.sample_rate,
        "cache_enabled": _cache.enabled if _cache else False,
    }


@app.get("/metrics")
def metrics() -> dict:
    if _metrics is None:
        return {}
    return _metrics.snapshot()


@app.get("/voices", response_model=list[VoiceInfo])
def voices() -> list[VoiceInfo]:
    return _VOICES


@app.post("/synthesize", response_class=Response)
def synthesize(req: SynthesizeRequest) -> Response:
    """
    Full WAV synthesis.

    Returns: audio/wav bytes
    Headers:
      X-TTS-Latency-Ms       : synthesis latency
      X-TTS-Cache-Hit        : true | false
      X-TTS-Resolved-State   : emotion state used
      X-TTS-Backend          : model backend name
    """
    engine = _get_engine()
    _metrics.record_request_start()
    t0 = time.monotonic()
    cache_hit = False
    error_occurred = False

    voice = req.voice or settings.default_voice
    prosody = _resolve_emotion(req.emotion)

    logger.info(
        "TTS synthesize | session=%s | chars=%d | state=%s | trend=%s | rate=%.2f | voice=%s",
        req.session_id or "-",
        len(req.text),
        prosody.resolved_state,
        prosody.resolved_trend,
        prosody.rate_multiplier,
        voice,
    )

    try:
        # Cache check
        cache_key = make_cache_key(req.text, prosody.resolved_state, prosody.resolved_trend, voice)
        if req.use_cache and _cache:
            cached = _cache.get(cache_key)
            if cached:
                cache_hit = True
                latency_ms = (time.monotonic() - t0) * 1000
                _metrics.record_request_end(latency_ms, cache_hit=True)
                return Response(
                    content=cached,
                    media_type="audio/wav",
                    headers={
                        "X-TTS-Latency-Ms": str(round(latency_ms, 1)),
                        "X-TTS-Cache-Hit": "true",
                        "X-TTS-Resolved-State": prosody.resolved_state,
                        "X-TTS-Backend": engine.backend,
                    },
                )

        wav_bytes = engine.wav_bytes(req.text, prosody, voice)
        latency_ms = (time.monotonic() - t0) * 1000

        # Store in cache
        if req.use_cache and _cache:
            _cache.set(cache_key, wav_bytes, req.text)

        logger.info(
            "TTS synthesis complete | latency=%.0fms | bytes=%d | state=%s | backend=%s",
            latency_ms, len(wav_bytes), prosody.resolved_state, engine.backend,
        )
        _metrics.record_request_end(latency_ms)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-TTS-Latency-Ms": str(round(latency_ms, 1)),
                "X-TTS-Cache-Hit": "false",
                "X-TTS-Resolved-State": prosody.resolved_state,
                "X-TTS-Backend": engine.backend,
            },
        )

    except Exception as exc:
        error_occurred = True
        latency_ms = (time.monotonic() - t0) * 1000
        _metrics.record_request_end(latency_ms, error=True)
        logger.exception("TTS synthesis failed | session=%s | error=%s", req.session_id, exc)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}")


@app.post("/synthesize/stream")
async def synthesize_stream(req: SynthesizeRequest) -> StreamingResponse:
    """
    Chunked streaming synthesis (HTTP chunked transfer).

    Returns: audio/wav — interleaved WAV header + PCM chunks.
    Suitable for clients that can play PCM progressively.

    Note: First chunk may take slightly longer (model warm-up for first sentence).
    """
    engine = _get_engine()
    voice = req.voice or settings.default_voice
    prosody = _resolve_emotion(req.emotion)

    logger.info(
        "TTS stream | session=%s | chars=%d | state=%s",
        req.session_id or "-", len(req.text), prosody.resolved_state,
    )

    async def _generate():
        async for chunk in engine.synthesize_stream(
            req.text, prosody, voice, chunk_bytes=settings.stream_chunk_size
        ):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="audio/pcm;rate=24000;bits=16;channels=1",
        headers={
            "X-TTS-Resolved-State": prosody.resolved_state,
            "X-TTS-Backend": engine.backend,
        },
    )


@app.websocket("/ws/synthesize")
async def ws_synthesize(websocket: WebSocket):
    """
    WebSocket streaming TTS.

    Client sends JSON:
      {
        "text": "...",
        "session_id": "...",
        "voice": "default",
        "emotion": { ... EmotionPayload fields ... }
      }

    Server streams back:
      - Multiple binary frames: raw int16 PCM bytes (24kHz, mono)
      - Final text frame: JSON {"done": true, "latency_ms": float, "state": str}

    Designed for real-time voice playback on the client side.
    """
    await websocket.accept()
    engine = _get_engine()
    _metrics.record_request_start()
    t0 = time.monotonic()

    try:
        data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
    except asyncio.TimeoutError:
        await websocket.close(code=1008)
        _metrics.record_request_end(0.0, error=True)
        return

    try:
        req = SynthesizeRequest(**data)
    except Exception as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close(code=1003)
        _metrics.record_request_end(0.0, error=True)
        return

    voice = req.voice or settings.default_voice
    prosody = _resolve_emotion(req.emotion)

    logger.info(
        "TTS WS stream | session=%s | chars=%d | state=%s | voice=%s",
        req.session_id or "-", len(req.text), prosody.resolved_state, voice,
    )

    try:
        async for chunk in engine.synthesize_stream(
            req.text, prosody, voice, chunk_bytes=settings.stream_chunk_size
        ):
            await websocket.send_bytes(chunk)

        latency_ms = (time.monotonic() - t0) * 1000
        await websocket.send_json({
            "done": True,
            "latency_ms": round(latency_ms, 1),
            "state": prosody.resolved_state,
            "trend": prosody.resolved_trend,
            "backend": engine.backend,
        })
        _metrics.record_request_end(latency_ms)

    except WebSocketDisconnect:
        logger.info("TTS WS client disconnected | session=%s", req.session_id)
        _metrics.record_request_end((time.monotonic() - t0) * 1000)
    except Exception as exc:
        logger.exception("TTS WS error | session=%s | error=%s", req.session_id, exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
        _metrics.record_request_end((time.monotonic() - t0) * 1000, error=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass