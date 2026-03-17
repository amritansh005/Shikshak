from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.stt_service import STTService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="STT Service")
stt_service = STTService()


class TranscriptionResponse(BaseModel):
    text: str
    language: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "streaming-ready"}


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/webm", "audio/ogg", "audio/flac",
    "audio/mpeg", "audio/mp3",
    "application/octet-stream",  # raw PCM uploads
}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...), partial: bool = False) -> TranscriptionResponse:
    # 1. Content-type sanity check
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported content type: {ct}")

    # 2. Read with size cap (stream in chunks so a 2GB upload doesn't OOM)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 256)  # 256 KB at a time
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)

    # 3. Empty check
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload")

    # 4. int16 requires even byte length
    if len(audio_bytes) % 2 != 0:
        raise HTTPException(400, "Audio byte length is odd — corrupted or wrong format (expected 16-bit PCM)")

    result = stt_service.transcribe_bytes(audio_bytes, partial=partial)
    return TranscriptionResponse(**result)
