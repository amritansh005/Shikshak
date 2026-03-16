from __future__ import annotations

import logging

from fastapi import FastAPI, File, UploadFile
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


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...), partial: bool = False) -> TranscriptionResponse:
    audio_bytes = await file.read()
    result = stt_service.transcribe_bytes(audio_bytes, partial=partial)
    return TranscriptionResponse(**result)
