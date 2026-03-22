"""
Pre-load heavy ML models on a background thread at app startup.
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PreloadedModels:
    """Container for pre-loaded model instances."""
    stt: Optional[object] = None
    ser_model: Optional[object] = None
    speaker_verifier: Optional[object] = None
    tts_client: Optional[object] = None
    tts_status: str = "disabled"
    sv_status: str = "disabled"
    ready: bool = False
    error: Optional[str] = None


_models = PreloadedModels()


def get_preloaded() -> PreloadedModels:
    return _models


def preload_models(on_progress=None) -> None:
    def report(msg: str) -> None:
        logger.info("Preload: %s", msg)
        if on_progress:
            try:
                on_progress(msg)
            except Exception:
                pass

    try:
        # ── Ensure bootstrap has run (idempotent guard) ──────────
        # main.py already calls setup_project_imports() before any
        # GUI imports, so the app namespace is correctly configured
        # with STT first.  However voice_chat_client.py inserts
        # tts_service into sys.path[0] at import time.  If we call
        # setup_project_imports() again here it nukes the cached
        # app.config module and re-resolves it — but now tts_service
        # sits ahead of STT on sys.path, so the WRONG Settings wins.
        #
        # Fix: only run bootstrap if it has never run (i.e. app.config
        # is not yet in sys.modules).  If it *is* already cached we
        # just reuse the correctly-resolved module.
        if "app.config" not in sys.modules:
            from gui_app.bootstrap import setup_project_imports
            setup_project_imports()

        from app.services.stt_service import STTService  # type: ignore
        from app.services.emotion_service import SERModel  # type: ignore
        from app.config import settings  # type: ignore

        try:
            from app.services.speaker_verification import SpeakerVerificationService  # type: ignore
        except Exception:
            SpeakerVerificationService = None

        try:
            from tts_client import TTSClient  # type: ignore
        except Exception:
            TTSClient = None

        # 1. STT (Whisper)
        report("Loading STT (Whisper)...")
        _models.stt = STTService()
        report("STT loaded")

        # 2. SER (emotion2vec)
        report("Loading emotion model...")
        _models.ser_model = SERModel()
        report("Emotion model loaded")

        # 3. Speaker verification
        if SpeakerVerificationService and getattr(settings, "speaker_verification_enabled", False):
            report("Loading speaker verification...")
            verifier = SpeakerVerificationService(
                similarity_threshold=settings.speaker_verification_threshold,
                auto_update_profile=settings.speaker_verification_auto_update,
                device="cpu",
            )
            if verifier.is_available:
                _models.speaker_verifier = verifier
                _models.sv_status = "enabled (will enroll on first speech)"
            else:
                _models.sv_status = "disabled (resemblyzer unavailable)"
        else:
            _models.sv_status = "disabled"
        report(f"Speaker verification: {_models.sv_status}")

        # 4. TTS client
        if TTSClient is not None:
            try:
                client = TTSClient(tts_service_url="http://127.0.0.1:5000", timeout=120)
                client.invalidate_availability_cache()
                if client.is_available():
                    _models.tts_client = client
                    _models.tts_status = "enabled"
                else:
                    _models.tts_status = "disabled (service not reachable)"
            except Exception as exc:
                logger.warning("TTS client init failed: %s", exc)
                _models.tts_status = "disabled"
        report(f"TTS: {_models.tts_status}")

        _models.ready = True
        report("All models preloaded — ready to start session")

    except Exception as exc:
        logger.exception("Preload failed")
        _models.error = str(exc)
        report(f"Preload error: {exc}")


def start_preload_thread(on_progress=None) -> threading.Thread:
    t = threading.Thread(
        target=preload_models,
        args=(on_progress,),
        daemon=True,
        name="model-preloader",
    )
    t.start()
    return t