"""
Memory Card Worker  (OPTIONAL — legacy fallback)
─────────────────────────────────────────────────
With the single-model strategy, memory card extraction now happens
INLINE inside the /chat endpoint (using the foreground Gemma 4b model
during TTS playback).  You do NOT need to run this worker anymore.

However, it still works as a safety-net:
  - If the inline extraction was cancelled (student interrupted quickly)
    the messages stay unprocessed in SQLite (memory_extracted = 0).
  - Running this worker will eventually pick them up and extract cards
    from the same Gemma model.

If you want belt-and-suspenders, run it.  Otherwise, skip it.

Usage:
    python memory_worker.py

Requires:
    - Ollama running on port 11434 (same instance as the teacher)
    - uvicorn app server running (for log forwarding)
"""

import logging
import threading
import time

import requests

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_card_service import MemoryCardService

# ── Logging (forward to uvicorn like terminal_chat.py) ─────────────

UVICORN_LOG_ENDPOINT = "http://127.0.0.1:8000/internal/log"

POLL_INTERVAL_SECONDS = 5


class UvicornHTTPHandler(logging.Handler):
    """Forwards log records to the FastAPI server so they appear in the uvicorn terminal."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "level": record.levelname,
                "name": record.name,
                "message": self.format(record),
            }
            t = threading.Thread(
                target=requests.post,
                args=(UVICORN_LOG_ENDPOINT,),
                kwargs={"json": payload, "timeout": 2},
                daemon=True,
            )
            t.start()
        except Exception:
            pass


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    # Forward to uvicorn terminal
    uvicorn_handler = UvicornHTTPHandler()
    uvicorn_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(uvicorn_handler)

    # Also print to this terminal so you can see progress here
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    root.addHandler(console_handler)


_setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Memory worker starting...")

    llm = LLMService()
    memory = ChatMemoryService()
    embedding_service = EmbeddingService(llm=llm)
    memory_card_service = MemoryCardService(
        llm=llm,
        memory=memory,
        embedding_service=embedding_service,
    )

    # Warm up the background model so it's loaded before polling starts.
    print("Loading background model into VRAM...", end="", flush=True)
    try:
        llm.bg_client.chat(model=llm.bg_model, messages=[{"role": "user", "content": "hi"}])
    except Exception as exc:
        logger.warning("Background model warmup failed | error=%s", exc)
    print(" done.")

    logger.info(
        "Memory worker ready | poll_interval=%ss | bg_model=%s",
        POLL_INTERVAL_SECONDS,
        llm.bg_model,
    )

    while True:
        try:
            messages = memory.get_unprocessed_turns()

            if not messages:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # All messages in this batch share the same session_id.
            session_id = messages[0].get("session_id", "")
            max_id = max(m["id"] for m in messages)

            logger.info(
                "Processing turn | session_id=%s | message_count=%s | max_id=%s",
                session_id,
                len(messages),
                max_id,
            )

            # Reuse the existing extraction logic from MemoryCardService.
            # We pass the messages directly instead of re-querying SQLite.
            memory_card_service.extract_and_store_memory_card_from_messages(
                session_id=session_id,
                messages=messages,
            )

            # Mark all messages up to max_id as processed.
            memory.mark_turn_as_extracted(up_to_message_id=max_id)

            logger.info(
                "Turn processed and marked | session_id=%s | max_id=%s",
                session_id,
                max_id,
            )

        except KeyboardInterrupt:
            logger.info("Memory worker stopped by user.")
            break
        except Exception as exc:
            logger.warning("Memory worker error: %s", exc, exc_info=True)
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()