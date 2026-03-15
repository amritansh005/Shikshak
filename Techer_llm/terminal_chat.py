import logging
import threading
from datetime import datetime

import requests

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_card_service import MemoryCardService
from app.services.recall_service import RecallService
from app.services.summary_service import SummaryService

UVICORN_LOG_ENDPOINT = "http://127.0.0.1:8000/internal/log"


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

    handler = UvicornHTTPHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)


_setup_logging()
logger = logging.getLogger(__name__)

SESSION_ID = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def main() -> None:
    llm = LLMService()
    memory = ChatMemoryService()
    summary_service = SummaryService()
    embedding_service = EmbeddingService()
    memory_card_service = MemoryCardService(
        llm=llm,
        memory=memory,
        embedding_service=embedding_service,
    )
    recall_service = RecallService(
        llm=llm,
        memory=memory,
        embedding_service=embedding_service,
    )

    print("AI Teacher is ready.")
    print(f"Session ID: {SESSION_ID}")
    print("Type 'exit' to stop.\n")
    print(
        f"Redis recent buffer: {memory.max_recent_messages} messages | "
        f"Prompt history sent to model: {memory.max_prompt_history_messages} messages"
    )
    print(
        f"Redis available: {'Yes' if memory.is_redis_available() else 'No (SQLite fallback active)'}"
    )
    print(
        f"Memory card Redis primary: {'Yes' if memory.is_redis_available() else 'No (SQLite fallback active)'}\n"
    )

    logger.info(
        "AI Teacher started | redis_available=%s | max_recent_messages=%s | max_prompt_history_messages=%s",
        memory.is_redis_available(),
        memory.max_recent_messages,
        memory.max_prompt_history_messages,
    )

    while True:
        user_input = input("Student: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            logger.info("Terminal chat stopped by user.")
            print("Goodbye.")
            break

        if not user_input:
            continue  # Silent re-prompt — avoids printing a message that looks like the teacher speaking

        try:
            logger.info(
                "New student message received | session_id=%s | text=%r",
                SESSION_ID,
                user_input,
            )

            conversation_summary = memory.get_conversation_summary_for_prompt(SESSION_ID)
            history_messages = memory.get_recent_history_for_prompt(SESSION_ID)

            recalled_memory = recall_service.get_recalled_memory_for_turn(
                session_id=SESSION_ID,
                user_message=user_input,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
            )

            logger.info(
                "Prompt context prepared | summary_chars=%s | recent_history_count=%s | recalled_memory=%s",
                len(conversation_summary or ""),
                len(history_messages),
                bool(recalled_memory),
            )

            print("\nTeacher: ", end="", flush=True)

            logger.info("Teacher response generation started.")
            full_response = ""
            for token in llm.stream_generate(
                user_message=user_input,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
                recalled_memory=recalled_memory,
            ):
                print(token, end="", flush=True)
                full_response += token

            full_response = full_response.strip()

            logger.info(
                "Teacher response generation completed | response_chars=%s",
                len(full_response),
            )

            memory.save_message(
                session_id=SESSION_ID,
                role="user",
                content=user_input,
            )
            memory.save_message(
                session_id=SESSION_ID,
                role="assistant",
                content=full_response,
            )

            logger.info("Current turn messages saved to Redis/SQLite.")

            logger.info("Older conversation summary update started.")
            memory.update_older_conversation_summary(
                session_id=SESSION_ID,
                summary_service=summary_service,
            )
            logger.info("Older conversation summary update completed.")

            logger.info("Memory card extraction started.")
            memory_card_service.extract_and_store_memory_card_for_latest_turn(
                session_id=SESSION_ID,
            )
            logger.info("Memory card extraction completed.")

            print("\n\n")

        except Exception as e:
            logger.exception("Error during terminal chat loop: %s", e)
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()