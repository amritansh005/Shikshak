import logging
import threading
from datetime import datetime

import requests

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
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


def _has_explicit_recall_cue(message: str) -> bool:
    lowered = message.lower().strip()
    explicit_cues = (
        "again",
        "before",
        "earlier",
        "previously",
        "last time",
        "same as before",
        "same as earlier",
        "like before",
        "as before",
        "old example",
        "same example",
        "same way",
        "the example you gave",
        "the way you explained before",
        "repeat",
    )
    return any(cue in lowered for cue in explicit_cues)


_setup_logging()
logger = logging.getLogger(__name__)

SESSION_ID = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def main() -> None:
    # ── Service initialisation ────────────────────────────────────
    # LLMService is created first because other services depend on
    # its bg_client (CPU Ollama instance) for background work.
    llm = LLMService()
    memory = ChatMemoryService()
    summary_service = SummaryService(llm=llm)
    embedding_service = EmbeddingService(llm=llm)
    recall_service = RecallService(
        llm=llm,
        memory=memory,
        embedding_service=embedding_service,
    )

    # Lock prevents overlapping runs of the summary background job
    # across consecutive fast turns.
    _summary_lock = threading.Lock()

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
            continue

        try:
            logger.info(
                "New student message received | session_id=%s | text=%r",
                SESSION_ID,
                user_input,
            )

            conversation_summary = memory.get_conversation_summary_for_prompt(SESSION_ID)
            history_messages = memory.get_recent_history_for_prompt(SESSION_ID)

            recall_decision = recall_service.get_recall_decision_for_turn(
                user_message=user_input,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
            )

            recalled_memory = None
            recall_clarification_mode = False
            recall_clarification_question = ""
            fresh_teach_topic = ""

            explicit_recall_cue_present = _has_explicit_recall_cue(user_input)

            if recall_decision.recall_needed and explicit_recall_cue_present:
                if (
                    recall_decision.needs_recall_clarification
                    or not recall_decision.topic_clear_for_recall
                ):
                    recall_clarification_mode = True
                    recall_clarification_question = (
                        recall_decision.clarification_question or ""
                    ).strip()
                    fresh_teach_topic = (
                        recall_decision.fresh_teach_topic
                        or recall_decision.likely_topic
                        or ""
                    ).strip()

                    logger.info(
                        "Recall clarification mode activated | likely_topic=%s | clarification_question=%r | fresh_teach_topic=%r",
                        recall_decision.likely_topic,
                        recall_clarification_question,
                        fresh_teach_topic,
                    )
                else:
                    recalled_memory = recall_service.get_recalled_memory_for_turn(
                        session_id=SESSION_ID,
                        user_message=user_input,
                        history_messages=history_messages,
                        conversation_summary=conversation_summary,
                    )
            elif recall_decision.recall_needed and not explicit_recall_cue_present:
                logger.info(
                    "Recall decision ignored because no explicit recall cue was found in user message."
                )

            logger.info(
                "Prompt context prepared | summary_chars=%s | recent_history_count=%s | recalled_memory=%s | recall_clarification_mode=%s",
                len(conversation_summary or ""),
                len(history_messages),
                bool(recalled_memory),
                recall_clarification_mode,
            )

            print("\nTeacher: ", end="", flush=True)

            logger.info("Teacher response generation started.")
            full_response = ""
            for token in llm.stream_generate(
                user_message=user_input,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
                recalled_memory=recalled_memory,
                recall_clarification_mode=recall_clarification_mode,
                recall_clarification_question=recall_clarification_question,
                fresh_teach_topic=fresh_teach_topic,
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

            # Snapshot session_id for background closures.
            _snapshot_session_id = SESSION_ID

            def _update_summary_background() -> None:
                if not _summary_lock.acquire(blocking=False):
                    logger.info(
                        "Summary update skipped — previous update still running."
                    )
                    return
                try:
                    logger.info("Older conversation summary update started.")
                    memory.update_older_conversation_summary(
                        session_id=_snapshot_session_id,
                        summary_service=summary_service,
                    )
                    logger.info("Older conversation summary update completed.")
                except Exception as exc:
                    logger.warning(
                        "Older conversation summary update failed | error=%s", exc
                    )
                finally:
                    _summary_lock.release()

            threading.Thread(target=_update_summary_background, daemon=True).start()

            print("\n\n")

        except Exception as e:
            logger.exception("Error during terminal chat loop: %s", e)
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()