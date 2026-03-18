import logging
import re
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.emotion_state_service import EmotionStateService
from app.services.interruption_state_service import InterruptionStateService
from app.services.llm_service import LLMService
from app.services.recall_service import RecallService
from app.services.summary_service import SummaryService

logger = logging.getLogger(__name__)
uvicorn_logger = logging.getLogger("uvicorn")

_llm: Optional[LLMService] = None
_memory: Optional[ChatMemoryService] = None
_summary_service: Optional[SummaryService] = None
_embedding_service: Optional[EmbeddingService] = None
_recall_service: Optional[RecallService] = None
_emotion_state: Optional[EmotionStateService] = None
_interruption_state: Optional[InterruptionStateService] = None
_summary_lock = threading.Lock()


def _has_explicit_recall_cue(message: str) -> bool:
    lowered = message.lower().strip()
    explicit_cues = (
        "again", "before", "earlier", "previously", "last time",
        "same as before", "same as earlier", "like before", "as before",
        "old example", "same example", "same way", "the example you gave",
        "the way you explained before", "repeat",
    )
    return any(cue in lowered for cue in explicit_cues)


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _llm, _memory, _summary_service, _embedding_service
    global _recall_service, _emotion_state, _interruption_state

    logger.info("Initialising shared services...")
    _llm = LLMService()
    _memory = ChatMemoryService()
    _summary_service = SummaryService(llm=_llm)
    _embedding_service = EmbeddingService(llm=_llm)
    _recall_service = RecallService(
        llm=_llm,
        memory=_memory,
        embedding_service=_embedding_service,
    )
    _emotion_state = EmotionStateService()
    _interruption_state = InterruptionStateService()

    logger.info("Warming up models...")
    try:
        _llm.fg_client.chat(model=_llm.model, messages=[{"role": "user", "content": "hi"}])
    except Exception as exc:
        logger.warning("Foreground model warmup failed | error=%s", exc)
    try:
        _llm.bg_client.chat(model=_llm.bg_model, messages=[{"role": "user", "content": "hi"}])
    except Exception as exc:
        logger.warning("Background model warmup failed | error=%s", exc)

    logger.info("Models warmed up. AI Teacher API is ready.")
    yield
    logger.info("Shutting down AI Teacher API.")


app = FastAPI(title="AI Teacher API", lifespan=lifespan)


class LogRecord(BaseModel):
    level: str
    name: str
    message: str


class ChatRequest(BaseModel):
    message: str
    session_id: str
    emotion: Optional[dict] = None
    interruption_meta: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    directive: Optional[dict] = None


@app.post("/internal/log")
def receive_log(record: LogRecord) -> dict:
    level = getattr(logging, record.level.upper(), logging.INFO)
    uvicorn_logger.log(level, "[terminal] %s | %s", record.name, record.message)
    return {"ok": True}


@app.get("/")
def root() -> dict:
    return {"message": "AI Teacher API is running"}


def _build_resume_question(topic: str) -> str:
    topic = topic.strip().rstrip(".")
    return f"\n\nShould I continue with {topic}, or would you like to talk about something else?"


# Regex to find the [PENDING_TOPIC: ...] tag the LLM appends on interruption turns.
_PENDING_TOPIC_RE = re.compile(
    r"\[PENDING_TOPIC:\s*(.+?)\]\s*$",
    re.IGNORECASE,
)


def _parse_and_strip_pending_topic(response: str) -> tuple:
    """
    Parse and remove the [PENDING_TOPIC: ...] tag from the LLM response.

    Returns:
        (clean_response, topic_or_none)

    topic_or_none is:
      - None  if no tag was found
      - ""    if the LLM wrote [PENDING_TOPIC: none]
      - str   the extracted topic otherwise
    """
    match = _PENDING_TOPIC_RE.search(response)
    if not match:
        return response, None

    raw_topic = match.group(1).strip()
    clean_response = response[: match.start()].rstrip()

    if raw_topic.lower() in ("none", "n/a", "nil", "nothing", "no topic", "greeting", "small talk", "small-talk"):
        return clean_response, ""

    return clean_response, raw_topic


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    user_input = req.message.strip()
    session_id = req.session_id.strip()

    if not user_input:
        return ChatResponse(response="", directive=None)

    logger.info(
        "New student message received | session_id=%s | text=%r",
        session_id, user_input,
    )

    # Pull context early.
    conversation_summary = _memory.get_conversation_summary_for_prompt(session_id)
    history_messages = _memory.get_recent_history_for_prompt(session_id)

    # ─────────────────────────────────────────────────────────────
    # 1) Interruption state / pending-topic control
    # ─────────────────────────────────────────────────────────────
    pending_state = _interruption_state.get_state(session_id)

    llm_user_message = user_input
    force_direct_response: Optional[str] = None

    if pending_state.waiting_for_resume_decision and pending_state.pending_topic:
        decision = _interruption_state.classify_resume_reply(user_input)

        logger.info(
            "Pending-topic resume decision | session_id=%s | pending_topic=%r | kind=%s | extracted_topic=%r",
            session_id,
            pending_state.pending_topic,
            decision["kind"],
            decision.get("topic"),
        )

        if decision["kind"] == "continue":
            topic = pending_state.pending_topic
            _interruption_state.clear(session_id)
            llm_user_message = (
                f"Please continue teaching this topic clearly and naturally from the beginning, "
                f"not from the middle of an interrupted sentence: {topic}"
            )

        elif decision["kind"] == "decline":
            _interruption_state.clear(session_id)
            force_direct_response = "Okay — what would you like to talk about now?"

        elif decision["kind"] == "new_topic":
            _interruption_state.clear(session_id)
            llm_user_message = decision["topic"] or user_input

        else:
            # Ambiguous reply while waiting for a resume decision.
            # Ask again rather than relying on a small model to infer too much.
            force_direct_response = (
                f"I still have {pending_state.pending_topic} pending. "
                f"Would you like me to continue with that, or would you like to switch to another topic?"
            )

    # If we already know the response, skip recall/LLM generation.
    if force_direct_response is not None:
        full_response = force_direct_response
        directive_dict = None

        # Save actual messages.
        _memory.save_message(session_id=session_id, role="user", content=user_input)
        _memory.save_message(session_id=session_id, role="assistant", content=full_response)
        logger.info("Direct interruption-state response returned without LLM generation.")

        _snap_session = session_id

        def _update_summary_background() -> None:
            if not _summary_lock.acquire(blocking=False):
                logger.info("Summary update skipped — previous update still running.")
                return
            try:
                logger.info("Older conversation summary update started.")
                _memory.update_older_conversation_summary(
                    session_id=_snap_session,
                    summary_service=_summary_service,
                )
                logger.info("Older conversation summary update completed.")
            except Exception as exc:
                logger.warning("Older conversation summary update failed | error=%s", exc)
            finally:
                _summary_lock.release()

        threading.Thread(target=_update_summary_background, daemon=True).start()
        return ChatResponse(response=full_response, directive=directive_dict)

    # ─────────────────────────────────────────────────────────────
    # 2) Standard recall pipeline
    # ─────────────────────────────────────────────────────────────
    recall_decision = _recall_service.get_recall_decision_for_turn(
        user_message=llm_user_message,
        history_messages=history_messages,
        conversation_summary=conversation_summary,
    )

    recalled_memory = None
    recall_clarification_mode = False
    recall_clarification_question = ""
    fresh_teach_topic = ""

    explicit_recall_cue_present = _has_explicit_recall_cue(llm_user_message)

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
            recalled_memory = _recall_service.get_recalled_memory_for_turn(
                session_id=session_id,
                user_message=llm_user_message,
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

    # ─────────────────────────────────────────────────────────────
    # 3) Emotion state tracking
    # ─────────────────────────────────────────────────────────────
    emotion_instruction = ""
    directive = None
    directive_dict = None

    if req.emotion:
        directive = _emotion_state.record_turn(session_id, req.emotion)
        emotion_instruction = directive.instruction or ""
        logger.info(
            "Emotion directive | state=%s | smoothed=%s | trend=%s | instruction_chars=%d",
            directive.teaching_state,
            directive.smoothed_state,
            directive.trend,
            len(emotion_instruction),
        )
        directive_dict = {
            "smoothed_state": directive.smoothed_state,
            "smoothed_confidence": directive.smoothed_confidence,
            "trend": directive.trend,
            "secondary_state": directive.smoothed_secondary_state,
            "secondary_confidence": directive.smoothed_secondary_confidence,
            "raw_text_label": directive.raw_text_label,
            "raw_audio_label": directive.raw_audio_label,
        }

    # ─────────────────────────────────────────────────────────────
    # 4) Normal LLM generation
    # ─────────────────────────────────────────────────────────────
    # Detect interruption early so we can pass context to the LLM.
    interruption_meta: Dict[str, Any] = req.interruption_meta or {}
    was_interruption = bool(interruption_meta.get("interrupted"))
    interrupted_assistant_text = (interruption_meta.get("interrupted_assistant_text") or "").strip()

    full_response = _llm.generate(
        user_message=llm_user_message,
        history_messages=history_messages,
        conversation_summary=conversation_summary,
        recalled_memory=recalled_memory,
        recall_clarification_mode=recall_clarification_mode,
        recall_clarification_question=recall_clarification_question,
        fresh_teach_topic=fresh_teach_topic,
        emotion_instruction=emotion_instruction,
        interruption_context=interrupted_assistant_text if was_interruption else "",
    )

    logger.info(
        "Teacher response generation completed | response_chars=%s",
        len(full_response),
    )

    # ─────────────────────────────────────────────────────────────
    # 5) If this user turn interrupted TTS, parse the LLM's
    #    [PENDING_TOPIC: ...] tag to decide whether to store a
    #    pending topic and append a resume question.
    # ─────────────────────────────────────────────────────────────
    if was_interruption:
        full_response, llm_topic = _parse_and_strip_pending_topic(full_response)

        if llm_topic is None:
            # LLM didn't produce the tag — fall back to not storing
            # a pending topic (safe default: no resume question).
            logger.info(
                "No [PENDING_TOPIC] tag found in LLM response | session_id=%s",
                session_id,
            )
        elif llm_topic == "":
            # LLM explicitly said "none" — greeting / small-talk, skip.
            logger.info(
                "LLM indicated interrupted content was greeting/small-talk | session_id=%s",
                session_id,
            )
        else:
            # Real topic — store it and append resume question.
            _interruption_state.mark_pending_topic(
                session_id=session_id,
                topic=llm_topic,
                interrupted_assistant_text=interrupted_assistant_text,
            )
            full_response = full_response.rstrip() + _build_resume_question(llm_topic)
            logger.info(
                "Pending topic marked from LLM tag | session_id=%s | topic=%r",
                session_id,
                llm_topic,
            )

    # Save actual spoken/user-visible turn, not the synthetic llm_user_message.
    _memory.save_message(session_id=session_id, role="user", content=user_input)
    _memory.save_message(session_id=session_id, role="assistant", content=full_response)
    logger.info("Current turn messages saved to Redis/SQLite.")

    _snap_session = session_id

    def _update_summary_background() -> None:
        if not _summary_lock.acquire(blocking=False):
            logger.info("Summary update skipped — previous update still running.")
            return
        try:
            logger.info("Older conversation summary update started.")
            _memory.update_older_conversation_summary(
                session_id=_snap_session,
                summary_service=_summary_service,
            )
            logger.info("Older conversation summary update completed.")
        except Exception as exc:
            logger.warning("Older conversation summary update failed | error=%s", exc)
        finally:
            _summary_lock.release()

    threading.Thread(target=_update_summary_background, daemon=True).start()

    return ChatResponse(response=full_response, directive=directive_dict)