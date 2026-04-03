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
from app.services.llm_service import LLMService
from app.services.memory_card_service import MemoryCardService
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
_memory_card_service: Optional[MemoryCardService] = None
_summary_lock = threading.Lock()


def _sanitize_for_tts(text: str) -> str:
    """
    Convert any LaTeX / math notation in *text* into plain speakable English.

    Strategy (universal — handles whatever formula the LLM produces):
    1. Strip known LaTeX delimiters and extract the inner expression.
    2. Replace common math symbols / commands with spoken words.
    3. Clean up leftover backslashes, braces, and whitespace.

    The replacements are ordered from most-specific to least-specific so
    that multi-character sequences are handled before their sub-parts.
    """

    # ── 1. Unwrap LaTeX display / inline delimiters ──────────────────────
    # \[ ... \]  and  \( ... \)  →  just the inner expression
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r" \1 ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r" \1 ", text, flags=re.DOTALL)
    # $$ ... $$  and  $ ... $
    text = re.sub(r"\$\$\s*(.*?)\s*\$\$", r" \1 ", text, flags=re.DOTALL)
    text = re.sub(r"\$\s*(.*?)\s*\$", r" \1 ", text)

    # ── 2. Named LaTeX commands → spoken words ───────────────────────────
    replacements = [
        # --- fractions ---
        # \frac{A}{B}  →  "A over B"
        (re.compile(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}"), r"\1 over \2"),

        # --- superscripts / subscripts ---
        # ^{...}  →  "to the power of ..."   (multi-char exponent)
        (re.compile(r"\^\{([^}]+)\}"), r" to the power of \1"),
        # ^2  →  "squared",  ^3  →  "cubed",  ^n  →  "to the power of n"
        (re.compile(r"\^2\b"), " squared "),
        (re.compile(r"\^3\b"), " cubed "),
        (re.compile(r"\^([A-Za-z0-9])"), r" to the power of \1 "),
        # _{...}  and  _x  →  "sub ..."
        (re.compile(r"_\{([^}]+)\}"), r" sub \1 "),
        (re.compile(r"_([A-Za-z0-9])"), r" sub \1 "),

        # --- square / nth roots ---
        (re.compile(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^}]*)\}"), r"\1th root of \2"),
        (re.compile(r"\\sqrt\s*\{([^}]*)\}"), r"square root of \1"),
        (re.compile(r"\\sqrt\b"), "square root of"),

        # --- named functions ---
        (re.compile(r"\\sin\b"), "sine"),
        (re.compile(r"\\cos\b"), "cosine"),
        (re.compile(r"\\tan\b"), "tangent"),
        (re.compile(r"\\sec\b"), "secant"),
        (re.compile(r"\\csc\b"), "cosecant"),
        (re.compile(r"\\cot\b"), "cotangent"),
        (re.compile(r"\\arcsin\b"), "arc sine"),
        (re.compile(r"\\arccos\b"), "arc cosine"),
        (re.compile(r"\\arctan\b"), "arc tangent"),
        (re.compile(r"\\ln\b"), "natural log"),
        (re.compile(r"\\log\b"), "log"),
        (re.compile(r"\\exp\b"), "e to the power of"),
        (re.compile(r"\\lim\b"), "the limit"),
        (re.compile(r"\\sum\b"), "the sum"),
        (re.compile(r"\\prod\b"), "the product"),
        (re.compile(r"\\int\b"), "the integral of"),
        (re.compile(r"\\infty\b"), "infinity"),

        # --- Greek letters ---
        (re.compile(r"\\alpha\b"), "alpha"),
        (re.compile(r"\\beta\b"), "beta"),
        (re.compile(r"\\gamma\b"), "gamma"),
        (re.compile(r"\\delta\b"), "delta"),
        (re.compile(r"\\epsilon\b"), "epsilon"),
        (re.compile(r"\\varepsilon\b"), "epsilon"),
        (re.compile(r"\\zeta\b"), "zeta"),
        (re.compile(r"\\eta\b"), "eta"),
        (re.compile(r"\\theta\b"), "theta"),
        (re.compile(r"\\lambda\b"), "lambda"),
        (re.compile(r"\\mu\b"), "mu"),
        (re.compile(r"\\nu\b"), "nu"),
        (re.compile(r"\\xi\b"), "xi"),
        (re.compile(r"\\pi\b"), "pi"),
        (re.compile(r"\\rho\b"), "rho"),
        (re.compile(r"\\sigma\b"), "sigma"),
        (re.compile(r"\\tau\b"), "tau"),
        (re.compile(r"\\phi\b"), "phi"),
        (re.compile(r"\\chi\b"), "chi"),
        (re.compile(r"\\psi\b"), "psi"),
        (re.compile(r"\\omega\b"), "omega"),
        (re.compile(r"\\Omega\b"), "Omega"),
        (re.compile(r"\\Delta\b"), "Delta"),
        (re.compile(r"\\Sigma\b"), "Sigma"),
        (re.compile(r"\\Lambda\b"), "Lambda"),
        (re.compile(r"\\Gamma\b"), "Gamma"),
        (re.compile(r"\\Pi\b"), "Pi"),
        (re.compile(r"\\Theta\b"), "Theta"),

        # --- operators and relations ---
        (re.compile(r"\\times\b"), "times"),
        (re.compile(r"\\cdot\b"), "times"),
        (re.compile(r"\\div\b"), "divided by"),
        (re.compile(r"\\pm\b"), "plus or minus"),
        (re.compile(r"\\mp\b"), "minus or plus"),
        (re.compile(r"\\leq\b"), "less than or equal to"),
        (re.compile(r"\\geq\b"), "greater than or equal to"),
        (re.compile(r"\\neq\b"), "not equal to"),
        (re.compile(r"\\approx\b"), "approximately"),
        (re.compile(r"\\equiv\b"), "is equivalent to"),
        (re.compile(r"\\propto\b"), "is proportional to"),
        (re.compile(r"\\rightarrow\b"), "goes to"),
        (re.compile(r"\\leftarrow\b"), "comes from"),
        (re.compile(r"\\Rightarrow\b"), "implies"),
        (re.compile(r"\\Leftarrow\b"), "is implied by"),
        (re.compile(r"\\leftrightarrow\b"), "if and only if"),

        # --- sets ---
        (re.compile(r"\\in\b"), "in"),
        (re.compile(r"\\notin\b"), "not in"),
        (re.compile(r"\\subset\b"), "is a subset of"),
        (re.compile(r"\\cup\b"), "union"),
        (re.compile(r"\\cap\b"), "intersection"),
        (re.compile(r"\\emptyset\b"), "the empty set"),

        # --- misc common commands ---
        (re.compile(r"\\text\s*\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\mathrm\s*\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\mathbf\s*\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\vec\s*\{([^}]*)\}"), r"vector \1"),
        (re.compile(r"\\hat\s*\{([^}]*)\}"), r"\1 hat"),
        (re.compile(r"\\bar\s*\{([^}]*)\}"), r"\1 bar"),
        (re.compile(r"\\dot\s*\{([^}]*)\}"), r"\1 dot"),
        (re.compile(r"\\ddot\s*\{([^}]*)\}"), r"\1 double-dot"),
        (re.compile(r"\\overline\s*\{([^}]*)\}"), r"\1 bar"),
        (re.compile(r"\\left\b"), ""),
        (re.compile(r"\\right\b"), ""),
        (re.compile(r"\\cdots\b"), "and so on"),
        (re.compile(r"\\ldots\b"), "and so on"),
        (re.compile(r"\\partial\b"), "partial"),
        (re.compile(r"\\nabla\b"), "nabla"),
        (re.compile(r"\\hbar\b"), "h-bar"),
    ]

    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)

    # ── 3. Strip remaining braces and lone backslashes ───────────────────
    text = re.sub(r"\\[A-Za-z]+", "", text)   # any unknown \command
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")

    # ── 4. Normalise whitespace ───────────────────────────────────────────
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    return text


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
    global _recall_service, _emotion_state, _memory_card_service

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
    _memory_card_service = MemoryCardService(
        llm=_llm,
        memory=_memory,
        embedding_service=_embedding_service,
    )

    logger.info("Warming up model...")
    try:
        _llm.fg_client.chat(model=_llm.model, messages=[{"role": "user", "content": "hi"}])
    except Exception as exc:
        logger.warning("Model warmup failed | error=%s", exc)

    logger.info("Model warmed up. AI Teacher API is ready.")
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

    # ─────────────────────────────────────────────────────────────
    # 0) Cancel any in-flight memory card extraction so Gemma is
    #    free for THIS turn's teacher response.  The cancel signal
    #    is checked inside foreground_structured_chat(); generate()
    #    also sets it and waits for the fg_model_lock.
    # ─────────────────────────────────────────────────────────────
    _llm.cancel_memory_card_extraction()

    # Pull context early.
    conversation_summary = _memory.get_conversation_summary_for_prompt(session_id)
    history_messages = _memory.get_recent_history_for_prompt(session_id)

    llm_user_message = user_input

    # ─────────────────────────────────────────────────────────────
    # 1) Standard recall pipeline
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
    # 2) Emotion state tracking
    # ─────────────────────────────────────────────────────────────
    emotion_instruction = ""
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
    # 3) LLM generation (acquires fg_model_lock internally)
    # ─────────────────────────────────────────────────────────────
    full_response = _llm.generate(
        user_message=llm_user_message,
        history_messages=history_messages,
        conversation_summary=conversation_summary,
        recalled_memory=recalled_memory,
        recall_clarification_mode=recall_clarification_mode,
        recall_clarification_question=recall_clarification_question,
        fresh_teach_topic=fresh_teach_topic,
        emotion_instruction=emotion_instruction,
    )

    logger.info(
        "Teacher response generation completed | response_chars=%s",
        len(full_response),
    )

    # Strip LaTeX / math notation so the response is clean for TTS.
    full_response = _sanitize_for_tts(full_response)

    # Save actual spoken/user-visible turn, not the synthetic llm_user_message.
    _memory.save_message(session_id=session_id, role="user", content=user_input)
    _memory.save_message(session_id=session_id, role="assistant", content=full_response)
    logger.info("Current turn messages saved to Redis/SQLite.")

    # ─────────────────────────────────────────────────────────────
    # 4) NO background GPU tasks here.
    #    Both summary update and memory card extraction run AFTER
    #    TTS completes, triggered by /extract_memory_card from
    #    voice_chat_client.  This avoids GPU contention between
    #    Gemma (Ollama) and MeloTTS/OpenVoice (both on CUDA).
    # ─────────────────────────────────────────────────────────────

    return ChatResponse(response=full_response, directive=directive_dict)


class MemoryCardRequest(BaseModel):
    session_id: str


@app.post("/extract_memory_card")
def extract_memory_card(req: MemoryCardRequest) -> dict:
    """Trigger memory card extraction + summary update AFTER TTS finishes.

    Called by voice_chat_client from a fire-and-forget thread once
    TTS playback completes.  This avoids GPU contention between
    Gemma (Ollama) and MeloTTS/OpenVoice since both share the same
    CUDA device.

    Both tasks run sequentially in a single background thread:
      1. Summary update (Gemma via bg_client)
      2. Memory card extraction (Gemma via foreground_structured_chat)

    The cancellation mechanism still works: if a new /chat request
    arrives, it calls cancel_memory_card_extraction() which aborts
    the in-flight memory card extraction.
    """
    session_id = req.session_id.strip()
    if not session_id:
        return {"ok": False, "reason": "empty session_id"}

    def _do_post_tts_tasks() -> None:
        # ── 1. Summary update ────────────────────────────────────
        if _summary_lock.acquire(blocking=False):
            try:
                logger.info("Summary update started (post-TTS) | session_id=%s", session_id)
                _memory.update_older_conversation_summary(
                    session_id=session_id,
                    summary_service=_summary_service,
                )
                logger.info("Summary update completed (post-TTS) | session_id=%s", session_id)
            except Exception as exc:
                logger.warning("Summary update failed | session_id=%s | error=%s", session_id, exc)
            finally:
                _summary_lock.release()
        else:
            logger.info("Summary update skipped — previous update still running.")

        # ── 2. Memory card extraction ────────────────────────────
        try:
            logger.info("Memory card extraction triggered (post-TTS) | session_id=%s", session_id)
            _memory_card_service.extract_and_store_inline(session_id=session_id)
        except Exception as exc:
            logger.warning("Memory card extraction failed | session_id=%s | error=%s", session_id, exc)

    threading.Thread(target=_do_post_tts_tasks, daemon=True, name="post-tts-tasks").start()
    return {"ok": True}