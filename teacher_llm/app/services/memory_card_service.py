from __future__ import annotations

import logging
from typing import List

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.structured_output import MemoryCardExtractionSchema

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_SYSTEM_PROMPT = """
You extract useful teaching memories for an AI teacher.

─── WHEN TO CREATE A MEMORY CARD ───────────────────────────────────────────

Set should_create_memory = true if the exchange contains actual teaching content:
- a concept or fact was explained
- a question was answered with substance
- an example or analogy was given
- a problem was solved
- a student confusion was addressed
- a comparison or relationship between ideas was made
- a follow-up explanation on an ongoing topic

Set should_create_memory = false if the exchange is:
- a greeting only ("hey", "how are you", "good morning", etc.)
- thanks, filler, or small talk only
- goodbye only
- the student asking what to study and the teacher asking back — but no actual teaching happened yet
- a topic suggestion with no explanation (e.g. "can we talk about X?" / "sure, what area?")

Examples that must be FALSE:
- Student: "Hey how are you" / Teacher: "I'm well, how can I help?" → false
- Student: "Can we talk about history?" / Teacher: "Sure! What area?" → false
- Student: "Thanks!" / Teacher: "You're welcome!" → false

Examples that must be TRUE:
- Student: "What is Newton's third law?" / Teacher: "Every action has an equal and opposite reaction..." → true
- Student: "Who were the villains of the French Revolution?" / Teacher: "Louis XVI and Marie Antoinette were..." → true
- Student: "I'm confused about velocity" / Teacher: "Velocity is speed with direction..." → true

When unsure, prefer true over false only if real information was exchanged.

─── HOW TO FILL THE FIELDS ─────────────────────────────────────────────────

- topic = short label for the subject (e.g. "Newton's Third Law", "French Revolution villains")
- snippet = 1-2 sentence summary of what was actually taught
- retrieval_text = key facts, answers, names, dates, equations, or ideas needed to recall this later. Write as complete sentences, not raw conversation.
- confusion = student confusion if clearly present, else leave empty
- helpful_example = copy the example or analogy EXACTLY as the teacher said it, word for word. Do not paraphrase. If no example was used, leave empty.
- student_preference = student's preferred learning style if expressed, else leave empty
- status = one of: "introduced", "concept explained", "question answered", "in progress", "confused", "mastered"

Keep all fields short, clear, and factual.
Return only valid JSON matching the schema.
""".strip()


class MemoryCardService:
    def __init__(
        self,
        llm,
        memory: ChatMemoryService,
        embedding_service: EmbeddingService,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.embedding_service = embedding_service

    def extract_and_store_memory_card_for_latest_turn(self, session_id: str) -> None:
        """Called from the chat loop background thread (legacy path)."""
        latest_messages = self.memory.get_latest_n_messages_from_sqlite(session_id, 4)
        if len(latest_messages) < 2:
            return
        self._extract_and_store(session_id, latest_messages)

    def extract_and_store_memory_card_from_messages(
        self, session_id: str, messages: List[dict]
    ) -> None:
        """Called from memory_worker.py with pre-fetched messages."""
        if len(messages) < 2:
            return
        self._extract_and_store(session_id, messages)

    def extract_and_store_inline(self, session_id: str) -> None:
        """Inline extraction using the FOREGROUND model (Gemma 4b).

        Called from a background thread in the /chat endpoint right after
        the teacher response is generated.  Runs while TTS is playing so
        the GPU is otherwise idle.

        Uses llm.foreground_structured_chat() which:
          - acquires fg_model_lock (serialises with teacher generate)
          - checks _memory_card_cancel before/after Ollama call
          - aborts immediately if a new turn arrives

        The embedding step also checks cancellation so we don't waste
        time on DB writes when the model is needed urgently.
        """
        latest_messages = self.memory.get_latest_n_messages_from_sqlite(session_id, 4)
        if len(latest_messages) < 2:
            return

        formatted = self._format_messages(latest_messages)

        logger.info(
            "\n"
            "┌─ Inline Memory Extraction Started (fg model) ───────────\n"
            "│  session_id : %s\n"
            "│  messages   : %d\n"
            "│  exchange   :\n%s\n"
            "└─────────────────────────────────────────────────────────",
            session_id,
            len(latest_messages),
            "\n".join(f"│    {line}" for line in formatted.splitlines()),
        )

        user_prompt = f"""
Latest messages:
{formatted}

Extract a memory card for this exchange.

Create memory (should_create_memory=true) ONLY if real teaching content was exchanged — a concept explained, a question answered with substance, a fact given, or a confusion addressed.

Skip (should_create_memory=false) if this is a greeting, small talk, topic suggestion with no explanation, or filler with no actual learning content.
""".strip()

        # ── Use FOREGROUND model with cancellation support ───────────
        parsed = self.llm.foreground_structured_chat(
            system_prompt=MEMORY_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_model=MemoryCardExtractionSchema,
            temperature=0.1,
        )

        if not parsed:
            logger.info("Inline memory extraction returned nothing (possibly cancelled).")
            return

        if not parsed.should_create_memory:
            if self._looks_like_teaching_exchange(latest_messages):
                logger.info(
                    "LLM declined memory creation, but teaching-exchange heuristic triggered. Creating fallback memory card."
                )
                parsed = self._build_fallback_memory(latest_messages)
            else:
                logger.info(
                    "\n"
                    "┌─ Inline Memory Card Skipped ────────────────────────────\n"
                    "│  reason : LLM decided no educational content\n"
                    "└─────────────────────────────────────────────────────────"
                )
                # Still mark as extracted so memory_worker doesn't retry.
                max_id = max(m["id"] for m in latest_messages)
                self.memory.mark_turn_as_extracted(up_to_message_id=max_id)
                return

        logger.info(
            "\n"
            "┌─ Inline Memory Card Extracted ──────────────────────────\n"
            "│  topic          : %s\n"
            "│  status         : %s\n"
            "│  snippet        : %s\n"
            "└─────────────────────────────────────────────────────────",
            parsed.topic or "(empty)",
            parsed.status or "(empty)",
            (parsed.snippet or "(empty)")[:120],
        )

        retrieval_text = (parsed.retrieval_text or "").strip()
        if not retrieval_text:
            retrieval_text = self._build_retrieval_text(parsed)

        if not retrieval_text:
            logger.info("Inline memory card skipped — retrieval_text empty after rebuild.")
            return

        # ── Check cancellation before embedding (avoid wasting time) ─
        if self.llm._memory_card_cancel.is_set():
            logger.info("Inline memory card aborted before embedding — new turn arrived.")
            return

        embedding = self.embedding_service.embed_text(retrieval_text)
        if not embedding:
            logger.warning("Inline memory card failed — could not create embedding.")
            return

        # ── Check cancellation before DB writes ──────────────────────
        if self.llm._memory_card_cancel.is_set():
            logger.info("Inline memory card aborted before DB write — new turn arrived.")
            return

        self.memory.save_memory_card(
            session_id=session_id,
            topic=parsed.topic,
            confusion=parsed.confusion,
            helpful_example=parsed.helpful_example,
            student_preference=parsed.student_preference,
            status=parsed.status,
            snippet=parsed.snippet,
            retrieval_text=retrieval_text,
            embedding=embedding,
        )

        # Mark messages as extracted so memory_worker (if running as
        # fallback) doesn't double-process them.
        max_id = max(m["id"] for m in latest_messages)
        self.memory.mark_turn_as_extracted(up_to_message_id=max_id)

        logger.info(
            "\n"
            "┌─ Inline Memory Card Stored ✓ ───────────────────────────\n"
            "│  topic          : %s\n"
            "│  status         : %s\n"
            "│  retrieval_text : %s\n"
            "└─────────────────────────────────────────────────────────",
            parsed.topic or "(empty)",
            parsed.status or "(empty)",
            retrieval_text[:120],
        )

    def _extract_and_store(self, session_id: str, messages: List[dict]) -> None:
        """Shared extraction logic used by both entry points."""
        formatted = self._format_messages(messages)

        logger.info(
            "\n"
            "┌─ Memory Extraction Started ─────────────────────────────\n"
            "│  session_id : %s\n"
            "│  messages   : %d\n"
            "│  exchange   :\n%s\n"
            "└─────────────────────────────────────────────────────────",
            session_id,
            len(messages),
            "\n".join(f"│    {line}" for line in formatted.splitlines()),
        )

        user_prompt = f"""
Latest messages:
{formatted}

Extract a memory card for this exchange.

Create memory (should_create_memory=true) ONLY if real teaching content was exchanged — a concept explained, a question answered with substance, a fact given, or a confusion addressed.

Skip (should_create_memory=false) if this is a greeting, small talk, topic suggestion with no explanation, or filler with no actual learning content.
""".strip()

        parsed = self.llm.structured_chat(
            system_prompt=MEMORY_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_model=MemoryCardExtractionSchema,
            temperature=0.1,
        )

        if not parsed:
            logger.info("Memory extraction returned nothing.")
            return

        if not parsed.should_create_memory:
            if self._looks_like_teaching_exchange(messages):
                logger.info(
                    "LLM declined memory creation, but teaching-exchange heuristic triggered. Creating fallback memory card."
                )
                parsed = self._build_fallback_memory(messages)
            else:
                logger.info(
                    "\n"
                    "┌─ Memory Card Skipped ───────────────────────────────────\n"
                    "│  reason : LLM decided no educational content\n"
                    "└─────────────────────────────────────────────────────────"
                )
                return

        # ── Log the raw extracted card before any post-processing ────
        logger.info(
            "\n"
            "┌─ Memory Card Extracted ─────────────────────────────────\n"
            "│  should_create  : %s\n"
            "│  topic          : %s\n"
            "│  status         : %s\n"
            "│  snippet        : %s\n"
            "│  retrieval_text : %s\n"
            "│  confusion      : %s\n"
            "│  helpful_example: %s\n"
            "│  student_pref   : %s\n"
            "└─────────────────────────────────────────────────────────",
            parsed.should_create_memory,
            parsed.topic or "(empty)",
            parsed.status or "(empty)",
            (parsed.snippet or "(empty)")[:120],
            (parsed.retrieval_text or "(empty)")[:120],
            parsed.confusion or "(none)",
            (parsed.helpful_example or "(none)")[:120],
            parsed.student_preference or "(none)",
        )

        retrieval_text = (parsed.retrieval_text or "").strip()
        if not retrieval_text:
            logger.info("retrieval_text was empty — rebuilding from fields.")
            retrieval_text = self._build_retrieval_text(parsed)

        if not retrieval_text:
            logger.info(
                "\n"
                "┌─ Memory Card Skipped ───────────────────────────────────\n"
                "│  reason : retrieval_text empty after rebuild\n"
                "└─────────────────────────────────────────────────────────"
            )
            return

        embedding = self.embedding_service.embed_text(retrieval_text)
        if not embedding:
            logger.warning(
                "\n"
                "┌─ Memory Card Failed ────────────────────────────────────\n"
                "│  reason : could not create embedding\n"
                "│  topic  : %s\n"
                "└─────────────────────────────────────────────────────────",
                parsed.topic or "(empty)",
            )
            return

        self.memory.save_memory_card(
            session_id=session_id,
            topic=parsed.topic,
            confusion=parsed.confusion,
            helpful_example=parsed.helpful_example,
            student_preference=parsed.student_preference,
            status=parsed.status,
            snippet=parsed.snippet,
            retrieval_text=retrieval_text,
            embedding=embedding,
        )

        logger.info(
            "\n"
            "┌─ Memory Card Stored ✓ ──────────────────────────────────\n"
            "│  topic          : %s\n"
            "│  status         : %s\n"
            "│  snippet        : %s\n"
            "│  retrieval_text : %s\n"
            "│  confusion      : %s\n"
            "│  helpful_example: %s\n"
            "└─────────────────────────────────────────────────────────",
            parsed.topic or "(empty)",
            parsed.status or "(empty)",
            (parsed.snippet or "(empty)")[:120],
            retrieval_text[:120],
            parsed.confusion or "(none)",
            (parsed.helpful_example or "(none)")[:120],
        )

    @staticmethod
    def _format_messages(messages: List[dict]) -> str:
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"Student: {content}")
            elif role == "assistant":
                lines.append(f"Teacher: {content}")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_retrieval_text(parsed: MemoryCardExtractionSchema) -> str:
        parts: List[str] = []
        if parsed.topic:
            parts.append(f"Topic studied: {parsed.topic}")
        if parsed.confusion:
            parts.append(f"Confusion: {parsed.confusion}")
        if parsed.helpful_example:
            parts.append(f"Helpful example: {parsed.helpful_example}")
        if parsed.student_preference:
            parts.append(f"Student preference: {parsed.student_preference}")
        if parsed.status:
            parts.append(f"Status: {parsed.status}")
        if parsed.snippet:
            parts.append(f"Snippet: {parsed.snippet}")
        return ". ".join(parts).strip()

    @staticmethod
    def _looks_like_teaching_exchange(messages: List[dict]) -> bool:
        """Fallback heuristic: detect any substantive teaching exchange."""
        user_parts: List[str] = []
        assistant_parts: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip().lower()
            if not content:
                continue
            if role == "user":
                user_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)

        combined_user = " ".join(user_parts)
        combined_assistant = " ".join(assistant_parts)

        # Skip if student message is too short and looks like filler
        noise_phrases = (
            "hi", "hello", "hey", "thanks", "thank you", "ok", "bye",
            "good morning", "good night", "hmm", "cool", "nice",
        )
        stripped_user = combined_user.strip()
        if stripped_user in noise_phrases or len(stripped_user) < 4:
            return False

        # Skip if teacher response is too short to be real teaching
        # e.g. "Sure! What area of history?" is not teaching
        if len(combined_assistant.strip()) < 80:
            return False

        # Student asking/exploring something
        user_question_cues = (
            "what is", "what are", "how", "why", "explain", "tell me",
            "describe", "define", "example", "difference between",
            "compare", "relation", "who is", "who was", "when did",
            "where", "can you", "show me", "help me", "teach me",
            # math cues
            "solve", "equation", "answer", "math", "calculate", "=",
            "find x", "what is the answer",
        )

        # Teacher actually teaching something
        assistant_teaching_cues = (
            "step", "example", "means", "is a", "is the", "refers to",
            "works by", "because", "therefore", "in other words",
            "for instance", "such as", "this is", "let's", "imagine",
            "think of", "consider", "summary",
            # math cues
            "the solution", "the answer", "x =", "divide both sides",
            "simplify", "result",
        )

        has_user_signal = any(cue in combined_user for cue in user_question_cues)
        has_assistant_signal = any(
            cue in combined_assistant for cue in assistant_teaching_cues
        )

        # Also trigger if assistant response is substantial (150+ chars = real explanation)
        assistant_is_substantial = len(combined_assistant) > 150

        return has_user_signal and (has_assistant_signal or assistant_is_substantial)

    @staticmethod
    def _build_fallback_memory(
        messages: List[dict],
    ) -> MemoryCardExtractionSchema:
        """Build a memory card from raw messages when the LLM declines but heuristic triggers."""
        user_texts: List[str] = []
        assistant_texts: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                user_texts.append(content)
            elif role == "assistant":
                assistant_texts.append(content)

        combined_user = " ".join(user_texts).strip()
        combined_assistant = " ".join(assistant_texts).strip()

        # Try to derive a topic from the student's question
        topic = combined_user[:80] if combined_user else "teaching exchange"

        # Truncate snippet to keep it reasonable
        snippet_parts: List[str] = []
        if combined_user:
            snippet_parts.append(f"Student asked: {combined_user[:200]}")
        if combined_assistant:
            snippet_parts.append(f"Teacher explained: {combined_assistant[:300]}")
        snippet = " | ".join(snippet_parts).strip()

        retrieval_text_parts: List[str] = [
            f"Topic studied: {topic}",
            f"Student question: {combined_user[:300]}",
            f"Teacher answer: {combined_assistant[:500]}",
            "Status: concept explained",
        ]

        return MemoryCardExtractionSchema(
            should_create_memory=True,
            topic=topic,
            confusion="",
            helpful_example="",
            student_preference="",
            status="concept explained",
            snippet=snippet,
            retrieval_text=". ".join(part for part in retrieval_text_parts if part).strip(),
        )