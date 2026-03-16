from __future__ import annotations

import logging
from typing import List

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.structured_output import MemoryCardExtractionSchema

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_SYSTEM_PROMPT = """
You extract useful teaching memories for an AI teacher.

Default rule:
Create a memory card for most student-teacher exchanges.

Set should_create_memory = true if the exchange contains any useful learning content, such as:
- a topic or concept
- a question and answer
- an explanation
- an example or analogy
- a solved problem
- a fact
- a student confusion
- a follow-up on the same topic
- a comparison or relationship between ideas

Set should_create_memory = false only if the exchange is only:
- greeting
- thanks
- filler
- goodbye
- no real learning content

When unsure, create the memory card.

When creating memory:
- topic = short label
- snippet = short summary of what was taught
- retrieval_text = key facts, answers, examples, equations, names, dates, or ideas needed to recall it later
- confusion = student confusion if present
- helpful_example = example or analogy if present
- status = short learning status

Keep it short, clear, and factual.
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
        latest_messages = self.memory.get_latest_n_messages_from_sqlite(session_id, 4)
        if len(latest_messages) < 2:
            return

        formatted = self._format_messages(latest_messages)

        user_prompt = f"""
Latest messages:
{formatted}

Extract a memory card for this exchange.

Create memory (should_create_memory=true) if the student and teacher discussed ANY topic, concept, question, or idea.
Only skip (should_create_memory=false) if this is purely a greeting, thanks, or filler with no educational content.
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
            if self._looks_like_teaching_exchange(latest_messages):
                logger.info(
                    "LLM declined memory creation, but teaching-exchange heuristic triggered. Creating fallback memory card."
                )
                parsed = self._build_fallback_memory(latest_messages)
            else:
                logger.info("Memory extraction decided not to create memory card.")
                return

        retrieval_text = (parsed.retrieval_text or "").strip()
        if not retrieval_text:
            retrieval_text = self._build_retrieval_text(parsed)

        if not retrieval_text:
            logger.info("Memory extraction produced empty retrieval_text. Skipping memory card.")
            return

        embedding = self.embedding_service.embed_text(retrieval_text)
        if not embedding:
            logger.warning("Could not create embedding for memory card.")
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
            "Memory card stored successfully | topic=%r | status=%r",
            parsed.topic,
            parsed.status,
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

        # Student asking/exploring something
        user_question_cues = (
            "what is", "what are", "how", "why", "explain", "tell me",
            "describe", "define", "example", "difference between",
            "compare", "relation", "who is", "who was", "when did",
            "where", "can you", "show me", "help me", "teach me",
            # math cues (kept from before)
            "solve", "equation", "answer", "math", "calculate", "=",
            "find x", "what is the answer",
        )

        # Teacher actually teaching something
        assistant_teaching_cues = (
            "step", "example", "means", "is a", "is the", "refers to",
            "works by", "because", "therefore", "in other words",
            "for instance", "such as", "this is", "let's", "imagine",
            "think of", "consider", "summary",
            # math cues (kept from before)
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