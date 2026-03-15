from __future__ import annotations

import logging
from typing import List

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.structured_output import MemoryCardExtractionSchema

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_SYSTEM_PROMPT = """
You are a memory extractor for an AI teacher.

Goal:
Create a memory only if the latest teaching moment is useful for future teaching.

Create memory when there is at least one of these:
- a clear topic studied
- a real confusion
- a helpful example or analogy
- a clear student learning preference
- a resolved or unresolved learning state

Do not create memory for:
- greetings
- filler
- thanks
- small talk
- generic chat

Keep outputs short and factual.
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

Extract one useful memory card only if this latest exchange is worth remembering for future teaching.
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
            logger.info("Memory extraction decided not to create memory card.")
            return

        retrieval_text = (parsed.retrieval_text or "").strip()
        if not retrieval_text:
            retrieval_text = self._build_retrieval_text(parsed)

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
        return ". ".join(parts).strip()