from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.structured_output import RecallDecisionSchema

logger = logging.getLogger(__name__)

RECALL_DECISION_SYSTEM_PROMPT = """
You decide whether the student's current message needs exact older teaching memory.

Say recall is needed when the student is clearly asking for:
- an old example again
- the previous way of explanation
- something explained earlier
- a past analogy
- the same method as before

Do not trigger recall for normal new questions.

Also give the most likely topic in short form if possible.
Return only valid JSON matching the schema.
""".strip()


class RecallService:
    def __init__(
        self,
        llm,
        memory: ChatMemoryService,
        embedding_service: EmbeddingService,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.embedding_service = embedding_service

    def get_recalled_memory_for_turn(
        self,
        session_id: str,
        user_message: str,
        history_messages: List[Dict[str, str]],
        conversation_summary: str,
    ) -> Optional[Dict[str, str]]:
        decision = self._detect_recall_intent(
            user_message=user_message,
            history_messages=history_messages,
            conversation_summary=conversation_summary,
        )

        if not decision or not decision.recall_needed:
            logger.info("Recall not needed for this turn.")
            return None

        if self._recent_history_already_covers_request(user_message, history_messages):
            logger.info("Recall skipped because recent history already likely covers it.")
            return None

        cards = self.memory.get_memory_cards_for_session(session_id)
        if not cards:
            logger.info("No memory cards available for session.")
            return None

        likely_topic = (decision.likely_topic or "").strip().lower()

        filtered_cards = self._filter_cards_by_topic(cards, likely_topic)
        if not filtered_cards:
            filtered_cards = cards

        query_text = self._build_query_text(
            user_message=user_message,
            history_messages=history_messages,
            conversation_summary=conversation_summary,
            likely_topic=decision.likely_topic,
        )

        query_embedding = self.embedding_service.embed_text(query_text)
        if not query_embedding:
            logger.info("No query embedding available, recall skipped.")
            return None

        best_card = None
        best_score = -1.0

        for card in filtered_cards:
            score = self.embedding_service.cosine_similarity(
                query_embedding,
                card.get("embedding", []),
            )
            if score > best_score:
                best_score = score
                best_card = card

        if not best_card:
            logger.info("No best memory card found.")
            return None

        logger.info(
            "Recalled memory selected | memory_id=%s | topic=%s | score=%.4f",
            best_card.get("memory_id"),
            best_card.get("topic"),
            best_score,
        )

        return {
            "topic": best_card.get("topic", ""),
            "confusion": best_card.get("confusion", ""),
            "helpful_example": best_card.get("helpful_example", ""),
            "student_preference": best_card.get("student_preference", ""),
            "status": best_card.get("status", ""),
            "snippet": best_card.get("snippet", ""),
        }

    def _detect_recall_intent(
        self,
        user_message: str,
        history_messages: List[Dict[str, str]],
        conversation_summary: str,
    ) -> Optional[RecallDecisionSchema]:
        formatted_history = self._format_history(history_messages)

        user_prompt = f"""
Current student message:
{user_message}

Recent chat:
{formatted_history if formatted_history else "None"}

Older summary:
{conversation_summary.strip() if conversation_summary.strip() else "None"}

Decide whether exact older memory is needed for this turn.
""".strip()

        parsed = self.llm.structured_chat(
            system_prompt=RECALL_DECISION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_model=RecallDecisionSchema,
            temperature=0.1,
        )

        if parsed:
            return parsed

        lowered = user_message.lower()
        fallback = any(
            phrase in lowered
            for phrase in (
                "like before",
                "as before",
                "earlier",
                "previously",
                "same example",
                "same way",
                "again",
                "old example",
                "last time",
            )
        )

        return RecallDecisionSchema(
            recall_needed=fallback,
            recall_reason="fallback_phrase_match" if fallback else "",
            likely_topic="",
            wants_old_example=fallback,
            wants_old_explanation_style=fallback,
        )

    @staticmethod
    def _format_history(history_messages: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for msg in history_messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"Student: {content}")
            elif role == "assistant":
                lines.append(f"Teacher: {content}")
        return "\n".join(lines).strip()

    @staticmethod
    def _filter_cards_by_topic(
        cards: List[Dict[str, object]],
        likely_topic: str,
    ) -> List[Dict[str, object]]:
        if not likely_topic:
            return cards

        filtered: List[Dict[str, object]] = []
        for card in cards:
            topic = str(card.get("topic", "")).lower()
            retrieval_text = str(card.get("retrieval_text", "")).lower()
            if likely_topic in topic or likely_topic in retrieval_text:
                filtered.append(card)
        return filtered

    @staticmethod
    def _build_query_text(
        user_message: str,
        history_messages: List[Dict[str, str]],
        conversation_summary: str,
        likely_topic: str,
    ) -> str:
        parts: List[str] = [f"Current request: {user_message.strip()}"]

        if likely_topic.strip():
            parts.append(f"Likely topic: {likely_topic.strip()}")

        recent_user_msgs = [
            (m.get("content") or "").strip()
            for m in history_messages
            if m.get("role") == "user" and (m.get("content") or "").strip()
        ]
        if recent_user_msgs:
            parts.append(f"Recent student context: {' | '.join(recent_user_msgs[-2:])}")

        if conversation_summary.strip():
            parts.append(f"Older summary: {conversation_summary.strip()}")

        return "\n".join(parts).strip()

    @staticmethod
    def _recent_history_already_covers_request(
        user_message: str,
        history_messages: List[Dict[str, str]],
    ) -> bool:
        lowered = user_message.lower()
        if "again" not in lowered and "before" not in lowered and "earlier" not in lowered:
            return False

        combined_recent = " ".join(
            (m.get("content") or "").lower() for m in history_messages[-2:]
        )
        if not combined_recent.strip():
            return False

        cues = ("example", "analogy", "like this", "imagine", "for example")
        return any(cue in combined_recent for cue in cues)