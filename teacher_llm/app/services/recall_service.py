from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.services.chat_memory import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.structured_output import RecallDecisionSchema

logger = logging.getLogger(__name__)

RECALL_DECISION_SYSTEM_PROMPT = """
You decide whether the student's current message needs exact older teaching memory.

Very important rule:
Only trigger recall when the student explicitly refers to something from earlier.

Explicit earlier-reference cues include phrases like:
- again
- before
- earlier
- previously
- last time
- same as before
- like before
- the example you gave
- the way you explained before
- explain that again
- tell me that example again

Do NOT trigger recall for normal ongoing teaching or topic progression.

Normal teaching / continuation requests are not recall. This includes:
- starting a topic
- asking to learn a concept
- asking for more examples
- asking the next question in the lesson
- asking to explain a concept directly
- asking about a subtopic currently being discussed

Examples of normal teaching / continuation:
- let's study this topic
- teach me quantum mechanics
- help me understand algebra
- explain photosynthesis
- give 2 more examples
- what's the next law?
- what is friction?
- explain this example

Your job is to detect 3 things carefully:
1. Whether the student is explicitly asking to revisit something from earlier.
2. Whether the earlier topic is clear enough for safe memory retrieval.
3. Whether the teacher should ask for clarification instead of recalling memory.

Rules:
- If there is NO explicit earlier-reference cue, then:
  - recall_needed = false
  - topic_clear_for_recall = false
  - needs_recall_clarification = false

- If the student explicitly refers to something earlier, but the exact topic is vague or unclear:
  - recall_needed = true
  - topic_clear_for_recall = false
  - needs_recall_clarification = true

Examples of vague recall:
- again
- explain again
- say that again
- do it like before
- same as earlier
- that part again

Examples of clear recall:
- explain Newton's first law again
- tell the atom example again
- explain photosynthesis in the same way as before
- give the toy car example again

In clear recall cases:
- recall_needed = true
- topic_clear_for_recall = true
- needs_recall_clarification = false

Also:
- likely_topic should be a short topic name if it can be identified
- clarification_question should be short and teacher-like when clarification is needed
- fresh_teach_topic should contain the most likely topic to offer fresh teaching, if possible

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

        if decision.needs_recall_clarification or not decision.topic_clear_for_recall:
            logger.info(
                "Recall skipped because earlier topic is unclear | likely_topic=%s",
                decision.likely_topic,
            )
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

    def get_recall_decision_for_turn(
        self,
        user_message: str,
        history_messages: List[Dict[str, str]],
        conversation_summary: str,
    ) -> RecallDecisionSchema:
        decision = self._detect_recall_intent(
            user_message=user_message,
            history_messages=history_messages,
            conversation_summary=conversation_summary,
        )
        return decision or RecallDecisionSchema()

    def _detect_recall_intent(
        self,
        user_message: str,
        history_messages: List[Dict[str, str]],
        conversation_summary: str,
    ) -> Optional[RecallDecisionSchema]:
        lowered = user_message.lower().strip()

        if not self._has_explicit_recall_cue(lowered):
            return RecallDecisionSchema(
                recall_needed=False,
                recall_reason="no_explicit_recall_cue",
                likely_topic="",
                wants_old_example=False,
                wants_old_explanation_style=False,
                topic_clear_for_recall=False,
                needs_recall_clarification=False,
                clarification_question="",
                fresh_teach_topic="",
            )

        formatted_history = self._format_history(history_messages)

        user_prompt = f"""
Current student message:
{user_message}

Recent chat:
{formatted_history if formatted_history else "None"}

Older summary:
{conversation_summary.strip() if conversation_summary.strip() else "None"}

Decide:
1. Is the student explicitly asking for older teaching memory?
2. If yes, is the earlier topic clear enough for safe recall?
3. If not clear, should the teacher ask for clarification and offer fresh teaching?

Be strict.
If the student is only continuing the lesson normally, then recall is not needed.
""".strip()

        parsed = self.llm.structured_chat(
            system_prompt=RECALL_DECISION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_model=RecallDecisionSchema,
            temperature=0.1,
        )

        if parsed:
            if not parsed.recall_needed:
                parsed.topic_clear_for_recall = False
                parsed.needs_recall_clarification = False
                parsed.clarification_question = parsed.clarification_question or ""
                return parsed

            return parsed

        strong_clear_recall_phrases = (
            "explain that again",
            "tell me that again",
            "say that again",
            "give that example again",
            "the example you gave",
            "the way you explained before",
            "same example",
            "same way",
            "like before",
            "as before",
            "last time",
            "previously",
            "earlier",
            "same as before",
            "same as earlier",
        )

        vague_recall_phrases = (
            "again",
            "explain again",
            "do it again",
            "that again",
            "that part again",
        )

        if lowered in vague_recall_phrases:
            return RecallDecisionSchema(
                recall_needed=True,
                recall_reason="fallback_vague_recall_phrase_match",
                likely_topic="",
                wants_old_example=False,
                wants_old_explanation_style=False,
                topic_clear_for_recall=False,
                needs_recall_clarification=True,
                clarification_question=(
                    "I’m not fully sure which earlier part you mean. Do you want the concept, example, or steps again?"
                ),
                fresh_teach_topic="",
            )

        if any(phrase in lowered for phrase in strong_clear_recall_phrases):
            return RecallDecisionSchema(
                recall_needed=True,
                recall_reason="fallback_explicit_recall_phrase_match",
                likely_topic="",
                wants_old_example=True,
                wants_old_explanation_style=True,
                topic_clear_for_recall=False,
                needs_recall_clarification=True,
                clarification_question=(
                    "Which exact earlier part do you want again? The concept, example, or steps?"
                ),
                fresh_teach_topic="",
            )

        return RecallDecisionSchema(
            recall_needed=False,
            recall_reason="fallback_no_safe_recall_match",
            likely_topic="",
            wants_old_example=False,
            wants_old_explanation_style=False,
            topic_clear_for_recall=False,
            needs_recall_clarification=False,
            clarification_question="",
            fresh_teach_topic="",
        )

    @staticmethod
    def _has_explicit_recall_cue(lowered_message: str) -> bool:
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
        return any(cue in lowered_message for cue in explicit_cues)

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

        explicit_recall_cues = (
            "before",
            "earlier",
            "last time",
            "like before",
            "same example",
            "same way",
            "previously",
            "again",
            "repeat",
        )

        if not any(cue in lowered for cue in explicit_recall_cues):
            return False

        combined_recent = " ".join(
            (m.get("content") or "").lower() for m in history_messages[-2:]
        )
        if not combined_recent.strip():
            return False

        cues = ("example", "analogy", "like this", "imagine", "for example")
        return any(cue in combined_recent for cue in cues)