from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PendingTopicState:
    pending_topic: Optional[str] = None
    waiting_for_resume_decision: bool = False
    interrupted_assistant_text: str = ""


class InterruptionStateService:
    """
    Keeps lightweight pending-topic state per session.

    Purpose:
    - when the student interrupts the teacher mid-explanation,
      remember the unfinished parent topic
    - after answering the interruption, ask whether to continue
    - interpret the student's yes / no / switch-topic reply

    This is intentionally heuristic and lightweight so it works
    well with a small model like qwen2.5:3b.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[str, PendingTopicState] = {}

        self._affirmative_exact = {
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "continue",
            "go on",
            "carry on",
            "please continue",
            "yes continue",
            "continue it",
            "continue please",
            "haan",
            "ha",
            "yes please",
            "go ahead",
            "please go on",
        }

        self._negative_exact = {
            "no",
            "nope",
            "nah",
            "not now",
            "leave it",
            "skip it",
            "don't continue",
            "dont continue",
            "stop",
            "something else",
            "other topic",
            "talk about something else",
        }

        self._continue_prefixes = (
            "yes",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "continue",
            "go on",
            "carry on",
            "go ahead",
        )

        self._decline_prefixes = (
            "no",
            "nope",
            "nah",
            "not now",
            "leave it",
            "skip it",
        )

        self._meta_phrases_to_strip = (
            "should i continue with",
            "continue with",
            "can you continue with",
            "please continue with",
            "go back to",
            "return to",
            "continue teaching",
            "continue the topic",
            "talk about",
            "teach me",
            "explain",
            "explain about",
        )

    def get_state(self, session_id: str) -> PendingTopicState:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return PendingTopicState()
            return PendingTopicState(
                pending_topic=state.pending_topic,
                waiting_for_resume_decision=state.waiting_for_resume_decision,
                interrupted_assistant_text=state.interrupted_assistant_text,
            )

    def mark_pending_topic(
        self,
        session_id: str,
        topic: str,
        interrupted_assistant_text: str = "",
    ) -> None:
        topic = self._clean_topic(topic)
        if not topic:
            return

        with self._lock:
            self._states[session_id] = PendingTopicState(
                pending_topic=topic,
                waiting_for_resume_decision=True,
                interrupted_assistant_text=interrupted_assistant_text.strip(),
            )

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)

    def classify_resume_reply(self, user_input: str) -> Dict[str, Optional[str]]:
        """
        Classify the student's reply after the teacher asks:

        "Should I continue with X, or would you like to talk about something else?"

        Returns:
            {"kind": "continue" | "decline" | "new_topic" | "unclear", "topic": Optional[str]}
        """
        raw = user_input.strip()
        text = self._normalize(raw)

        if not text:
            return {"kind": "unclear", "topic": None}

        if text in self._affirmative_exact:
            return {"kind": "continue", "topic": None}

        if text in self._negative_exact:
            return {"kind": "decline", "topic": None}

        if self._starts_with_any(text, self._continue_prefixes):
            # Examples:
            # "yes"
            # "yes continue"
            # "okay continue"
            return {"kind": "continue", "topic": None}

        # Handle "no, explain friction" or "nah teach me force"
        decline_match = re.match(
            r"^(no|nope|nah|not now|leave it|skip it)\b[\s,.-]*(.*)",
            text,
        )
        if decline_match:
            tail = (decline_match.group(2) or "").strip()
            if tail:
                cleaned_topic = self._clean_topic_phrase(tail)
                if cleaned_topic:
                    return {"kind": "new_topic", "topic": cleaned_topic}
            return {"kind": "decline", "topic": None}

        # Handle explicit switches:
        # "teach me force"
        # "explain friction"
        # "lets talk about momentum"
        if self._looks_like_direct_topic_switch(text):
            cleaned_topic = self._clean_topic_phrase(raw)
            if cleaned_topic:
                return {"kind": "new_topic", "topic": cleaned_topic}

        # Single-word ambiguous replies.
        if text in {"hmm", "umm", "maybe", "not sure", "idk", "i don't know", "dont know"}:
            return {"kind": "unclear", "topic": None}

        # Default: if the student typed something meaningful,
        # treat it as a new topic instead of asking the tiny model to infer too much.
        cleaned_topic = self._clean_topic_phrase(raw)
        if cleaned_topic:
            return {"kind": "new_topic", "topic": cleaned_topic}

        return {"kind": "unclear", "topic": None}

    def infer_pending_topic(
        self,
        current_user_message: str,
        history_messages: List[Dict[str, Any]],
        conversation_summary: str,
        interrupted_assistant_text: str = "",
    ) -> str:
        """
        Infer the unfinished parent topic when the user interrupts.

        Priority:
        1. previous user message from recent history
        2. interrupted assistant text snapshot
        3. conversation summary tail
        4. fallback generic topic

        Returns empty string if nothing useful can be inferred (caller
        should skip storing a pending topic in that case).
        """
        current_norm = self._normalize(current_user_message)

        # Most useful heuristic: previous user question/topic before the interruption.
        for msg in reversed(history_messages or []):
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue

            if role == "user":
                content_norm = self._normalize(content)
                if content_norm and content_norm != current_norm:
                    cleaned = self._clean_topic(content)
                    if cleaned:
                        return cleaned

        # Second best: assistant text that was being spoken when interrupted.
        if interrupted_assistant_text.strip():
            cleaned = self._extract_topic_from_assistant_text(interrupted_assistant_text)
            if cleaned:
                return cleaned

        # Third fallback: tail of summary.
        if conversation_summary.strip():
            cleaned = self._clean_topic(conversation_summary[-240:])
            if cleaned:
                return cleaned

        # FIX (bug #2): return the generic fallback only here, at the top-level
        # caller, rather than baking it into _clean_topic.  This keeps _clean_topic
        # honest (empty means "nothing useful") while still giving the /chat handler
        # a usable string for the resume question.
        return "the previous topic"

    def _extract_topic_from_assistant_text(self, text: str) -> str:
        """
        Try to compress the interrupted assistant text into a usable topic.
        """
        cleaned = self._clean_topic(text)
        if not cleaned:
            return ""

        # If it is long explanatory text, reduce it to the first sentence or clause.
        sentence = re.split(r"[.!?]\s+", cleaned, maxsplit=1)[0].strip()
        if sentence:
            cleaned = sentence

        if len(cleaned) > 100:
            cleaned = cleaned[:100].rsplit(" ", 1)[0].strip()

        return cleaned

    def _looks_like_direct_topic_switch(self, text: str) -> bool:
        switch_patterns = (
            r"^(teach me)\b",
            r"^(explain)\b",
            r"^(explain about)\b",
            r"^(tell me about)\b",
            r"^(lets talk about)\b",
            r"^(let's talk about)\b",
            r"^(talk about)\b",
            r"^(i want to learn)\b",
            r"^(i want to study)\b",
            r"^(can we do)\b",
            r"^(can you teach)\b",
            r"^(can you explain)\b",
        )
        return any(re.match(pattern, text) for pattern in switch_patterns)

    def _clean_topic_phrase(self, text: str) -> str:
        cleaned = text.strip()

        for phrase in self._meta_phrases_to_strip:
            pattern = r"^" + re.escape(phrase) + r"\b[\s:,-]*"
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;-")
        if len(cleaned) > 120:
            cleaned = cleaned[:120].rsplit(" ", 1)[0].strip()

        return cleaned

    def _clean_topic(self, text: str) -> str:
        """
        General cleaner for topic strings extracted from user/assistant/history.

        FIX (bug #2): returns empty string when cleaning produces nothing,
        instead of the old "the previous topic" fallback.  This lets callers
        distinguish "we found a real topic" from "cleaning gave us nothing"
        and handle the fallback at the right level.
        """
        cleaned = text.strip()

        # Remove common prompt-like prefixes.
        cleaned = re.sub(
            r"^(please\s+)?(teach|explain|tell me about|can you explain|can you teach)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;-")

        # Prefer first sentence if text is long.
        cleaned = re.split(r"[.!?]\s+", cleaned, maxsplit=1)[0].strip()

        # Trim if too long.
        if len(cleaned) > 120:
            cleaned = cleaned[:120].rsplit(" ", 1)[0].strip()

        return cleaned

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
        return any(text == prefix or text.startswith(prefix + " ") for prefix in prefixes)