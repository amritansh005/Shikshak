from __future__ import annotations

import json
import logging
import os
from typing import Dict, Iterator, List, Optional, Type

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

# Two-instance Ollama strategy
# ─────────────────────────────
# We run TWO separate Ollama processes on different ports:
#
#   Instance 1 (GPU, port 11434 — default):
#     • stream_generate() / generate() — foreground teacher responses.
#     • Runs qwen2.5:3b on GPU for fast, low-latency answers.
#
#   Instance 2 (CPU, port 11435):
#     • structured_chat() — background tasks like memory-card extraction,
#       recall decisions, summary generation, etc.
#     • Runs gemma2:2b on CPU.
#
# Memory card extraction is handled by a SEPARATE worker process
# (memory_worker.py) that reads unprocessed turns from SQLite and
# sends them to Instance 2. This means the chat loop never skips
# memory cards, no matter how fast the student types.

AI_TEACHER_SYSTEM_PROMPT = """
You are an excellent teacher.

You can teach many subjects such as science, mathematics, history, arts, and general knowledge.

Your goal is to help students truly understand ideas, not just memorize answers.

Teaching behavior:

- Explain ideas in simple language.
- Teach step by step like a good teacher.
- Keep explanations clear and not too long.
- Focus on the key idea first, then add details if needed.
- Use examples or analogies when they help understanding.
- If the student asks for more examples, give multiple short examples instead of repeating the same explanation.
- If the student is confused, explain again in a simpler way.
- If you do not know something, say you do not know.

Conversation style:

- Talk like a friendly and supportive teacher.
- Be natural, not like a textbook.
- Do NOT ask a follow-up question every time.
- Ask questions only when it genuinely helps learning.

When the student asks a question:

1. Explain the concept clearly.
2. Give a simple example or analogy if useful.
3. Give a short summary in 1–2 lines.

If the student says something vague like:
"I want to study physics" or "teach me math"

Do NOT start a random lesson.  
Instead ask what specific topic they want to learn and suggest 2–3 possible topics.

Emotion-aware teaching:

You may receive a note about the student's current emotional state.
When you do, adapt your teaching style naturally:
- Do NOT announce the student's emotion. Never say "I can see you are frustrated".
- Do NOT label or name the emotion in your response.
- Instead, silently adjust your tone, pace, and approach based on the instruction.
- If the student is struggling, be warmer and simpler without pointing it out.
- If the student is confident, you can move faster without over-explaining.
- The goal is for the student to feel understood, not analyzed.

Your mission is to make learning clear, simple, and enjoyable.
""".strip()


class LLMService:
    def __init__(self) -> None:
        # ── Foreground (GPU) ──────────────────────────────────────────
        self.model = os.getenv("LLM_MODEL", "qwen2.5:3b")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        fg_host = os.getenv("OLLAMA_FG_HOST", "http://127.0.0.1:11434")
        self.fg_client = ollama.Client(host=fg_host)

        # ── Background (CPU) ─────────────────────────────────────────
        self.bg_model = os.getenv("BG_LLM_MODEL", "gemma2:2b")
        bg_host = os.getenv("OLLAMA_BG_HOST", "http://127.0.0.1:11435")
        self.bg_client = ollama.Client(host=bg_host)

        logger.info(
            "LLMService initialised | fg_model=%s | fg_host=%s | bg_model=%s | bg_host=%s | temperature=%s | streaming=%s",
            self.model,
            fg_host,
            self.bg_model,
            bg_host,
            self.temperature,
            self.enable_streaming,
        )

    def build_messages(
        self,
        user_message: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
        recalled_memory: Optional[Dict[str, str]] = None,
        recall_clarification_mode: bool = False,
        recall_clarification_question: str = "",
        fresh_teach_topic: str = "",
        emotion_instruction: str = "",
        interruption_context: str = "",
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": AI_TEACHER_SYSTEM_PROMPT},
        ]

        if conversation_summary.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Older conversation summary:\n"
                        f"{conversation_summary.strip()}\n\n"
                        "Use this only as background. Give priority to recent chat and current student message."
                    ),
                }
            )

        # ── Emotion-aware teaching directive ─────────────────────
        if emotion_instruction.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Current student emotional state note:\n"
                        f"{emotion_instruction.strip()}\n\n"
                        "Adapt your teaching style based on this. "
                        "Do not mention, label, or announce the student's emotion in your response."
                    ),
                }
            )

        if recall_clarification_mode:
            clarification_parts: List[str] = [
                "The student seems to be referring to something from earlier, but the exact earlier topic is unclear.",
                "Do not pretend that you clearly remember the earlier part.",
                "Do not use or invent recalled memory.",
                "Respond naturally like a teacher.",
            ]

            if recall_clarification_question.strip():
                clarification_parts.append(
                    f"Helpful clarification question: {recall_clarification_question.strip()}"
                )

            if fresh_teach_topic.strip():
                clarification_parts.append(
                    f"If helpful, offer to teach this topic fresh from the beginning: {fresh_teach_topic.strip()}"
                )
            else:
                clarification_parts.append(
                    "If helpful, offer to teach the topic fresh from the beginning."
                )

            clarification_parts.append("Keep the reply short, honest, and supportive.")

            messages.append(
                {
                    "role": "system",
                    "content": "\n".join(clarification_parts).strip(),
                }
            )

        if recalled_memory and not recall_clarification_mode:
            memory_lines: List[str] = []
            if recalled_memory.get("topic"):
                memory_lines.append(f"Topic: {recalled_memory['topic']}")
            if recalled_memory.get("confusion"):
                memory_lines.append(f"Past confusion: {recalled_memory['confusion']}")
            if recalled_memory.get("helpful_example"):
                memory_lines.append(
                    f"Helpful past example: {recalled_memory['helpful_example']}"
                )
            if recalled_memory.get("student_preference"):
                memory_lines.append(
                    f"Student preference: {recalled_memory['student_preference']}"
                )
            if recalled_memory.get("status"):
                memory_lines.append(f"Past status: {recalled_memory['status']}")

            block = "\n".join(memory_lines).strip()
            if block:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Relevant past teaching memory:\n"
                            f"{block}\n\n"
                            "Use it only if it truly helps with the current question."
                        ),
                    }
                )

            snippet = (recalled_memory.get("snippet") or "").strip()
            if snippet:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Relevant past conversation snippet:\n"
                            f"{snippet}\n\n"
                            "Use this only as supporting context."
                        ),
                    }
                )

        if history_messages:
            for item in history_messages:
                role = item.get("role")
                content = item.get("content", "")
                if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content.strip()})

        # ── Interruption: ask the LLM to identify the pending topic ──
        if interruption_context.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The student just interrupted your previous response. "
                        "Here is what you were saying when interrupted:\n"
                        f'"{interruption_context.strip()[:300]}"\n\n'
                        "First, answer the student's current question normally.\n"
                        "Then, on the VERY LAST LINE of your response, write exactly:\n"
                        "[PENDING_TOPIC: <topic>]\n"
                        "where <topic> is the specific subject you were teaching when interrupted "
                        "(for example: Newton's first law of motion, quadratic equations, the French Revolution).\n"
                        "If you were just greeting, making small-talk, or not teaching any specific topic, write:\n"
                        "[PENDING_TOPIC: none]\n"
                        "This line must always be the very last line. Do not add anything after it."
                    ),
                }
            )

        messages.append({"role": "user", "content": user_message})

        logger.info(
            "Final LLM prompt built | total_messages=%s | recalled_memory=%s | clarification_mode=%s | current_user_chars=%s",
            len(messages),
            bool(recalled_memory),
            recall_clarification_mode,
            len(user_message),
        )
        return messages

    # ─────────────────────────────────────────────────────────────────
    # FOREGROUND — runs on GPU Ollama instance (port 11434)
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        user_message: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
        recalled_memory: Optional[Dict[str, str]] = None,
        recall_clarification_mode: bool = False,
        recall_clarification_question: str = "",
        fresh_teach_topic: str = "",
        emotion_instruction: str = "",
        interruption_context: str = "",
    ) -> str:
        response = self.fg_client.chat(
            model=self.model,
            messages=self.build_messages(
                user_message=user_message,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
                recalled_memory=recalled_memory,
                recall_clarification_mode=recall_clarification_mode,
                recall_clarification_question=recall_clarification_question,
                fresh_teach_topic=fresh_teach_topic,
                emotion_instruction=emotion_instruction,
                interruption_context=interruption_context,
            ),
            stream=False,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"].strip()

    def stream_generate(
        self,
        user_message: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
        recalled_memory: Optional[Dict[str, str]] = None,
        recall_clarification_mode: bool = False,
        recall_clarification_question: str = "",
        fresh_teach_topic: str = "",
        emotion_instruction: str = "",
        interruption_context: str = "",
    ) -> Iterator[str]:
        if not self.enable_streaming:
            yield self.generate(
                user_message=user_message,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
                recalled_memory=recalled_memory,
                recall_clarification_mode=recall_clarification_mode,
                recall_clarification_question=recall_clarification_question,
                fresh_teach_topic=fresh_teach_topic,
                emotion_instruction=emotion_instruction,
                interruption_context=interruption_context,
            )
            return

        logger.info("Foreground stream_generate started.")
        try:
            stream = self.fg_client.chat(
                model=self.model,
                messages=self.build_messages(
                    user_message=user_message,
                    history_messages=history_messages,
                    conversation_summary=conversation_summary,
                    recalled_memory=recalled_memory,
                    recall_clarification_mode=recall_clarification_mode,
                    recall_clarification_question=recall_clarification_question,
                    fresh_teach_topic=fresh_teach_topic,
                    emotion_instruction=emotion_instruction,
                    interruption_context=interruption_context,
                ),
                stream=True,
                options={"temperature": self.temperature},
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        finally:
            logger.info("Foreground stream_generate completed.")

    # ─────────────────────────────────────────────────────────────────
    # BACKGROUND — runs on CPU Ollama instance (port 11435)
    # ─────────────────────────────────────────────────────────────────

    def structured_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[BaseModel],
        temperature: float = 0.1,
    ) -> Optional[BaseModel]:
        """Structured call for background services.

        Runs on the separate CPU Ollama instance so it never blocks
        foreground teacher responses.
        """
        try:
            logger.info(
                "Background structured_chat started | schema=%s | model=%s",
                schema_model.__name__,
                self.bg_model,
            )
            response = self.bg_client.chat(
                model=self.bg_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                options={"temperature": temperature},
                format=schema_model.model_json_schema(),
            )
            logger.info(
                "Background structured_chat completed | schema=%s",
                schema_model.__name__,
            )

            raw = response.get("message", {}).get("content", "").strip()
            if not raw:
                return None

            data = json.loads(raw)
            return schema_model(**data)
        except Exception as e:
            logger.warning("structured_chat failed | error=%s", e)
            return None