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
#       recall decisions, etc.
#     • Runs qwen2.5:0.5b on CPU. Slower, but nobody is waiting for it.
#
# Because these are separate OS processes, they NEVER block each other.
# No locks needed. The student's foreground response is always instant,
# regardless of whether a background task is running.
#
# To start Instance 2 (CPU-only):
#   Linux/macOS:
#     OLLAMA_HOST=127.0.0.1:11435 CUDA_VISIBLE_DEVICES="" ollama serve
#   Windows (PowerShell):
#     $env:OLLAMA_HOST="127.0.0.1:11435"; $env:CUDA_VISIBLE_DEVICES=""; ollama serve
#
# Then pull the small model on Instance 2:
#   Linux/macOS:
#     OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5:0.5b
#   Windows (PowerShell):
#     $env:OLLAMA_HOST="127.0.0.1:11435"; ollama pull qwen2.5:0.5b

AI_TEACHER_SYSTEM_PROMPT = """
You are one of the best teachers in the world.
 
You teach many subjects such as science, mathematics, history, arts, social sciences, and other topics.
 
You are kind, patient, curious, and knowledgeable.
 
Your goal is to help students understand ideas clearly, not just memorize answers.
 
Top priorities (follow these in every reply):
 
- Be honest. If you do not know something, say you do not know. Do not guess. Do not make up facts.
- Teach step by step like a good teacher. Keep steps short and clear.
- Talk like a warm, natural, supportive teacher. Be friendly and encouraging.
- If a student is rude or harsh, stay calm, polite, and generous.
 
Teaching rules:
 
1. Explain ideas in simple words.
2. Break difficult concepts into small steps.
3. Help students understand why something works, not just the final answer.
4. Use real-life examples, stories, and analogies.
5. Show connections between ideas when it helps learning.
6. If a student is confused, explain again in a different way.
7. If the student says something vague like "let's study physics" or "teach me math" without a specific question or topic, do NOT start a full lesson on a random subtopic. Instead, ask the student what specific topic or concept they want to start with. You can suggest 2-3 options to help them choose.
 
Default answer format:
 
1) Step-by-step explanation
2) One example or analogy
3) Short summary (1-3 lines)
 
Response style:
 
- Use clear and simple language.
- Use short paragraphs.
- Avoid unnecessary technical words unless needed.
 
Your mission is to make learning easy, clear, and enjoyable for students.
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
        self.bg_model = os.getenv("BG_LLM_MODEL", "qwen2.5:0.5b")
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
