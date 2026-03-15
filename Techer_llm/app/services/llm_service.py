from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Type

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

AI_TEACHER_SYSTEM_PROMPT = """
You are a warm and clear AI teacher.

Main job:
Help the student understand, not memorize.

Rules:
- Be honest. If unsure, say so.
- Use simple English.
- Teach step by step.
- Keep each step short.
- Use one good real-life example when useful.
- If the student seems confused, explain again in a different way.
- Do not make the answer too long unless needed.
- Stay focused on the student's current question.

Reply format:
1. Clear explanation in short steps
2. One example or analogy when helpful
3. Very short recap at the end

Style:
- Friendly
- Calm
- Encouraging
- Natural
""".strip()


class LLMService:
    def __init__(self) -> None:
        self.model = os.getenv("LLM_MODEL", "qwen2.5:3b")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        logger.info(
            "LLMService initialised | model=%s | temperature=%s | streaming=%s",
            self.model,
            self.temperature,
            self.enable_streaming,
        )

    def build_messages(
        self,
        user_message: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
        recalled_memory: Optional[Dict[str, str]] = None,
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

        if recalled_memory:
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
            "Final LLM prompt built | total_messages=%s | recalled_memory=%s | current_user_chars=%s",
            len(messages),
            bool(recalled_memory),
            len(user_message),
        )
        return messages

    def generate(
        self,
        user_message: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
        recalled_memory: Optional[Dict[str, str]] = None,
    ) -> str:
        response = ollama.chat(
            model=self.model,
            messages=self.build_messages(
                user_message=user_message,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
                recalled_memory=recalled_memory,
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
    ) -> Iterator[str]:
        if not self.enable_streaming:
            yield self.generate(
                user_message=user_message,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
                recalled_memory=recalled_memory,
            )
            return

        stream = ollama.chat(
            model=self.model,
            messages=self.build_messages(
                user_message=user_message,
                history_messages=history_messages,
                conversation_summary=conversation_summary,
                recalled_memory=recalled_memory,
            ),
            stream=True,
            options={"temperature": self.temperature},
        )

        for chunk in stream:
            message = chunk.get("message", {})
            content = message.get("content", "")
            if content:
                yield content

    def structured_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[BaseModel],
        temperature: float = 0.1,
    ) -> Optional[BaseModel]:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                options={"temperature": temperature},
                format=schema_model.model_json_schema(),
            )

            raw = response.get("message", {}).get("content", "").strip()
            if not raw:
                return None

            data = json.loads(raw)
            return schema_model(**data)
        except Exception as e:
            logger.warning("structured_chat failed | error=%s", e)
            return None