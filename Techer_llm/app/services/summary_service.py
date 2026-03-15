from __future__ import annotations

import logging
import os
from typing import Dict, List

import ollama
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """
You update the older conversation summary for an AI teacher.

Goal:
Keep only important learning context from older messages.

Keep:
- topic studied
- confusions
- helpful examples
- student preferences
- whether something was understood or still unclear

Remove:
- greetings
- filler
- repetition
- small talk

Rules:
- plain text only
- short and useful
- no bullet points
- no markdown
- no invented facts
- ideally under 120 words
""".strip()


class SummaryService:
    def __init__(self) -> None:
        self.model = os.getenv("SUMMARY_MODEL", os.getenv("LLM_MODEL", "qwen2.5:3b"))
        self.temperature = float(os.getenv("SUMMARY_TEMPERATURE", "0.2"))

        logger.info(
            "SummaryService initialised | model=%s | temperature=%s",
            self.model,
            self.temperature,
        )

    def update_summary(
        self,
        previous_summary: str,
        new_messages: List[Dict[str, str]],
    ) -> str:
        if not new_messages:
            return previous_summary.strip()

        formatted_messages = self._format_messages(new_messages)

        user_prompt = f"""
Previous summary:
{previous_summary.strip() if previous_summary.strip() else "None"}

Newly older messages:
{formatted_messages}

Update the summary.
Return only the updated summary text.
""".strip()

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            options={"temperature": self.temperature},
        )

        content = response["message"]["content"].strip()
        return content or previous_summary.strip()

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        lines: List[str] = []

        for item in messages:
            role = item.get("role", "").strip().lower()
            content = item.get("content", "").strip()

            if not content:
                continue

            if role == "user":
                lines.append(f"Student: {content}")
            elif role == "assistant":
                lines.append(f"Teacher: {content}")

        return "\n".join(lines).strip()