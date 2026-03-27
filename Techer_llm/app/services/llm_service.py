from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Iterator, List, Optional, Type

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

# Single-instance Ollama strategy (GPU)
# ──────────────────────────────────────
# Both foreground and background clients point to the SAME Ollama
# instance (port 11434, GPU).  OLLAMA_MAX_LOADED_MODELS=2 must be set
# before starting Ollama so both models stay resident in VRAM:
#
#   $env:OLLAMA_MAX_LOADED_MODELS = "2"; ollama serve
#
#   Instance (GPU, port 11434):
#     • stream_generate() / generate() — foreground teacher responses.
#       Runs qwen2.5:3b on GPU for fast, low-latency answers.
#     • structured_chat() — background tasks like memory-card extraction,
#       recall decisions, summary generation, etc.
#       Runs gemma2:2b on GPU with a small num_ctx (512) so its KV cache
#       stays tiny and both models fit comfortably in 6 GiB VRAM.
#
# VRAM budget (approximate, num_ctx=4096 for qwen, 512 for gemma):
#   qwen2.5:3b  weights ~1.8 GiB  +  KV ~144 MiB  +  compute ~301 MiB
#   gemma2:2b   weights ~1.5 GiB  +  KV  ~26 MiB  +  compute ~505 MiB
#   ─────────────────────────────────────────────────────────────────
#   Total                                              ≈ 4.3 GiB  ✅
#
# Memory card extraction is handled by a SEPARATE worker process
# (memory_worker.py) that reads unprocessed turns from SQLite and
# sends them to the background model.  The chat loop therefore never
# blocks on memory cards no matter how fast the student types.

AI_TEACHER_SYSTEM_PROMPT = """
You are an excellent teacher.

Identity:
- You teach clearly, simply, and naturally.
- Your goal is to help the student understand, not just memorize.

Instruction priority:
- First, answer the student's current message.
- Second, follow the response length rules.
- Third, use examples only when they truly help.

Permanent rules:
- Use simple language.
- Be friendly, calm, and supportive.
- Focus on the main idea first.
- Teach step by step when needed.
- Be honest. If you do not know something, say so.
- If the student's request is unclear, ask one short clarification question instead of guessing.
- Do not pretend to remember earlier conversation unless it appears in the provided context.
- Do not sound like a textbook.
- Do not ask follow-up questions unless they truly help.
- Do not explain extra side topics unless the student asks.
- Avoid repetition.

Response length rules:
- Default response: about 2 to 5 short sentences.
- For simple questions: 1 to 3 sentences.
- Use at most one short example unless the student asks for more.
- Do not give long detailed teaching unless the student explicitly asks for detail.
- Avoid long introductions and long conclusions.

Default answer pattern:
- Start with the direct answer or explanation.
- Add one short example only if it truly helps.
- End with one short summary line only if it adds value.

Broad-request rule:
- If the student says something broad like "teach me physics" or "I want to study math", do not begin a full lesson immediately.
- Instead, ask what specific topic they want and suggest 2 or 3 options.

Reply style:
- Sound natural and conversational.
- Do not use headings, bullet points, or numbered lists in the reply unless the student asks for a structured answer.
- Never use LaTeX, markdown math, or any formula notation such as \(...\), \[...\], $$...$$, or $...$. Write all formulas in plain English. For example, write "F equals m times a" instead of "\( F = ma \)", and "the square root of 16" instead of "\( \sqrt{16} \)".

Emotion-aware behavior:
- You may receive a separate teaching-style note about the student's current state.
- If you receive one, adapt your tone and pace naturally.
- Do not mention or label the student's emotion.
- Silently become warmer, simpler, slower, or more encouraging when needed.

Your mission is to make learning clear, simple, and enjoyable.
""".strip()


class LLMService:
    def __init__(self) -> None:
        # ── Foreground (GPU) ──────────────────────────────────────────
        self.model = os.getenv("LLM_MODEL", "qwen2.5:3b")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.top_p = float(os.getenv("LLM_TOP_P", "0.85"))
        self.repeat_penalty = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        fg_host = os.getenv("OLLAMA_FG_HOST", "http://127.0.0.1:11434")
        self.fg_client = ollama.Client(host=fg_host)

        # ── Background (GPU, same instance) ──────────────────────────
        # Points to the same Ollama instance as the foreground client.
        # num_ctx is kept small (512) so gemma2:2b's KV cache stays
        # tiny and both models remain resident in VRAM simultaneously.
        self.bg_model = os.getenv("BG_LLM_MODEL", "gemma2:2b")
        self.bg_num_ctx = int(os.getenv("OLLAMA_BG_NUM_CTX", "512"))
        bg_host = os.getenv("OLLAMA_BG_HOST", "http://127.0.0.1:11434")
        self.bg_client = ollama.Client(host=bg_host)

        logger.info(
            (
                "LLMService initialised | fg_model=%s | fg_host=%s | "
                "bg_model=%s | bg_host=%s | bg_num_ctx=%s | temperature=%s | "
                "top_p=%s | repeat_penalty=%s | streaming=%s"
            ),
            self.model,
            fg_host,
            self.bg_model,
            bg_host,
            self.bg_num_ctx,
            self.temperature,
            self.top_p,
            self.repeat_penalty,
            self.enable_streaming,
        )

    def _foreground_options(self) -> Dict[str, float]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }

    def _background_options(self, temperature: float) -> Dict:
        return {
            "temperature": temperature,
            "num_ctx": self.bg_num_ctx,
        }

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
        pending_topics: Optional[List[str]] = None,
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

        # ── Pending topics from earlier interruptions ─────────────
        if pending_topics:
            if len(pending_topics) == 1:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Pending topic note:\n"
                            f"The student previously interrupted you while you were explaining: \"{pending_topics[0]}\".\n"
                            "After answering the student's current message, naturally ask if they would like to "
                            "continue with that topic or talk about something else.\n"
                            "Keep it short and conversational. Do not force it."
                        ),
                    }
                )
            else:
                topic_list = "\n".join(f"- {t}" for t in pending_topics)
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Pending topics note:\n"
                            "The student interrupted you on multiple occasions. The following topics were left unfinished:\n"
                            f"{topic_list}\n\n"
                            "After answering the student's current message, naturally mention these pending topics "
                            "and ask if they would like to continue with any of them or talk about something else.\n"
                            "Keep it short and conversational. Do not force it."
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
                if (
                    role in {"user", "assistant"}
                    and isinstance(content, str)
                    and content.strip()
                ):
                    messages.append({"role": role, "content": content.strip()})

        messages.append({"role": "user", "content": user_message})

        logger.info(
            "Final LLM prompt built | total_messages=%s | recalled_memory=%s | "
            "clarification_mode=%s | pending_topics=%s | current_user_chars=%s",
            len(messages),
            bool(recalled_memory),
            recall_clarification_mode,
            len(pending_topics) if pending_topics else 0,
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
        pending_topics: Optional[List[str]] = None,
    ) -> str:
        messages = self.build_messages(
            user_message=user_message,
            history_messages=history_messages,
            conversation_summary=conversation_summary,
            recalled_memory=recalled_memory,
            recall_clarification_mode=recall_clarification_mode,
            recall_clarification_question=recall_clarification_question,
            fresh_teach_topic=fresh_teach_topic,
            emotion_instruction=emotion_instruction,
            pending_topics=pending_topics,
        )

        response = self.fg_client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options=self._foreground_options(),
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
        pending_topics: Optional[List[str]] = None,
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
                pending_topics=pending_topics,
            )
            return

        messages = self.build_messages(
            user_message=user_message,
            history_messages=history_messages,
            conversation_summary=conversation_summary,
            recalled_memory=recalled_memory,
            recall_clarification_mode=recall_clarification_mode,
            recall_clarification_question=recall_clarification_question,
            fresh_teach_topic=fresh_teach_topic,
            emotion_instruction=emotion_instruction,
            pending_topics=pending_topics,
        )

        logger.info("Foreground stream_generate started.")
        try:
            stream = self.fg_client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=self._foreground_options(),
            )

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue
                yield content

        finally:
            logger.info("Foreground stream_generate completed.")

    # ─────────────────────────────────────────────────────────────────
    # BACKGROUND — runs on GPU Ollama instance (same port 11434)
    # ─────────────────────────────────────────────────────────────────

    def structured_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[BaseModel],
        temperature: float = 0.1,
    ) -> Optional[BaseModel]:
        """Structured call for background services.

        Runs gemma2:2b on the same GPU Ollama instance as the foreground
        model.  num_ctx is kept at 512 (OLLAMA_BG_NUM_CTX) so gemma's
        KV cache stays small (~26 MiB vs ~416 MiB at 4096 ctx) and both
        models remain resident in VRAM without evicting each other.
        """
        try:
            logger.info(
                "Background structured_chat started | schema=%s | model=%s | num_ctx=%s",
                schema_model.__name__,
                self.bg_model,
                self.bg_num_ctx,
            )
            response = self.bg_client.chat(
                model=self.bg_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                options=self._background_options(temperature),
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