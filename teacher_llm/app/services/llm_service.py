from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Dict, Iterator, List, Optional, Type

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

# Single-model Ollama strategy (GPU)
# ───────────────────────────────────
# One model (gemma3:4b) handles EVERYTHING:
#   • Teacher responses (generate / stream_generate)
#   • Recall decisions, summary updates (structured_chat via bg_client)
#   • Memory card extraction (foreground_structured_chat — runs during
#     TTS playback while the model is otherwise idle)
#
# Only ONE Ollama instance is needed (port 11434).
# OLLAMA_MAX_LOADED_MODELS=1 is sufficient.
#
# Memory card extraction is triggered INLINE after the teacher response
# is generated, while TTS is playing.  A threading.Event
# (_memory_card_cancel) lets the next teacher turn abort an in-flight
# extraction so the model is immediately available.
#
# The bg_client / bg_model are kept as config aliases pointing to the
# SAME model so recall_service and summary_service continue to work
# unchanged (they call structured_chat / bg_client.chat directly).

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

        # ── Background (same model, same instance) ───────────────────
        # bg_model now defaults to the SAME model as the foreground.
        # This eliminates the need for a second model in VRAM.
        # bg_client still exists as an alias so recall_service and
        # summary_service keep working without changes.
        self.bg_model = os.getenv("BG_LLM_MODEL", self.model)
        self.bg_num_ctx = int(os.getenv("OLLAMA_BG_NUM_CTX", "512"))
        bg_host = os.getenv("OLLAMA_BG_HOST", fg_host)
        self.bg_client = ollama.Client(host=bg_host)

        # ── Foreground model lock ────────────────────────────────────
        # Serialises access to the foreground model so that memory card
        # extraction (running during TTS) and the next teacher response
        # never race on the same Ollama model.
        self.fg_model_lock = threading.Lock()

        # ── Memory card cancellation ─────────────────────────────────
        # Set by the /chat endpoint when a new turn arrives.  The
        # inline memory card extraction checks this BEFORE calling
        # Ollama; if set, it aborts immediately so Gemma is free.
        self._memory_card_cancel = threading.Event()

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
            "num_predict": 512,
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
        # Signal any in-flight memory card extraction to abort, then
        # wait for the lock so we don't overlap on the same model.
        self._memory_card_cancel.set()

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

        with self.fg_model_lock:
            # Clear the cancel flag now that WE hold the lock — the
            # next memory card extraction (after this turn) should
            # start with a clean slate.
            self._memory_card_cancel.clear()

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

        # Signal any in-flight memory card extraction to abort.
        self._memory_card_cancel.set()

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
        with self.fg_model_lock:
            self._memory_card_cancel.clear()
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
        """Structured call for lightweight background services (recall, summary).

        Uses bg_client which now points to the SAME model as the foreground.
        Kept separate so recall_service / summary_service work unchanged.
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

    # ─────────────────────────────────────────────────────────────────
    # MEMORY CARD EXTRACTION — runs on foreground model during TTS
    # ─────────────────────────────────────────────────────────────────

    def cancel_memory_card_extraction(self) -> None:
        """Signal any in-flight memory card extraction to abort.

        Called by the /chat endpoint as soon as a new student turn arrives,
        BEFORE acquiring fg_model_lock.  If a memory card extraction is
        currently running (or about to call Ollama), it will see the cancel
        event and bail out immediately, releasing the lock.
        """
        self._memory_card_cancel.set()
        logger.info("Memory card cancellation requested.")

    def foreground_structured_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[BaseModel],
        temperature: float = 0.1,
    ) -> Optional[BaseModel]:
        """Structured call that runs on the FOREGROUND model (Gemma 4b).

        Used for memory card extraction during TTS playback.
        Acquires fg_model_lock so it serialises with teacher response
        generation.  Checks _memory_card_cancel before and after the
        Ollama call — if a new turn has arrived, it aborts to free the
        model.
        """
        # ── Check cancellation before even trying to acquire lock ────
        if self._memory_card_cancel.is_set():
            logger.info(
                "foreground_structured_chat skipped (cancelled before lock) | schema=%s",
                schema_model.__name__,
            )
            return None

        acquired = self.fg_model_lock.acquire(timeout=30)
        if not acquired:
            logger.warning(
                "foreground_structured_chat timed out waiting for fg_model_lock | schema=%s",
                schema_model.__name__,
            )
            return None

        try:
            # ── Re-check cancellation after acquiring lock ───────────
            if self._memory_card_cancel.is_set():
                logger.info(
                    "foreground_structured_chat skipped (cancelled after lock) | schema=%s",
                    schema_model.__name__,
                )
                return None

            logger.info(
                "foreground_structured_chat started | schema=%s | model=%s",
                schema_model.__name__,
                self.model,
            )

            response = self.fg_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                options={
                    "temperature": temperature,
                    "num_ctx": self.bg_num_ctx,
                    "num_predict": 512,  # enough for structured JSON output
                },
                format=schema_model.model_json_schema(),
            )

            # ── Check cancellation after Ollama returns ──────────────
            # If cancelled while Ollama was running, discard the result
            # so the caller doesn't do unnecessary post-processing
            # (embedding, DB writes) while the next turn is waiting.
            if self._memory_card_cancel.is_set():
                logger.info(
                    "foreground_structured_chat result discarded (cancelled during inference) | schema=%s",
                    schema_model.__name__,
                )
                return None

            logger.info(
                "foreground_structured_chat completed | schema=%s",
                schema_model.__name__,
            )

            raw = response.get("message", {}).get("content", "").strip()
            if not raw:
                return None

            data = json.loads(raw)
            return schema_model(**data)

        except Exception as e:
            logger.warning("foreground_structured_chat failed | error=%s", e)
            return None
        finally:
            self.fg_model_lock.release()