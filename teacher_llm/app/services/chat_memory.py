from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None

load_dotenv()
logger = logging.getLogger(__name__)


class ChatMemoryService:
    """
    Redis (primary fast store):
        - recent rolling chat messages
        - summary cache
        - memory cards
        - memory embeddings

    SQLite (fallback + durable store):
        - full chat history
        - summary state
        - memory cards
        - memory embeddings
    """

    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.sqlite_db_path = os.getenv("SQLITE_DB_PATH", "chat_history.db")

        self.max_recent_messages = int(os.getenv("MAX_RECENT_MESSAGES", "12"))
        self.max_prompt_history_messages = int(
            os.getenv("MAX_PROMPT_HISTORY_MESSAGES", "4")
        )

        self.redis_client = self._build_redis_client()
        self._init_sqlite()

        logger.info(
            "ChatMemoryService initialised | redis_available=%s | sqlite_db_path=%s | max_recent=%s | max_prompt=%s",
            self.redis_client is not None,
            self.sqlite_db_path,
            self.max_recent_messages,
            self.max_prompt_history_messages,
        )

    def _build_redis_client(self):
        if redis is None:
            logger.warning("redis package not installed. Redis disabled.")
            return None

        try:
            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
            )
            client.ping()
            logger.info("Connected to Redis successfully | redis_url=%s", self.redis_url)
            return client
        except Exception as e:
            logger.warning("Redis unavailable. Falling back to SQLite | error=%s", e)
            return None

    def _init_sqlite(self) -> None:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    memory_extracted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Migration: add column to existing databases that lack it.
            try:
                conn.execute(
                    "ALTER TABLE chat_messages ADD COLUMN memory_extracted INTEGER NOT NULL DEFAULT 0"
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                ON chat_messages(session_id, created_at, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL DEFAULT '',
                    last_summarized_message_id INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_cards (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    confusion TEXT NOT NULL DEFAULT '',
                    helpful_example TEXT NOT NULL DEFAULT '',
                    student_preference TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '',
                    snippet TEXT NOT NULL DEFAULT '',
                    retrieval_text TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_cards_session_created
                ON memory_cards(session_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_session_created
                ON memory_embeddings(session_id, created_at)
                """
            )
            conn.commit()
            logger.info("SQLite initialised successfully | db=%s", self.sqlite_db_path)
        finally:
            conn.close()

    def _redis_history_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    def _redis_summary_key(self, session_id: str) -> str:
        return f"chat_summary:{session_id}"

    def _redis_summary_state_key(self, session_id: str) -> str:
        return f"chat_summary_state:{session_id}"

    def _redis_memory_card_key(self, memory_id: str) -> str:
        return f"memory_card:{memory_id}"

    def _redis_memory_cards_by_session_key(self, session_id: str) -> str:
        return f"memory_cards_by_session:{session_id}"

    def _redis_embedding_key(self, memory_id: str) -> str:
        return f"memory_embedding:{memory_id}"

    def save_message(self, session_id: str, role: str, content: str) -> None:
        message = {
            "role": role,
            "content": content,
            "created_at": time.time(),
        }

        logger.info(
            "Saving message | session_id=%s | role=%s | content_chars=%s",
            session_id,
            role,
            len(content),
        )

        self._save_to_sqlite(session_id=session_id, message=message)
        self._save_to_redis(session_id=session_id, message=message)

    def _save_to_sqlite(self, session_id: str, message: Dict[str, object]) -> None:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(message["role"]),
                    str(message["content"]),
                    float(message["created_at"]),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_to_redis(self, session_id: str, message: Dict[str, object]) -> None:
        if not self.redis_client:
            return

        try:
            key = self._redis_history_key(session_id)
            self.redis_client.rpush(key, json.dumps(message))
            self.redis_client.ltrim(key, -self.max_recent_messages, -1)
        except Exception as e:
            logger.warning(
                "Failed to save message to Redis | session_id=%s | error=%s",
                session_id,
                e,
            )

    def get_recent_history_for_prompt(self, session_id: str) -> List[Dict[str, str]]:
        history = self._get_recent_history_from_redis(session_id)

        if history:
            return history[-self.max_prompt_history_messages :]

        history = self._get_recent_history_from_sqlite(session_id)
        return history[-self.max_prompt_history_messages :]

    def _get_recent_history_from_redis(self, session_id: str) -> List[Dict[str, str]]:
        if not self.redis_client:
            return []

        try:
            key = self._redis_history_key(session_id)
            raw_messages = self.redis_client.lrange(key, 0, -1)

            parsed_messages: List[Dict[str, str]] = []
            for raw in raw_messages:
                item = json.loads(raw)
                role = item.get("role")
                content = item.get("content")

                if role in {"user", "assistant"} and isinstance(content, str):
                    parsed_messages.append({"role": role, "content": content})

            return parsed_messages
        except Exception as e:
            logger.warning(
                "Failed to load recent history from Redis | session_id=%s | error=%s",
                session_id,
                e,
            )
            return []

    def _get_recent_history_from_sqlite(self, session_id: str) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, self.max_prompt_history_messages),
            )
            rows = cursor.fetchall()
            rows.reverse()
            return [
                {"role": role, "content": content}
                for role, content in rows
                if role in {"user", "assistant"}
            ]
        finally:
            conn.close()

    def get_latest_n_messages_from_sqlite(
        self,
        session_id: str,
        n: int,
    ) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, n),
            )
            rows = cursor.fetchall()
            rows.reverse()
            return [
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "created_at": row[3],
                }
                for row in rows
                if row[1] in {"user", "assistant"}
            ]
        finally:
            conn.close()

    # ─────────────────────────────────────────────────────────────────
    # MEMORY WORKER METHODS
    # ─────────────────────────────────────────────────────────────────

    def get_unprocessed_turns(self) -> List[Dict[str, Any]]:
        """Return the oldest assistant message that has not been processed for memory extraction.

        Returns up to 1 turn (user + assistant pair) at a time so the worker
        processes them sequentially.
        """
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            # Find the oldest unprocessed assistant message.
            cursor = conn.execute(
                """
                SELECT id, session_id, content, created_at
                FROM chat_messages
                WHERE memory_extracted = 0 AND role = 'assistant'
                ORDER BY id ASC
                LIMIT 1
                """,
            )
            row = cursor.fetchone()
            if not row:
                return []

            assistant_id = row[0]
            session_id = row[1]

            # Grab up to 4 messages ending at this assistant message for context.
            cursor2 = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE session_id = ? AND id <= ?
                ORDER BY id DESC
                LIMIT 4
                """,
                (session_id, assistant_id),
            )
            rows = cursor2.fetchall()
            rows.reverse()
            return [
                {
                    "id": r[0],
                    "role": r[1],
                    "content": r[2],
                    "created_at": r[3],
                    "session_id": session_id,
                }
                for r in rows
                if r[1] in {"user", "assistant"}
            ]
        finally:
            conn.close()

    def mark_turn_as_extracted(self, up_to_message_id: int) -> None:
        """Mark user and assistant messages up to the given id as memory_extracted = 1."""
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            conn.execute(
                """
                UPDATE chat_messages
                SET memory_extracted = 1
                WHERE id <= ? AND memory_extracted = 0
                """,
                (up_to_message_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_conversation_summary_for_prompt(self, session_id: str) -> str:
        summary = self._get_summary_from_redis(session_id)
        if summary:
            return summary

        return self._get_summary_from_sqlite(session_id)

    def _get_summary_from_redis(self, session_id: str) -> str:
        if not self.redis_client:
            return ""

        try:
            value = self.redis_client.get(self._redis_summary_key(session_id))
            return value.strip() if isinstance(value, str) else ""
        except Exception as e:
            logger.warning(
                "Failed to load summary from Redis | session_id=%s | error=%s",
                session_id,
                e,
            )
            return ""

    def _get_summary_from_sqlite(self, session_id: str) -> str:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT summary_text
                FROM conversation_memory
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return ""
            return row[0].strip() if isinstance(row[0], str) else ""
        finally:
            conn.close()

    def _get_summary_state(self, session_id: str) -> Tuple[str, int]:
        redis_state = self._get_summary_state_from_redis(session_id)
        if redis_state is not None:
            return redis_state

        return self._get_summary_state_from_sqlite(session_id)

    def _get_summary_state_from_redis(
        self,
        session_id: str,
    ) -> Optional[Tuple[str, int]]:
        if not self.redis_client:
            return None

        try:
            summary = self.redis_client.get(self._redis_summary_key(session_id))
            state_raw = self.redis_client.get(self._redis_summary_state_key(session_id))

            if summary is None and state_raw is None:
                return None

            last_summarized_message_id = 0
            if isinstance(state_raw, str) and state_raw.strip():
                state = json.loads(state_raw)
                last_summarized_message_id = int(
                    state.get("last_summarized_message_id", 0)
                )

            return (
                summary.strip() if isinstance(summary, str) else "",
                last_summarized_message_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to load summary state from Redis | session_id=%s | error=%s",
                session_id,
                e,
            )
            return None

    def _get_summary_state_from_sqlite(self, session_id: str) -> Tuple[str, int]:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT summary_text, last_summarized_message_id
                FROM conversation_memory
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return "", 0

            summary_text = row[0].strip() if isinstance(row[0], str) else ""
            last_summarized_message_id = int(row[1]) if row[1] is not None else 0
            return summary_text, last_summarized_message_id
        finally:
            conn.close()

    def _save_summary_state(
        self,
        session_id: str,
        summary_text: str,
        last_summarized_message_id: int,
    ) -> None:
        self._save_summary_state_to_sqlite(
            session_id=session_id,
            summary_text=summary_text,
            last_summarized_message_id=last_summarized_message_id,
        )
        self._save_summary_state_to_redis(
            session_id=session_id,
            summary_text=summary_text,
            last_summarized_message_id=last_summarized_message_id,
        )

    def _save_summary_state_to_sqlite(
        self,
        session_id: str,
        summary_text: str,
        last_summarized_message_id: int,
    ) -> None:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            conn.execute(
                """
                INSERT INTO conversation_memory (
                    session_id,
                    summary_text,
                    last_summarized_message_id,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    last_summarized_message_id = excluded.last_summarized_message_id,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    summary_text,
                    last_summarized_message_id,
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_summary_state_to_redis(
        self,
        session_id: str,
        summary_text: str,
        last_summarized_message_id: int,
    ) -> None:
        if not self.redis_client:
            return

        try:
            self.redis_client.set(self._redis_summary_key(session_id), summary_text)
            self.redis_client.set(
                self._redis_summary_state_key(session_id),
                json.dumps(
                    {"last_summarized_message_id": last_summarized_message_id}
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to save summary state to Redis | session_id=%s | error=%s",
                session_id,
                e,
            )

    def update_older_conversation_summary(self, session_id: str, summary_service) -> None:
        summary_text, last_summarized_message_id = self._get_summary_state(session_id)
        cutoff_message_id = self._get_summary_cutoff_message_id(session_id)

        if cutoff_message_id <= last_summarized_message_id:
            return

        messages_to_merge = self._get_messages_for_summary_update(
            session_id=session_id,
            after_message_id=last_summarized_message_id,
            up_to_message_id=cutoff_message_id,
        )

        if not messages_to_merge:
            return

        updated_summary = summary_service.update_summary(
            previous_summary=summary_text,
            new_messages=messages_to_merge,
        )

        self._save_summary_state(
            session_id=session_id,
            summary_text=updated_summary,
            last_summarized_message_id=cutoff_message_id,
        )

    def _get_summary_cutoff_message_id(self, session_id: str) -> int:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, self.max_prompt_history_messages),
            )
            rows = cursor.fetchall()

            if len(rows) < self.max_prompt_history_messages:
                return 0

            rows.reverse()
            oldest_recent_message_id = int(rows[0][0])
            return oldest_recent_message_id - 1
        finally:
            conn.close()

    def _get_messages_for_summary_update(
        self,
        session_id: str,
        after_message_id: int,
        up_to_message_id: int,
    ) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                  AND id > ?
                  AND id <= ?
                ORDER BY id ASC
                """,
                (session_id, after_message_id, up_to_message_id),
            )
            rows = cursor.fetchall()

            return [
                {"role": role, "content": content}
                for role, content in rows
                if role in {"user", "assistant"}
            ]
        finally:
            conn.close()

    # -----------------------------
    # MEMORY CARD METHODS
    # -----------------------------
    def save_memory_card(
        self,
        session_id: str,
        topic: str,
        confusion: str,
        helpful_example: str,
        student_preference: str,
        status: str,
        snippet: str,
        retrieval_text: str,
        embedding: List[float],
    ) -> str:
        memory_id = f"mem_{uuid.uuid4().hex}"
        created_at = time.time()

        card = {
            "memory_id": memory_id,
            "session_id": session_id,
            "topic": topic.strip(),
            "confusion": confusion.strip(),
            "helpful_example": helpful_example.strip(),
            "student_preference": student_preference.strip(),
            "status": status.strip(),
            "snippet": snippet.strip(),
            "retrieval_text": retrieval_text.strip(),
            "created_at": created_at,
        }

        self._save_memory_card_to_sqlite(card)
        self._save_memory_embedding_to_sqlite(
            memory_id=memory_id,
            session_id=session_id,
            embedding=embedding,
            created_at=created_at,
        )

        self._save_memory_card_to_redis(card)
        self._save_memory_embedding_to_redis(memory_id=memory_id, embedding=embedding)

        logger.info(
            "Memory card saved | session_id=%s | memory_id=%s | topic=%s",
            session_id,
            memory_id,
            topic,
        )
        return memory_id

    def _save_memory_card_to_sqlite(self, card: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_cards (
                    memory_id, session_id, topic, confusion, helpful_example,
                    student_preference, status, snippet, retrieval_text, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card["memory_id"],
                    card["session_id"],
                    card["topic"],
                    card["confusion"],
                    card["helpful_example"],
                    card["student_preference"],
                    card["status"],
                    card["snippet"],
                    card["retrieval_text"],
                    card["created_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_memory_embedding_to_sqlite(
        self,
        memory_id: str,
        session_id: str,
        embedding: List[float],
        created_at: float,
    ) -> None:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (
                    memory_id, session_id, embedding_json, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    memory_id,
                    session_id,
                    json.dumps(embedding),
                    created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_memory_card_to_redis(self, card: Dict[str, Any]) -> None:
        if not self.redis_client:
            return

        try:
            self.redis_client.set(
                self._redis_memory_card_key(card["memory_id"]),
                json.dumps(card),
            )
            self.redis_client.rpush(
                self._redis_memory_cards_by_session_key(card["session_id"]),
                card["memory_id"],
            )
        except Exception as e:
            logger.warning(
                "Failed to save memory card to Redis | memory_id=%s | error=%s",
                card["memory_id"],
                e,
            )

    def _save_memory_embedding_to_redis(
        self,
        memory_id: str,
        embedding: List[float],
    ) -> None:
        if not self.redis_client:
            return

        try:
            self.redis_client.set(
                self._redis_embedding_key(memory_id),
                json.dumps(embedding),
            )
        except Exception as e:
            logger.warning(
                "Failed to save memory embedding to Redis | memory_id=%s | error=%s",
                memory_id,
                e,
            )

    def get_memory_cards_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        cards = self._get_memory_cards_for_session_from_redis(session_id)
        if cards:
            return cards
        return self._get_memory_cards_for_session_from_sqlite(session_id)

    def _get_memory_cards_for_session_from_redis(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        if not self.redis_client:
            return []

        try:
            ids = self.redis_client.lrange(
                self._redis_memory_cards_by_session_key(session_id),
                0,
                -1,
            )
            results: List[Dict[str, Any]] = []
            for memory_id in ids:
                raw_card = self.redis_client.get(self._redis_memory_card_key(memory_id))
                raw_embedding = self.redis_client.get(self._redis_embedding_key(memory_id))
                if not raw_card:
                    continue

                card = json.loads(raw_card)
                embedding = json.loads(raw_embedding) if raw_embedding else []
                card["embedding"] = embedding
                results.append(card)

            return results
        except Exception as e:
            logger.warning(
                "Failed to load memory cards from Redis | session_id=%s | error=%s",
                session_id,
                e,
            )
            return []

    def _get_memory_cards_for_session_from_sqlite(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.execute(
                """
                SELECT
                    c.memory_id, c.session_id, c.topic, c.confusion, c.helpful_example,
                    c.student_preference, c.status, c.snippet, c.retrieval_text,
                    c.created_at, e.embedding_json
                FROM memory_cards c
                LEFT JOIN memory_embeddings e
                  ON c.memory_id = e.memory_id
                WHERE c.session_id = ?
                ORDER BY c.created_at ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                results.append(
                    {
                        "memory_id": row[0],
                        "session_id": row[1],
                        "topic": row[2] or "",
                        "confusion": row[3] or "",
                        "helpful_example": row[4] or "",
                        "student_preference": row[5] or "",
                        "status": row[6] or "",
                        "snippet": row[7] or "",
                        "retrieval_text": row[8] or "",
                        "created_at": row[9],
                        "embedding": json.loads(row[10]) if row[10] else [],
                    }
                )
            return results
        finally:
            conn.close()

    def is_redis_available(self) -> bool:
        return self.redis_client is not None