from __future__ import annotations

from typing import Any, Optional

import requests


class TeacherClient:
    def __init__(self, chat_url: str, timeout: int = 180) -> None:
        self.chat_url = chat_url
        self.timeout = timeout

    def send(
        self,
        *,
        session_id: str,
        text: str,
        emotion_data: Optional[dict[str, Any]] = None,
        interruption_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"message": text, "session_id": session_id}
        if emotion_data:
            payload["emotion"] = emotion_data
        if interruption_meta:
            payload["interruption_meta"] = interruption_meta

        response = requests.post(self.chat_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return {
            "text": (data.get("response") or "").strip(),
            "directive": data.get("directive"),
        }
