from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ChatMessage:
    role: str
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    meta: Optional[dict[str, Any]] = None
