"""
Text-based emotion classifier for terminal chat.

Same keyword/pattern logic as the STT emotion_service but without
numpy or audio dependencies. Used when there is no voice input.
"""

from __future__ import annotations

import re
from typing import Dict, List


_TEXT_EMOTION_RULES: List[Dict] = [
    {
        "label": "frustrated",
        "patterns": [
            r"\bi\s*(still\s+)?don'?t\s+(get|understand)\b",
            r"\bthis\s+(is\s+)?(so\s+)?(hard|difficult|confusing|complicated)\b",
            r"\bnothing\s+makes\s+sense\b",
            r"\bi\s+give\s+up\b",
            r"\bwhat\s+even\b",
            r"\bthis\s+is\s+(too|very)\s+(much|hard)\b",
            r"\bi\s+can'?t\s+(do|figure|understand|solve)\b",
            r"\bugh\b",
            r"\bwhy\s+is\s+this\s+so\b",
            r"\bi\s+keep\s+getting\s+(it\s+)?wrong\b",
        ],
        "confidence": 0.75,
    },
    {
        "label": "confused",
        "patterns": [
            r"\bi\s+don'?t\s+(understand|get\s+it)\b",
            r"\bwhat\s+do\s+you\s+mean\b",
            r"\bi'?m\s+(confused|lost)\b",
            r"\bcan\s+you\s+(explain|say)\s+(that\s+)?again\b",
            r"\bhow\s+(does|is)\s+that\s+(work|possible)\b",
            r"\bwait\s+what\b",
            r"\bthat\s+doesn'?t\s+make\s+sense\b",
            r"\bwhat\s+is\s+the\s+difference\b",
            r"\bhuh\b",
            r"\bwhy\s+(does|is|do)\b",
        ],
        "confidence": 0.70,
    },
    {
        "label": "bored",
        "patterns": [
            r"\bthis\s+is\s+(boring|dull)\b",
            r"\bi\s+(already\s+)?know\s+(this|that|all\s+this)\b",
            r"\bcan\s+we\s+(move\s+on|skip)\b",
            r"\bok\s*(ay)?\s*\.?\s*$",
            r"\byeah\s*(\.?\s*)$",
            r"\bwhatever\b",
            r"\btoo\s+(easy|simple|basic)\b",
        ],
        "confidence": 0.60,
    },
    {
        "label": "confident",
        "patterns": [
            r"\bi\s+(get|got|understand)\s+(it|this|that)\b",
            r"\boh\s+(i\s+see|ok|okay|that\s+makes\s+sense)\b",
            r"\bthat\s+makes\s+sense\b",
            r"\bi\s+think\s+i\s+(understand|got\s+it)\b",
            r"\beasy\b",
            r"\bi\s+can\s+do\s+(this|that|it)\b",
            r"\bnow\s+i\s+(get|understand)\b",
        ],
        "confidence": 0.65,
    },
    {
        "label": "curious",
        "patterns": [
            r"\bbut\s+what\s+(if|about|happens)\b",
            r"\bwhat\s+would\s+happen\b",
            r"\bcan\s+you\s+tell\s+me\s+more\b",
            r"\bthat'?s\s+interesting\b",
            r"\bwow\b",
            r"\bhow\s+come\b",
            r"\bwhat\s+else\b",
            r"\btell\s+me\s+more\b",
        ],
        "confidence": 0.60,
    },
    {
        "label": "anxious",
        "patterns": [
            r"\bi'?m\s+(scared|nervous|worried)\s+(about|of|for)\b",
            r"\bwhat\s+if\s+i\s+(fail|get\s+it\s+wrong|can'?t)\b",
            r"\bis\s+this\s+(going\s+to\s+be\s+)?(on|in)\s+the\s+(exam|test)\b",
            r"\bi'?m\s+not\s+(sure|ready|good\s+enough)\b",
            r"\bthis\s+is\s+stressful\b",
        ],
        "confidence": 0.65,
    },
]


def classify_text_emotion(text: str) -> Dict[str, object]:
    """Classify emotion from transcript text using pattern matching.

    Returns:
        {"label": str, "confidence": float}
    """
    if not text or not text.strip():
        return {"label": "neutral", "confidence": 0.5}

    lowered = text.lower().strip()

    for rule in _TEXT_EMOTION_RULES:
        for pattern in rule["patterns"]:
            if re.search(pattern, lowered):
                return {
                    "label": rule["label"],
                    "confidence": rule["confidence"],
                }

    return {"label": "neutral", "confidence": 0.5}
