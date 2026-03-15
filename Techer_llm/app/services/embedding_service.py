from __future__ import annotations

import logging
import math
import os
from typing import List

import ollama
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        logger.info("EmbeddingService initialised | model=%s", self.model)

    def embed_text(self, text: str) -> List[float]:
        clean_text = (text or "").strip()
        if not clean_text:
            return []

        response = ollama.embeddings(model=self.model, prompt=clean_text)
        embedding = response.get("embedding", [])
        if not isinstance(embedding, list):
            return []
        return [float(x) for x in embedding]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return -1.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return -1.0

        return dot / (norm_a * norm_b)