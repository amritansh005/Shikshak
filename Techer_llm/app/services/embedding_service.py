from __future__ import annotations

import logging
import math
import os
from typing import List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, llm) -> None:
        """Accept an LLMService instance to reuse its background Ollama client.

        Embeddings are always a background task (memory card storage, recall
        similarity search), so they run on the CPU Ollama instance to keep
        the GPU free for the teacher.
        """
        self.model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.llm = llm
        logger.info("EmbeddingService initialised | model=%s", self.model)

    def embed_text(self, text: str) -> List[float]:
        clean_text = (text or "").strip()
        if not clean_text:
            return []

        try:
            response = self.llm.bg_client.embeddings(model=self.model, prompt=clean_text)
            embedding = response.get("embedding", [])
            if not isinstance(embedding, list):
                return []
            return [float(x) for x in embedding]
        except Exception as e:
            logger.warning("Embedding generation failed | error=%s", e)
            return []

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
