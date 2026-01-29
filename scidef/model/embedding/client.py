import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from openai import OpenAI

from scidef.model.dataclass import CacheModel
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


class EmbeddingClient(CacheModel):
    """OpenAI-compatible embedding client with caching."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        cache_dir: Optional[Path] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_embedding(
        self,
        text: str,
    ) -> Tuple[List[float], Optional[str]]:
        """Generate embedding vector for text."""
        try:
            # Check cache first
            cached_embedding = self.load_from_cache(text)
            if cached_embedding is not None:
                return cached_embedding, None

            response = self.client.embeddings.create(
                input=text,
                model=self.model_name,
            )

            if not response.data:
                return [], "No embedding data in response"

            embedding = response.data[0].embedding

            self.save_to_cache(text, embedding)

            return embedding, None

        except Exception as e:
            error_msg = f"Error generating embedding: {e}"
            logger.error(error_msg)
            return [], error_msg

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0

        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(np.clip(similarity, -1, 1))

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def load_from_cache(self, text: str) -> Optional[List[float]]:
        """Load embedding from cache if ava_load_from_cacheilable."""
        if not self.cache_dir:
            return None

        try:
            cache_key = self.get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                if "embedding" in cache_data and "model" in cache_data:
                    if cache_data["model"] == self.model_name:
                        return cache_data["embedding"]

        except Exception as e:
            logger.debug(f"Error loading embedding from cache: {e}")

        return None

    def save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        if not self.cache_dir:
            return

        try:
            cache_key = self.get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.json"

            cache_data = {
                "text_preview": text[:200] + "..."
                if len(text) > 200
                else text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "embedding": embedding,
                "model": self.model_name,
                "timestamp": time.time(),
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)

        except Exception as e:
            logger.debug(f"Error saving embedding to cache: {e}")
