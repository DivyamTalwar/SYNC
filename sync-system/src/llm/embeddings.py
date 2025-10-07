import hashlib
import time
from typing import List, Optional, Union, Dict
import cohere
import torch
from pathlib import Path
import pickle
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.utils.logging import get_logger
from src.utils.helpers import ensure_dir
from config.base import config

logger = get_logger("embeddings")


class EmbeddingCache:

    def __init__(self, cache_dir: Path = Path("./data/embedding_cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        logger.info(f"Initialized embedding cache at {cache_dir}")

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[torch.Tensor]:
        cache_file = self.cache_dir / f"{self._get_hash(text)}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, text: str, embedding: torch.Tensor):
        cache_file = self.cache_dir / f"{self._get_hash(text)}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(embedding, f)

    def clear(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cleared embedding cache")


class CohereEmbeddingsClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        self.api_key = api_key or config.api.cohere_api_key
        if not self.api_key:
            raise ValueError("Cohere API key not provided")

        self.model = model
        self.client = cohere.Client(self.api_key)

        # Caching
        self.use_cache = use_cache
        if use_cache:
            actual_cache_dir = cache_dir if cache_dir is not None else Path("./data/embedding_cache")
            self.cache = EmbeddingCache(actual_cache_dir)
        else:
            self.cache = None

        logger.info(f"Initialized Cohere embeddings client with model: {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        reraise=True
    )
    def _embed_with_retry(self, texts: List[str], input_type: str):
        logger.debug(f"Calling Cohere API for {len(texts)} texts")
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
        )
        return response

    def embed(
        self,
        texts: Union[str, List[str]],
        input_type: str = "search_document",
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        embeddings = []
        texts_to_embed = []
        cached_indices = []

        for i, text in enumerate(texts):
            if self.use_cache:
                cached_emb = self.cache.get(text)
                if cached_emb is not None:
                    embeddings.append(cached_emb)
                    cached_indices.append(i)
                    continue

            texts_to_embed.append(text)

        if texts_to_embed:
            try:
                response = self._embed_with_retry(
                    texts=texts_to_embed,
                    input_type=input_type,
                )

                new_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in response.embeddings]

                if self.use_cache:
                    for text, emb in zip(texts_to_embed, new_embeddings):
                        self.cache.set(text, emb)

                embeddings.extend(new_embeddings)

                logger.debug(f"Embedded {len(texts_to_embed)} texts ({len(cached_indices)} from cache)")

            except Exception as e:
                logger.error(f"Embedding error: {e}")
                raise

        result = torch.stack(embeddings)

        return result[0] if single else result

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 96,
        input_type: str = "search_document",
    ) -> torch.Tensor:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed(batch, input_type=input_type)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def embed_for_reasoning(self, reasoning_text: str) -> torch.Tensor:
        return self.embed(reasoning_text, input_type="clustering")

    def embed_for_message(self, message: str) -> torch.Tensor:
        return self.embed(message, input_type="search_document")

    def similarity(self, text1: str, text2: str) -> float:
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        )

        return similarity.item()


_client: Optional[CohereEmbeddingsClient] = None


def get_client() -> CohereEmbeddingsClient:
    global _client
    if _client is None:
        _client = CohereEmbeddingsClient()
    return _client
