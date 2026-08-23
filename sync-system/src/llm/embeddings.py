import hashlib
from typing import List, Optional, Union
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
from src.llm.batching import DeduplicatedBatch

logger = get_logger("embeddings")


class EmbeddingCache:

    def __init__(self, cache_dir: Path = Path("./data/embedding_cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        logger.info(f"Initialized embedding cache at {cache_dir}")

    def _get_hash(self, text: str, namespace: str) -> str:
        payload = f"{namespace}\0{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def get(self, text: str, namespace: str) -> Optional[torch.Tensor]:
        cache_file = self.cache_dir / f"{self._get_hash(text, namespace)}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, text: str, namespace: str, embedding: torch.Tensor):
        cache_file = self.cache_dir / f"{self._get_hash(text, namespace)}.pkl"
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

        plan = DeduplicatedBatch.from_texts(texts)
        namespace = f"{self.model}:{input_type}"
        unique_embeddings: list[Optional[torch.Tensor]] = [None] * len(plan.unique_texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for index, text in enumerate(plan.unique_texts):
            cached = self.cache.get(text, namespace) if self.use_cache else None
            if cached is None:
                missing_texts.append(text)
                missing_indices.append(index)
            else:
                unique_embeddings[index] = cached

        if missing_texts:
            response = self._embed_with_retry(texts=missing_texts, input_type=input_type)
            new_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in response.embeddings]
            if len(new_embeddings) != len(missing_texts):
                raise RuntimeError("embedding provider returned an unexpected number of vectors")
            for index, text, embedding in zip(missing_indices, missing_texts, new_embeddings):
                unique_embeddings[index] = embedding
                if self.use_cache:
                    self.cache.set(text, namespace, embedding)

        if any(embedding is None for embedding in unique_embeddings):
            raise RuntimeError("embedding batch was not fully resolved")
        resolved = plan.restore(unique_embeddings)
        result = torch.stack(resolved)
        logger.debug(
            "Resolved %d texts from %d unique values in one provider batch",
            len(texts),
            len(missing_texts),
        )

        return result[0] if single else result

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 96,
        input_type: str = "search_document",
    ) -> torch.Tensor:
        if batch_size < 1 or batch_size > 96:
            raise ValueError("batch_size must be between 1 and Cohere's 96-text limit")
        if len(texts) <= batch_size:
            return self.embed(texts, input_type=input_type)

        # Deduplicate across the entire round before splitting provider calls.
        plan = DeduplicatedBatch.from_texts(texts)
        unique_batches = []
        for start in range(0, len(plan.unique_texts), batch_size):
            unique_batches.append(
                self.embed(list(plan.unique_texts[start:start + batch_size]), input_type=input_type)
            )
        unique_embeddings = torch.cat(unique_batches, dim=0)
        restored = plan.restore(list(unique_embeddings))
        return torch.stack(restored)

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
