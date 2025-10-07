import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm

from src.data.datasets import TrainingExample, DialogueExample, DialogueTurn
from src.llm.embeddings import CohereEmbeddingsClient
from src.utils.logging import get_logger

logger = get_logger("data.preprocessing")


class TextCleaner:

    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_extra_spaces: bool = True,
        max_length: Optional[int] = None,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_extra_spaces = remove_extra_spaces
        self.max_length = max_length

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove extra spaces
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Truncate
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts"""
        return [self.clean(text) for text in texts]


class EmbeddingPreprocessor:
    """Pre-compute embeddings for efficiency"""

    def __init__(
        self,
        embedding_client: Optional[CohereEmbeddingsClient] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize embedding preprocessor

        Args:
            embedding_client: Cohere client (creates new if None)
            cache_dir: Directory to cache embeddings
        """
        self.embedding_client = embedding_client or CohereEmbeddingsClient()
        self.cache_dir = cache_dir or Path("./data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def precompute_embeddings(
        self,
        texts: List[str],
        cache_name: str,
        batch_size: int = 96,
        force_recompute: bool = False,
    ) -> torch.Tensor:
        """
        Pre-compute embeddings for a list of texts

        Args:
            texts: List of texts to embed
            cache_name: Name for cached embeddings
            batch_size: Batch size for embedding
            force_recompute: Force recomputation even if cached

        Returns:
            Tensor of shape (N, embedding_dim)
        """
        cache_path = self.cache_dir / f"{cache_name}.pkl"

        # Check cache
        if cache_path.exists() and not force_recompute:
            logger.info(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            return torch.tensor(embeddings, dtype=torch.float32)

        logger.info(f"Pre-computing embeddings for {len(texts)} texts...")

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_client.embed_batch(
                batch_texts,
                input_type="search_document"
            )
            embeddings.extend(batch_embeddings)

        embeddings_np = np.array(embeddings)

        # Cache
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings_np, f)
        logger.info(f"Cached embeddings to {cache_path}")

        return torch.tensor(embeddings_np, dtype=torch.float32)

    def precompute_dataset_embeddings(
        self,
        dataset: List[TrainingExample],
        cache_name: str = "training_data",
        force_recompute: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-compute embeddings for entire training dataset

        Args:
            dataset: List of training examples
            cache_name: Cache name
            force_recompute: Force recomputation

        Returns:
            Dict with task_id -> embedding mapping
        """
        # Extract queries
        task_ids = [ex.task_id for ex in dataset]
        queries = [ex.query for ex in dataset]

        # Compute embeddings
        embeddings = self.precompute_embeddings(
            queries,
            cache_name=cache_name,
            force_recompute=force_recompute
        )

        # Create mapping
        embedding_dict = {
            task_id: embeddings[i]
            for i, task_id in enumerate(task_ids)
        }

        logger.info(f"Pre-computed embeddings for {len(embedding_dict)} examples")
        return embedding_dict

    def precompute_dialogue_embeddings(
        self,
        dialogues: List[DialogueExample],
        cache_name: str = "dialogue_data",
        force_recompute: bool = False,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Pre-compute embeddings for dialogue utterances

        Args:
            dialogues: List of dialogue examples
            cache_name: Cache name
            force_recompute: Force recomputation

        Returns:
            Dict with dialogue_id -> [turn_embeddings] mapping
        """
        # Extract all utterances
        all_utterances = []
        dialogue_turn_counts = []

        for dialogue in dialogues:
            utterances = [turn.utterance for turn in dialogue.turns]
            all_utterances.extend(utterances)
            dialogue_turn_counts.append(len(utterances))

        # Compute embeddings
        all_embeddings = self.precompute_embeddings(
            all_utterances,
            cache_name=cache_name,
            force_recompute=force_recompute
        )

        # Split back into dialogues
        embedding_dict = {}
        current_idx = 0

        for dialogue, turn_count in zip(dialogues, dialogue_turn_counts):
            turn_embeddings = all_embeddings[current_idx:current_idx + turn_count]
            embedding_dict[dialogue.dialogue_id] = list(turn_embeddings)
            current_idx += turn_count

        logger.info(f"Pre-computed embeddings for {len(embedding_dict)} dialogues")
        return embedding_dict


class DataAugmenter:
    """Data augmentation for training"""

    def __init__(self, augmentation_factor: int = 2):
        """
        Initialize data augmenter

        Args:
            augmentation_factor: How many augmented versions per example
        """
        self.augmentation_factor = augmentation_factor

    def augment_query(self, query: str) -> List[str]:
        """
        Augment a query with variations

        Args:
            query: Original query

        Returns:
            List of augmented queries including original
        """
        augmented = [query]  # Original

        # Variation 1: Add "Please"
        if not query.lower().startswith("please"):
            augmented.append(f"Please {query}")

        # Variation 2: Add "Can you"
        if not query.lower().startswith("can you"):
            augmented.append(f"Can you {query.lower()}")

        # Variation 3: Rephrase as question
        if not query.endswith("?"):
            augmented.append(f"{query}?")

        return augmented[:self.augmentation_factor]

    def augment_dataset(
        self,
        dataset: List[TrainingExample]
    ) -> List[TrainingExample]:
        """
        Augment entire dataset

        Args:
            dataset: Original dataset

        Returns:
            Augmented dataset (original + variations)
        """
        augmented_dataset = []

        for example in dataset:
            # Add original
            augmented_dataset.append(example)

            # Add variations
            variations = self.augment_query(example.query)
            for i, variation in enumerate(variations[1:], 1):  # Skip original
                augmented_example = TrainingExample(
                    task_id=f"{example.task_id}_aug{i}",
                    query=variation,
                    context=example.context,
                    reference_answer=example.reference_answer,
                    difficulty=example.difficulty,
                    category=example.category,
                    metadata={
                        **example.metadata,
                        "augmented": True,
                        "original_id": example.task_id
                    }
                )
                augmented_dataset.append(augmented_example)

        logger.info(
            f"Augmented dataset: {len(dataset)} -> {len(augmented_dataset)} examples"
        )
        return augmented_dataset


class BatchCollator:
    """Collate function for DataLoader"""

    def __init__(
        self,
        embedding_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize batch collator

        Args:
            embedding_dict: Pre-computed embeddings (optional)
        """
        self.embedding_dict = embedding_dict

    def collate_training_batch(
        self,
        batch: List[TrainingExample]
    ) -> Dict[str, any]:
        """
        Collate a batch of training examples

        Args:
            batch: List of TrainingExample

        Returns:
            Dict with batched data
        """
        batch_dict = {
            "task_ids": [ex.task_id for ex in batch],
            "queries": [ex.query for ex in batch],
            "contexts": [ex.context for ex in batch],
            "reference_answers": [ex.reference_answer for ex in batch],
            "difficulties": [ex.difficulty for ex in batch],
            "categories": [ex.category for ex in batch],
        }

        # Add pre-computed embeddings if available
        if self.embedding_dict:
            embeddings = []
            for ex in batch:
                if ex.task_id in self.embedding_dict:
                    embeddings.append(self.embedding_dict[ex.task_id])
                else:
                    embeddings.append(None)
            batch_dict["embeddings"] = embeddings

        return batch_dict

    def collate_dialogue_batch(
        self,
        batch: List[DialogueExample]
    ) -> Dict[str, any]:
        """
        Collate a batch of dialogue examples

        Args:
            batch: List of DialogueExample

        Returns:
            Dict with batched data
        """
        batch_dict = {
            "dialogue_ids": [d.dialogue_id for d in batch],
            "turns": [d.turns for d in batch],
            "topics": [d.topic for d in batch],
            "num_speakers": [d.num_speakers for d in batch],
        }

        return batch_dict


def create_preprocessor(
    clean_text: bool = True,
    precompute_embeddings: bool = False,
    augment_data: bool = False,
    **kwargs
) -> Tuple[Optional[TextCleaner], Optional[EmbeddingPreprocessor], Optional[DataAugmenter]]:
    """
    Factory function to create preprocessing pipeline

    Args:
        clean_text: Enable text cleaning
        precompute_embeddings: Enable embedding pre-computation
        augment_data: Enable data augmentation
        **kwargs: Additional args for each preprocessor

    Returns:
        (cleaner, embedding_preprocessor, augmenter)
    """
    cleaner = TextCleaner(**kwargs.get("cleaner_args", {})) if clean_text else None

    embedding_preprocessor = EmbeddingPreprocessor(
        **kwargs.get("embedding_args", {})
    ) if precompute_embeddings else None

    augmenter = DataAugmenter(
        **kwargs.get("augmenter_args", {})
    ) if augment_data else None

    return cleaner, embedding_preprocessor, augmenter


if __name__ == "__main__":
    # Test preprocessing
    print("=" * 80)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 80)

    # Test text cleaning
    print("\n[1/3] Testing text cleaner...")
    cleaner = TextCleaner(remove_urls=True, remove_extra_spaces=True)
    dirty_text = "Check this out:  https://example.com   Multiple    spaces!"
    clean_text = cleaner.clean(dirty_text)
    print(f"Original: {dirty_text}")
    print(f"Cleaned: {clean_text}")
    print("[OK] Text cleaner working")

    # Test data augmentation
    print("\n[2/3] Testing data augmenter...")
    augmenter = DataAugmenter(augmentation_factor=3)
    query = "Explain machine learning"
    augmented = augmenter.augment_query(query)
    print(f"Original: {query}")
    print(f"Augmented: {augmented}")
    print("[OK] Data augmenter working")

    # Test batch collator
    print("\n[3/3] Testing batch collator...")
    from src.data.datasets import TrainingExample
    batch = [
        TrainingExample(task_id="1", query="What is AI?", difficulty="easy"),
        TrainingExample(task_id="2", query="How does ML work?", difficulty="medium"),
    ]
    collator = BatchCollator()
    batch_dict = collator.collate_training_batch(batch)
    print(f"Batch keys: {list(batch_dict.keys())}")
    print(f"Batch size: {len(batch_dict['queries'])}")
    print("[OK] Batch collator working")

    print("\n[OK] All preprocessing components working!")
