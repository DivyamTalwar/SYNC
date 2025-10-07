import json
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm

from src.utils.logging import get_logger
from config.base import config

logger = get_logger("data.datasets")


@dataclass
class TrainingExample:
    """Single training example"""
    task_id: str
    query: str
    context: Optional[str] = None
    reference_answer: Optional[str] = None
    difficulty: str = "medium"
    category: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class DialogueTurn:
    """Single turn in a dialogue"""
    speaker: str
    utterance: str
    turn_index: int


@dataclass
class DialogueExample:
    """Multi-turn dialogue for CKM training"""
    dialogue_id: str
    turns: List[DialogueTurn]
    topic: Optional[str] = None
    num_speakers: int = 2


class AlpacaEvalDataset(Dataset):
    """
    AlpacaEval dataset loader

    Format: Instruction following tasks with reference outputs
    Used for: RL training and evaluation
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Load AlpacaEval dataset

        Args:
            split: Dataset split (train/val/test)
            max_samples: Limit number of samples
            cache_dir: Cache directory for downloaded data
        """
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading AlpacaEval dataset ({split})...")
        self.examples = self._load_dataset()
        logger.info(f"Loaded {len(self.examples)} examples")

    def _load_dataset(self) -> List[TrainingExample]:
        """Load and parse AlpacaEval dataset"""
        examples = []

        try:
            # Try to load from HuggingFace datasets
            dataset = load_dataset(
                "tatsu-lab/alpaca_eval",
                split="eval" if self.split == "test" else "train",
                cache_dir=str(self.cache_dir)
            )

            for i, item in enumerate(dataset):
                if self.max_samples and i >= self.max_samples:
                    break

                example = TrainingExample(
                    task_id=f"alpaca_{i}",
                    query=item.get("instruction", ""),
                    context=item.get("input", None) if item.get("input") else None,
                    reference_answer=item.get("output", None),
                    difficulty="medium",
                    category=item.get("category", "general"),
                    metadata={"source": "alpaca_eval", "index": i}
                )
                examples.append(example)

        except Exception as e:
            logger.warning(f"Could not load AlpacaEval from HF: {e}")
            logger.info("Using synthetic examples for testing...")
            examples = self._create_synthetic_examples()

        return examples

    def _create_synthetic_examples(self) -> List[TrainingExample]:
        """Create synthetic examples for testing when dataset unavailable"""
        synthetic_tasks = [
            {
                "query": "Explain the concept of machine learning in simple terms.",
                "category": "education",
                "difficulty": "easy"
            },
            {
                "query": "Write a Python function to check if a number is prime.",
                "category": "coding",
                "difficulty": "medium"
            },
            {
                "query": "What are the key factors to consider when designing a sustainable city?",
                "context": "Focus on environmental, social, and economic aspects.",
                "category": "planning",
                "difficulty": "hard"
            },
            {
                "query": "Compare and contrast renewable and non-renewable energy sources.",
                "category": "science",
                "difficulty": "medium"
            },
            {
                "query": "Describe the process of photosynthesis step by step.",
                "category": "biology",
                "difficulty": "easy"
            },
        ]

        examples = []
        num_copies = self.max_samples // len(synthetic_tasks) + 1 if self.max_samples else 10

        for copy_idx in range(num_copies):
            for task_idx, task in enumerate(synthetic_tasks):
                if self.max_samples and len(examples) >= self.max_samples:
                    break

                example = TrainingExample(
                    task_id=f"synthetic_{copy_idx}_{task_idx}",
                    query=task["query"],
                    context=task.get("context"),
                    reference_answer=None,
                    difficulty=task["difficulty"],
                    category=task["category"],
                    metadata={"source": "synthetic", "copy": copy_idx}
                )
                examples.append(example)

        return examples[:self.max_samples] if self.max_samples else examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get PyTorch DataLoader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x  # Return list of TrainingExamples
        )


class ShareGPTDataset(Dataset):
    """
    ShareGPT dataset loader

    Format: Multi-turn dialogues (user-assistant)
    Used for: CKM pre-training (learning to model conversation partners)
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        min_turns: int = 3,
        max_turns: int = 20,
        cache_dir: Optional[Path] = None,
    ):
        """
        Load ShareGPT dataset

        Args:
            split: Dataset split
            max_samples: Limit number of dialogues
            min_turns: Minimum turns per dialogue
            max_turns: Maximum turns per dialogue
            cache_dir: Cache directory
        """
        self.split = split
        self.max_samples = max_samples
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading ShareGPT dataset ({split})...")
        self.dialogues = self._load_dataset()
        logger.info(f"Loaded {len(self.dialogues)} dialogues")

    def _load_dataset(self) -> List[DialogueExample]:
        """Load and parse ShareGPT dataset"""
        dialogues = []

        try:
            # Try to load from HuggingFace
            dataset = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                split="train",
                cache_dir=str(self.cache_dir)
            )

            for i, item in enumerate(dataset):
                if self.max_samples and i >= self.max_samples:
                    break

                # Parse conversations
                conversations = item.get("conversations", [])
                if len(conversations) < self.min_turns:
                    continue

                turns = []
                for turn_idx, conv in enumerate(conversations[:self.max_turns]):
                    turn = DialogueTurn(
                        speaker=conv.get("from", "unknown"),
                        utterance=conv.get("value", ""),
                        turn_index=turn_idx
                    )
                    turns.append(turn)

                dialogue = DialogueExample(
                    dialogue_id=f"sharegpt_{i}",
                    turns=turns,
                    topic=None,
                    num_speakers=len(set(t.speaker for t in turns))
                )
                dialogues.append(dialogue)

        except Exception as e:
            logger.warning(f"Could not load ShareGPT from HF: {e}")
            logger.info("Using synthetic dialogues for testing...")
            dialogues = self._create_synthetic_dialogues()

        return dialogues

    def _create_synthetic_dialogues(self) -> List[DialogueExample]:
        """Create synthetic dialogues for testing"""
        synthetic_dialogues = [
            [
                ("user", "What is machine learning?"),
                ("assistant", "Machine learning is a subset of AI where systems learn from data."),
                ("user", "Can you give me an example?"),
                ("assistant", "Sure! Email spam filters learn to identify spam by analyzing patterns."),
            ],
            [
                ("user", "How does photosynthesis work?"),
                ("assistant", "Plants use sunlight to convert CO2 and water into glucose and oxygen."),
                ("user", "What role does chlorophyll play?"),
                ("assistant", "Chlorophyll absorbs light energy, which drives the chemical reactions."),
            ],
        ]

        dialogues = []
        num_copies = self.max_samples // len(synthetic_dialogues) + 1 if self.max_samples else 50

        for copy_idx in range(num_copies):
            for dialogue_idx, raw_dialogue in enumerate(synthetic_dialogues):
                if self.max_samples and len(dialogues) >= self.max_samples:
                    break

                turns = [
                    DialogueTurn(speaker=speaker, utterance=text, turn_index=i)
                    for i, (speaker, text) in enumerate(raw_dialogue)
                ]

                dialogue = DialogueExample(
                    dialogue_id=f"synthetic_{copy_idx}_{dialogue_idx}",
                    turns=turns,
                    topic="general",
                    num_speakers=2
                )
                dialogues.append(dialogue)

        return dialogues[:self.max_samples] if self.max_samples else dialogues

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> DialogueExample:
        return self.dialogues[idx]

    def get_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get PyTorch DataLoader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x  # Return list of DialogueExamples
        )


class CustomTaskDataset(Dataset):
    """
    Custom task dataset loader

    Format: JSON file with tasks
    Used for: Domain-specific RL training
    """

    def __init__(
        self,
        data_path: Path,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        Load custom task dataset from JSON

        Args:
            data_path: Path to JSON file
            split: Dataset split
            max_samples: Limit number of samples
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_samples = max_samples

        logger.info(f"Loading custom dataset from {data_path}...")
        self.examples = self._load_dataset()
        logger.info(f"Loaded {len(self.examples)} examples")

    def _load_dataset(self) -> List[TrainingExample]:
        """Load custom dataset from JSON"""
        if not self.data_path.exists():
            logger.warning(f"Dataset file not found: {self.data_path}")
            return []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []
        items = data.get(self.split, data.get("data", []))

        for i, item in enumerate(items):
            if self.max_samples and i >= self.max_samples:
                break

            example = TrainingExample(
                task_id=item.get("id", f"custom_{i}"),
                query=item["query"],
                context=item.get("context"),
                reference_answer=item.get("answer"),
                difficulty=item.get("difficulty", "medium"),
                category=item.get("category", "custom"),
                metadata=item.get("metadata", {})
            )
            examples.append(example)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation

    Args:
        dataset: Full dataset
        val_ratio: Validation set ratio
        seed: Random seed

    Returns:
        (train_dataset, val_dataset)
    """
    random.seed(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_size = int(len(dataset) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")

    return train_dataset, val_dataset


def get_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs
) -> Dataset:
    """
    Factory function to get dataset by name

    Args:
        dataset_name: Dataset name (alpaca/sharegpt/custom)
        split: Dataset split
        max_samples: Limit samples
        **kwargs: Additional dataset-specific args

    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "alpaca" or dataset_name == "alpacaeval":
        return AlpacaEvalDataset(split=split, max_samples=max_samples, **kwargs)

    elif dataset_name == "sharegpt":
        return ShareGPTDataset(split=split, max_samples=max_samples, **kwargs)

    elif dataset_name == "custom":
        data_path = kwargs.get("data_path")
        if not data_path:
            raise ValueError("Must provide data_path for custom dataset")
        return CustomTaskDataset(data_path=data_path, split=split, max_samples=max_samples)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test dataset loaders
    print("=" * 80)
    print("TESTING DATASET LOADERS")
    print("=" * 80)

    # Test AlpacaEval
    print("\n[1/2] Testing AlpacaEval dataset...")
    alpaca_dataset = AlpacaEvalDataset(split="train", max_samples=10)
    print(f"Loaded {len(alpaca_dataset)} examples")
    print(f"Example 0: {alpaca_dataset[0].query[:100]}...")

    # Test ShareGPT
    print("\n[2/2] Testing ShareGPT dataset...")
    sharegpt_dataset = ShareGPTDataset(split="train", max_samples=10)
    print(f"Loaded {len(sharegpt_dataset)} dialogues")
    dialogue = sharegpt_dataset[0]
    print(f"Example dialogue: {len(dialogue.turns)} turns")
    print(f"First turn: {dialogue.turns[0].utterance[:100]}...")

    print("\n[OK] All dataset loaders working!")
