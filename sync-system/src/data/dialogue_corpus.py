from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from src.data.datasets import DialogueExample, DialogueTurn
from src.utils.logging import get_logger

logger = get_logger("data.dialogue_corpus")


@dataclass
class CKMTrainingPair:
    dialogue_id: str
    target_speaker: str
    utterance_sequence: List[str]  
    context_utterances: List[str]  
    turn_indices: List[int]
    next_utterance: Optional[str] = None


class DialogueCorpusHandler:

    def __init__(
        self,
        context_window: int = 5,
        min_utterances: int = 2,
        max_utterances: int = 10,
        include_next_utterance: bool = True,
    ):
        """
        Initialize dialogue corpus handler

        Args:
            context_window: Number of recent turns to include
            min_utterances: Minimum utterances per training pair
            max_utterances: Maximum utterances per training pair
            include_next_utterance: Include next utterance for self-supervision
        """
        self.context_window = context_window
        self.min_utterances = min_utterances
        self.max_utterances = max_utterances
        self.include_next_utterance = include_next_utterance

    def extract_training_pairs(
        self,
        dialogue: DialogueExample
    ) -> List[CKMTrainingPair]:
        """
        Extract CKM training pairs from a dialogue

        For each speaker, we create training pairs at different points
        in the dialogue, simulating the process of updating CKM as
        new utterances arrive.

        Args:
            dialogue: Dialogue example

        Returns:
            List of CKM training pairs
        """
        pairs = []

        # Group turns by speaker
        speaker_turns = defaultdict(list)
        for turn in dialogue.turns:
            speaker_turns[turn.speaker].append(turn)

        # For each speaker, create training pairs
        for speaker, turns in speaker_turns.items():
            if len(turns) < self.min_utterances:
                continue

            # Create pairs at different dialogue points
            for end_idx in range(self.min_utterances, len(turns) + 1):
                # Get utterances from this speaker up to this point
                speaker_utterances = turns[:end_idx]

                # Get context from other speakers
                context_turns = [
                    t for t in dialogue.turns
                    if t.speaker != speaker and t.turn_index < speaker_utterances[-1].turn_index
                ]
                context_turns = context_turns[-self.context_window:]

                # Extract utterance texts
                utterance_sequence = [t.utterance for t in speaker_utterances[-self.max_utterances:]]
                context_utterances = [t.utterance for t in context_turns]
                turn_indices = [t.turn_index for t in speaker_utterances[-self.max_utterances:]]

                # Get next utterance if available (for self-supervision)
                next_utterance = None
                if self.include_next_utterance and end_idx < len(turns):
                    next_utterance = turns[end_idx].utterance

                pair = CKMTrainingPair(
                    dialogue_id=dialogue.dialogue_id,
                    target_speaker=speaker,
                    utterance_sequence=utterance_sequence,
                    context_utterances=context_utterances,
                    turn_indices=turn_indices,
                    next_utterance=next_utterance
                )
                pairs.append(pair)

        return pairs

    def extract_training_pairs_batch(
        self,
        dialogues: List[DialogueExample],
        verbose: bool = True
    ) -> List[CKMTrainingPair]:
        """
        Extract training pairs from multiple dialogues

        Args:
            dialogues: List of dialogue examples
            verbose: Show progress bar

        Returns:
            List of all training pairs
        """
        all_pairs = []

        iterator = tqdm(dialogues, desc="Extracting pairs") if verbose else dialogues

        for dialogue in iterator:
            pairs = self.extract_training_pairs(dialogue)
            all_pairs.extend(pairs)

        logger.info(
            f"Extracted {len(all_pairs)} training pairs from {len(dialogues)} dialogues "
            f"({len(all_pairs) / len(dialogues):.1f} pairs/dialogue)"
        )

        return all_pairs

    def create_speaker_profiles(
        self,
        dialogues: List[DialogueExample]
    ) -> Dict[str, Dict]:
        """
        Create speaker profiles for analysis

        Args:
            dialogues: List of dialogues

        Returns:
            Dict with speaker statistics
        """
        speaker_stats = defaultdict(lambda: {
            "total_utterances": 0,
            "total_dialogues": 0,
            "avg_utterance_length": [],
            "dialogues": []
        })

        for dialogue in dialogues:
            speakers_in_dialogue = set()

            for turn in dialogue.turns:
                speaker = turn.speaker
                speakers_in_dialogue.add(speaker)

                speaker_stats[speaker]["total_utterances"] += 1
                speaker_stats[speaker]["avg_utterance_length"].append(len(turn.utterance))

            for speaker in speakers_in_dialogue:
                speaker_stats[speaker]["total_dialogues"] += 1
                speaker_stats[speaker]["dialogues"].append(dialogue.dialogue_id)

        # Compute averages
        for speaker, stats in speaker_stats.items():
            if stats["avg_utterance_length"]:
                stats["avg_utterance_length"] = np.mean(stats["avg_utterance_length"])

        logger.info(f"Created profiles for {len(speaker_stats)} speakers")

        return dict(speaker_stats)


class CKMDatasetWrapper(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for CKM training pairs

    Converts CKM training pairs into format suitable for training.
    """

    def __init__(
        self,
        training_pairs: List[CKMTrainingPair],
        embedding_dict: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
        """
        Initialize CKM dataset wrapper

        Args:
            training_pairs: List of CKM training pairs
            embedding_dict: Pre-computed embeddings (dialogue_id -> [turn_embeddings])
        """
        self.training_pairs = training_pairs
        self.embedding_dict = embedding_dict

    def __len__(self) -> int:
        return len(self.training_pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get training pair

        Returns:
            Dict with:
            - utterance_sequence: List[str]
            - context_utterances: List[str]
            - next_utterance: Optional[str]
            - embeddings: Optional[torch.Tensor] if pre-computed
        """
        pair = self.training_pairs[idx]

        item = {
            "dialogue_id": pair.dialogue_id,
            "target_speaker": pair.target_speaker,
            "utterance_sequence": pair.utterance_sequence,
            "context_utterances": pair.context_utterances,
            "turn_indices": pair.turn_indices,
            "next_utterance": pair.next_utterance,
        }

        # Add pre-computed embeddings if available
        if self.embedding_dict and pair.dialogue_id in self.embedding_dict:
            dialogue_embeddings = self.embedding_dict[pair.dialogue_id]
            # Get embeddings for the utterances in this pair
            pair_embeddings = [
                dialogue_embeddings[turn_idx]
                for turn_idx in pair.turn_indices
                if turn_idx < len(dialogue_embeddings)
            ]
            if pair_embeddings:
                item["embeddings"] = torch.stack(pair_embeddings)

        return item

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        """Get PyTorch DataLoader"""
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader"""
        # Simple collation - return list of dicts
        # More sophisticated collation can be done during training
        return {
            "dialogue_ids": [item["dialogue_id"] for item in batch],
            "target_speakers": [item["target_speaker"] for item in batch],
            "utterance_sequences": [item["utterance_sequence"] for item in batch],
            "context_utterances": [item["context_utterances"] for item in batch],
            "turn_indices": [item["turn_indices"] for item in batch],
            "next_utterances": [item["next_utterance"] for item in batch],
            "embeddings": [item.get("embeddings") for item in batch],
        }


def prepare_ckm_training_data(
    dialogues: List[DialogueExample],
    context_window: int = 5,
    min_utterances: int = 2,
    max_utterances: int = 10,
    include_next_utterance: bool = True,
    embedding_dict: Optional[Dict] = None,
) -> CKMDatasetWrapper:
    """
    Prepare CKM training data from dialogues

    Args:
        dialogues: List of dialogue examples
        context_window: Context window size
        min_utterances: Minimum utterances per pair
        max_utterances: Maximum utterances per pair
        include_next_utterance: Include next utterance
        embedding_dict: Pre-computed embeddings

    Returns:
        CKMDatasetWrapper ready for training
    """
    handler = DialogueCorpusHandler(
        context_window=context_window,
        min_utterances=min_utterances,
        max_utterances=max_utterances,
        include_next_utterance=include_next_utterance
    )

    training_pairs = handler.extract_training_pairs_batch(dialogues)

    dataset = CKMDatasetWrapper(
        training_pairs=training_pairs,
        embedding_dict=embedding_dict
    )

    logger.info(f"Prepared CKM dataset with {len(dataset)} training pairs")

    return dataset


if __name__ == "__main__":
    # Test dialogue corpus handler
    print("=" * 80)
    print("TESTING DIALOGUE CORPUS HANDLER")
    print("=" * 80)

    # Create test dialogue
    print("\n[1/3] Creating test dialogue...")
    from src.data.datasets import DialogueExample, DialogueTurn

    test_dialogue = DialogueExample(
        dialogue_id="test_1",
        turns=[
            DialogueTurn("user", "What is machine learning?", 0),
            DialogueTurn("assistant", "ML is a subset of AI where systems learn from data.", 1),
            DialogueTurn("user", "Can you give me an example?", 2),
            DialogueTurn("assistant", "Sure! Email spam filters use ML to identify spam.", 3),
            DialogueTurn("user", "How does it learn?", 4),
            DialogueTurn("assistant", "By analyzing patterns in labeled training data.", 5),
        ],
        num_speakers=2
    )
    print(f"[OK] Created dialogue with {len(test_dialogue.turns)} turns")

    # Extract training pairs
    print("\n[2/3] Extracting training pairs...")
    handler = DialogueCorpusHandler(context_window=3, min_utterances=2)
    pairs = handler.extract_training_pairs(test_dialogue)
    print(f"[OK] Extracted {len(pairs)} training pairs")
    print(f"Example pair:")
    print(f"  Speaker: {pairs[0].target_speaker}")
    print(f"  Utterances: {len(pairs[0].utterance_sequence)}")
    print(f"  Context: {len(pairs[0].context_utterances)}")

    # Create dataset
    print("\n[3/3] Creating PyTorch dataset...")
    dataset = CKMDatasetWrapper(pairs)
    print(f"[OK] Created dataset with {len(dataset)} examples")
    print(f"Example item keys: {list(dataset[0].keys())}")

    print("\n[OK] All dialogue corpus components working!")
