import os
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.core.ckm import CollaboratorKnowledgeModel, CKMState
from src.data.datasets import ShareGPTDataset
from src.data.dialogue_corpus import prepare_ckm_training_data, CKMDatasetWrapper
from src.data.preprocessing import EmbeddingPreprocessor
from src.llm.embeddings import CohereEmbeddingsClient
from src.utils.logging import get_logger
from src.utils.checkpoints import CheckpointManager
from config.training import CKMPretrainingConfig

logger = get_logger("training.pretraining")


class CKMPretrainer:
    def __init__(
        self,
        model: CollaboratorKnowledgeModel,
        config: Optional[CKMPretrainingConfig] = None,
        device: str = "cpu",
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.config = config or CKMPretrainingConfig()
        self.device = device
        self.use_wandb = use_wandb

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.min_learning_rate
        )

        # Loss function
        self.criterion = nn.MSELoss()  # For embedding prediction

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=Path("./checkpoints/ckm_pretraining"),
            keep_best_n=3
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info(f"Initialized CKMPretrainer with config: {self.config}")

    def train(
        self,
        train_dataset: CKMDatasetWrapper,
        val_dataset: Optional[CKMDatasetWrapper] = None,
    ) -> Dict[str, List[float]]:
        """
        Train CKM model

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Returns:
            Dict with training history
        """
        logger.info("=" * 80)
        logger.info("STARTING CKM PRE-TRAINING")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info("=" * 80)

        # Create dataloaders
        train_loader = train_dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = None
        if val_dataset:
            val_loader = val_dataset.get_dataloader(
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"ckm_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(self.config)
            )

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

        # Training loop
        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            logger.info(f"{'='*80}")

            # Train
            train_loss = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            # Validate
            if val_loader:
                val_loss = self._validate_epoch(val_loader, epoch)
                history["val_loss"].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metric=val_loss,
                        is_best=True
                    )
                    logger.info(f"[OK] Saved best model (val_loss={val_loss:.4f})")

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history["learning_rate"].append(current_lr)
            self.scheduler.step()

            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": current_lr
                }
                if val_loader:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)

            # Summary
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            if val_loader:
                logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")

        # Save final model
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.config.epochs,
            metric=history["train_loss"][-1],
            is_best=False
        )

        logger.info("\n" + "=" * 80)
        logger.info("CKM PRE-TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.use_wandb:
            wandb.finish()

        return history

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Training")

        for batch in progress_bar:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    def _train_step(self, batch: Dict) -> float:
        """Single training step"""
        self.optimizer.zero_grad()

        # Get batch data
        utterance_sequences = batch["utterance_sequences"]
        embeddings_list = batch["embeddings"]

        # Compute loss
        batch_loss = 0.0
        valid_samples = 0

        for utterance_seq, embeddings in zip(utterance_sequences, embeddings_list):
            if embeddings is None or len(embeddings) < 2:
                continue

            # Move to device
            embeddings = embeddings.to(self.device)

            # Input: all but last embedding
            # Target: last embedding (prediction target)
            input_embeddings = embeddings[:-1].unsqueeze(0)  # (1, seq_len-1, 1024)
            target_embedding = embeddings[-1]  # (1024,)

            # Forward pass through CKM
            ckm_output, hidden = self.model(input_embeddings, previous_ckm_state=None)

            # Predict next embedding from CKM state
            # Add a prediction head (or use CKM output directly)
            predicted_embedding = self._predict_next_embedding(ckm_output)

            # Loss: MSE between predicted and target
            loss = self.criterion(predicted_embedding, target_embedding)
            batch_loss += loss
            valid_samples += 1

        if valid_samples == 0:
            return 0.0

        # Average loss
        batch_loss = batch_loss / valid_samples

        # Backward pass
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self.global_step += 1

        return batch_loss.item()

    def _predict_next_embedding(self, ckm_output: torch.Tensor) -> torch.Tensor:
        """
        Predict next embedding from CKM output

        For now, use a simple linear projection.
        Could be replaced with a more sophisticated predictor.

        Args:
            ckm_output: CKM output (B, 128)

        Returns:
            Predicted embedding (B, 1024)
        """
        # Create predictor if not exists
        if not hasattr(self, 'embedding_predictor'):
            self.embedding_predictor = nn.Linear(128, 1024).to(self.device)

        return self.embedding_predictor(ckm_output.squeeze(0))

    @torch.no_grad()
    def _validate_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validating"):
            # Same as training step but without gradients
            utterance_sequences = batch["utterance_sequences"]
            embeddings_list = batch["embeddings"]

            batch_loss = 0.0
            valid_samples = 0

            for utterance_seq, embeddings in zip(utterance_sequences, embeddings_list):
                if embeddings is None or len(embeddings) < 2:
                    continue

                embeddings = embeddings.to(self.device)
                input_embeddings = embeddings[:-1].unsqueeze(0)
                target_embedding = embeddings[-1]

                ckm_output, hidden = self.model(input_embeddings, previous_ckm_state=None)
                predicted_embedding = self._predict_next_embedding(ckm_output)

                loss = self.criterion(predicted_embedding, target_embedding)
                batch_loss += loss
                valid_samples += 1

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                total_loss += batch_loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss


def pretrain_ckm(
    num_dialogues: int = 1000,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    device: str = "cpu",
    use_wandb: bool = False,
    save_dir: Optional[Path] = None,
) -> CollaboratorKnowledgeModel:
    """
    Complete CKM pre-training pipeline

    Args:
        num_dialogues: Number of dialogues to use
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
        use_wandb: Log to wandb
        save_dir: Save directory

    Returns:
        Trained CKM model
    """
    logger.info("=" * 80)
    logger.info("CKM PRE-TRAINING PIPELINE")
    logger.info("=" * 80)

    # 1. Load dialogue dataset
    logger.info("\n[1/5] Loading ShareGPT dataset...")
    dataset = ShareGPTDataset(split="train", max_samples=num_dialogues)
    logger.info(f"[OK] Loaded {len(dataset)} dialogues")

    # 2. Pre-compute embeddings
    logger.info("\n[2/5] Pre-computing embeddings...")
    embedding_preprocessor = EmbeddingPreprocessor()
    embedding_dict = embedding_preprocessor.precompute_dialogue_embeddings(
        dataset.dialogues,
        cache_name=f"sharegpt_{num_dialogues}",
        force_recompute=False
    )
    logger.info(f"[OK] Pre-computed embeddings for {len(embedding_dict)} dialogues")

    # 3. Prepare CKM training data
    logger.info("\n[3/5] Preparing CKM training data...")
    ckm_dataset = prepare_ckm_training_data(
        dialogues=dataset.dialogues,
        context_window=5,
        min_utterances=2,
        max_utterances=10,
        include_next_utterance=True,
        embedding_dict=embedding_dict
    )
    logger.info(f"[OK] Prepared {len(ckm_dataset)} training pairs")

    # Split train/val
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        list(range(len(ckm_dataset))),
        test_size=0.1,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(ckm_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(ckm_dataset, val_indices)

    logger.info(f"[OK] Split: {len(train_dataset)} train, {len(val_dataset)} val")

    # 4. Create and train model
    logger.info("\n[4/5] Creating CKM model...")
    model = CollaboratorKnowledgeModel(
        input_dim=1024,  # Cohere embedding dim
        output_dim=128,
        num_layers=2,
        num_heads=8
    )
    logger.info(f"[OK] Created CKM with {sum(p.numel() for p in model.parameters())} parameters")

    # Create config
    config = CKMPretrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Train
    logger.info("\n[5/5] Training CKM...")
    trainer = CKMPretrainer(model, config, device, use_wandb)
    history = trainer.train(train_dataset, val_dataset)

    logger.info("\n[OK] CKM pre-training complete!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")

    return model


if __name__ == "__main__":
    # Test CKM pretraining (without real training)
    print("=" * 80)
    print("TESTING CKM PRETRAINING SETUP")
    print("=" * 80)

    # Load small dataset
    print("\n[1/2] Loading test dataset...")
    dataset = ShareGPTDataset(split="train", max_samples=10)
    print(f"[OK] Loaded {len(dataset)} dialogues")

    # Prepare training data
    print("\n[2/2] Preparing training data...")
    ckm_dataset = prepare_ckm_training_data(
        dialogues=dataset.dialogues,
        context_window=3,
        min_utterances=2
    )
    print(f"[OK] Prepared {len(ckm_dataset)} training pairs")

    print("\n[OK] CKM pretraining setup working!")
    print("NOTE: Full training requires running pretrain_ckm() with real data")
