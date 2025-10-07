import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil
from datetime import datetime
from src.utils.logging import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger("checkpoints")


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_best_n: int = 3,
        metric_name: str = "loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.mode = mode
        self.best_checkpoints: List[Dict[str, Any]] = []

        ensure_dir(self.checkpoint_dir)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        if extra_data is not None:
            checkpoint_data.update(extra_data)

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        self._update_best_checkpoints(checkpoint_path, metrics)

        return checkpoint_path

    def _update_best_checkpoints(
        self,
        checkpoint_path: Path,
        metrics: Dict[str, float],
    ) -> None:
        if self.metric_name not in metrics:
            return

        metric_value = metrics[self.metric_name]

        checkpoint_info = {
            "path": checkpoint_path,
            "metric_value": metric_value,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.best_checkpoints.append(checkpoint_info)

        reverse = self.mode == "max"
        self.best_checkpoints.sort(key=lambda x: x["metric_value"], reverse=reverse)

        if len(self.best_checkpoints) > self.keep_best_n:
            removed = self.best_checkpoints.pop()
            removed_path = removed["path"]
            if removed_path.exists():
                removed_path.unlink()
                logger.info(f"Removed checkpoint: {removed_path}")

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_data["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data

    def get_best_checkpoint(self) -> Optional[Path]:
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[0]["path"]

    def get_latest_checkpoint(self) -> Optional[Path]:
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)


def save_model(
    model: torch.nn.Module,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(save_path.parent)

    save_data = {
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if metadata is not None:
        save_data["metadata"] = metadata

    torch.save(save_data, save_path)
    logger.info(f"Saved model to {save_path}")


def load_model(
    model: torch.nn.Module,
    load_path: Path,
) -> Dict[str, Any]:
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    model_data = torch.load(load_path)
    model.load_state_dict(model_data["model_state_dict"])
    logger.info(f"Loaded model from {load_path}")

    return model_data
