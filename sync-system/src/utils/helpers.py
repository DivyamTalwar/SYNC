import torch
import numpy as np
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import hashlib


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)


def euclidean_distance(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    return torch.norm(vec1 - vec2, p=2, dim=-1)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num / 1000:.1f}K"
    elif num < 1_000_000_000:
        return f"{num / 1_000_000:.1f}M"
    else:
        return f"{num / 1_000_000_000:.1f}B"


def get_timestamp() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: Path) -> None:
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
