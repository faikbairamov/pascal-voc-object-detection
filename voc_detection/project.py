"""Project paths, configuration, random seeds, and data loaders."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import VOCDetectionDataset, detection_collate_fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Load JSON settings; resolve repository-relative paths consistently."""
    path = PROJECT_ROOT / "configs/baseline.json" if path is None else Path(path)
    config = json.loads(path.read_text())
    return config


def seed_everything(seed):
    """Seed initialization; reproducibility is limited to compatible environments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(config, split, shuffle=False):
    """Build a loader using saved train/dev manifests, never the final held-out set."""
    if split not in {"train", "dev"}:
        raise ValueError("Choose train or dev; final evaluation is kept separate.")
    data = VOCDetectionDataset(
        PROJECT_ROOT / config["dataset_dir"],
        PROJECT_ROOT / "splits" / f"{split}.txt",
        horizontal_flip_prob=config.get("horizontal_flip_prob", 0.0) if split == "train" else 0.0,
    )
    loader = DataLoader(
        data, batch_size=config["batch_size"], shuffle=shuffle,
        num_workers=0, collate_fn=detection_collate_fn,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    return data, loader
