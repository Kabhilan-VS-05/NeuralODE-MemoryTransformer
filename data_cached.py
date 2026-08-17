"""
data_cached.py
--------------
Lightweight dataset and loader for pre-cached DINOv2 embeddings.

Exposes `build_task_loaders()` with an identical interface to `data.py`,
returning `(train_loaders, val_loaders)` where each batch yields
`(features, global_labels)` with shapes `((B, 384), (B,))`.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

from data import NUM_TASKS, CLASSES_PER_TASK

DINOV2_EMBED_DIM = 384


class CachedTaskDataset(Dataset):
    """
    In-memory dataset holding pre-extracted (features, labels) for a single task.
    """

    def __init__(self, pt_path: str):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Cached file '{pt_path}' not found. "
                f"Please run `python data_dinov2.py` first to pre-extract features."
            )
        data = torch.load(pt_path, weights_only=True)
        self.features = data['features']  # (N, 384)
        self.labels = data['labels']      # (N,) in global 0..99 range

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def build_task_loaders(cache_dir: str = "./cached_features",
                       batch_size: int = 64,
                       num_workers: int = 0):
    """
    Returns:
        train_loaders: list of 10 DataLoaders yielding (features, global_label)
        val_loaders:   list of 10 DataLoaders yielding (features, global_label)
    `features` is a (B, 384) tensor; `global_label` is in 0..99.
    """
    train_loaders, val_loaders = [], []

    for task_id in range(NUM_TASKS):
        train_path = os.path.join(cache_dir, f"task_{task_id}_train.pt")
        val_path = os.path.join(cache_dir, f"task_{task_id}_val.pt")

        train_ds = CachedTaskDataset(train_path)
        val_ds = CachedTaskDataset(val_path)

        train_loaders.append(DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True
        ))
        val_loaders.append(DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        ))

    return train_loaders, val_loaders


def build_streaming_loader(cache_dir: str = "./cached_features",
                           batch_size: int = 64,
                           epochs_per_stream: int = 1,
                           num_workers: int = 0):
    """
    Returns a generator that yields batches seamlessly across all 10 tasks,
    repeating the entire sequence `epochs_per_stream` times to simulate continuous data.
    Also returns `val_loaders` for periodic evaluation.
    """
    train_loaders, val_loaders = build_task_loaders(cache_dir, batch_size, num_workers)
    
    total_batches = sum(len(loader) for loader in train_loaders) * epochs_per_stream

    def stream_generator():
        for _ in range(epochs_per_stream):
            for loader in train_loaders:
                for x, y in loader:
                    yield x, y

    return stream_generator(), val_loaders, total_batches


if __name__ == "__main__":
    import sys
    print("Testing data_cached loader...")
    try:
        train_loaders, val_loaders = build_task_loaders()
        print(f"Successfully loaded {len(train_loaders)} train loaders and {len(val_loaders)} val loaders.")
        fb, yb = next(iter(train_loaders[0]))
        print(f"Task 0 batch shape: features {fb.shape}, labels {yb.shape}")
    except FileNotFoundError as e:
        print(f"Notice: {e}")
