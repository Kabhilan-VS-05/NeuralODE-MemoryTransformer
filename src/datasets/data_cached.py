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

from src.datasets.data import NUM_TASKS, CLASSES_PER_TASK

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
                           num_workers: int = 0,
                           blur_window_std: float = 2000.0,
                           seed: int = 42):
    """
    Returns a single DataLoader that yields batches across all 10 tasks,
    repeating `epochs_per_stream` times.
    
    CRITICAL CHANGE (Phase 4): 
    Instead of discrete task boundaries, we apply a Gaussian noise shuffle to the
    strict sequential order of the 50,000 samples. This blurs the boundaries so 
    Task N smoothly overlaps with Task N-1 and N+1, simulating true continuous streaming.
    """
    import random
    from torch.utils.data import TensorDataset
    
    # Fix seed for identical apple-to-apples streams across different scoring runs
    gen = torch.Generator()
    gen.manual_seed(seed)
    random.seed(seed)
    
    # We still need val_loaders for periodic evaluation
    _, val_loaders = build_task_loaders(cache_dir, batch_size, num_workers)
    
    all_x, all_y = [], []
    for task_id in range(NUM_TASKS):
        train_path = os.path.join(cache_dir, f"task_{task_id}_train.pt")
        data = torch.load(train_path, weights_only=True)
        # Randomly shuffle within the task itself first to break ordered class blocks using the generator
        idx = torch.randperm(data['features'].shape[0], generator=gen)
        all_x.append(data['features'][idx])
        all_y.append(data['labels'][idx])
        
    global_x = torch.cat(all_x, dim=0)
    global_y = torch.cat(all_y, dim=0)
    
    total_samples = len(global_x)
    
    def get_blurred_indices():
        # Add Gaussian noise to the sequential index to blur boundaries
        noisy_indices = [(i, i + random.gauss(0, blur_window_std)) for i in range(total_samples)]
        noisy_indices.sort(key=lambda x: x[1])
        return [x[0] for x in noisy_indices]

    stream_x, stream_y = [], []
    for _ in range(epochs_per_stream):
        shuffled_idx = torch.tensor(get_blurred_indices(), dtype=torch.long)
        stream_x.append(global_x[shuffled_idx])
        stream_y.append(global_y[shuffled_idx])
        
    final_x = torch.cat(stream_x, dim=0)
    final_y = torch.cat(stream_y, dim=0)
    
    streaming_dataset = TensorDataset(final_x, final_y)
    streaming_loader = DataLoader(
        streaming_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, drop_last=True
    )
    
    total_batches = len(streaming_loader)
    
    return streaming_loader, val_loaders, total_batches


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
