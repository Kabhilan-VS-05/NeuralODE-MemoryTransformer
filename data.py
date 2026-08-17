"""
data.py
-------
Split CIFAR-100 into 10 sequential tasks of 10 classes each, as described
in Table 3 of the base paper (Task number: 10, Class number: 100).

The class-to-task assignment is fixed and logged for reproducibility:
Task 0 -> classes [0..9], Task 1 -> classes [10..19], ... Task 9 -> [90..99]
(CIFAR-100's default label ordering, no shuffling — the simplest, most
literal reading since the paper does not specify a particular task order).
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T

NUM_TASKS = 10
CLASSES_PER_TASK = 10
NUM_CLASSES_TOTAL = NUM_TASKS * CLASSES_PER_TASK  # 100, matches Table 3

# Standard CIFAR-100 normalization stats
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def _class_indices_for_task(dataset: Dataset, task_id: int):
    """Return dataset indices whose label falls in this task's class range."""
    lo = task_id * CLASSES_PER_TASK
    hi = lo + CLASSES_PER_TASK
    targets = dataset.targets  # list[int]
    return [i for i, y in enumerate(targets) if lo <= y < hi]


def build_task_loaders(data_root: str = "./data", batch_size: int = 64,
                        num_workers: int = 2):
    """
    Returns:
        train_loaders: list of 10 DataLoaders, one per task
        val_loaders:   list of 10 DataLoaders, one per task (held-out val split)
    Each loader yields (image, label) with label in the GLOBAL 0..99 range
    (i.e. NOT remapped to 0..9 per task) — this matches a single shared
    100-way output head, which is the literal reading of Table 3's
    "Class number: 100" and is consistent with the instant-collapse-to-0%
    behavior observed in the Phase 0 run (a class-incremental head, not a
    task-incremental multi-head setup).
    """
    train_full = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=get_transforms(True))
    test_full = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=get_transforms(False))

    train_loaders, val_loaders = [], []
    for task_id in range(NUM_TASKS):
        train_idx = _class_indices_for_task(train_full, task_id)
        val_idx = _class_indices_for_task(test_full, task_id)

        train_loaders.append(DataLoader(
            Subset(train_full, train_idx), batch_size=batch_size,
            shuffle=True, num_workers=num_workers, drop_last=True))
        val_loaders.append(DataLoader(
            Subset(test_full, val_idx), batch_size=batch_size,
            shuffle=False, num_workers=num_workers))

    return train_loaders, val_loaders


if __name__ == "__main__":
    # Quick sanity check — run this locally before anything else.
    train_loaders, val_loaders = build_task_loaders(batch_size=8, num_workers=0)
    for t, (tl, vl) in enumerate(zip(train_loaders, val_loaders)):
        xb, yb = next(iter(tl))
        print(f"Task {t}: train batch {xb.shape}, labels {yb.min().item()}-{yb.max().item()}, "
              f"train size {len(tl.dataset)}, val size {len(vl.dataset)}")
