"""
data_dinov2.py
--------------
Pre-computes and caches frozen DINOv2 (ViT-S/14) embeddings for Split CIFAR-100.

Extracts feature vectors once, saving task splits to disk in `cached_features/`.
Subsequent training runs load pre-extracted 384-dimensional features directly,
drastically reducing VRAM usage and training time.
"""

import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import NUM_TASKS, CLASSES_PER_TASK, _class_indices_for_task

DINOV2_EMBED_DIM = 384  # dinov2_vits14 CLS token dimension


def get_dinov2_transforms():
    """
    Standard DINOv2 input transform:
    Resize 32x32 CIFAR images to 224x224 (bicubic) and normalize with ImageNet stats.
    """
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dinov2_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads frozen dinov2_vits14 from PyTorch Hub.
    Ensures model is in eval mode and parameters do not track gradients.
    """
    print("Loading frozen DINOv2 (dinov2_vits14) from torch.hub...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_features_from_loader(model, loader: DataLoader, device: str):
    """
    Passes all images through frozen DINOv2 and returns (features, labels) tensors.
    """
    all_features = []
    all_labels = []

    for images, targets in tqdm(loader, desc="  Extracting", leave=False):
        images = images.to(device)
        feats = model(images)  # (batch_size, 384)
        all_features.append(feats.cpu())
        all_labels.append(targets.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def cache_split_cifar100_dinov2(data_root: str = "./data",
                                cache_dir: str = "./cached_features",
                                batch_size: int = 128,
                                num_workers: int = 2):
    """
    Extracts and caches DINOv2 embeddings for all 10 tasks in Split CIFAR-100.
    Saves:
      cache_dir/task_{i}_train.pt -> {'features': tensor(N, 384), 'labels': tensor(N,)}
      cache_dir/task_{i}_val.pt   -> {'features': tensor(N, 384), 'labels': tensor(N,)}
    """
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for DINOv2 extraction: {device}")

    # Check if all tasks are already cached
    all_exist = True
    for task_id in range(NUM_TASKS):
        train_path = os.path.join(cache_dir, f"task_{task_id}_train.pt")
        val_path = os.path.join(cache_dir, f"task_{task_id}_val.pt")
        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            all_exist = False
            break

    if all_exist:
        print(f"All {NUM_TASKS} tasks are already cached in '{cache_dir}'. Skipping extraction.")
        return

    model = load_dinov2_model(device=device)
    transforms = get_dinov2_transforms()

    print("Loading raw CIFAR-100 with DINOv2 transforms...")
    train_full = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transforms)
    test_full = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transforms)

    for task_id in range(NUM_TASKS):
        train_path = os.path.join(cache_dir, f"task_{task_id}_train.pt")
        val_path = os.path.join(cache_dir, f"task_{task_id}_val.pt")

        if os.path.exists(train_path) and os.path.exists(val_path):
            print(f"Task {task_id} already cached. Skipping.")
            continue

        print(f"\n--- Caching Task {task_id}/{NUM_TASKS - 1} (Classes {task_id * CLASSES_PER_TASK}..{(task_id + 1) * CLASSES_PER_TASK - 1}) ---")

        train_idx = _class_indices_for_task(train_full, task_id)
        val_idx = _class_indices_for_task(test_full, task_id)

        train_loader = DataLoader(
            Subset(train_full, train_idx), batch_size=batch_size,
            shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(
            Subset(test_full, val_idx), batch_size=batch_size,
            shuffle=False, num_workers=num_workers)

        print("  Extracting train split...")
        train_feats, train_labels = extract_features_from_loader(model, train_loader, device)
        print("  Extracting val split...")
        val_feats, val_labels = extract_features_from_loader(model, val_loader, device)

        torch.save({'features': train_feats, 'labels': train_labels}, train_path)
        torch.save({'features': val_feats, 'labels': val_labels}, val_path)
        print(f"  Saved Task {task_id}: train {train_feats.shape}, val {val_feats.shape}")

    print("\nDINOv2 feature caching complete!")


if __name__ == "__main__":
    cache_split_cifar100_dinov2()
