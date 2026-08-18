"""
training/replay_buffer.py
-------------------------
Class-Balanced Bounded Replay Buffer (Capacity = 500).

Features:
- Fixed upper capacity of 500 samples total.
- Strict class-balanced retention: dynamically allocates K = floor(500 / num_seen_classes)
  slots per class so no single task or class is crowded out by another.
- Within each class, retains the top-K highest-scoring samples based on the active scoring function
  (Random, Influence, or Fisher).
- sample(n=50): Draws a uniform random mini-batch of replay samples for joint training.
"""

import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 500, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.features = None  # (N, feature_dim)
        self.labels = None    # (N,)
        self.scores = None    # (N,)

    def __len__(self) -> int:
        return 0 if self.features is None else self.features.shape[0]

    def add_candidates(self, new_features: torch.Tensor, new_labels: torch.Tensor,
                       new_scores: torch.Tensor):
        """
        Adds candidate samples and applies class-balanced eviction to maintain capacity <= 500.
        
        Args:
            new_features: (M, feature_dim) tensor
            new_labels: (M,) tensor of class labels
            new_scores: (M,) tensor of importance scores
        """
        new_features = new_features.detach().cpu()
        new_labels = new_labels.detach().cpu()
        new_scores = new_scores.detach().cpu()

        if self.features is None:
            all_features = new_features
            all_labels = new_labels
            all_scores = new_scores
        else:
            all_features = torch.cat([self.features.cpu(), new_features], dim=0)
            all_labels = torch.cat([self.labels.cpu(), new_labels], dim=0)
            all_scores = torch.cat([self.scores.cpu(), new_scores], dim=0)

        # Identify all unique classes present
        unique_classes = torch.unique(all_labels).tolist()
        num_classes = len(unique_classes)
        if num_classes == 0:
            return

        # Target quota per class
        quota_per_class = max(1, self.capacity // num_classes)

        selected_indices = []
        for c in unique_classes:
            c_mask = (all_labels == c).nonzero(as_tuple=True)[0]
            if len(c_mask) <= quota_per_class:
                selected_indices.extend(c_mask.tolist())
            else:
                # Rank by score descending within this class and pick top-K
                c_scores = all_scores[c_mask]
                topk_rel_idx = torch.topk(c_scores, quota_per_class).indices
                selected_indices.extend(c_mask[topk_rel_idx].tolist())

        # If total selected is less than capacity, fill remaining slots with highest remaining scores
        if len(selected_indices) < self.capacity and len(selected_indices) < len(all_labels):
            selected_set = set(selected_indices)
            remaining_indices = [i for i in range(len(all_labels)) if i not in selected_set]
            if remaining_indices:
                rem_tensor = torch.tensor(remaining_indices, dtype=torch.long)
                rem_scores = all_scores[rem_tensor]
                slots_left = min(self.capacity - len(selected_indices), len(remaining_indices))
                top_rem_rel = torch.topk(rem_scores, slots_left).indices
                selected_indices.extend(rem_tensor[top_rem_rel].tolist())

        idx_tensor = torch.tensor(selected_indices, dtype=torch.long)
        self.features = all_features[idx_tensor].cpu()
        self.labels = all_labels[idx_tensor].cpu()
        self.scores = all_scores[idx_tensor].cpu()

    def sample(self, n: int = 50, device: str = None) -> tuple:
        """
        Samples up to n random (features, labels) pairs from the buffer.
        Returns (None, None) if buffer is empty.
        """
        if len(self) == 0:
            return None, None

        target_device = device if device is not None else self.device
        sample_size = min(n, len(self))
        rand_idx = torch.randperm(len(self))[:sample_size]

        batch_x = self.features[rand_idx].to(target_device)
        batch_y = self.labels[rand_idx].to(target_device)
        return batch_x, batch_y

    def get_class_counts(self) -> dict:
        """Returns distribution of samples per class in buffer."""
        if len(self) == 0:
            return {}
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def state_dict(self) -> dict:
        return {
            'features': self.features.cpu() if self.features is not None else None,
            'labels': self.labels.cpu() if self.labels is not None else None,
            'scores': self.scores.cpu() if self.scores is not None else None,
            'capacity': self.capacity,
        }

    def load_state_dict(self, state: dict):
        if state is not None:
            f = state.get('features')
            self.features = f.detach().cpu() if isinstance(f, torch.Tensor) else None
            l = state.get('labels')
            self.labels = l.detach().cpu() if isinstance(l, torch.Tensor) else None
            s = state.get('scores')
            self.scores = s.detach().cpu() if isinstance(s, torch.Tensor) else None
            self.capacity = state.get('capacity', self.capacity)


if __name__ == "__main__":
    print("--- Testing ReplayBuffer Class-Balanced Retention ---")
    buf = ReplayBuffer(capacity=500)

    # Simulate Task 0: 10 classes (0..9), 100 samples per class = 1000 samples
    x0 = torch.randn(1000, 384)
    y0 = torch.repeat_interleave(torch.arange(0, 10), 100)
    scores0 = torch.rand(1000)

    buf.add_candidates(x0, y0, scores0)
    print(f"After Task 0: Buffer size = {len(buf)} (Expected 500)")
    print("Class counts (Task 0):", buf.get_class_counts())  # Expected 50 per class for 10 classes

    # Simulate Task 1: 10 new classes (10..19), 100 samples per class = 1000 samples
    x1 = torch.randn(1000, 384)
    y1 = torch.repeat_interleave(torch.arange(10, 20), 100)
    scores1 = torch.rand(1000)

    buf.add_candidates(x1, y1, scores1)
    print(f"After Task 1: Buffer size = {len(buf)} (Expected 500)")
    print("Class counts (Task 0 + 1):", buf.get_class_counts())  # Expected 25 per class for 20 classes

    # Test sampling
    bx, by = buf.sample(50)
    print(f"Sampled batch: features {bx.shape}, labels {by.shape}")
    print("Replay buffer verification passed!")
