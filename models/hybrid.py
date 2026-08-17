"""
models/hybrid.py
-----------------
Assembles the full architecture per Figure 1:

    Input Embedding -> Neural ODE Layer -> Memory Module (read/write)
        -> Transformer Attention -> Output Layer

ASSUMPTION (Phase 0 only, no DINOv2 yet): "Input Embedding" is a small CNN
that maps raw 32x32x3 CIFAR images to a hidden_dim vector. This matches
the paper's literal pipeline (raw images -> embedding -> ODE) since the
paper never mentions a pretrained backbone. DINOv2 replaces this module
in Phase 1+ of the extended project, not here.

Output head: task-incremental multi-head (10 separate 10-way linear
classifiers, one per task, selected by known task_id at train and test time).
This matches the evaluation protocol used by the baselines in Table 4
(EWC/GEM/A-GEM/PackNet), where a fine-tuning baseline achieves 41.2% --
impossible under a shared 100-way head with no replay (which collapses to 0%).
"""

import torch
import torch.nn as nn

from models.ode import NeuralODELayer
from models.memory import MemoryModule
from models.transformer import TransformerFusion


class ImageEmbedding(nn.Module):
    """Small CNN, literal minimum reading of 'Input Embedding' in Fig. 1."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                           # -> (256,1,1)
        )
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x).flatten(1)   # (batch, 256)
        return self.proj(feat)           # (batch, hidden_dim)


class HybridModel(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_tasks: int = 10,
                 classes_per_task: int = 10, backbone: str = "dinov2",
                 head: str = "multi", dinov2_dim: int = 384,
                 num_slots: int = 200, ode_integration_time: float = 1.0,
                 ode_tolerance: float = 1e-4, transformer_heads: int = 8,
                 transformer_layers: int = 6, memory_decay: float = 0.95):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task

        if self.backbone == "cnn":
            self.embedding = ImageEmbedding(hidden_dim)
        elif self.backbone == "dinov2":
            # Linear projection layer mapping DINOv2's 384-dim CLS embedding to hidden_dim (512)
            self.proj = nn.Linear(dinov2_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose 'cnn' or 'dinov2'.")

        self.ode = NeuralODELayer(hidden_dim, ode_integration_time, ode_tolerance)
        self.memory = MemoryModule(num_slots, hidden_dim, decay_rate=memory_decay)
        self.fusion = TransformerFusion(hidden_dim, transformer_heads, transformer_layers)

        if self.head == "multi":
            # Task-incremental: 10 separate 10-way heads
            self.classifiers = nn.ModuleList(
                [nn.Linear(hidden_dim, classes_per_task) for _ in range(num_tasks)])
        elif self.head == "shared":
            # Class-incremental / Streaming: single shared 100-way output head
            self.classifier = nn.Linear(hidden_dim, num_tasks * classes_per_task)
        else:
            raise ValueError(f"Unknown head '{head}'. Choose 'multi' or 'shared'.")

    def forward(self, x: torch.Tensor, write_memory: bool = True,
                task_id: int = 0) -> torch.Tensor:
        """
        NOTE ON write_memory: this flag only STAGES a memory write, it does
        NOT mutate the memory buffer here. Mutating self.memory.M in-place
        during forward() would corrupt the autograd graph that read() just
        built (PyTorch tracks a version counter on M; changing it before
        loss.backward() runs raises "modified by an inplace operation").
        Call model.commit_memory_write() AFTER optimizer.step(), once the
        backward pass no longer needs the old M values. See
        training/engine_baseline.py's train_one_task() for the call site.
        """
        if self.backbone == "dinov2":
            h0 = self.proj(x)                           # DINOv2 Feature Projection: (B, 384) -> (B, 512)
        else:
            h0 = self.embedding(x)                      # CNN Input Embedding: (B, 3, 32, 32) -> (B, 512)

        h_ode = self.ode(h0)                            # Neural ODE Layer
        r_mem = self.memory.read(h_ode)                 # Memory Module (read only)
        fused = self.fusion(h_ode, r_mem)                # Transformer Attention

        if self.head == "shared":
            logits = self.classifier(fused)             # Single shared 100-way head -> (B, 100)
        else:
            logits = self.classifiers[task_id](fused)   # Task-specific 10-way head -> (B, 10)

        if write_memory:
            # Detach: the write is a non-differentiable bookkeeping update,
            # and must not hold a reference into the graph we still need
            # for backward().
            self._pending_write = h_ode.detach()
        return logits

    def commit_memory_write(self):
        """Call this AFTER optimizer.step(), once per training step that
        used write_memory=True in forward(). Safe no-op if nothing staged."""
        pending = getattr(self, "_pending_write", None)
        if pending is not None:
            self.memory.write(pending)
            self._pending_write = None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("--- 1. Testing DINOv2 + Multi-Head ---")
    model_dm = HybridModel(backbone="dinov2", head="multi")
    x_dinov2 = torch.randn(4, 384)
    logits_dm = model_dm(x_dinov2, task_id=3)
    print("Output shape:", logits_dm.shape)  # expect (4, 10)
    logits_dm.sum().backward()
    print("Grad OK:", model_dm.proj.weight.grad is not None)

    print("\n--- 2. Testing DINOv2 + Shared-Head (100-way) ---")
    model_ds = HybridModel(backbone="dinov2", head="shared")
    logits_ds = model_ds(x_dinov2)
    print("Output shape:", logits_ds.shape)  # expect (4, 100)
    logits_ds.sum().backward()
    print("Grad OK:", model_ds.classifier.weight.grad is not None)

    print("\n--- 3. Testing CNN + Multi-Head ---")
    model_cm = HybridModel(backbone="cnn", head="multi")
    x_cnn = torch.randn(4, 3, 32, 32)
    logits_cm = model_cm(x_cnn, task_id=3)
    print("Output shape:", logits_cm.shape)  # expect (4, 10)
    logits_cm.sum().backward()
    print("Grad OK:", model_cm.embedding.proj.weight.grad is not None)

    print("\n--- 4. Testing CNN + Shared-Head ---")
    model_cs = HybridModel(backbone="cnn", head="shared")
    logits_cs = model_cs(x_cnn)
    print("Output shape:", logits_cs.shape)  # expect (4, 100)
    logits_cs.sum().backward()
    print("Grad OK:", model_cs.classifier.weight.grad is not None)
