"""
models/memory.py
-----------------
Content-addressable external memory bank, literal reading of Eq. 7-8.

Read (Eq. 7):
    r_t = sum_i w_t^r(i) * M_t(i)

Write (Eq. 7):
    M_t(i) = M_{t-1}(i) + w_t^w(i) * a_t

Attention weights (Eq. 8):
    w_t(i) = softmax_i( beta * k_t . K(i) )

ASSUMPTION (paper does not give a value for beta): beta = 1.0, exposed as
a constructor argument so it can be tuned. Document this in your writeup.

Hyperparameters from Table 2 (defaults used):
    Memory slot number    : 200
    Memory slot dimension : 512
    Memory decay rate     : 0.95 (applied per training step, see decay())
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    def __init__(self, num_slots: int = 200, slot_dim: int = 512,
                 beta: float = 1.0, decay_rate: float = 0.95):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.beta = beta
        self.decay_rate = decay_rate

        # Memory values M(i) and their addressing keys K(i).
        # Registered as buffers (not nn.Parameters) since the paper treats
        # memory as an explicit store updated by the write rule (Eq. 7),
        # not learned directly via backprop -- the read path IS differentiable
        # (attention over M), so gradients still flow through *usage* of memory.
        self.register_buffer("M", torch.zeros(num_slots, slot_dim))
        self.register_buffer("K", torch.randn(num_slots, slot_dim) * 0.01)

        # Learned projections for the query key k_t and the write add-vector a_t.
        self.query_proj = nn.Linear(slot_dim, slot_dim)
        self.write_proj = nn.Linear(slot_dim, slot_dim)

    def _attention_weights(self, k_t: torch.Tensor) -> torch.Tensor:
        """
        k_t: (batch, slot_dim)
        returns: (batch, num_slots) attention weights, Eq. 8
        """
        # dot product between each query and every stored key
        logits = self.beta * torch.matmul(k_t, self.K.t())  # (batch, num_slots)
        return F.softmax(logits, dim=-1)

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (batch, slot_dim) current hidden state (post-ODE)
        returns: r_t, (batch, slot_dim) retrieved memory context, Eq. 7
        """
        k_t = self.query_proj(h)
        w_r = self._attention_weights(k_t)              # (batch, num_slots)
        r_t = torch.matmul(w_r, self.M)                  # (batch, slot_dim)
        return r_t

    @torch.no_grad()
    def write(self, h: torch.Tensor):
        """
        h: (batch, slot_dim) hidden state to consolidate into memory.
        Non-differentiable in-place update (Eq. 7's write rule is a running
        update to the memory store itself, not a backprop target).
        Applies decay first (Table 2's memory_decay_rate) then writes the
        batch-averaged add-vector to every slot weighted by write attention.
        """
        self.M.mul_(self.decay_rate)  # temporal decay before new writes

        a_t = self.write_proj(h).mean(dim=0, keepdim=True)  # (1, slot_dim)
        k_t = self.query_proj(h).mean(dim=0, keepdim=True)  # (1, slot_dim)
        w_w = self._attention_weights(k_t)                  # (1, num_slots)

        # M(i) += w_w(i) * a_t   (Eq. 7)
        self.M.add_(w_w.t() * a_t)

    def forward(self, h: torch.Tensor, write: bool = True) -> torch.Tensor:
        r_t = self.read(h)
        if write and self.training:
            self.write(h)
        return r_t


if __name__ == "__main__":
    mem = MemoryModule(num_slots=200, slot_dim=512)
    h = torch.randn(4, 512)
    r = mem(h)
    print("Memory read output:", r.shape)
    print("Memory slot norm after write:", mem.M.norm().item())
