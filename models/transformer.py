"""
models/transformer.py
----------------------
Transformer attention stage, literal reading of Eq. 6 plus the paper's
prose description: "The architectural components achieve synergistic
collaboration through residual connections and gated fusion mechanisms."

Attention (Eq. 6):
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

We treat [ode_output_token, memory_retrieved_token] as a length-2 sequence
and run it through a standard multi-head self-attention Transformer
encoder, then pool and gate-fuse the two paths -- this is the simplest
literal reading of "transformer layers that compute self-attention over
both current and memory-retrieved contexts" (paper's "Hybrid architecture
design" section).

Hyperparameters from Table 2 (defaults used):
    Transformer heads  : 8
    Transformer layers : 6
"""

import torch
import torch.nn as nn


class TransformerFusion(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gated fusion: learn how much to trust the transformer-attended
        # path vs. the raw ODE path (residual connection + gate).
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, ode_out: torch.Tensor, mem_out: torch.Tensor) -> torch.Tensor:
        """
        ode_out: (batch, hidden_dim) -- current ODE-evolved representation
        mem_out: (batch, hidden_dim) -- memory-retrieved context (Eq. 7 output)
        returns: (batch, hidden_dim) fused representation
        """
        seq = torch.stack([ode_out, mem_out], dim=1)     # (batch, 2, hidden_dim)
        attended = self.encoder(seq)                      # (batch, 2, hidden_dim)
        attended_pooled = attended.mean(dim=1)             # (batch, hidden_dim)

        # Gated residual fusion between the raw ODE path and the
        # transformer-attended path.
        gate_in = torch.cat([ode_out, attended_pooled], dim=-1)
        g = self.gate(gate_in)
        fused = g * attended_pooled + (1 - g) * ode_out
        return fused


if __name__ == "__main__":
    fusion = TransformerFusion(hidden_dim=512, num_heads=8, num_layers=6)
    a = torch.randn(4, 512)
    b = torch.randn(4, 512)
    out = fusion(a, b)
    print("Transformer fusion output:", out.shape)
