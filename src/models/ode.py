"""
models/ode.py
-------------
Neural ODE layer, literal reading of the base paper's Eq. 2-5.

    dh(t)/dt = f(h(t), t, theta)                              (Eq. 2)
    h(t1) = h(t0) + integral_{t0}^{t1} f(h(t), t, theta) dt    (Eq. 3)

Gradients w.r.t. theta computed via the adjoint sensitivity method
(Eq. 4-5) -- we use torchdiffeq's odeint_adjoint rather than hand-deriving
the backward ODE, since that's the standard, numerically-stable way to
implement this and the paper doesn't specify a custom adjoint variant.

ASSUMPTION (paper does not specify f's internal architecture): f is a
small 2-layer MLP taking the concatenation of [h(t), t] as input. This is
the simplest literal reading of "f denotes a neural network parameterized
by theta". Document this assumption in your writeup.

Hyperparameters from Table 2 (defaults used):
    ODE integration time range : [0.5, 2.0], default 1.0  -> integrate t in [0, 1.0]
    ODE solver tolerance       : 1e-4
    solver                     : Dormand-Prince ('dopri5')
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    """f(h(t), t, theta) -- the dynamics function integrated over time."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Small init helps ODE integration stability at the start of training.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # t is a scalar tensor (odeint calls f(t, h)); broadcast it into the batch.
        t_vec = t.expand(h.shape[0], 1)
        return self.net(torch.cat([h, t_vec], dim=-1))


class NeuralODELayer(nn.Module):
    """
    Wraps ODEFunc with an adaptive-step Dormand-Prince solver.
    Integrates from t0=0 to t1=integration_time (default 1.0, Table 2).
    """

    def __init__(self, hidden_dim: int, integration_time: float = 1.0,
                 tolerance: float = 1e-4):
        super().__init__()
        self.odefunc = ODEFunc(hidden_dim)
        self.register_buffer(
            "t_span", torch.tensor([0.0, integration_time], dtype=torch.float32))
        self.tolerance = tolerance

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        """
        h0: (batch, hidden_dim) initial hidden state (from the embedding layer)
        returns: (batch, hidden_dim) hidden state at t1
        """
        out = odeint(
            self.odefunc, h0, self.t_span,
            rtol=self.tolerance, atol=self.tolerance, method="dopri5",
        )
        # out shape: (len(t_span)=2, batch, hidden_dim) -> take the t1 state
        return out[-1]


if __name__ == "__main__":
    # Dummy-tensor sanity check (run locally, CPU is fine for this size).
    layer = NeuralODELayer(hidden_dim=512)
    x = torch.randn(4, 512, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    print("ODE forward OK, output shape:", y.shape)
    print("Gradient reached input:", x.grad is not None and x.grad.abs().sum().item() > 0)
