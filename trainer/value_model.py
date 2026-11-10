#!/usr/bin/env python3
"""New value network: 450 -> 1024 -> 64 -> 32 -> 1, black-absolute value.

One output head, no side-to-move head: the side to move is converted into the
target (see ``value_data``) and the search converts back when it negates. This
module only holds the float model; ``value_format`` owns the deployment format
and the integer reference forward.

Hidden activations use a clipped ReLU with a fixed ceiling (``act_max``) so the
exported integer network has exactly the same shape. The output is
``tanh(z)`` in ``[-1, 1]``; ``z`` is what the integer file stores.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

import value_format as vf
import features


@dataclass(frozen=True)
class ValueConfig:
    n_features: int = vf.N_FEATURES
    n_acc: int = vf.N_ACC
    n_h1: int = vf.N_H1
    n_h2: int = vf.N_H2
    act_max: float = vf.ACT_MAX_DEFAULT
    ea: int = vf.EA_DEFAULT

    def arch(self) -> list[int]:
        return [self.n_features, self.n_acc, self.n_h1, self.n_h2, 1]


class ValueNet(nn.Module):
    """Black-absolute value network with a single scalar output."""

    def __init__(self, seed: int = 0, config: ValueConfig | None = None) -> None:
        super().__init__()
        self.config = config or ValueConfig()
        c = self.config
        self.W0 = nn.Parameter(torch.empty(c.n_features, c.n_acc))
        self.b0 = nn.Parameter(torch.zeros(c.n_acc))
        self.W1 = nn.Parameter(torch.empty(c.n_acc, c.n_h1))
        self.b1 = nn.Parameter(torch.zeros(c.n_h1))
        self.W2 = nn.Parameter(torch.empty(c.n_h1, c.n_h2))
        self.b2 = nn.Parameter(torch.zeros(c.n_h2))
        self.W3 = nn.Parameter(torch.empty(c.n_h2, 1))
        self.b3 = nn.Parameter(torch.zeros(1))
        self.reset_parameters(seed)

    @torch.no_grad()
    def reset_parameters(self, seed: int = 0) -> None:
        """Random initialisation (from-scratch training starts here)."""
        generator = torch.Generator().manual_seed(int(seed))
        for name, weight in (("W0", self.W0), ("W1", self.W1),
                             ("W2", self.W2), ("W3", self.W3)):
            fan_in = weight.shape[0]
            bound = (2.0 / fan_in) ** 0.5
            weight.copy_(torch.empty(weight.shape)
                         .uniform_(-bound, bound, generator=generator))
        self.b0.zero_()
        self.b1.zero_()
        self.b2.zero_()
        self.b3.zero_()

    def forward_z(self, x: Tensor) -> Tensor:
        """Pre-tanh output z, [B]."""
        c = self.config
        a0 = torch.clamp(x @ self.W0 + self.b0, 0.0, c.act_max)
        a1 = torch.clamp(a0 @ self.W1 + self.b1, 0.0, c.act_max)
        a2 = torch.clamp(a1 @ self.W2 + self.b2, 0.0, c.act_max)
        return (a2 @ self.W3 + self.b3)[:, 0]

    def forward(self, x: Tensor) -> Tensor:
        """Black-absolute value in [-1, 1], [B]."""
        return torch.tanh(self.forward_z(x))

    # -- export helpers ---------------------------------------------------

    def flat_parameters(self) -> dict[str, list[float]]:
        with torch.no_grad():
            return {name: tensor.reshape(-1).cpu().tolist()
                    for name, tensor in (
                        ("W0", self.W0), ("b0", self.b0), ("W1", self.W1),
                        ("b1", self.b1), ("W2", self.W2), ("b2", self.b2),
                        ("W3", self.W3), ("b3", self.b3))}

    @classmethod
    def from_quantized(cls, weights: dict) -> "ValueNet":
        """Float model whose weights ARE the exported integers (dequantised)."""
        scale = float(1 << int(weights["ea"]))
        act_max = int(weights["act_max_int"]) / scale
        config = ValueConfig(act_max=act_max, ea=int(weights["ea"]))
        model = cls(config=config)
        with torch.no_grad():
            for name, tensor in (
                    ("W0", model.W0), ("b0", model.b0), ("W1", model.W1),
                    ("b1", model.b1), ("W2", model.W2), ("b2", model.b2),
                    ("W3", model.W3), ("b3", model.b3)):
                values = torch.tensor(list(weights[name]), dtype=torch.float32)
                tensor.copy_(values.reshape(tensor.shape) / scale)
        return model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def value_loss(prediction: Tensor, target: Tensor, kind: str = "mse",
               beta: float = 1.0, weights: Tensor | None = None) -> Tensor:
    if kind == "mse":
        losses = (prediction - target) ** 2
    elif kind == "smooth_l1":
        losses = torch.nn.functional.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    else:
        raise ValueError(f"unknown loss {kind!r}")
    return losses.mean() if weights is None else (losses * weights).sum() / weights.sum()
