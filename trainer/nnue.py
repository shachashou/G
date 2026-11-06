"""PyTorch NNUE model (ticket 08).

Architecture matches `trainer/weights_format.py` exactly::

    450 absolute-colour features -> accumulator 1024 (int16 domain)
      -> ClippedReLU(0, QA=127) -> L1 15 -> ClippedReLU(0, QB=64)
      -> 2 output heads (head 0 = black to move, head 1 = white to move)

Two forward modes share one topology:

* ``quantized=False`` (default training): continuous float weights, plain
  division. A quantisation regularizer encourages weights near integers;
  measured export fidelity, not arithmetic parity, determines the actual loss.
* ``quantized=True`` (optional QAT / parity check): weights are rounded with a
  straight-through estimator and divisions use ``round_div`` (also STE). Every
  intermediate value is then an exact integer, so this path is numerically
  identical to the C++ integer forward in `weights_format.infer_int`.

Parameters start as small integers. Training can move them away from integers;
original-model to exported-model fidelity must be measured independently.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import weights_format as wf


class RoundSte(torch.autograd.Function):
    """Round half away from zero, identity backward (straight-through)."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


class RoundDivSte(torch.autograd.Function):
    """`round_div` with the gradient of plain division (straight-through)."""

    @staticmethod
    def forward(ctx, x: Tensor, d: int) -> Tensor:
        ctx.d = d
        n = x.detach().to(torch.int64)
        sign = torch.where(n >= 0, torch.ones_like(n), -torch.ones_like(n))
        quotient = sign * ((2 * torch.abs(n) + d) // (2 * d))
        return quotient.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output / ctx.d, None


def quantize(x: Tensor, lo: int, hi: int) -> Tensor:
    return RoundSte.apply(x).clamp(lo, hi)


@dataclass(frozen=True)
class ModelConfig:
    format_version: int = 1
    output_gain: int = 1
    n_features: int = wf.N_FEATURES
    n_acc: int = wf.N_ACC
    n_l1: int = wf.N_L1
    n_heads: int = wf.N_HEADS
    qa: int = wf.QA
    qb: int = wf.QB
    score_scale: int = wf.SCORE_SCALE
    # WDL logits are derived from the single eval output; 400 would squash a
    # typical eval (~100) into ~0.25 and leave CE flat at ln(3). 100 keeps the
    # CE term informative. This is a training-only constant, never exported.
    wdl_temperature: float = 100.0
    weight_clamp: int = wf.WEIGHT_CLAMP


class NNUEModel(nn.Module):
    def __init__(self, seed: int = 0, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.format_version not in (1, 2) or c.output_gain not in wf.OUTPUT_GAINS or (c.format_version == 1 and c.output_gain != 1):
            raise ValueError("invalid format version/output gain")
        self.W0 = nn.Parameter(torch.empty(c.n_features, c.n_acc))
        self.b0 = nn.Parameter(torch.empty(c.n_acc))
        self.W1 = nn.Parameter(torch.empty(c.n_acc, c.n_l1))
        self.b1 = nn.Parameter(torch.empty(c.n_l1))
        self.W2 = nn.Parameter(torch.empty(c.n_heads, c.n_l1))
        self.b2 = nn.Parameter(torch.empty(c.n_heads))
        # Training-only class priors for the WDL CE head. The inference
        # network still emits exactly ONE scalar per position; these three
        # biases never enter the exported weight file.
        self.wdl_bias = nn.Parameter(torch.zeros(3))
        self.reset_parameters(seed)

    @torch.no_grad()
    def reset_parameters(self, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        c = self.config

        def integers(lo: int, hi: int, shape: tuple[int, ...]) -> Tensor:
            return torch.randint(lo, hi + 1, shape, generator=generator).float()

        # Positive biases keep both ClippedReLU units inside their active
        # (non-saturated) region at initialisation.
        self.W0.copy_(integers(-4, 4, self.W0.shape))
        self.b0.fill_(32.0)
        self.W1.copy_(integers(-2, 2, self.W1.shape))
        self.b1.fill_(32.0)
        self.W2.copy_(integers(-32, 32, self.W2.shape))
        self.b2.zero_()
        self.wdl_bias.zero_()

    def _tensors(self, quantized: bool):
        wc = self.config.weight_clamp
        if quantized:
            return (
                quantize(self.W0, -wc, wc), quantize(self.b0, wf.INT16_MIN, wf.INT16_MAX),
                quantize(self.W1, -wc, wc), quantize(self.b1, wf.INT16_MIN, wf.INT16_MAX),
                quantize(self.W2, -wc, wc), quantize(self.b2, wf.INT16_MIN, wf.INT16_MAX),
            )
        return self.W0, self.b0, self.W1, self.b1, self.W2, self.b2

    def forward(self, x: Tensor, stm: Tensor, quantized: bool = False) -> Tensor:
        """x: float 0/1 [B,450], stm: int64 [B] (0 black, 1 white)."""
        c = self.config
        w0, b0, w1, b1, w2, b2 = self._tensors(quantized)
        head = stm.long()

        acc = (x @ w0 + b0).clamp(wf.INT16_MIN, wf.INT16_MAX)
        a = acc.clamp(0, c.qa)
        z = a @ w1
        zq = (RoundDivSte.apply(z, c.qa) if quantized else z / c.qa) + b1
        a2 = zq.clamp(0, c.qb)
        y = (a2 * w2[head]).sum(dim=-1)
        gain = c.output_gain
        eval_score = (RoundDivSte.apply(y * gain, c.qb) if quantized
                      else y * gain / c.qb) + gain * b2[head]
        if c.format_version == 2:
            eval_score = eval_score.clamp(-wf.STATIC_LIMIT, wf.STATIC_LIMIT)
        return eval_score

    def quantization_regularizer(self) -> Tensor:
        """Keeps master weights close to integers for near-lossless export."""
        return sum((p - p.detach().round()).square().mean()
                   for p in (self.W0, self.b0, self.W1, self.b1, self.W2, self.b2))

    @classmethod
    @torch.no_grad()
    def from_quantized_weights(cls, weights: dict,
                               config: ModelConfig | None = None) -> "NNUEModel":
        """Rebuild a float model whose parameters ARE the exported integers.

        Ticket 09 uses this to compare PyTorch against the C++ scalar inference
        on exactly the weights shipped in the .nnue file.
        """
        config = config or ModelConfig(format_version=weights.get("format_version", 1),
                                        output_gain=weights.get("output_gain", 1))
        model = cls(config=config)
        c = model.config
        shapes = {
            "W0": (c.n_features, c.n_acc), "b0": (c.n_acc,),
            "W1": (c.n_acc, c.n_l1), "b1": (c.n_l1,),
            "W2": (c.n_heads, c.n_l1), "b2": (c.n_heads,),
        }
        for name, shape in shapes.items():
            values = torch.tensor(list(weights[name]), dtype=torch.float32)
            getattr(model, name).copy_(values.reshape(shape))
        return model

    @torch.no_grad()
    def quantized_weights(self) -> dict:
        """Integer weights exactly as used by `forward(..., quantized=True)`."""
        wc = self.config.weight_clamp
        specs = (
            ("W0", self.W0, -wc, wc),
            ("b0", self.b0, wf.INT16_MIN, wf.INT16_MAX),
            ("W1", self.W1, -wc, wc),
            ("b1", self.b1, wf.INT16_MIN, wf.INT16_MAX),
            ("W2", self.W2, -wc, wc),
            ("b2", self.b2, wf.INT16_MIN, wf.INT16_MAX),
        )
        weights = {
            name: RoundSte.apply(p).clamp(lo, hi).cpu().numpy().astype(int)
                   .reshape(-1).tolist()
            for name, p, lo, hi in specs
        }
        weights.update(format_version=self.config.format_version,
                       output_gain=self.config.output_gain)
        return weights


def wdl_logits(eval_score: Tensor, temperature: float = 100.0,
               bias: Tensor | None = None) -> Tensor:
    """Three WDL logits [loss, draw, win] derived from the single eval output.

    `bias` is a training-only class-prior parameter (never exported); without
    it a single symmetric scalar has a CE floor close to ln(3).
    """
    zero = torch.zeros_like(eval_score)
    logits = torch.stack((-eval_score / temperature, zero,
                          eval_score / temperature), dim=-1)
    return logits + bias if bias is not None else logits


def mixed_losses(eval_score: Tensor, score: Tensor, wdl: Tensor,
                 score_scale: int = wf.SCORE_SCALE,
                 wdl_temperature: float = 100.0,
                 wdl_bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """MSE(tanh(score/400)) + CE(WDL), returned separately for logging."""
    prediction = torch.tanh(eval_score / score_scale)
    target = torch.tanh(score / score_scale)
    mse = F.mse_loss(prediction, target)
    ce = F.cross_entropy(
        wdl_logits(eval_score, wdl_temperature, wdl_bias), wdl.long())
    return mse, ce
