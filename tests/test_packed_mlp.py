from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu

from dflash_mlx.runtime import _PackedGateUpMLP


def _quantized_linear(in_dims: int, out_dims: int) -> nn.QuantizedLinear:
    linear = nn.Linear(in_dims, out_dims, bias=False)
    linear.weight = mx.random.normal((out_dims, in_dims)).astype(mx.bfloat16) * 0.1
    return nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _quantized_linear(128, 256)
        self.up_proj = _quantized_linear(128, 256)
        self.down_proj = _quantized_linear(256, 128)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def test_packed_gate_up_mlp_matches_separate_quantized_projections():
    mlp = _TinyMLP()
    packed = _PackedGateUpMLP(mlp)
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16) * 0.2
    expected = mlp(x)
    actual = packed(x)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, atol=1e-5, rtol=1e-5).item()
