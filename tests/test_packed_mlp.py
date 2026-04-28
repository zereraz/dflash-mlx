from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu

from dflash_mlx.runtime import _HybridLargeMLinear
from dflash_mlx.runtime import _HybridPrefillMLP
from dflash_mlx.runtime import _PackedGateUpMLP
from dflash_mlx.runtime import _pack_gdn_input_projections
from dflash_mlx.runtime import _pack_attention_kv_projection
from dflash_mlx.runtime import _project_gdn_inputs
from dflash_mlx.runtime import _pack_mlp_gate_up_enabled
from dflash_mlx.runtime import _use_packed_gdn_inputs


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


def test_packed_mlp_is_opt_in_by_default(monkeypatch):
    monkeypatch.delenv("DFLASH_PACK_MLP_GATE_UP", raising=False)
    assert not _pack_mlp_gate_up_enabled()
    monkeypatch.setenv("DFLASH_PACK_MLP_GATE_UP", "1")
    assert _pack_mlp_gate_up_enabled()


def test_packed_gate_up_mlp_matches_separate_quantized_projections():
    mlp = _TinyMLP()
    packed = _PackedGateUpMLP(mlp)
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16) * 0.2
    expected = mlp(x)
    actual = packed(x)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, atol=1e-5, rtol=1e-5).item()


def test_hybrid_mlp_uses_quantized_path_below_threshold():
    mlp = _TinyMLP()
    hybrid = _HybridPrefillMLP(mlp, threshold=17)
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16) * 0.2
    expected = mlp(x)
    actual = hybrid(x)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, atol=1e-5, rtol=1e-5).item()


def test_hybrid_mlp_uses_bf16_path_at_effective_row_threshold():
    mlp = _TinyMLP()
    hybrid = _HybridPrefillMLP(mlp, threshold=16)
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16) * 0.2
    gate_up_proj = getattr(hybrid, "bf16_gate_up_proj", None)
    if gate_up_proj is not None:
        gate_up = gate_up_proj(x)
        gate, up = mx.split(gate_up, [hybrid.hidden_dim], axis=-1)
    else:
        gate = hybrid.bf16_gate_proj(x)
        up = hybrid.bf16_up_proj(x)
    expected = hybrid.bf16_down_proj(swiglu(gate, up))
    actual = hybrid(x)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, atol=1e-5, rtol=1e-5).item()


def test_hybrid_mlp_bf16_path_stays_close_to_quantized_mlp():
    mlp = _TinyMLP()
    hybrid = _HybridPrefillMLP(mlp, threshold=16)
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16) * 0.2
    expected = mlp(x)
    actual = hybrid(x)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, atol=5e-3, rtol=5e-3).item()


def test_hybrid_large_m_linear_switches_by_effective_rows():
    linear = _quantized_linear(128, 64)
    hybrid = _HybridLargeMLinear(linear, threshold=4)
    small = mx.random.normal((1, 3, 128)).astype(mx.bfloat16) * 0.2
    large = mx.random.normal((2, 2, 128)).astype(mx.bfloat16) * 0.2
    small_expected = linear(small)
    small_actual = hybrid(small)
    large_expected = hybrid.bf16_linear(large)
    large_actual = hybrid(large)
    mx.eval(small_expected, small_actual, large_expected, large_actual)
    assert mx.allclose(small_expected, small_actual, atol=1e-5, rtol=1e-5).item()
    assert mx.allclose(large_expected, large_actual, atol=1e-5, rtol=1e-5).item()


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = _quantized_linear(128, 64)
        self.v_proj = _quantized_linear(128, 64)


def test_packed_attention_kv_projection_matches_separate_quantized_projections():
    attn = _TinyAttention()
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16) * 0.2
    expected_k = attn.k_proj(x)
    expected_v = attn.v_proj(x)
    assert _pack_attention_kv_projection(attn)
    packed_kv = attn.kv_proj(x)
    actual_k, actual_v = mx.split(packed_kv, [64], axis=-1)
    mx.eval(expected_k, expected_v, actual_k, actual_v)
    assert mx.allclose(expected_k, actual_k, atol=1e-5, rtol=1e-5).item()
    assert mx.allclose(expected_v, actual_v, atol=1e-5, rtol=1e-5).item()


class _TinyGDNInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = _quantized_linear(128, 192)
        self.in_proj_z = _quantized_linear(128, 96)
        self.in_proj_b = _quantized_linear(128, 64)
        self.in_proj_a = _quantized_linear(128, 64)


def test_packed_gdn_input_projections_match_separate_quantized_projections():
    gdn = _TinyGDNInputs()
    x = mx.random.normal((2, 8, 128)).astype(mx.bfloat16) * 0.2
    expected = (
        gdn.in_proj_qkv(x),
        gdn.in_proj_z(x),
        gdn.in_proj_b(x),
        gdn.in_proj_a(x),
    )
    assert _pack_gdn_input_projections(gdn, threshold=4)
    actual = _project_gdn_inputs(gdn, x, seq_len=8)
    mx.eval(*expected, *actual)
    for expected_value, actual_value in zip(expected, actual, strict=True):
        assert mx.allclose(expected_value, actual_value, atol=1e-5, rtol=1e-5).item()


def test_packed_gdn_input_projections_keep_small_m_on_separate_path():
    gdn = _TinyGDNInputs()
    x = mx.random.normal((2, 2, 128)).astype(mx.bfloat16) * 0.2
    assert _pack_gdn_input_projections(gdn, threshold=5)
    assert not _use_packed_gdn_inputs(gdn, x)
    qkv, z, b, a = _project_gdn_inputs(gdn, x, seq_len=2)
    expected = (
        gdn.in_proj_qkv(x),
        gdn.in_proj_z(x),
        gdn.in_proj_b(x),
        gdn.in_proj_a(x),
    )
    mx.eval(qkv, z, b, a, *expected)
    for expected_value, actual_value in zip(expected, (qkv, z, b, a), strict=True):
        assert mx.allclose(expected_value, actual_value, atol=1e-5, rtol=1e-5).item()


def test_packed_gdn_input_projections_switch_by_effective_rows():
    gdn = _TinyGDNInputs()
    x = mx.random.normal((2, 2, 128)).astype(mx.bfloat16) * 0.2
    assert _pack_gdn_input_projections(gdn, threshold=4)
    assert _use_packed_gdn_inputs(gdn, x)
