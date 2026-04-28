"""Synchronized component profile for one Qwen3.5 GatedDeltaNet layer.

This is attribution-only: synchronizing after each subcomponent changes MLX
scheduling. Use the totals to choose experiments, not as end-to-end timing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gated_delta as gated_delta_mod

from dflash_mlx.runtime import (
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
)


def _timed(name: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, float | str]]:
    start = time.perf_counter_ns()
    value = fn()
    if isinstance(value, tuple):
        mx.eval(*[item for item in value if isinstance(item, mx.array)])
    elif isinstance(value, mx.array):
        mx.eval(value)
    elapsed_us = (time.perf_counter_ns() - start) / 1_000.0
    return value, {"name": name, "elapsed_us": elapsed_us}


def _summary(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["name"]), []).append(float(row["elapsed_us"]))
    return {
        name: {
            "count": float(len(values)),
            "total_us": float(sum(values)),
            "median_us": float(statistics.median(values)),
        }
        for name, values in grouped.items()
    }


def _first_gdn_layer(model: Any, layer_index: int | None) -> tuple[int, Any]:
    text_model = _target_text_model(model)
    if layer_index is not None:
        layer = text_model.layers[int(layer_index)]
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            raise ValueError(f"layer {layer_index} is not a linear-attention layer")
        return int(layer_index), attn
    for index, layer in enumerate(text_model.layers):
        attn = getattr(layer, "linear_attn", None)
        if attn is not None:
            return index, attn
    raise ValueError("could not find a linear-attention layer")


def _component_profile(attn: Any, x: mx.array) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    B, S, _ = x.shape

    qkv, row = _timed("in_proj_qkv", lambda: attn.in_proj_qkv(x))
    rows.append(row)
    z_proj, row = _timed("in_proj_z", lambda: attn.in_proj_z(x))
    rows.append(row)
    b, row = _timed("in_proj_b", lambda: attn.in_proj_b(x))
    rows.append(row)
    a, row = _timed("in_proj_a", lambda: attn.in_proj_a(x))
    rows.append(row)

    z, row = _timed(
        "reshape_z",
        lambda: z_proj.reshape(B, S, attn.num_v_heads, attn.head_v_dim),
    )
    rows.append(row)

    conv_state = mx.zeros(
        (B, int(attn.conv_kernel_size) - 1, int(attn.conv_dim)),
        dtype=x.dtype,
    )
    conv_input, row = _timed(
        "concat_conv_state",
        lambda: mx.concatenate([conv_state, qkv], axis=1),
    )
    rows.append(row)
    conv_out, row = _timed("conv1d_silu", lambda: nn.silu(attn.conv1d(conv_input)))
    rows.append(row)

    split, row = _timed(
        "split_qkv",
        lambda: tuple(
            tensor.reshape(B, S, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], -1),
                [attn.num_k_heads, attn.num_k_heads, attn.num_v_heads],
                [attn.head_k_dim, attn.head_k_dim, attn.head_v_dim],
                strict=True,
            )
        ),
    )
    rows.append(row)
    q, k, v = split

    inv_scale = k.shape[-1] ** -0.5
    q, row = _timed(
        "rms_q",
        lambda: (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6),
    )
    rows.append(row)
    k, row = _timed("rms_k", lambda: inv_scale * mx.fast.rms_norm(k, None, 1e-6))
    rows.append(row)
    g, row = _timed("compute_g", lambda: gated_delta_mod.compute_g(attn.A_log, a, attn.dt_bias))
    rows.append(row)
    beta, row = _timed("sigmoid_b", lambda: mx.sigmoid(b))
    rows.append(row)

    state = mx.zeros((B, attn.num_v_heads, attn.head_v_dim, attn.head_k_dim), dtype=q.dtype)
    out, row = _timed(
        "gated_delta_kernel",
        lambda: gated_delta_mod.gated_delta_kernel(q, k, v, g, beta, state, None)[0],
    )
    rows.append(row)
    normed, row = _timed("rms_norm_gated", lambda: attn.norm(out, z))
    rows.append(row)
    out_proj, row = _timed("out_proj", lambda: attn.out_proj(normed.reshape(B, S, -1)))
    rows.append(row)
    del out_proj
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--hybrid-gdn-linear", action="store_true")
    parser.add_argument("--hybrid-gdn-linear-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-linear-attrs", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.hybrid_gdn_linear:
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(
            int(args.hybrid_gdn_linear_threshold)
        )
        if args.hybrid_gdn_linear_attrs:
            os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = str(
                args.hybrid_gdn_linear_attrs
            )

    memory_config = configure_mlx_memory_limits()
    model, _tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    layer_index, attn = _first_gdn_layer(model, args.layer)
    hidden_size = int(getattr(attn, "hidden_size"))
    x = mx.random.normal((1, int(args.tokens), hidden_size)).astype(mx.bfloat16)
    mx.eval(x)

    rows: list[dict[str, float | str]] = []
    for _ in range(max(1, int(args.repeats))):
        rows.extend(_component_profile(attn, x))

    result = {
        "model": str(Path(args.model).expanduser()),
        "layer": int(layer_index),
        "tokens": int(args.tokens),
        "repeats": int(args.repeats),
        "memory_config": memory_config,
        "target_meta": target_meta,
        "summary_us": _summary(rows),
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
