#!/usr/bin/env python3
"""Probe bf16-packed GDN qkv+z projection for one target linear-attention layer."""

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

from dflash_mlx.runtime import (
    _dequantize_linear,
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
)


def _first_linear_attn(model: Any, layer_index: int | None) -> tuple[int, Any]:
    text_model = _target_text_model(model)
    if layer_index is not None:
        layer = text_model.layers[int(layer_index)]
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            raise ValueError(f"layer {layer_index} is not a linear-attention layer")
        return int(layer_index), linear_attn
    for index, layer in enumerate(text_model.layers):
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is not None:
            return index, linear_attn
    raise ValueError("could not find a linear-attention layer")


def _input_width(linear: Any) -> int:
    scales = getattr(linear, "scales", None)
    if scales is not None:
        return int(scales.shape[1]) * int(linear.group_size)
    return int(linear.weight.shape[1])


def _time_ms(fn: Callable[[mx.array], mx.array], x: mx.array, *, repeats: int, warmup: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        mx.eval(fn(x))
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter_ns()
        y = fn(x)
        mx.eval(y)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "best_ms": min(samples),
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--m-values", default="512,1024,2048")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["DFLASH_HYBRID_MLP"] = "0"
    os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "0"
    os.environ["DFLASH_VERIFY_LINEAR"] = "0"

    memory_config = configure_mlx_memory_limits()
    model, _tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    layer_index, linear_attn = _first_linear_attn(model, args.layer)

    qkv_bf16, qkv_nbytes = _dequantize_linear(linear_attn.in_proj_qkv)
    z_bf16, z_nbytes = _dequantize_linear(linear_attn.in_proj_z)
    packed = nn.Linear(
        qkv_bf16.weight.shape[1],
        qkv_bf16.weight.shape[0] + z_bf16.weight.shape[0],
        bias=False,
    )
    packed.weight = mx.concatenate([qkv_bf16.weight, z_bf16.weight], axis=0)
    mx.eval(qkv_bf16.weight, z_bf16.weight, packed.weight)

    qkv_dim = int(qkv_bf16.weight.shape[0])

    def separate(x: mx.array) -> mx.array:
        return mx.concatenate([qkv_bf16(x), z_bf16(x)], axis=-1)

    def packed_split_concat(x: mx.array) -> mx.array:
        qkvz = packed(x)
        qkv, z = mx.split(qkvz, [qkv_dim], axis=-1)
        return mx.concatenate([qkv, z], axis=-1)

    hidden_size = _input_width(linear_attn.in_proj_qkv)
    rows: dict[str, Any] = {}
    for m_value in [int(value) for value in str(args.m_values).split(",") if value.strip()]:
        x = mx.random.normal((1, int(m_value), hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        expected = separate(x)
        actual = packed_split_concat(x)
        mx.eval(expected, actual)
        diff = mx.abs(expected - actual)
        separate_timing = _time_ms(
            separate,
            x,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        packed_timing = _time_ms(
            packed_split_concat,
            x,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        rows[str(m_value)] = {
            "max_abs_diff": float(mx.max(diff).item()),
            "separate": separate_timing,
            "packed": packed_timing,
            "packed_best_speedup": float(separate_timing["best_ms"])
            / float(packed_timing["best_ms"]),
            "packed_median_speedup": float(separate_timing["median_ms"])
            / float(packed_timing["median_ms"]),
        }
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    result = {
        "model": str(Path(args.model).expanduser()),
        "layer": int(layer_index),
        "m_values": [int(value) for value in str(args.m_values).split(",") if value.strip()],
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "memory_config": memory_config,
        "target_meta": {
            key: value for key, value in target_meta.items() if key != "config"
        },
        "separate_bf16_weight_nbytes": int(qkv_nbytes + z_nbytes),
        "packed_bf16_weight_nbytes": int(packed.weight.nbytes),
        "qkv_dim": qkv_dim,
        "z_dim": int(z_bf16.weight.shape[0]),
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
