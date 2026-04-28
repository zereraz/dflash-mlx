#!/usr/bin/env python3
"""Benchmark q4 vs bf16 full-attention projection linears.

This is a narrow Apple/MLX probe for the Qwen hybrid target. The deployed
model keeps projections quantized, but large-M prefill can sometimes run faster
with dequantized bf16 weights. Decode-like small-M work should stay q4.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from dflash_mlx.runtime import (
    _dequantize_linear,
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
)


def _time_eval(fn, *, repeats: int, warmup: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        y = fn()
        mx.eval(y)
    times_ms: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter_ns()
        y = fn()
        mx.eval(y)
        times_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "best_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "times_ms": times_ms,
    }


def _first_full_attention_layer(model: Any, layer_index: int | None) -> tuple[int, Any]:
    text_model = _target_text_model(model)
    if layer_index is not None:
        layer = text_model.layers[int(layer_index)]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise ValueError(f"layer {layer_index} is not a full-attention layer")
        return int(layer_index), attn
    for index, layer in enumerate(text_model.layers):
        if not getattr(layer, "is_linear", False):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                return index, attn
    raise ValueError("could not find a full-attention layer")


def _input_width(linear: nn.Module) -> int:
    if isinstance(linear, nn.QuantizedLinear):
        return int(linear.scales.shape[1]) * int(linear.group_size)
    return int(linear.weight.shape[1])


def _bench_linear(
    linear: nn.Module,
    *,
    m_values: list[int],
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    bf16_linear, weight_nbytes = _dequantize_linear(linear)
    if bf16_linear is linear:
        raise ValueError(f"{type(linear).__name__} is not quantized")
    mx.eval(bf16_linear.weight)
    width = _input_width(linear)
    rows = []
    for tokens in m_values:
        x = mx.random.normal((1, int(tokens), width)).astype(mx.bfloat16)
        mx.eval(x)
        q4 = _time_eval(lambda: linear(x), repeats=repeats, warmup=warmup)
        bf16 = _time_eval(lambda: bf16_linear(x), repeats=repeats, warmup=warmup)
        rows.append(
            {
                "tokens": int(tokens),
                "q4": q4,
                "bf16": bf16,
                "speedup_best": float(q4["best_ms"]) / float(bf16["best_ms"]),
                "speedup_median": float(q4["median_ms"]) / float(bf16["median_ms"]),
            }
        )
    return {
        "input_width": width,
        "weight_nbytes": int(weight_nbytes),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--m-values", default="128,256,512,1024,2048")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["DFLASH_HYBRID_MLP"] = "0"
    os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "0"
    os.environ["DFLASH_VERIFY_LINEAR"] = "0"

    memory_config = configure_mlx_memory_limits()
    model, _tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    layer_index, attn = _first_full_attention_layer(model, args.layer)
    m_values = [int(value) for value in str(args.m_values).split(",") if value.strip()]

    rows = {}
    for attr_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        module = getattr(attn, attr_name, None)
        if module is not None:
            rows[attr_name] = _bench_linear(
                module,
                m_values=m_values,
                repeats=int(args.repeats),
                warmup=int(args.warmup),
            )
            mx.clear_cache()

    result = {
        "model": str(Path(args.model).expanduser()),
        "layer": int(layer_index),
        "m_values": m_values,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "memory_config": memory_config,
        "target_meta": {
            key: value for key, value in target_meta.items() if key != "config"
        },
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
