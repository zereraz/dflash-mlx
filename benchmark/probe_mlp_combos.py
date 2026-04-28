#!/usr/bin/env python3
"""Benchmark mixed q4/bf16 choices for one target MLP."""

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
from mlx_lm.models.activations import swiglu

from dflash_mlx.runtime import (
    _dequantize_linear,
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
)


class _ComboMLP(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        name: str,
    ):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.name = name

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


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


def _first_mlp(model: Any, layer_index: int | None) -> tuple[int, Any]:
    text_model = _target_text_model(model)
    if layer_index is not None:
        return int(layer_index), text_model.layers[int(layer_index)].mlp
    for index, layer in enumerate(text_model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate_proj"):
            return index, mlp
    raise ValueError("could not find target MLP")


def _bench_combo(
    combo: nn.Module,
    *,
    hidden_size: int,
    m_values: list[int],
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    rows = []
    for tokens in m_values:
        x = mx.random.normal((1, int(tokens), int(hidden_size))).astype(mx.bfloat16)
        mx.eval(x)
        timing = _time_eval(lambda: combo(x), repeats=repeats, warmup=warmup)
        rows.append(
            {
                "tokens": int(tokens),
                **timing,
                "best_us_per_token": timing["best_ms"] * 1000.0 / float(tokens),
                "median_us_per_token": timing["median_ms"] * 1000.0 / float(tokens),
            }
        )
    return rows


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
    os.environ["DFLASH_DEQUANTIZE_MLP"] = "0"
    os.environ["DFLASH_VERIFY_LINEAR"] = "0"

    memory_config = configure_mlx_memory_limits()
    model, _tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    layer_index, mlp = _first_mlp(model, args.layer)

    q4_gate = mlp.gate_proj
    q4_up = mlp.up_proj
    q4_down = mlp.down_proj
    bf16_gate, gate_nbytes = _dequantize_linear(q4_gate)
    bf16_up, up_nbytes = _dequantize_linear(q4_up)
    bf16_down, down_nbytes = _dequantize_linear(q4_down)
    mx.eval(bf16_gate.weight, bf16_up.weight, bf16_down.weight)

    hidden_size = int(q4_gate.scales.shape[1]) * int(q4_gate.group_size)
    m_values = [int(value) for value in str(args.m_values).split(",") if value.strip()]
    combos = [
        _ComboMLP(
            name="q4_all",
            gate_proj=q4_gate,
            up_proj=q4_up,
            down_proj=q4_down,
        ),
        _ComboMLP(
            name="bf16_all",
            gate_proj=bf16_gate,
            up_proj=bf16_up,
            down_proj=bf16_down,
        ),
        _ComboMLP(
            name="bf16_gate_up_q4_down",
            gate_proj=bf16_gate,
            up_proj=bf16_up,
            down_proj=q4_down,
        ),
        _ComboMLP(
            name="q4_gate_up_bf16_down",
            gate_proj=q4_gate,
            up_proj=q4_up,
            down_proj=bf16_down,
        ),
        _ComboMLP(
            name="bf16_gate_q4_up_down",
            gate_proj=bf16_gate,
            up_proj=q4_up,
            down_proj=q4_down,
        ),
        _ComboMLP(
            name="q4_gate_bf16_up_q4_down",
            gate_proj=q4_gate,
            up_proj=bf16_up,
            down_proj=q4_down,
        ),
    ]
    rows = {
        combo.name: _bench_combo(
            combo,
            hidden_size=hidden_size,
            m_values=m_values,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        for combo in combos
    }
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
        "bf16_weight_nbytes": int(gate_nbytes + up_nbytes + down_nbytes),
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
