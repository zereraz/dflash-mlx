#!/usr/bin/env python3
"""Validate final-layer last-token-only prefill against full final-layer logits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from dflash_mlx.runtime import (
    configure_mlx_memory_limits,
    load_target_bundle,
    make_target_cache,
    target_forward_with_hidden_states,
)


PROMPT = (
    "You are validating a final-layer prefill shortcut. The cache after the "
    "prompt and the next-token logits must remain identical.\n"
)


def _target_prompt_tokens(tokenizer: Any, target_tokens: int) -> list[int]:
    pieces: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target_tokens:
        pieces.append(PROMPT)
        tokens = list(tokenizer.encode("".join(pieces)))
    return tokens[:target_tokens]


def _prefill_last_logits(
    model: Any,
    prompt_tokens: list[int],
    *,
    enabled: bool,
) -> tuple[mx.array, list[Any]]:
    os.environ["DFLASH_PREFILL_FINAL_LAYER_LAST_TOKEN"] = "1" if enabled else "0"
    cache = make_target_cache(model, enable_speculative_linear_cache=False)
    input_ids = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    logits, _ = target_forward_with_hidden_states(
        model,
        input_ids=input_ids,
        cache=cache,
        capture_layer_ids=set(),
        last_logits_only=True,
    )
    if logits is None:
        raise RuntimeError("prefill did not return logits")
    mx.eval(logits)
    return logits, cache


def _next_logits(model: Any, cache: list[Any], token: int) -> mx.array:
    input_ids = mx.array([int(token)], dtype=mx.uint32)[None]
    logits, _ = target_forward_with_hidden_states(
        model,
        input_ids=input_ids,
        cache=cache,
        capture_layer_ids=set(),
        last_logits_only=True,
    )
    if logits is None:
        raise RuntimeError("decode did not return logits")
    mx.eval(logits)
    return logits


def _compare(control: mx.array, experiment: mx.array) -> dict[str, Any]:
    diff = mx.abs(control - experiment)
    mx.eval(diff)
    control_top = mx.argpartition(control[0, -1], kth=-10)[-10:]
    experiment_top = mx.argpartition(experiment[0, -1], kth=-10)[-10:]
    mx.eval(control_top, experiment_top)
    return {
        "max_abs_diff": float(mx.max(diff).item()),
        "mean_abs_diff": float(mx.mean(diff).item()),
        "same_argmax": int(mx.argmax(control[0, -1]).item())
        == int(mx.argmax(experiment[0, -1]).item()),
        "top10_overlap": len(
            set(int(x) for x in control_top.tolist())
            & set(int(x) for x in experiment_top.tolist())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--hybrid-mlp", action="store_true")
    parser.add_argument("--hybrid-mlp-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-linear", action="store_true")
    parser.add_argument("--hybrid-gdn-linear-attrs", default="in_proj_qkv,in_proj_z,out_proj")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.hybrid_mlp:
        os.environ["DFLASH_HYBRID_MLP"] = "1"
        os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(int(args.hybrid_mlp_threshold))
    if args.hybrid_gdn_linear:
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(
            int(args.hybrid_mlp_threshold)
        )
        os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = str(args.hybrid_gdn_linear_attrs)

    memory_config = configure_mlx_memory_limits()
    model, tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    tokens = _target_prompt_tokens(tokenizer, int(args.prompt_tokens) + 1)
    prompt_tokens = tokens[:-1]
    next_token = tokens[-1]

    control_prefill, control_cache = _prefill_last_logits(
        model,
        prompt_tokens,
        enabled=False,
    )
    experiment_prefill, experiment_cache = _prefill_last_logits(
        model,
        prompt_tokens,
        enabled=True,
    )
    control_next = _next_logits(model, control_cache, next_token)
    experiment_next = _next_logits(model, experiment_cache, next_token)

    result = {
        "model": str(Path(args.model).expanduser()),
        "prompt_tokens": len(prompt_tokens),
        "next_token": int(next_token),
        "hybrid_mlp": bool(args.hybrid_mlp),
        "hybrid_gdn_linear": bool(args.hybrid_gdn_linear),
        "hybrid_gdn_linear_attrs": str(args.hybrid_gdn_linear_attrs),
        "memory_config": memory_config,
        "target_meta": {
            key: value for key, value in target_meta.items() if key != "config"
        },
        "prefill_logits": _compare(control_prefill, experiment_prefill),
        "next_logits_after_cache": _compare(control_next, experiment_next),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
