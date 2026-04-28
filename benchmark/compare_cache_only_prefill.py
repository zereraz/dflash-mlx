#!/usr/bin/env python3
"""Check cache-only prefill shortcuts against next-token logits.

The cache-only prefill path may skip work whose hidden output is discarded.
This script validates the important contract: after the cached prefix is built,
the next-token target logits must match the non-shortcut chunked cache path.
"""

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
    target_prefill_without_logits,
)


PROMPT = (
    "You are validating a long-context cache optimization. Explain the "
    "invariant and keep every detail deterministic.\n"
)


def _target_prompt_tokens(tokenizer: Any, target_tokens: int) -> list[int]:
    pieces: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target_tokens:
        pieces.append(PROMPT)
        tokens = list(tokenizer.encode("".join(pieces)))
    return tokens[:target_tokens]


def _next_logits_after_cached_prefix(
    model: Any,
    *,
    prefix_tokens: list[int],
    next_token: int,
    skip_final_attention: bool,
) -> mx.array:
    cache = make_target_cache(model, enable_speculative_linear_cache=False)
    prefix_ids = mx.array(prefix_tokens, dtype=mx.uint32)[None]
    h, h_dependencies = target_prefill_without_logits(
        model,
        input_ids=prefix_ids,
        cache=cache,
        skip_final_layer_mlp=True,
        skip_final_layer_attention=skip_final_attention,
        return_dependencies=True,
    )
    mx.eval(h, *h_dependencies)

    next_ids = mx.array([int(next_token)], dtype=mx.uint32)[None]
    logits, _ = target_forward_with_hidden_states(
        model,
        input_ids=next_ids,
        cache=cache,
        capture_layer_ids=set(),
        last_logits_only=True,
    )
    if logits is None:
        raise RuntimeError("target forward unexpectedly returned no logits")
    mx.eval(logits)
    return logits


def _next_logits_after_capture_prefill(
    model: Any,
    *,
    prefix_tokens: list[int],
    next_token: int,
    capture_layer_id: int,
    skip_final_output: bool,
) -> tuple[mx.array, dict[int, mx.array]]:
    cache = make_target_cache(model, enable_speculative_linear_cache=False)
    prefix_ids = mx.array(prefix_tokens, dtype=mx.uint32)[None]
    _, captured = target_forward_with_hidden_states(
        model,
        input_ids=prefix_ids,
        cache=cache,
        capture_layer_ids={int(capture_layer_id)},
        skip_logits=True,
        skip_final_layer_mlp=bool(skip_final_output),
        skip_final_layer_attention=bool(skip_final_output),
    )
    if not isinstance(captured, dict):
        raise RuntimeError("target forward unexpectedly returned list captures")
    mx.eval(*captured.values())

    next_ids = mx.array([int(next_token)], dtype=mx.uint32)[None]
    logits, _ = target_forward_with_hidden_states(
        model,
        input_ids=next_ids,
        cache=cache,
        capture_layer_ids=set(),
        last_logits_only=True,
    )
    if logits is None:
        raise RuntimeError("target forward unexpectedly returned no logits")
    mx.eval(logits)
    return logits, captured


def _logit_comparison(control: mx.array, experiment: mx.array) -> dict[str, Any]:
    diff = mx.abs(control - experiment)
    mx.eval(diff)
    control_top = mx.argpartition(control[0, -1], kth=-10)[-10:]
    experiment_top = mx.argpartition(experiment[0, -1], kth=-10)[-10:]
    mx.eval(control_top, experiment_top)
    return {
        "max_abs_logit_diff": float(mx.max(diff).item()),
        "mean_abs_logit_diff": float(mx.mean(diff).item()),
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
    parser.add_argument("--hybrid-gdn-out-proj", action="store_true")
    parser.add_argument(
        "--capture-layer-id",
        type=int,
        default=None,
        help=(
            "Also validate target_forward_with_hidden_states(skip_logits=True) "
            "with this captured layer id, matching prompt-cache checkpoint chunks."
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.hybrid_mlp:
        os.environ["DFLASH_HYBRID_MLP"] = "1"
        os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(int(args.hybrid_mlp_threshold))
    if args.hybrid_gdn_out_proj:
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(
            int(args.hybrid_mlp_threshold)
        )
        os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = "out_proj"

    memory_config = configure_mlx_memory_limits()
    model, tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    tokens = _target_prompt_tokens(tokenizer, int(args.prompt_tokens) + 1)
    prefix_tokens = tokens[:-1]
    next_token = tokens[-1]

    control = _next_logits_after_cached_prefix(
        model,
        prefix_tokens=prefix_tokens,
        next_token=next_token,
        skip_final_attention=False,
    )
    experiment = _next_logits_after_cached_prefix(
        model,
        prefix_tokens=prefix_tokens,
        next_token=next_token,
        skip_final_attention=True,
    )
    cache_only = _logit_comparison(control, experiment)
    capture_path: dict[str, Any] | None = None
    if args.capture_layer_id is not None:
        capture_control, captured_control = _next_logits_after_capture_prefill(
            model,
            prefix_tokens=prefix_tokens,
            next_token=next_token,
            capture_layer_id=int(args.capture_layer_id),
            skip_final_output=False,
        )
        capture_experiment, captured_experiment = _next_logits_after_capture_prefill(
            model,
            prefix_tokens=prefix_tokens,
            next_token=next_token,
            capture_layer_id=int(args.capture_layer_id),
            skip_final_output=True,
        )
        capture_path = _logit_comparison(capture_control, capture_experiment)
        control_hidden = captured_control[int(args.capture_layer_id)]
        experiment_hidden = captured_experiment[int(args.capture_layer_id)]
        hidden_diff = mx.abs(control_hidden - experiment_hidden)
        mx.eval(hidden_diff)
        capture_path.update(
            {
                "capture_layer_id": int(args.capture_layer_id),
                "max_abs_capture_diff": float(mx.max(hidden_diff).item()),
                "mean_abs_capture_diff": float(mx.mean(hidden_diff).item()),
            }
        )
    result = {
        "model": str(Path(args.model).expanduser()),
        "prompt_tokens": len(prefix_tokens),
        "next_token": int(next_token),
        "hybrid_mlp": bool(args.hybrid_mlp),
        "hybrid_gdn_out_proj": bool(args.hybrid_gdn_out_proj),
        "memory_config": memory_config,
        "target_meta": target_meta,
        **cache_only,
    }
    if capture_path is not None:
        result["capture_prefill"] = capture_path
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
