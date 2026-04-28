"""Compare q4 and hybrid-prefill target logits on the same prompt."""

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
    "You are reviewing a performance patch for a local LLM runtime. "
    "Be precise about correctness, speed, and memory tradeoffs.\n"
)


def _target_prompt_tokens(tokenizer: Any, target_tokens: int) -> list[int]:
    pieces: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target_tokens:
        pieces.append(PROMPT)
        tokens = list(tokenizer.encode("".join(pieces)))
    return tokens[:target_tokens]


def _load_logits(
    model_ref: str,
    prompt_tokens: list[int],
    *,
    hybrid_mlp: bool,
    hybrid_gdn_proj: bool,
    hybrid_gdn_linear: bool,
    hybrid_gdn_linear_attrs: str | None,
    hybrid_gdn_state_dtype: str | None,
    threshold: int,
) -> tuple[mx.array, dict[str, Any]]:
    if hybrid_mlp:
        os.environ["DFLASH_HYBRID_MLP"] = "1"
    else:
        os.environ.pop("DFLASH_HYBRID_MLP", None)
    if hybrid_gdn_proj:
        os.environ["DFLASH_HYBRID_GDN_PROJ"] = "1"
    else:
        os.environ.pop("DFLASH_HYBRID_GDN_PROJ", None)
    if hybrid_gdn_linear:
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        if hybrid_gdn_linear_attrs:
            os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = hybrid_gdn_linear_attrs
    else:
        os.environ.pop("DFLASH_HYBRID_GDN_LINEAR", None)
        os.environ.pop("DFLASH_HYBRID_GDN_LINEAR_ATTRS", None)
    if hybrid_gdn_state_dtype:
        os.environ["DFLASH_GDN_STATE_DTYPE"] = hybrid_gdn_state_dtype
    else:
        os.environ.pop("DFLASH_GDN_STATE_DTYPE", None)
    os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(int(threshold))
    os.environ["DFLASH_HYBRID_GDN_PROJ_THRESHOLD"] = str(int(threshold))
    os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(int(threshold))

    model, _tokenizer, meta = load_target_bundle(model_ref, lazy=True)
    input_ids = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(model, enable_speculative_linear_cache=False)
    logits, _ = target_forward_with_hidden_states(
        model,
        input_ids=input_ids,
        cache=cache,
        capture_layer_ids=set(),
        last_logits_only=True,
    )
    mx.eval(logits)
    return logits[:, -1, :], meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-proj", action="store_true")
    parser.add_argument("--hybrid-gdn-linear", action="store_true")
    parser.add_argument("--hybrid-gdn-linear-attrs", default=None)
    parser.add_argument(
        "--hybrid-gdn-state-dtype",
        choices=("bf16", "bfloat16", "fp16", "float16"),
        default=None,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    memory_config = configure_mlx_memory_limits()
    os.environ.pop("DFLASH_HYBRID_MLP", None)
    os.environ.pop("DFLASH_HYBRID_GDN_PROJ", None)
    os.environ.pop("DFLASH_HYBRID_GDN_LINEAR", None)
    os.environ.pop("DFLASH_HYBRID_GDN_LINEAR_ATTRS", None)
    os.environ.pop("DFLASH_GDN_STATE_DTYPE", None)
    q4_model, tokenizer, q4_meta = load_target_bundle(args.model, lazy=True)
    prompt_tokens = _target_prompt_tokens(tokenizer, int(args.prompt_tokens))
    q4_input_ids = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    q4_cache = make_target_cache(q4_model, enable_speculative_linear_cache=False)
    q4_logits, _ = target_forward_with_hidden_states(
        q4_model,
        input_ids=q4_input_ids,
        cache=q4_cache,
        capture_layer_ids=set(),
        last_logits_only=True,
    )
    mx.eval(q4_logits)
    q4_last = q4_logits[:, -1, :]

    hybrid_last, hybrid_meta = _load_logits(
        args.model,
        prompt_tokens,
        hybrid_mlp=True,
        hybrid_gdn_proj=bool(args.hybrid_gdn_proj),
        hybrid_gdn_linear=bool(args.hybrid_gdn_linear),
        hybrid_gdn_linear_attrs=args.hybrid_gdn_linear_attrs,
        hybrid_gdn_state_dtype=args.hybrid_gdn_state_dtype,
        threshold=int(args.threshold),
    )

    diff = mx.abs(q4_last - hybrid_last)
    q4_top = mx.argsort(q4_last, axis=-1)[:, -10:]
    hybrid_top = mx.argsort(hybrid_last, axis=-1)[:, -10:]
    q4_top_list = [int(x) for x in q4_top.reshape(-1).tolist()]
    hybrid_top_list = [int(x) for x in hybrid_top.reshape(-1).tolist()]

    result = {
        "model": str(Path(args.model).expanduser()),
        "prompt_tokens": len(prompt_tokens),
        "memory_config": memory_config,
        "q4_meta": {
            key: value
            for key, value in q4_meta.items()
            if key not in {"config"}
        },
        "hybrid_meta": {
            key: value
            for key, value in hybrid_meta.items()
            if key not in {"config"}
        },
        "max_abs_logit_diff": float(mx.max(diff).item()),
        "mean_abs_logit_diff": float(mx.mean(diff).item()),
        "q4_argmax": int(mx.argmax(q4_last, axis=-1).item()),
        "hybrid_argmax": int(mx.argmax(hybrid_last, axis=-1).item()),
        "top10_overlap": len(set(q4_top_list) & set(hybrid_top_list)),
        "q4_top10": q4_top_list,
        "hybrid_top10": hybrid_top_list,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
