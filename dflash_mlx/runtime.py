# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import os
import sys
import time
from contextlib import nullcontext
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.activations import swiglu
from mlx_lm.utils import load, load_model

from dflash_mlx.adapter import detect_engine
from dflash_mlx.draft_backend import make_draft_backend
from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    extract_context_feature,
)
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache


_DFLASH_GENERATION_STREAM = None
_KV_CACHE_BITS_DEFAULT = 8
_KV_CACHE_GROUP_SIZE_DEFAULT = 64
_MLX_CACHE_FRACTION_DEFAULT = 0.25
# M3 Max tuning: isolated MLP/GDN projections can cross over earlier, but
# whole-path cached suffixes at 128-255 tokens were faster staying on q4.
_HYBRID_MLP_THRESHOLD_DEFAULT = 256
_HYBRID_GDN_PROJ_THRESHOLD_DEFAULT = 256
_HYBRID_GDN_LINEAR_THRESHOLD_DEFAULT = 256


def _dflash_stream_context():
    global _DFLASH_GENERATION_STREAM
    raw = os.environ.get("DFLASH_THREAD_STREAM", "").strip().lower()
    if raw in {"", "0", "false", "no"}:
        return nullcontext()
    if not hasattr(mx, "new_thread_local_stream"):
        return nullcontext()
    if _DFLASH_GENERATION_STREAM is None:
        _DFLASH_GENERATION_STREAM = mx.new_thread_local_stream(mx.default_device())
    return mx.stream(_DFLASH_GENERATION_STREAM)


def _resolve_mlx_cache_fraction(device_info: dict[str, Any]) -> float:
    del device_info
    raw = os.environ.get("DFLASH_MLX_CACHE_FRACTION", "").strip()
    if raw:
        try:
            return min(1.0, max(0.0, float(raw)))
        except ValueError:
            return _MLX_CACHE_FRACTION_DEFAULT

    # Hardware note: on this 40-core M3 Max MacBook Pro
    # (applegpu_g15s, 128 GB unified memory), DFLASH_MLX_CACHE_FRACTION=0.125
    # was faster for 4096-token stable-cache prefill, but 10k-token prefill
    # stayed better at MLX-LM's 0.25 default. Keep 0.25 as the safe default;
    # use the env var for device/prompt-specific sweeps.
    return _MLX_CACHE_FRACTION_DEFAULT


def configure_mlx_memory_limits() -> dict[str, Any]:
    if not mx.metal.is_available():
        return {"metal": False}
    device_info = mx.device_info()
    wired_limit = int(device_info["max_recommended_working_set_size"])
    cache_fraction = _resolve_mlx_cache_fraction(device_info)
    mx.set_wired_limit(wired_limit)
    mx.set_cache_limit(int(wired_limit * cache_fraction))
    return {
        "metal": True,
        "wired_limit": wired_limit,
        "cache_fraction": cache_fraction,
        "device_name": str(device_info.get("device_name", "")),
        "architecture": str(device_info.get("architecture", "")),
    }


def _resolve_kv_cache_bits(kv_cache_bits: int = _KV_CACHE_BITS_DEFAULT) -> int:
    bits = int(kv_cache_bits)
    if bits not in {2, 4, 8}:
        raise ValueError("kv_cache_bits must be one of 2, 4, or 8")
    return bits


def _resolve_kv_cache_group_size(
    kv_cache_group_size: int = _KV_CACHE_GROUP_SIZE_DEFAULT,
) -> int:
    group_size = int(kv_cache_group_size)
    if group_size not in {32, 64, 128}:
        raise ValueError("kv_cache_group_size must be one of 32, 64, or 128")
    return group_size


def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")


def _get_dflash_model_classes(config: dict[str, Any]):
    return DFlashDraftModel, DFlashDraftModelArgs


def _resolve_local_model_path(model_ref: str | Path) -> Path:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise FileNotFoundError(f"Model path does not exist and huggingface_hub is unavailable: {model_ref}") from exc

    snapshot_path = snapshot_download(
        repo_id=str(model_ref),
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer*"],
    )
    return Path(snapshot_path)


def _prepare_prompt_tokens(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))


def _require_non_empty_prompt_tokens(prompt_tokens: list[int]) -> None:
    if not prompt_tokens:
        raise ValueError("DFlash generation requires at least one prompt token")


def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: Optional[list[int]],
) -> Optional[mx.array]:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)


def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def _match_acceptance_length(
    drafted_tokens: mx.array,
    posterior_tokens: mx.array,
) -> mx.array:
    if int(drafted_tokens.shape[0]) == 0:
        return mx.array(0, dtype=mx.int32)
    matches = mx.equal(drafted_tokens, posterior_tokens).astype(mx.int32)
    return mx.sum(mx.cumprod(matches, axis=0))


def _record_acceptance_position_stats(
    attempts: list[int],
    accepts: list[int],
    *,
    drafted_count: int,
    acceptance_length: int,
) -> None:
    for index in range(min(int(drafted_count), len(attempts))):
        attempts[index] += 1
        if index < int(acceptance_length):
            accepts[index] += 1


def _acceptance_position_rates(attempts: list[int], accepts: list[int]) -> list[float]:
    return [
        (accepted / attempted) if attempted else 0.0
        for attempted, accepted in zip(attempts, accepts)
    ]


def _adaptive_fallback_enabled() -> bool:
    raw = os.environ.get("DFLASH_ADAPTIVE_FALLBACK", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _adaptive_fallback_probe_cycles() -> int:
    raw = os.environ.get("DFLASH_ADAPTIVE_FALLBACK_PROBE_CYCLES", "").strip()
    if not raw:
        return 4
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _adaptive_fallback_window() -> int:
    raw = os.environ.get("DFLASH_ADAPTIVE_FALLBACK_WINDOW", "").strip()
    if not raw:
        return 8
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _adaptive_fallback_min_tokens_per_cycle() -> float:
    raw = os.environ.get("DFLASH_ADAPTIVE_FALLBACK_MIN_TOKENS_PER_CYCLE", "").strip()
    if not raw:
        return 3.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 3.0


def _adaptive_fallback_cooldown_tokens() -> int:
    raw = os.environ.get("DFLASH_ADAPTIVE_FALLBACK_COOLDOWN_TOKENS", "").strip()
    if not raw:
        return 64
    try:
        return max(1, int(raw))
    except ValueError:
        return 64


def _adaptive_fallback_reprobe_block_tokens() -> int:
    raw = os.environ.get("DFLASH_ADAPTIVE_FALLBACK_REPROBE_BLOCK_TOKENS", "").strip()
    if not raw:
        return 4
    try:
        return max(2, int(raw))
    except ValueError:
        return 4


def _adaptive_fallback_recent_tokens_per_cycle(
    acceptance_history: list[int],
    *,
    window: int,
) -> float:
    if not acceptance_history:
        return 0.0
    recent = acceptance_history[-max(1, int(window)) :]
    return sum(1 + int(accepted) for accepted in recent) / len(recent)


def _should_adaptive_fallback_to_ar(
    acceptance_history: list[int],
    *,
    probe_cycles: int,
    window: int,
    min_tokens_per_cycle: float,
) -> tuple[bool, float]:
    if len(acceptance_history) < max(1, int(probe_cycles)):
        return False, _adaptive_fallback_recent_tokens_per_cycle(
            acceptance_history,
            window=window,
        )
    recent_tokens_per_cycle = _adaptive_fallback_recent_tokens_per_cycle(
        acceptance_history,
        window=window,
    )
    return recent_tokens_per_cycle < float(min_tokens_per_cycle), recent_tokens_per_cycle


def _concat_hidden_state_chunks(
    hidden_state_chunks: list[list[mx.array]],
) -> list[mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return [
        mx.concatenate([chunk[index] for chunk in hidden_state_chunks], axis=1)
        for index in range(len(hidden_state_chunks[0]))
    ]


def _concat_hidden_state_chunk_dicts(
    hidden_state_chunks: list[dict[int, mx.array]],
    capture_layer_ids: set[int],
) -> dict[int, mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return {
        layer_id: mx.concatenate([chunk[layer_id] for chunk in hidden_state_chunks], axis=1)
        for layer_id in sorted(capture_layer_ids)
    }


def _eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)


def _target_text_wrapper(target_model: Any) -> Any:
    if hasattr(target_model, "model"):
        return target_model
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")


def _target_text_model(target_model: Any) -> Any:
    wrapper = _target_text_wrapper(target_model)
    if hasattr(wrapper, "model"):
        return wrapper.model
    raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")


def detect_target_family(target_model: Any) -> str:
    inner = _target_text_model(target_model)
    has_linear = any(
        hasattr(layer, "linear_attn") or hasattr(layer, "is_linear")
        for layer in inner.layers
    )
    return "hybrid_gdn" if has_linear else "pure_attention"


def _target_embed_tokens(target_model: Any) -> Any:
    return _target_text_model(target_model).embed_tokens


def _lm_head_logits(target_model: Any, hidden_states: mx.array) -> mx.array:
    wrapper = _target_text_wrapper(target_model)
    if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
        return wrapper.model.embed_tokens.as_linear(hidden_states)
    return wrapper.lm_head(hidden_states)


def _lm_head_argmax_enabled() -> bool:
    raw = os.environ.get("DFLASH_LM_HEAD_ARGMAX", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _quantized_lm_head_argmax(
    module: Any,
    hidden_states: mx.array,
) -> Optional[mx.array]:
    if hidden_states.ndim != 3:
        return None
    try:
        max_tokens = int(os.environ.get("DFLASH_LM_HEAD_ARGMAX_MAX_TOKENS", "32"))
    except ValueError:
        max_tokens = 32
    if int(hidden_states.shape[1]) > max(1, max_tokens):
        return None

    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    bits = getattr(module, "bits", None)
    group_size = getattr(module, "group_size", None)
    if (
        weight is None
        or scales is None
        or biases is None
        or bits is None
        or group_size is None
    ):
        return None

    try:
        chunk_rows = int(os.environ.get("DFLASH_LM_HEAD_ARGMAX_CHUNK_ROWS", "32768"))
    except ValueError:
        chunk_rows = 32768
    chunk_rows = max(1, chunk_rows)

    flat_hidden = mx.contiguous(hidden_states.reshape(-1, hidden_states.shape[-1]))
    row_count = int(weight.shape[0])
    best_values: Optional[mx.array] = None
    best_indices: Optional[mx.array] = None
    for start in range(0, row_count, chunk_rows):
        end = min(start + chunk_rows, row_count)
        logits = mx.quantized_matmul(
            flat_hidden,
            weight[start:end],
            scales=scales[start:end],
            biases=biases[start:end],
            transpose=True,
            group_size=int(group_size),
            bits=int(bits),
        )
        chunk_values = mx.max(logits, axis=-1)
        chunk_indices = mx.argmax(logits, axis=-1).astype(mx.int32) + start
        if best_values is None or best_indices is None:
            best_values = chunk_values
            best_indices = chunk_indices
        else:
            take_chunk = chunk_values > best_values
            best_values = mx.where(take_chunk, chunk_values, best_values)
            best_indices = mx.where(take_chunk, chunk_indices, best_indices)

    if best_indices is None:
        return None
    return best_indices.reshape(hidden_states.shape[:-1]).astype(mx.uint32)


def _lm_head_argmax(
    target_model: Any,
    hidden_states: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if _lm_head_argmax_enabled() and suppress_token_mask is None:
        wrapper = _target_text_wrapper(target_model)
        module = (
            wrapper.model.embed_tokens
            if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True)
            else wrapper.lm_head
        )
        posterior = _quantized_lm_head_argmax(module, hidden_states)
        if posterior is not None:
            return posterior
    return greedy_tokens_with_mask(
        _lm_head_logits(target_model, hidden_states),
        suppress_token_mask,
    )


def extract_context_feature_from_dict(
    captured_dict: dict[int, mx.array],
    target_layer_ids: list[int],
) -> mx.array:
    selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
    return mx.concatenate(selected, axis=-1)


def extract_context_feature_range_from_dict(
    captured_dict: dict[int, mx.array],
    target_layer_ids: list[int],
    *,
    start: int,
    end: int,
) -> mx.array:
    selected = [
        captured_dict[layer_id + 1][:, int(start):int(end), :]
        for layer_id in target_layer_ids
    ]
    return mx.concatenate(selected, axis=-1)


def _eval_hidden_state_container(
    hidden_states: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(hidden_states, dict):
        mx.eval(*hidden_states.values())
    else:
        mx.eval(*hidden_states)


def _resolve_verify_len_cap(target_model: Any, block_tokens: int) -> int:
    override_raw = os.environ.get("DFLASH_VERIFY_LEN", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(int(block_tokens), override))
    return int(block_tokens)


def _resolve_dflash_max_ctx() -> int:
    raw = os.environ.get("DFLASH_MAX_CTX", "0").strip()
    try:
        max_ctx = int(raw)
    except ValueError:
        max_ctx = 0
    if max_ctx <= 0:
        return sys.maxsize
    return max_ctx


def _resolve_draft_window() -> tuple[int, int]:
    sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64").strip())
    window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024").strip())
    return max(0, sink), max(1, window)


def _prefill_cache_fastpath_enabled() -> bool:
    raw = os.environ.get("DFLASH_PREFILL_CACHE_FASTPATH", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _prefill_skip_capture_enabled() -> bool:
    raw = os.environ.get("DFLASH_PREFILL_SKIP_CAPTURE", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _prefill_last_logits_only_enabled() -> bool:
    raw = os.environ.get("DFLASH_PREFILL_LAST_LOGITS_ONLY", "1").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _prefill_defer_draft_context_enabled() -> bool:
    raw_value = os.environ.get("DFLASH_PREFILL_DEFER_DRAFT_CONTEXT")
    if raw_value is None:
        return not _prefill_cache_fastpath_enabled()
    raw = raw_value.strip().lower()
    return raw not in {"", "0", "false", "no"}


def _prefill_middle_no_logits_enabled() -> bool:
    raw = os.environ.get("DFLASH_PREFILL_MIDDLE_NO_LOGITS", "1").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _prefill_slice_capture_enabled() -> bool:
    raw = os.environ.get("DFLASH_PREFILL_SLICE_CAPTURE", "1").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _prefill_skip_final_mlp_enabled() -> bool:
    # Cache-only prefill chunks need every layer's attention cache, but the
    # final layer MLP output is only consumed by logits/hidden capture. Skipping
    # it is exact for later decode and saves one large-M MLP on middle chunks.
    raw = os.environ.get("DFLASH_PREFILL_SKIP_FINAL_MLP", "1").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _prefill_skip_final_attention_enabled() -> bool:
    # Cache-only prefill chunks discard the final hidden output. On Qwen3.5
    # hybrid models with a final full-attention layer, only that layer's K/V
    # cache survives into the next chunk/decode. Updating just K/V avoids final
    # Q/SDPA/O/MLP work and measured +1.5-2.1% on 10k-20k M3 Max prefill.
    raw = os.environ.get("DFLASH_PREFILL_SKIP_FINAL_ATTENTION", "1").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _clear_cache_after_prefill_enabled() -> bool:
    # Decode starts immediately after prefill, so keeping MLX's allocator cache
    # warm avoids first-cycle reallocations on long prompts. Set
    # DFLASH_CLEAR_CACHE_AFTER_PREFILL=1 on memory-constrained systems.
    raw = os.environ.get("DFLASH_CLEAR_CACHE_AFTER_PREFILL", "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}

def _draft_context_retain_ranges(
    prompt_len: int,
    *,
    sink_size: int,
    window_size: int,
) -> list[tuple[int, int]]:
    if prompt_len <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    if sink_size > 0:
        ranges.append((0, min(int(sink_size), prompt_len)))
    window_start = max(0, prompt_len - max(0, int(window_size)))
    if window_start < prompt_len:
        ranges.append((window_start, prompt_len))
    ranges.sort()

    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _range_overlaps_chunk(
    ranges: list[tuple[int, int]],
    *,
    chunk_start: int,
    chunk_end: int,
) -> list[tuple[int, int]]:
    overlaps: list[tuple[int, int]] = []
    for start, end in ranges:
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)
        if overlap_start < overlap_end:
            overlaps.append((overlap_start, overlap_end))
    return overlaps


def _local_capture_slice_for_ranges(
    ranges: list[tuple[int, int]],
    *,
    chunk_start: int,
) -> Optional[tuple[int, int]]:
    if not ranges:
        return None
    local_start = min(int(start) for start, _ in ranges) - int(chunk_start)
    local_end = max(int(end) for _, end in ranges) - int(chunk_start)
    return max(0, local_start), max(0, local_end)


def _merge_context_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _iter_uncached_prefill_chunks(
    *,
    cached_prompt_len: int,
    uncached_prompt_len: int,
    prefill_step_size: int,
) -> Iterator[tuple[int, int, int, int]]:
    step_size = max(1, int(prefill_step_size))
    cached_prompt_len = max(0, int(cached_prompt_len))
    uncached_prompt_len = max(0, int(uncached_prompt_len))
    for chunk_start in range(0, uncached_prompt_len, step_size):
        chunk_end = min(chunk_start + step_size, uncached_prompt_len)
        yield (
            chunk_start,
            chunk_end,
            cached_prompt_len + chunk_start,
            cached_prompt_len + chunk_end,
        )


def _supports_draft_context_prefill(draft_cache: list[Any]) -> bool:
    return all(isinstance(cache, ContextOnlyDraftKVCache) for cache in draft_cache)


def _draft_hidden_dtype(draft_model: DFlashDraftModel) -> Any:
    weight = getattr(getattr(draft_model, "fc", None), "weight", None)
    return getattr(weight, "dtype", mx.float16)


def _empty_projected_target_hidden(draft_model: DFlashDraftModel) -> mx.array:
    return mx.zeros(
        (1, 0, int(draft_model.args.hidden_size)),
        dtype=_draft_hidden_dtype(draft_model),
    )


def _draft_cache_arrays(draft_cache: list[Any]) -> list[mx.array]:
    arrays: list[mx.array] = []
    for cache in draft_cache:
        if isinstance(cache, ContextOnlyDraftKVCache):
            if cache.keys is not None:
                arrays.append(cache.keys)
            if cache.values is not None:
                arrays.append(cache.values)
    return arrays


def _split_dflash_prompt_cache(
    target_model: Any,
    prompt_cache: list[Any],
) -> tuple[list[Any], list[Any]]:
    target_layers = len(_target_text_model(target_model).layers)
    if len(prompt_cache) < target_layers:
        raise ValueError(
            f"DFlash prompt cache has {len(prompt_cache)} entries; expected at least {target_layers}"
        )
    return list(prompt_cache[:target_layers]), list(prompt_cache[target_layers:])


def _combined_dflash_prompt_cache(
    target_cache: list[Any],
    draft_cache: list[Any],
) -> list[Any]:
    return list(target_cache) + list(draft_cache)


def _target_only_dflash_prompt_cache(
    target_cache: list[Any],
    draft_cache: list[Any],
    *,
    context_len: int,
) -> list[Any]:
    empty_draft_cache: list[Any] = []
    for cache in draft_cache:
        if isinstance(cache, ContextOnlyDraftKVCache):
            empty = ContextOnlyDraftKVCache(
                sink_size=cache.sink_size,
                window_size=cache.window_size,
            )
            # Target-only checkpoints are exact for target prefill reuse, but
            # intentionally skip persisted draft context. Keep the absolute
            # offset so later suffix/draft RoPE positions remain correct.
            empty.offset = int(context_len)
            empty_draft_cache.append(empty)
        else:
            raise TypeError(
                "target-only prompt cache checkpoints require ContextOnlyDraftKVCache"
            )
    return _combined_dflash_prompt_cache(target_cache, empty_draft_cache)


def _context_cache_offset(draft_cache: list[Any]) -> int:
    offsets = [
        int(getattr(cache, "offset", 0) or 0)
        for cache in draft_cache
        if isinstance(cache, ContextOnlyDraftKVCache)
    ]
    return min(offsets) if offsets else 0


def _finalize_draft_context_cache(
    *,
    draft_model: DFlashDraftModel,
    draft_cache: list[Any],
    target_hidden: Optional[mx.array],
    target_hidden_is_projected: bool,
    total_context_len: int,
) -> None:
    if target_hidden is None or int(target_hidden.shape[1]) <= 0:
        return
    current_offset = _context_cache_offset(draft_cache)
    if current_offset >= int(total_context_len):
        return

    segment_len = int(target_hidden.shape[1])
    segment_start = int(total_context_len) - segment_len
    if current_offset < segment_start:
        return

    local_start = current_offset - segment_start
    segment = target_hidden[:, local_start:, :]
    with _dflash_stream_context():
        if not target_hidden_is_projected:
            segment = _project_target_feature_for_draft(draft_model, segment)
        draft_model.prefill_context_cache(
            target_hidden_segments=[(segment, current_offset)],
            cache=draft_cache,
            total_context_len=int(total_context_len),
        )
        draft_cache_arrays = _draft_cache_arrays(draft_cache)
        if draft_cache_arrays:
            mx.eval(*draft_cache_arrays)


def _materialize_deferred_draft_context(
    *,
    draft_model: DFlashDraftModel,
    draft_cache: list[Any],
    target_hidden_segments: list[tuple[mx.array, int]],
    total_context_len: int,
) -> int:
    if not target_hidden_segments:
        return 0

    start_ns = time.perf_counter_ns()
    with _dflash_stream_context():
        projected_segments = [
            (_project_target_feature_for_draft(draft_model, segment), int(offset))
            for segment, offset in target_hidden_segments
            if int(segment.shape[1]) > 0
        ]
        if not projected_segments:
            return time.perf_counter_ns() - start_ns

        draft_model.prefill_context_cache(
            target_hidden_segments=projected_segments,
            cache=draft_cache,
            total_context_len=int(total_context_len),
        )
        draft_cache_arrays = _draft_cache_arrays(draft_cache)
        if draft_cache_arrays:
            mx.eval(*draft_cache_arrays)
    return time.perf_counter_ns() - start_ns


def _materialize_projected_draft_context(
    *,
    draft_model: DFlashDraftModel,
    draft_cache: list[Any],
    projected_hidden_segments: list[tuple[mx.array, int]],
    total_context_len: int,
) -> int:
    if not projected_hidden_segments:
        return 0

    start_ns = time.perf_counter_ns()
    with _dflash_stream_context():
        non_empty_segments = [
            (segment, int(offset))
            for segment, offset in projected_hidden_segments
            if int(segment.shape[1]) > 0
        ]
        if not non_empty_segments:
            return time.perf_counter_ns() - start_ns

        draft_model.prefill_context_cache(
            target_hidden_segments=non_empty_segments,
            cache=draft_cache,
            total_context_len=int(total_context_len),
        )
        draft_cache_arrays = _draft_cache_arrays(draft_cache)
        if draft_cache_arrays:
            mx.eval(*draft_cache_arrays)
    return time.perf_counter_ns() - start_ns


def _context_segments_after_offset(
    target_hidden_segments: list[tuple[mx.array, int]],
    current_offset: int,
) -> list[tuple[mx.array, int]]:
    current_offset = max(0, int(current_offset))
    filtered: list[tuple[mx.array, int]] = []
    for segment, offset in sorted(
        target_hidden_segments,
        key=lambda item: int(item[1]),
    ):
        segment_len = int(segment.shape[1])
        segment_start = int(offset)
        segment_end = segment_start + segment_len
        if segment_len <= 0 or segment_end <= current_offset:
            continue
        if segment_start < current_offset:
            local_start = current_offset - segment_start
            filtered.append((segment[:, local_start:, :], current_offset))
        else:
            filtered.append((segment, segment_start))
    return filtered


def _project_target_feature_for_draft(
    draft_model: DFlashDraftModel,
    target_feature: mx.array,
) -> mx.array:
    return draft_model._project_target_hidden(target_feature)


def _profile_dflash_cycles_enabled() -> bool:
    raw = os.environ.get("DFLASH_PROFILE", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _profile_prefill_chunks_enabled() -> bool:
    raw = os.environ.get("DFLASH_PROFILE_PREFILL", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _mlx_memory_snapshot_gb() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for key, attr in (
        ("active_gb", "get_active_memory"),
        ("cache_gb", "get_cache_memory"),
        ("peak_gb", "get_peak_memory"),
    ):
        fn = getattr(mx, attr, None)
        if not callable(fn):
            continue
        try:
            snapshot[key] = float(fn()) / 1e9
        except Exception:
            pass
    return snapshot


def _ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0


def _should_quantize_draft(quantize_draft: bool = False) -> bool:
    if quantize_draft:
        return True
    raw = os.environ.get("DFLASH_QUANTIZE_DRAFT", "").strip().lower()
    return raw not in {"", "0", "false", "no"}




def _linear_forward(x: mx.array, weight: mx.array, bias: Optional[mx.array]) -> mx.array:
    out = x @ weight.T
    return out if bias is None else out + bias


_EXACT_SMALL_PROJ_PAD_M = 16


class _ExactSmallProjPad(nn.Module):
    def __init__(self, linear: nn.Module, *, pad_m: int = _EXACT_SMALL_PROJ_PAD_M):
        super().__init__()
        self.linear = linear
        self.pad_m = int(pad_m)
        self._dflash_exact_small_proj_wrapped = True

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    @weight.setter
    def weight(self, value: mx.array) -> None:
        self.linear.weight = value

    @property
    def bias(self) -> Optional[mx.array]:
        return getattr(self.linear, "bias", None)

    @bias.setter
    def bias(self, value: Optional[mx.array]) -> None:
        self.linear.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] < self.pad_m:
            batch_size, seq_len, hidden_dim = x.shape
            pad = mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim), dtype=x.dtype)
            out = self.linear(mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)


def _install_exact_small_proj_hooks(
    linear_attn: Any,
    *,
    pad_m: int = _EXACT_SMALL_PROJ_PAD_M,
) -> None:
    for attr_name in ("in_proj_b", "in_proj_a"):
        proj = getattr(linear_attn, attr_name, None)
        if proj is None or getattr(proj, "_dflash_exact_small_proj_wrapped", False):
            continue
        setattr(linear_attn, attr_name, _ExactSmallProjPad(proj, pad_m=pad_m))


def _attention_num_heads(attn: Any) -> int:
    for attr in ("num_attention_heads", "n_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing attention head count attribute")


def _attention_num_kv_heads(attn: Any) -> int:
    for attr in ("num_key_value_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing KV head count attribute")


def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    q_proj_weight = getattr(q_proj, "weight", None)
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_proj_weight is None or q_norm_weight is None:
        return False
    try:
        num_attention_heads = _attention_num_heads(attn)
    except AttributeError:
        return False
    expected_out_dim = 2 * num_attention_heads * int(q_norm_weight.shape[0])
    return int(q_proj_weight.shape[0]) == expected_out_dim


def _pack_mlp_gate_up_enabled() -> bool:
    raw = os.environ.get("DFLASH_PACK_MLP_GATE_UP", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _dequantize_mlp_enabled() -> bool:
    raw = os.environ.get("DFLASH_DEQUANTIZE_MLP", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _hybrid_mlp_enabled() -> bool:
    raw = os.environ.get("DFLASH_HYBRID_MLP", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _hybrid_mlp_threshold() -> int:
    raw = os.environ.get("DFLASH_HYBRID_MLP_THRESHOLD", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return _HYBRID_MLP_THRESHOLD_DEFAULT
    return _HYBRID_MLP_THRESHOLD_DEFAULT


def _hybrid_mlp_pack_bf16_gate_up_enabled() -> bool:
    raw = os.environ.get("DFLASH_HYBRID_MLP_PACK_BF16_GATE_UP", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _hybrid_gdn_proj_enabled() -> bool:
    raw = os.environ.get("DFLASH_HYBRID_GDN_PROJ", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _hybrid_gdn_proj_threshold() -> int:
    raw = os.environ.get("DFLASH_HYBRID_GDN_PROJ_THRESHOLD", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return _HYBRID_GDN_PROJ_THRESHOLD_DEFAULT
    return _HYBRID_GDN_PROJ_THRESHOLD_DEFAULT


def _hybrid_gdn_linear_enabled() -> bool:
    raw = os.environ.get("DFLASH_HYBRID_GDN_LINEAR", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _hybrid_gdn_linear_threshold() -> int:
    raw = os.environ.get("DFLASH_HYBRID_GDN_LINEAR_THRESHOLD", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return _HYBRID_GDN_LINEAR_THRESHOLD_DEFAULT
    return _HYBRID_GDN_LINEAR_THRESHOLD_DEFAULT


def _hybrid_gdn_linear_attrs() -> tuple[str, ...]:
    allowed = {"in_proj_qkv", "in_proj_z", "out_proj"}
    raw = os.environ.get("DFLASH_HYBRID_GDN_LINEAR_ATTRS", "").strip()
    if not raw:
        return ("in_proj_qkv", "in_proj_z", "out_proj")
    attrs = tuple(
        attr for attr in (item.strip() for item in raw.split(",")) if attr in allowed
    )
    return attrs or ("in_proj_qkv", "in_proj_z", "out_proj")


def _gdn_state_dtype_override() -> Any | None:
    raw = os.environ.get("DFLASH_GDN_STATE_DTYPE", "").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        return mx.bfloat16
    if raw in {"fp16", "float16"}:
        return mx.float16
    return None


def _pack_attention_kv_enabled() -> bool:
    raw = os.environ.get("DFLASH_PACK_ATTENTION_KV", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _quantized_linears_compatible(lhs: nn.Module, rhs: nn.Module) -> bool:
    if not isinstance(lhs, nn.QuantizedLinear) or not isinstance(
        rhs,
        nn.QuantizedLinear,
    ):
        return False
    lhs_has_biases = getattr(lhs, "biases", None) is not None
    rhs_has_biases = getattr(rhs, "biases", None) is not None
    return (
        int(getattr(lhs, "group_size", 0)) == int(getattr(rhs, "group_size", -1))
        and int(getattr(lhs, "bits", 0)) == int(getattr(rhs, "bits", -1))
        and getattr(lhs, "mode", "affine") == getattr(rhs, "mode", "affine")
        and int(lhs.weight.shape[1]) == int(rhs.weight.shape[1])
        and ("bias" in lhs) == ("bias" in rhs)
        and lhs_has_biases == rhs_has_biases
    )


def _concat_quantized_linears(
    lhs: nn.QuantizedLinear,
    rhs: nn.QuantizedLinear,
) -> nn.QuantizedLinear:
    ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(ql)
    ql.group_size = lhs.group_size
    ql.bits = lhs.bits
    ql.mode = getattr(lhs, "mode", "affine")
    ql.weight = mx.concatenate([lhs.weight, rhs.weight], axis=0)
    ql.scales = mx.concatenate([lhs.scales, rhs.scales], axis=0)
    lhs_biases = getattr(lhs, "biases", None)
    rhs_biases = getattr(rhs, "biases", None)
    if lhs_biases is not None and rhs_biases is not None:
        ql.biases = mx.concatenate([lhs_biases, rhs_biases], axis=0)
    if "bias" in lhs and "bias" in rhs:
        ql.bias = mx.concatenate([lhs.bias, rhs.bias], axis=0)
    ql.freeze()
    return ql


class _PackedGateUpMLP(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if (
            gate_proj is None
            or up_proj is None
            or down_proj is None
            or not _quantized_linears_compatible(gate_proj, up_proj)
        ):
            raise ValueError("MLP gate/up projections are not compatible for packing")
        self.gate_up_proj = _concat_quantized_linears(gate_proj, up_proj)
        self.down_proj = down_proj
        self.hidden_dim = int(gate_proj.weight.shape[0])
        self._dflash_gate_up_packed = True

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, [self.hidden_dim], axis=-1)
        return self.down_proj(swiglu(gate, up))


def _dequantize_linear(ql: nn.Module) -> tuple[nn.Module, int]:
    if not isinstance(ql, nn.QuantizedLinear):
        return ql, 0
    weight = mx.dequantize(
        ql.weight,
        ql.scales,
        ql.biases,
        ql.group_size,
        ql.bits,
        getattr(ql, "mode", "affine"),
    )
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=("bias" in ql))
    linear.weight = weight
    if "bias" in ql:
        linear.bias = ql.bias
    return linear, int(weight.nbytes)


def _effective_linear_rows(x: mx.array) -> int:
    if len(x.shape) < 2:
        return 1
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    return max(1, rows)


class _HybridLargeMLinear(nn.Module):
    def __init__(self, linear: nn.Module, *, threshold: int):
        super().__init__()
        if not isinstance(linear, nn.QuantizedLinear):
            raise ValueError("hybrid large-M linear requires a quantized linear")
        bf16_linear, weight_nbytes = _dequantize_linear(linear)
        self.q4_linear = linear
        self.bf16_linear = bf16_linear
        self.threshold = max(1, int(threshold))
        self.weight_nbytes = int(weight_nbytes)
        self._dflash_hybrid_large_m_linear = True

    @property
    def weight(self) -> mx.array:
        return self.q4_linear.weight

    def __call__(self, x: mx.array) -> mx.array:
        if _effective_linear_rows(x) >= self.threshold:
            return self.bf16_linear(x)
        return self.q4_linear(x)

    def bf16_weight(self) -> mx.array:
        return self.bf16_linear.weight


class _HybridPrefillMLP(nn.Module):
    def __init__(self, mlp: nn.Module, *, threshold: int):
        super().__init__()
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if (
            gate_proj is None
            or up_proj is None
            or down_proj is None
            or not isinstance(gate_proj, nn.QuantizedLinear)
            or not isinstance(up_proj, nn.QuantizedLinear)
            or not isinstance(down_proj, nn.QuantizedLinear)
        ):
            raise ValueError("hybrid MLP requires unpacked quantized projections")

        bf16_gate_proj, gate_nbytes = _dequantize_linear(gate_proj)
        bf16_up_proj, up_nbytes = _dequantize_linear(up_proj)
        bf16_down_proj, down_nbytes = _dequantize_linear(down_proj)

        self.q4_gate_proj = gate_proj
        self.q4_up_proj = up_proj
        self.q4_down_proj = down_proj
        self.hidden_dim = int(gate_proj.weight.shape[0])
        if _hybrid_mlp_pack_bf16_gate_up_enabled():
            self.bf16_gate_up_proj = nn.Linear(
                bf16_gate_proj.weight.shape[1],
                bf16_gate_proj.weight.shape[0] + bf16_up_proj.weight.shape[0],
                bias=False,
            )
            self.bf16_gate_up_proj.weight = mx.concatenate(
                [bf16_gate_proj.weight, bf16_up_proj.weight],
                axis=0,
            )
        else:
            self.bf16_gate_proj = bf16_gate_proj
            self.bf16_up_proj = bf16_up_proj
        self.bf16_down_proj = bf16_down_proj
        self.threshold = max(1, int(threshold))
        self.weight_nbytes = gate_nbytes + up_nbytes + down_nbytes
        self._dflash_hybrid_mlp = True

    def __call__(self, x: mx.array) -> mx.array:
        if _effective_linear_rows(x) >= self.threshold:
            # Apple/MLX shape tuning: on this M3 Max, q4 wins decode and
            # cached-chat suffixes below the configured threshold, while large
            # prefill chunks can use bf16 weights to avoid q4 dequant overhead.
            gate_up_proj = getattr(self, "bf16_gate_up_proj", None)
            if gate_up_proj is not None:
                gate_up = gate_up_proj(x)
                gate, up = mx.split(gate_up, [self.hidden_dim], axis=-1)
            else:
                gate = self.bf16_gate_proj(x)
                up = self.bf16_up_proj(x)
            return self.bf16_down_proj(swiglu(gate, up))
        return self.q4_down_proj(swiglu(self.q4_gate_proj(x), self.q4_up_proj(x)))

    def bf16_weights(self) -> tuple[mx.array, ...]:
        gate_up_proj = getattr(self, "bf16_gate_up_proj", None)
        if gate_up_proj is not None:
            return (gate_up_proj.weight, self.bf16_down_proj.weight)
        return (
            self.bf16_gate_proj.weight,
            self.bf16_up_proj.weight,
            self.bf16_down_proj.weight,
        )


def install_hybrid_target_mlp_layers(
    target_model: Any,
    *,
    threshold: Optional[int] = None,
) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    resolved_threshold = (
        _HYBRID_MLP_THRESHOLD_DEFAULT if threshold is None else int(threshold)
    )
    info = {
        "enabled": True,
        "threshold": max(1, resolved_threshold),
        "layers": [],
        "linears": 0,
        "weight_nbytes": 0,
    }
    for layer_index, layer in enumerate(text_model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "_dflash_hybrid_mlp", False):
            continue
        try:
            hybrid = _HybridPrefillMLP(mlp, threshold=info["threshold"])
        except ValueError:
            continue
        layer.mlp = hybrid
        # This path deliberately keeps q4 and bf16 MLP weights resident. It is
        # intended for large unified-memory Macs where memory buys prefill
        # throughput without slowing decode's small-M q4 matmuls.
        mx.eval(*hybrid.bf16_weights())
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        info["layers"].append(layer_index)
        info["linears"] += 3
        info["weight_nbytes"] += int(hybrid.weight_nbytes)
    return info


def _pack_gdn_input_projections(
    linear_attn: Any,
    *,
    threshold: int,
) -> bool:
    if getattr(linear_attn, "_dflash_gdn_proj_packed", False):
        return False
    in_proj_qkv = getattr(linear_attn, "in_proj_qkv", None)
    in_proj_z = getattr(linear_attn, "in_proj_z", None)
    in_proj_b = getattr(linear_attn, "in_proj_b", None)
    in_proj_a = getattr(linear_attn, "in_proj_a", None)
    if not _quantized_linears_compatible(in_proj_qkv, in_proj_z):
        return False

    linear_attn.in_proj_qkvz = _concat_quantized_linears(in_proj_qkv, in_proj_z)
    linear_attn._dflash_qkv_dim = int(in_proj_qkv.weight.shape[0])
    if _quantized_linears_compatible(in_proj_b, in_proj_a):
        linear_attn.in_proj_ba = _concat_quantized_linears(in_proj_b, in_proj_a)
        linear_attn._dflash_b_dim = int(in_proj_b.weight.shape[0])
        linear_attn._dflash_gdn_ba_packed = True
    else:
        linear_attn._dflash_gdn_ba_packed = False
    linear_attn._dflash_gdn_proj_threshold = max(1, int(threshold))
    linear_attn._dflash_gdn_proj_packed = True
    eval_args = [
        linear_attn.in_proj_qkvz.weight,
        linear_attn.in_proj_qkvz.scales,
    ]
    if getattr(linear_attn, "_dflash_gdn_ba_packed", False):
        eval_args.extend([linear_attn.in_proj_ba.weight, linear_attn.in_proj_ba.scales])
    mx.eval(*eval_args)
    return True


def install_hybrid_gdn_input_projections(
    target_model: Any,
    *,
    threshold: Optional[int] = None,
) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    resolved_threshold = (
        _HYBRID_GDN_PROJ_THRESHOLD_DEFAULT if threshold is None else int(threshold)
    )
    info = {
        "enabled": True,
        "threshold": max(1, resolved_threshold),
        "layers": [],
    }
    for layer_index, layer in enumerate(text_model.layers):
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        if _pack_gdn_input_projections(linear_attn, threshold=info["threshold"]):
            # Mirrors the newer MLX Qwen3-Next packed projection shape for
            # large-M prefill, while runtime dispatch keeps small-M decode on
            # the original narrower projections.
            info["layers"].append(layer_index)
    return info


def install_hybrid_gdn_linears(
    target_model: Any,
    *,
    threshold: Optional[int] = None,
    attrs: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    resolved_threshold = (
        _HYBRID_GDN_LINEAR_THRESHOLD_DEFAULT if threshold is None else int(threshold)
    )
    selected_attrs = tuple(attrs or _hybrid_gdn_linear_attrs())
    info = {
        "enabled": True,
        "threshold": max(1, resolved_threshold),
        "attrs": list(selected_attrs),
        "layers": [],
        "linears": 0,
        "weight_nbytes": 0,
    }
    for layer_index, layer in enumerate(text_model.layers):
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        eval_weights: list[mx.array] = []
        changed = False
        for attr_name in selected_attrs:
            module = getattr(linear_attn, attr_name, None)
            if module is None or getattr(module, "_dflash_hybrid_large_m_linear", False):
                continue
            try:
                hybrid = _HybridLargeMLinear(module, threshold=info["threshold"])
            except ValueError:
                continue
            setattr(linear_attn, attr_name, hybrid)
            eval_weights.append(hybrid.bf16_weight())
            info["linears"] += 1
            info["weight_nbytes"] += int(hybrid.weight_nbytes)
            changed = True
        if changed:
            # M3 Max / large-unified-memory experiment: q4 remains available
            # for decode-like M, while large prefill can use bf16 projection
            # matmuls if MLX schedules them better than quantized matmuls.
            # DFLASH_HYBRID_GDN_LINEAR_ATTRS lets us avoid the memory pressure
            # from dequantizing every projection on smaller Apple GPUs.
            mx.eval(*eval_weights)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            info["layers"].append(layer_index)
    return info


def _use_packed_gdn_inputs(linear_attn: Any, inputs: mx.array) -> bool:
    if not getattr(linear_attn, "_dflash_gdn_proj_packed", False):
        return False
    threshold = int(
        getattr(
            linear_attn,
            "_dflash_gdn_proj_threshold",
            _HYBRID_GDN_PROJ_THRESHOLD_DEFAULT,
        )
    )
    return _effective_linear_rows(inputs) >= max(1, threshold)


def _project_gdn_inputs(
    linear_attn: Any,
    inputs: mx.array,
    seq_len: Optional[int] = None,
):
    del seq_len
    if _use_packed_gdn_inputs(linear_attn, inputs):
        qkvz = linear_attn.in_proj_qkvz(inputs)
        qkv, z_proj = mx.split(qkvz, [int(linear_attn._dflash_qkv_dim)], axis=-1)
        if getattr(linear_attn, "_dflash_gdn_ba_packed", False):
            ba = linear_attn.in_proj_ba(inputs)
            b, a = mx.split(ba, [int(linear_attn._dflash_b_dim)], axis=-1)
        else:
            b = linear_attn.in_proj_b(inputs)
            a = linear_attn.in_proj_a(inputs)
        return qkv, z_proj, b, a
    return (
        linear_attn.in_proj_qkv(inputs),
        linear_attn.in_proj_z(inputs),
        linear_attn.in_proj_b(inputs),
        linear_attn.in_proj_a(inputs),
    )


def dequantize_target_mlp_layers(target_model: Any) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    info = {
        "enabled": True,
        "layers": [],
        "linears": 0,
        "weight_nbytes": 0,
    }
    for layer_index, layer in enumerate(text_model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        eval_weights: list[mx.array] = []
        changed = False
        for attr_name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(mlp, attr_name, None)
            linear, nbytes = _dequantize_linear(module)
            if linear is module:
                continue
            setattr(mlp, attr_name, linear)
            eval_weights.append(linear.weight)
            info["linears"] += 1
            info["weight_nbytes"] += nbytes
            changed = True
        if changed:
            # This is a memory-for-speed M3 Max/large-unified-memory tuning:
            # materialize each layer before moving on so peak memory does not
            # hold every lazy dequantization graph at once.
            mx.eval(*eval_weights)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            info["layers"].append(layer_index)
    return info


def _pack_attention_kv_projection(attn: nn.Module) -> bool:
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if (
        k_proj is None
        or v_proj is None
        or getattr(attn, "_dflash_kv_packed", False)
        or not _quantized_linears_compatible(k_proj, v_proj)
    ):
        return False
    attn.kv_proj = _concat_quantized_linears(k_proj, v_proj)
    attn._dflash_k_dim = int(k_proj.weight.shape[0])
    attn._dflash_kv_packed = True
    return True


def pack_target_model_weights_selective(
    target_model: Any,
    *,
    validate: bool = True,
    pack_mlp: bool = True,
    pack_attention: bool = False,
) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_pack_info", None) is not None:
        return text_model._dflash_pack_info

    pack_info = {
        "enabled": True,
        "validated": validate,
        "pack_mlp": pack_mlp,
        "pack_attention": pack_attention,
        "packed_mlp_layers": [],
        "packed_attention_layers": [],
    }
    if pack_mlp:
        for layer_index, layer in enumerate(text_model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or getattr(mlp, "_dflash_gate_up_packed", False):
                continue
            try:
                layer.mlp = _PackedGateUpMLP(mlp)
            except ValueError:
                continue
            pack_info["packed_mlp_layers"].append(layer_index)
    if pack_attention:
        for layer_index, layer in enumerate(text_model.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and _pack_attention_kv_projection(attn):
                pack_info["packed_attention_layers"].append(layer_index)
    text_model._dflash_pack_info = pack_info
    return pack_info


def _install_speculative_linear_cache_hook(linear_attn: Any) -> None:
    cls = type(linear_attn)
    if getattr(cls, "_dflash_speculative_call_installed", False):
        return

    original_call = cls.__call__

    def speculative_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        from mlx.nn.layers.distributed import sum_gradients

        B, S, _ = inputs.shape
        armed = isinstance(cache, RecurrentRollbackCache) and getattr(
            cache, "_armed", False
        )
        use_packed_inputs = _use_packed_gdn_inputs(self, inputs)
        state_dtype_override = _gdn_state_dtype_override()
        force_custom_state_dtype = state_dtype_override is not None
        if not armed and not use_packed_inputs and not force_custom_state_dtype:
            return original_call(self, inputs, mask=mask, cache=cache)

        sharding_group = getattr(self, "sharding_group", None)

        if sharding_group is not None:
            inputs = sum_gradients(sharding_group)(inputs)

        qkv, z_proj, b, a = _project_gdn_inputs(self, inputs)
        z = z_proj.reshape(B, S, self.num_v_heads, self.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(B, S, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = gated_delta_mod.compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        state = cache[1] if cache is not None else None

        if state is None:
            _, _, h_k, d_k = q.shape
            h_v, d_v = v.shape[-2:]
            # Upstream Qwen3.5 keeps GDN recurrent state in float32 even when
            # activations are bf16. Preserve that unless an explicit low-level
            # experiment overrides it.
            state_dtype = state_dtype_override or mx.float32
            state = mx.zeros((B, h_v, d_v, d_k), dtype=state_dtype)
        state_in = state

        if armed and mx.default_device() == mx.gpu and mx.metal.is_available() and not self.training:
            if getattr(cache, "_armed", False):
                from dflash_mlx.kernels import gated_delta_kernel_with_tape

                out, state, innovation_tape = gated_delta_kernel_with_tape(
                    q, k, v, g, beta, state, mask
                )
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )
            else:
                out, state = gated_delta_mod.gated_delta_kernel(q, k, v, g, beta, state, mask)
        else:
            if mx.default_device() == mx.gpu and mx.metal.is_available() and not self.training:
                out, state = gated_delta_mod.gated_delta_kernel(q, k, v, g, beta, state, mask)
            else:
                out, state = gated_delta_mod.gated_delta_ops(q, k, v, g, beta, state, mask)
            if armed:
                decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
                decayed_state = state_in[:, None, ...] * decay
                kv_mem = (decayed_state * k[..., None, :]).sum(axis=-1)
                innovation_tape = (v - kv_mem) * beta[..., None]
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )

        if cache is not None:
            cache[1] = state
            if not armed and hasattr(cache, "advance"):
                cache.advance(S)
        out = self.norm(out, z)
        out_flat = out.reshape(B, S, -1)
        out = self.out_proj(out_flat)

        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True


def _split_sdpa_mask(
    mask: Optional[Any],
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Optional[Any]:
    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]


def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    q_len = int(queries.shape[2])
    if q_len <= chunk_size:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    outputs: list[mx.array] = []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        key_end = cached_prefix_len + end
        chunk_mask = _split_sdpa_mask(mask, query_start=start, query_end=end, key_end=key_end)
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)


_HYBRID_SDPA_EXACT_KV_THRESHOLD = 1024
_HYBRID_SDPA_SPLIT_MAX_Q_LEN = 64
_HYBRID_SDPA_PREFILL_SPLIT_THRESHOLD_DEFAULT = 0
_HYBRID_SDPA_PREFILL_SPLIT_CHUNK_DEFAULT = 2048


def _prefill_split_sdpa_threshold() -> int:
    raw = os.environ.get("DFLASH_PREFILL_SPLIT_SDPA_THRESHOLD", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return _HYBRID_SDPA_PREFILL_SPLIT_THRESHOLD_DEFAULT
    return _HYBRID_SDPA_PREFILL_SPLIT_THRESHOLD_DEFAULT


def _prefill_split_sdpa_chunk_size() -> int:
    raw = os.environ.get("DFLASH_PREFILL_SPLIT_SDPA_CHUNK", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return _HYBRID_SDPA_PREFILL_SPLIT_CHUNK_DEFAULT
    return _HYBRID_SDPA_PREFILL_SPLIT_CHUNK_DEFAULT


def _install_split_full_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_split_full_attention_installed", False):
        return

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not getattr(self, "_dflash_split_sdpa_enabled", False):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        if getattr(self, "_dflash_kv_packed", False):
            kv = self.kv_proj(x)
            keys, values = mx.split(kv, [int(self._dflash_k_dim)], axis=-1)
        else:
            keys = self.k_proj(x)
            values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_len)
            keys = self.rope(keys, offset=cached_prefix_len)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        total_kv_len = int(keys.shape[2])
        exact_prefix_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_exact_kv_threshold",
                _HYBRID_SDPA_EXACT_KV_THRESHOLD,
            )
        )
        split_max_q_len = int(
            getattr(
                self,
                "_dflash_split_sdpa_max_q_len",
                _HYBRID_SDPA_SPLIT_MAX_Q_LEN,
            )
        )
        q_len = int(queries.shape[2])
        should_split = (
            cache is not None
            and cached_prefix_len >= exact_prefix_threshold
            and q_len <= split_max_q_len
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        prefill_split_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_prefill_threshold",
                _HYBRID_SDPA_PREFILL_SPLIT_THRESHOLD_DEFAULT,
            )
        )
        should_split_prefill = (
            cache is not None
            and prefill_split_threshold > 0
            and q_len >= prefill_split_threshold
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        should_use_batched_2pass = (
            should_split
            and q_len == 16
            and queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if should_use_batched_2pass:
            from dflash_mlx.kernels import batched_sdpa_2pass_exact

            output = batched_sdpa_2pass_exact(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
            )
            if output is None:
                output = _split_sdpa_output(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=self.scale,
                    mask=mask,
                    cache=cache,
                    chunk_size=1,
                    cached_prefix_len=cached_prefix_len,
                )
        elif should_split:
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=1,
                cached_prefix_len=cached_prefix_len,
            )
        elif should_split_prefill:
            # Large first-prefill path for Apple GPUs with enough unified
            # memory: run the prompt as one model chunk so MLP/GDN weights are
            # loaded once per layer, but split only the full-attention SDPA
            # into causal query tiles to avoid a huge single attention kernel.
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=int(
                    getattr(
                        self,
                        "_dflash_split_sdpa_prefill_chunk_size",
                        _HYBRID_SDPA_PREFILL_SPLIT_CHUNK_DEFAULT,
                    )
                ),
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = split_call
    cls._dflash_split_full_attention_installed = True


def _install_target_speculative_hooks(target_model: Any) -> None:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_speculative_hooks_installed", False):
        return
    if detect_target_family(target_model) == "pure_attention":
        text_model._dflash_speculative_hooks_installed = True
        return
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            _install_exact_small_proj_hooks(layer.linear_attn)
            _install_speculative_linear_cache_hook(layer.linear_attn)
        elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            _install_split_full_attention_hook(layer.self_attn)
    text_model._dflash_speculative_hooks_installed = True


def configure_full_attention_split(
    target_model: Any,
    *,
    enabled: bool,
    chunk_size: int = 8,
) -> None:
    text_model = _target_text_model(target_model)
    if detect_target_family(target_model) == "pure_attention":
        return
    _install_target_speculative_hooks(target_model)
    for layer in text_model.layers:
        if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = enabled
            layer.self_attn._dflash_split_sdpa_chunk_size = int(chunk_size)
            layer.self_attn._dflash_split_sdpa_exact_kv_threshold = (
                _HYBRID_SDPA_EXACT_KV_THRESHOLD
            )
            layer.self_attn._dflash_split_sdpa_max_q_len = (
                _HYBRID_SDPA_SPLIT_MAX_Q_LEN
            )
            layer.self_attn._dflash_split_sdpa_prefill_threshold = (
                _prefill_split_sdpa_threshold()
            )
            layer.self_attn._dflash_split_sdpa_prefill_chunk_size = (
                _prefill_split_sdpa_chunk_size()
            )


def make_target_cache(
    target_model: Any,
    *,
    enable_speculative_linear_cache: bool,
    quantize_kv_cache: bool = False,
    kv_cache_bits: int = _KV_CACHE_BITS_DEFAULT,
    kv_cache_group_size: int = _KV_CACHE_GROUP_SIZE_DEFAULT,
) -> list[Any]:
    resolved_kv_cache_bits = _resolve_kv_cache_bits(kv_cache_bits)
    resolved_kv_cache_group_size = _resolve_kv_cache_group_size(kv_cache_group_size)
    text_model = _target_text_model(target_model)
    caches: list[Any] = []
    for layer_index, layer in enumerate(text_model.layers):
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            if enable_speculative_linear_cache:
                _install_target_speculative_hooks(target_model)
                conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                caches.append(
                    RecurrentRollbackCache(size=2, conv_kernel_size=conv_kernel_size)
                )
            else:
                caches.append(cache_mod.ArraysCache(size=2))
        else:
            if quantize_kv_cache:
                caches.append(
                    cache_mod.QuantizedKVCache(
                        group_size=resolved_kv_cache_group_size,
                        bits=resolved_kv_cache_bits,
                    )
                )
            else:
                caches.append(cache_mod.KVCache())
    return caches


def load_target_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    pack_target_weights: bool = False,
    pack_attention_weights: bool = False,
    validate_packing: bool = True,
    split_full_attention_sdpa: bool = True,
    split_full_attention_chunk_size: int = 8,
    quantize_kv_cache: bool = False,
    kv_cache_bits: int = _KV_CACHE_BITS_DEFAULT,
    kv_cache_group_size: int = _KV_CACHE_GROUP_SIZE_DEFAULT,
):
    resolved_kv_cache_bits = _resolve_kv_cache_bits(kv_cache_bits)
    resolved_kv_cache_group_size = _resolve_kv_cache_group_size(kv_cache_group_size)
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_family = detect_target_family(model)
    if target_family == "hybrid_gdn":
        _install_target_speculative_hooks(model)
        configure_full_attention_split(
            model,
            enabled=split_full_attention_sdpa and not quantize_kv_cache,
            chunk_size=split_full_attention_chunk_size,
        )
    meta = {
        "resolved_model_ref": resolved_ref,
        "config": config,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "kv_cache_bits": resolved_kv_cache_bits,
        "kv_cache_group_size": resolved_kv_cache_group_size,
        "target_family": target_family,
    }
    dequantize_mlp_weights = _dequantize_mlp_enabled()
    hybrid_mlp_weights = _hybrid_mlp_enabled()
    hybrid_gdn_linear_weights = _hybrid_gdn_linear_enabled()
    hybrid_gdn_proj_weights = _hybrid_gdn_proj_enabled()
    if dequantize_mlp_weights and hybrid_mlp_weights:
        raise ValueError(
            "DFLASH_DEQUANTIZE_MLP and DFLASH_HYBRID_MLP are mutually exclusive"
        )
    if hybrid_gdn_linear_weights and hybrid_gdn_proj_weights:
        raise ValueError(
            "DFLASH_HYBRID_GDN_LINEAR and DFLASH_HYBRID_GDN_PROJ are mutually exclusive"
        )
    pack_mlp_weights = (
        pack_target_weights or _pack_mlp_gate_up_enabled()
    ) and not dequantize_mlp_weights and not hybrid_mlp_weights
    pack_attention_weights = pack_attention_weights or _pack_attention_kv_enabled()
    if pack_mlp_weights or pack_attention_weights:
        meta["packing"] = pack_target_model_weights_selective(
            model,
            validate=validate_packing,
            pack_mlp=pack_mlp_weights,
            pack_attention=pack_attention_weights,
        )
    if dequantize_mlp_weights:
        meta["dequantized_mlp"] = dequantize_target_mlp_layers(model)
    if hybrid_mlp_weights:
        meta["hybrid_mlp"] = install_hybrid_target_mlp_layers(
            model,
            threshold=_hybrid_mlp_threshold(),
        )
    if hybrid_gdn_linear_weights:
        meta["hybrid_gdn_linear"] = install_hybrid_gdn_linears(
            model,
            threshold=_hybrid_gdn_linear_threshold(),
            attrs=_hybrid_gdn_linear_attrs(),
        )
    if hybrid_gdn_proj_weights:
        meta["hybrid_gdn_proj"] = install_hybrid_gdn_input_projections(
            model,
            threshold=_hybrid_gdn_proj_threshold(),
        )
    if _verify_enabled_for(config):
        # verify_matmul is gated by DFLASH_VERIFY_QMM at call time; without
        # the flag it falls back to mx.quantized_matmul, making the swap a
        # no-op. setdefault preserves any explicit user override.
        os.environ.setdefault("DFLASH_VERIFY_QMM", "1")
        if _verify_decode_only_enabled():
            setattr(model, "_dflash_verify_linear_pending", True)
            meta["verify_linear_pending"] = True
            meta["verify_linear_swapped"] = 0
        else:
            from dflash_mlx.verify_linear import install_verify_linears

            n_swapped = install_verify_linears(model)
            setattr(model, "_dflash_verify_linear_swapped", n_swapped)
            meta["verify_linear_swapped"] = n_swapped
    return model, tokenizer, meta


def _verify_enabled_for(config: Any) -> bool:
    """Decide whether to swap `nn.QuantizedLinear` → `VerifyQuantizedLinear`.

    Explicit env var wins:
      DFLASH_VERIFY_LINEAR=1  → always on
      DFLASH_VERIFY_LINEAR=0  → always off

    Default (env unset or empty): auto-enable on models where verify M=16
    kernel measurably helps end-to-end. Conservative rule:
      - MoE (num_experts > 0) → ON. Bench shows +6-11 % tps on 35B-A3B.
      - Dense with num_hidden_layers >= 40 → ON. 27B (64 layers) gains +8 %.
      - Dense with num_hidden_layers < 40 → OFF. 9B/4B regress (-17 to -29 %)
        because small-shape projections are Python-overhead-bound and the
        M=16 specialization does not amortize.
    """
    override = os.environ.get("DFLASH_VERIFY_LINEAR", "").strip()
    if override == "1":
        return True
    if override == "0":
        return False
    try:
        text_cfg = config.get("text_config", config) if isinstance(config, dict) else config
        num_experts = int(text_cfg.get("num_experts", 0) or 0)
        num_layers = int(text_cfg.get("num_hidden_layers", 0) or 0)
    except Exception:
        return False
    if num_experts > 0:
        return True
    return num_layers >= 40


def _verify_decode_only_enabled() -> bool:
    raw = os.environ.get("DFLASH_VERIFY_DECODE_ONLY", "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def ensure_verify_linears_for_decode(target_model: Any) -> int:
    if not getattr(target_model, "_dflash_verify_linear_pending", False):
        return 0
    from dflash_mlx.verify_linear import install_verify_linears

    n_swapped = install_verify_linears(target_model)
    setattr(target_model, "_dflash_verify_linear_pending", False)
    setattr(target_model, "_dflash_verify_linear_swapped", n_swapped)
    return int(n_swapped)


def load_draft_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    quantize_draft: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="draft")
    model_path = _resolve_local_model_path(resolved_ref)
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    quantized = _should_quantize_draft(quantize_draft)
    if quantized:
        nn.quantize(model, bits=4, group_size=64)
    if hasattr(model, "prepare_fused_context_kv"):
        model.prepare_fused_context_kv()
    return model, {
        "resolved_model_ref": str(model_ref) if model_ref is not None else str(resolved_ref),
        "config": config,
        "quantize_draft": bool(quantized),
    }


def target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: Optional[mx.array] = None,
    cache: Optional[list[Any]] = None,
    input_embeddings: Optional[mx.array] = None,
    capture_layer_ids: Optional[set[int]] = None,
    skip_logits: bool = False,
    force_hidden_state: bool = False,
    return_normalized: bool = False,
    last_logits_only: bool = False,
    capture_token_slice: Optional[tuple[int, int]] = None,
    skip_final_layer_mlp: bool = False,
    skip_final_layer_attention: bool = False,
) -> tuple[Optional[mx.array], list[mx.array] | dict[int, mx.array]]:
    inner = _target_text_model(target_model)
    hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(inner.layers)
    capture_start = 0
    capture_end: Optional[int] = None
    if capture_token_slice is not None:
        capture_start = max(0, int(capture_token_slice[0]))
        capture_end = max(capture_start, int(capture_token_slice[1]))

    def _captured_value(value: mx.array) -> mx.array:
        if capture_end is None:
            return value
        return value[:, capture_start:capture_end, :]

    capture_all = capture_layer_ids is None
    if capture_all:
        captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
    else:
        capture_layer_ids = set(capture_layer_ids)
        captured = {0: _captured_value(hidden_states)} if 0 in capture_layer_ids else {}
    h = hidden_states

    def _can_skip_final_output(capture_key: int) -> bool:
        # This path is used by prompt-cache checkpoints: we need earlier hidden
        # captures plus cache side effects, but no final hidden output/logits.
        # Keep it conservative so normal generation and final-layer capture are
        # unchanged.
        return (
            skip_logits
            and not force_hidden_state
            and not capture_all
            and capture_layer_ids is not None
            and int(capture_key) not in capture_layer_ids
        )

    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
        last_layer_index = len(inner.layers) - 1
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            capture_key = layer_index + 1
            skip_final_output = layer_index == last_layer_index and _can_skip_final_output(
                capture_key
            )
            if (
                skip_final_output
                and skip_final_layer_attention
                and not getattr(layer, "is_linear", False)
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "self_attn")
            ):
                dependency = _update_full_attention_kv_cache_only(
                    layer.self_attn,
                    layer.input_layernorm(h),
                    layer_cache,
                    return_dependencies=True,
                )
                if dependency is not None:
                    # Earlier hidden captures do not depend on the final layer's
                    # K/V-only update. Keep the K/V arrays in the returned
                    # capture container so mx.eval(captured) forces the cache
                    # mutation without adding a zero-valued reduction.
                    if isinstance(captured, dict):
                        captured[-1] = dependency
                    else:
                        captured.append(dependency)
                    continue
            if (
                skip_final_output
                and skip_final_layer_mlp
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "mlp")
            ):
                normed = layer.input_layernorm(h)
                if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
                    r = layer.linear_attn(normed, mask, layer_cache)
                elif hasattr(layer, "self_attn"):
                    r = layer.self_attn(normed, mask, layer_cache)
                else:
                    h = layer(h, mask=mask, cache=layer_cache)
                    continue
                h = h + r
            else:
                h = layer(h, mask=mask, cache=layer_cache)
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = _captured_value(h)
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        last_layer_index = len(inner.layers) - 1
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            capture_key = layer_index + 1
            skip_final_output = layer_index == last_layer_index and _can_skip_final_output(
                capture_key
            )
            if (
                skip_final_output
                and skip_final_layer_attention
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "self_attn")
            ):
                dependency = _update_full_attention_kv_cache_only(
                    layer.self_attn,
                    layer.input_layernorm(h),
                    layer_cache,
                    return_dependencies=True,
                )
                if dependency is not None:
                    # Preserve the cache-write dependency even when the final
                    # hidden output/logits are intentionally skipped.
                    if isinstance(captured, dict):
                        captured[-1] = dependency
                    else:
                        captured.append(dependency)
                    continue
            if (
                skip_final_output
                and skip_final_layer_mlp
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "mlp")
                and hasattr(layer, "self_attn")
            ):
                normed = layer.input_layernorm(h)
                r = layer.self_attn(normed, mask, layer_cache)
                h = h + r
            else:
                h = layer(h, mask, layer_cache)
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = _captured_value(h)
    if force_hidden_state:
        if isinstance(captured, dict):
            captured[-1] = h
        else:
            captured.append(h)
    if skip_logits:
        return None, captured
    logits_hidden = h[:, -1:, :] if last_logits_only else h
    normalized = inner.norm(logits_hidden)
    if return_normalized:
        return normalized, captured
    logits = _lm_head_logits(target_model, normalized)
    return logits, captured


def _flatten_mx_arrays(value: Any) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        arrays: list[mx.array] = []
        for item in value:
            arrays.extend(_flatten_mx_arrays(item))
        return arrays
    if isinstance(value, dict):
        arrays = []
        for item in value.values():
            arrays.extend(_flatten_mx_arrays(item))
        return arrays
    return []


def _cache_new_token_slices(cache: Any, *, start: int, end: int) -> list[mx.array]:
    if end <= start:
        return []
    arrays: list[mx.array] = []
    for field_name in ("keys", "values"):
        value = getattr(cache, field_name, None)
        for array in _flatten_mx_arrays(value):
            if array.ndim >= 2:
                arrays.append(array[..., start:end, :])
    return arrays


def _zero_dependency(dtype: Any, arrays: list[mx.array]) -> Optional[mx.array]:
    dependency: Optional[mx.array] = None
    for array in arrays:
        if int(array.size) <= 0:
            continue
        term = mx.sum(array.astype(mx.float32)) * 0.0
        dependency = term if dependency is None else dependency + term
    return None if dependency is None else dependency.astype(dtype)


def _update_full_attention_kv_cache_only(
    attn: Any,
    x: mx.array,
    cache: Any,
    *,
    return_dependencies: bool = False,
) -> Optional[mx.array] | list[mx.array]:
    if cache is None or not hasattr(cache, "update_and_fetch"):
        return None

    batch_size, seq_len, _ = x.shape
    cache_start = int(getattr(cache, "offset", 0) or 0)
    num_key_value_heads = _attention_num_kv_heads(attn)
    if getattr(attn, "_dflash_kv_packed", False):
        kv = attn.kv_proj(x)
        keys, values = mx.split(kv, [int(attn._dflash_k_dim)], axis=-1)
    else:
        keys = attn.k_proj(x)
        values = attn.v_proj(x)

    keys = attn.k_norm(keys.reshape(batch_size, seq_len, num_key_value_heads, -1))
    keys = keys.transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, num_key_value_heads, -1).transpose(
        0,
        2,
        1,
        3,
    )
    if hasattr(attn, "rope"):
        keys = attn.rope(keys, offset=cache_start)
    cache.update_and_fetch(keys, values)
    cache_end = int(getattr(cache, "offset", cache_start + seq_len) or 0)
    dependencies = _cache_new_token_slices(cache, start=cache_start, end=cache_end)
    if return_dependencies:
        return dependencies
    return _zero_dependency(x.dtype, dependencies)


def target_prefill_without_logits(
    target_model: Any,
    *,
    input_ids: mx.array,
    cache: Optional[list[Any]] = None,
    skip_final_layer_mlp: bool = False,
    skip_final_layer_attention: bool = False,
    return_dependencies: bool = False,
) -> mx.array | tuple[mx.array, list[mx.array]]:
    inner = _target_text_model(target_model)
    h = inner.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(inner.layers)
    eval_dependencies: list[mx.array] = []

    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        last_layer_index = len(inner.layers) - 1
        for layer_index, (layer, layer_cache) in enumerate(
            zip(inner.layers, cache, strict=True)
        ):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            if (
                skip_final_layer_attention
                and layer_index == last_layer_index
                and not getattr(layer, "is_linear", False)
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "self_attn")
            ):
                # The returned hidden state is discarded on this path. Mutate
                # only the final layer K/V cache and attach a zero dependency so
                # the caller's mx.eval(h) still forces the cache write.
                dependency = _update_full_attention_kv_cache_only(
                    layer.self_attn,
                    layer.input_layernorm(h),
                    layer_cache,
                    return_dependencies=return_dependencies,
                )
                if return_dependencies and dependency is not None:
                    eval_dependencies.extend(dependency or [])
                    continue
                if dependency is not None:
                    h = h + dependency
                    continue
            if (
                skip_final_layer_mlp
                and layer_index == last_layer_index
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "mlp")
            ):
                normed = layer.input_layernorm(h)
                if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
                    r = layer.linear_attn(normed, mask, layer_cache)
                elif hasattr(layer, "self_attn"):
                    r = layer.self_attn(normed, mask, layer_cache)
                else:
                    h = layer(h, mask=mask, cache=layer_cache)
                    continue
                h = h + r
            else:
                h = layer(h, mask=mask, cache=layer_cache)
    else:
        mask = create_attention_mask(h, cache[0])
        last_layer_index = len(inner.layers) - 1
        for layer_index, (layer, layer_cache) in enumerate(
            zip(inner.layers, cache, strict=True)
        ):
            if (
                skip_final_layer_attention
                and layer_index == last_layer_index
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "self_attn")
            ):
                dependency = _update_full_attention_kv_cache_only(
                    layer.self_attn,
                    layer.input_layernorm(h),
                    layer_cache,
                    return_dependencies=return_dependencies,
                )
                if return_dependencies and dependency is not None:
                    eval_dependencies.extend(dependency or [])
                    continue
                if dependency is not None:
                    h = h + dependency
                    continue
            if (
                skip_final_layer_mlp
                and layer_index == last_layer_index
                and hasattr(layer, "input_layernorm")
                and hasattr(layer, "mlp")
                and hasattr(layer, "self_attn")
            ):
                normed = layer.input_layernorm(h)
                r = layer.self_attn(normed, mask, layer_cache)
                h = h + r
            else:
                h = layer(h, mask, layer_cache)
    if return_dependencies:
        return h, eval_dependencies
    return h


def trim_cache_to(cache_entries: list[Any], size: int) -> int:
    if not cache_entries:
        return 0
    current_size = int(getattr(cache_entries[0], "offset", 0) or 0)
    if current_size <= size:
        return 0
    return int(cache_mod.trim_prompt_cache(cache_entries, current_size - size) or 0)


def _arm_target_rollback(cache_entries: list[Any]) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback()


def _arm_target_rollback_with_prefix(
    cache_entries: list[Any],
    *,
    prefix_len: int,
) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback(prefix_len=int(prefix_len))


def _clear_rollback_state(cache_entry: Any) -> None:
    if hasattr(cache_entry, "clear_transients"):
        cache_entry.clear_transients()
        return
    if hasattr(cache_entry, "_armed"):
        cache_entry._armed = False
    if hasattr(cache_entry, "_tape"):
        cache_entry._tape = None
    if hasattr(cache_entry, "_tape_k"):
        cache_entry._tape_k = None
    if hasattr(cache_entry, "_tape_g"):
        cache_entry._tape_g = None
    if hasattr(cache_entry, "_tape_qkv"):
        cache_entry._tape_qkv = None
    if hasattr(cache_entry, "_snapshot"):
        cache_entry._snapshot = None


def _cleanup_generation_caches(
    target_cache: list[Any],
    draft_cache: list[Any],
) -> None:
    for cache_entry in target_cache:
        if hasattr(cache_entry, "clear_transients"):
            cache_entry.clear_transients()
    draft_cache.clear()
    target_cache.clear()


def _restore_target_cache_after_acceptance(
    cache_entries: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int = 0,
) -> int:
    replay_ns_total = 0
    fully_accepted = drafted_tokens > 0 and acceptance_length == drafted_tokens
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            if fully_accepted:
                _clear_rollback_state(cache_entry)
                continue
            replay_start_ns = time.perf_counter_ns()
            cache_entry.rollback(acceptance_length)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "trim"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                replay_start_ns = time.perf_counter_ns()
                cache_entry.trim(offset - target_len)
                replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "offset"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                cache_entry.offset = target_len
        elif hasattr(cache_entry, "crop"):
            cache_entry.crop(target_len)
    return replay_ns_total


def _verify_target_block(
    *,
    target_model: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    verify_chunk_tokens: Optional[int],
    capture_layer_ids: Optional[set[int]] = None,
    return_normalized: bool = False,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    total_tokens = int(verify_ids.shape[1])
    if total_tokens <= 0:
        raise ValueError("verify block must contain at least one token")

    chunk_size = max(1, int(verify_chunk_tokens or total_tokens))
    if chunk_size >= total_tokens:
        verify_logits, verify_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
            return_normalized=return_normalized,
        )
        return verify_logits, verify_hidden_states

    logits_chunks: list[mx.array] = []
    hidden_state_chunks: list[list[mx.array]] | list[dict[int, mx.array]]
    hidden_state_chunks = []
    for offset in range(0, total_tokens, chunk_size):
        verify_chunk = verify_ids[:, offset : offset + chunk_size]
        chunk_logits, chunk_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_chunk,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
            return_normalized=return_normalized,
        )
        logits_chunks.append(chunk_logits)
        hidden_state_chunks.append(chunk_hidden_states)

    if capture_layer_ids is None:
        return mx.concatenate(logits_chunks, axis=1), _concat_hidden_state_chunks(hidden_state_chunks)
    return (
        mx.concatenate(logits_chunks, axis=1),
        _concat_hidden_state_chunk_dicts(hidden_state_chunks, capture_layer_ids),
    )


def generate_baseline_once(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    kv_cache_bits: int = _KV_CACHE_BITS_DEFAULT,
    kv_cache_group_size: int = _KV_CACHE_GROUP_SIZE_DEFAULT,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    _require_non_empty_prompt_tokens(prompt_tokens)
    stop_token_ids = list(stop_token_ids or [])

    if max_new_tokens <= 0:
        return {
            "elapsed_us": 0.0,
            "prompt_token_count": len(prompt_tokens),
            "generated_token_ids": [],
            "generation_tokens": 0,
        }

    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
        kv_cache_bits=kv_cache_bits,
        kv_cache_group_size=kv_cache_group_size,
    )
    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    logits = target_model(prompt_array, cache=cache)
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    return {
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": len(prompt_tokens),
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "kv_cache_bits": int(_resolve_kv_cache_bits(kv_cache_bits)),
        "kv_cache_group_size": int(_resolve_kv_cache_group_size(kv_cache_group_size)),
    }


def stream_baseline_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    kv_cache_bits: int = _KV_CACHE_BITS_DEFAULT,
    kv_cache_group_size: int = _KV_CACHE_GROUP_SIZE_DEFAULT,
    fallback_reason: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    _require_non_empty_prompt_tokens(prompt_tokens)
    prompt_len = len(prompt_tokens)
    stop_token_ids = list(stop_token_ids or [])
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
        kv_cache_bits=kv_cache_bits,
        kv_cache_group_size=kv_cache_group_size,
    )
    start_ns = time.perf_counter_ns()
    _yield_pause_ns = 0

    prefill_start_ns = time.perf_counter_ns()
    logits = target_model(prompt_array, cache=cache)
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    _pre_yield = time.perf_counter_ns()
    yield {
        "event": "prefill",
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": prompt_len,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }
    _yield_pause_ns += time.perf_counter_ns() - _pre_yield

    _pre_yield = time.perf_counter_ns()
    yield {
        "event": "token",
        "token_id": next_token,
        "generated_tokens": 1,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }
    _yield_pause_ns += time.perf_counter_ns() - _pre_yield

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)
        _pre_yield = time.perf_counter_ns()
        yield {
            "event": "token",
            "token_id": next_token,
            "generated_tokens": len(generated_tokens),
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "fallback_ar": True,
            "fallback_reason": fallback_reason,
        }
        _yield_pause_ns += time.perf_counter_ns() - _pre_yield

    elapsed_us = (time.perf_counter_ns() - start_ns - _yield_pause_ns) / 1_000.0
    del cache
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    yield {
        "event": "summary",
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "accepted_from_draft": 0,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": 0.0,
            "draft_prefill": 0.0,
            "draft_incremental": 0.0,
            "verify": 0.0,
            "replay": 0.0,
            "commit": 0.0,
        },
        "verify_len_cap": None,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "kv_cache_bits": int(_resolve_kv_cache_bits(kv_cache_bits)),
        "kv_cache_group_size": int(_resolve_kv_cache_group_size(kv_cache_group_size)),
    }


def generate_dflash_once(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    verify_chunk_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    kv_cache_bits: int = _KV_CACHE_BITS_DEFAULT,
    kv_cache_group_size: int = _KV_CACHE_GROUP_SIZE_DEFAULT,
    prefill_step_size: int = 2048,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    _require_non_empty_prompt_tokens(prompt_tokens)
    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        baseline = generate_baseline_once(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            kv_cache_bits=kv_cache_bits,
            kv_cache_group_size=kv_cache_group_size,
        )
        baseline.update(
            {
                "accepted_from_draft": 0,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
                "phase_timings_us": {
                    "prefill": baseline["elapsed_us"],
                    "draft": 0.0,
                    "draft_prefill": 0.0,
                    "draft_incremental": 0.0,
                    "verify": 0.0,
                    "replay": 0.0,
                    "commit": 0.0,
                },
                "speculative_linear_cache": False,
                "verify_chunk_tokens": None,
                "verify_len_cap": None,
                "quantize_kv_cache": bool(quantize_kv_cache),
                "kv_cache_bits": int(_resolve_kv_cache_bits(kv_cache_bits)),
                "kv_cache_group_size": int(_resolve_kv_cache_group_size(kv_cache_group_size)),
                "fallback_ar": True,
                "fallback_reason": fallback_reason,
            }
        )
        return baseline
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    use_speculative_linear_cache = verify_chunk_tokens is None
    engine = detect_engine(target_model)
    draft_backend = make_draft_backend()
    target_cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=use_speculative_linear_cache,
        quantize_kv_cache=quantize_kv_cache,
        kv_cache_bits=kv_cache_bits,
        kv_cache_group_size=kv_cache_group_size,
    )

    draft_cache = draft_backend.make_cache(
        draft_model=draft_model,
        sink_size=draft_sink_size,
        window_size=draft_window_size,
    )
    target_layer_id_list = list(draft_model.target_layer_ids)
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}
    mask_token_id = int(draft_model.mask_token_id)

    try:
        start_ns = time.perf_counter_ns()
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = max(1, int(prefill_step_size))
        cache_only_prefill = max_new_tokens <= 0
        prefill_logits = None
        target_hidden: Optional[mx.array] = None
        target_hidden_chunks: list[mx.array] = []
        target_hidden_is_projected = False
        supports_draft_context_prefill = _supports_draft_context_prefill(draft_cache)
        prefill_defer_context = (
            _prefill_defer_draft_context_enabled()
            and supports_draft_context_prefill
        )
        prefill_fastpath = (
            _prefill_cache_fastpath_enabled()
            and supports_draft_context_prefill
            and not prefill_defer_context
        )
        skip_prefill_capture = prefill_defer_context or (
            prefill_fastpath and _prefill_skip_capture_enabled()
        )
        retained_context_ranges = _draft_context_retain_ranges(
            prompt_len,
            sink_size=draft_sink_size,
            window_size=draft_window_size,
        )
        retained_context_tokens = sum(
            end - start for start, end in retained_context_ranges
        )
        prompt_cache_checkpoint_tokens = 0
        checkpoint_count = 0
        checkpoint_ns_total = 0
        prefill_last_logits_only = _prefill_last_logits_only_enabled()
        prefill_slice_capture = _prefill_slice_capture_enabled()
        retained_context_segments: list[tuple[mx.array, int]] = []
        deferred_context_segments: list[tuple[mx.array, int]] = []
        for chunk_start in range(0, prompt_len, prefill_step_size):
            chunk_end = min(chunk_start + prefill_step_size, prompt_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            is_last_chunk = chunk_end >= prompt_len
            chunk_retained_ranges = _range_overlaps_chunk(
                retained_context_ranges,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            needs_prefill_features = bool(chunk_retained_ranges) or (
                not skip_prefill_capture
                and not (prefill_fastpath or prefill_defer_context)
            )
            slice_candidate_ranges = (
                chunk_retained_ranges
                if (prefill_fastpath or prefill_defer_context)
                else []
            )
            capture_token_slice = (
                _local_capture_slice_for_ranges(
                    slice_candidate_ranges,
                    chunk_start=chunk_start,
                )
                if prefill_slice_capture and slice_candidate_ranges
                else None
            )
            capture_offset = int(capture_token_slice[0]) if capture_token_slice else 0
            if (
                (cache_only_prefill or _prefill_middle_no_logits_enabled())
                and not needs_prefill_features
                and (cache_only_prefill or not is_last_chunk)
            ):
                with _dflash_stream_context():
                    h, h_dependencies = target_prefill_without_logits(
                        target_model,
                        input_ids=chunk_ids,
                        cache=target_cache,
                        skip_final_layer_mlp=_prefill_skip_final_mlp_enabled(),
                        skip_final_layer_attention=_prefill_skip_final_attention_enabled(),
                        return_dependencies=True,
                    )
                    if h_dependencies:
                        # Cache-only chunks discard the final hidden state. The
                        # direct K/V dependencies already force all preceding
                        # layer work on this M3 Max-tuned prefill path, so avoid
                        # making the discarded hidden tensor an eval root.
                        mx.eval(*h_dependencies)
                    else:
                        mx.eval(h)
                del h
                continue
            with _dflash_stream_context():
                prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
                    target_model,
                    input_ids=chunk_ids,
                    cache=target_cache,
                    capture_layer_ids=(
                        capture_layer_ids if needs_prefill_features else set()
                    ),
                    skip_logits=cache_only_prefill or not is_last_chunk,
                    force_hidden_state=(
                        not needs_prefill_features
                        and (cache_only_prefill or not is_last_chunk)
                    ),
                    last_logits_only=(
                        not cache_only_prefill
                        and is_last_chunk
                        and prefill_last_logits_only
                    ),
                    capture_token_slice=capture_token_slice,
                    skip_final_layer_mlp=_prefill_skip_final_mlp_enabled(),
                    skip_final_layer_attention=_prefill_skip_final_attention_enabled(),
                )
                if not needs_prefill_features:
                    if prefill_logits is not None:
                        _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
                    else:
                        _eval_hidden_state_container(prefill_hidden_states)
                    del prefill_hidden_states
                    continue
                eval_targets: list[mx.array] = []
                if prefill_logits is not None:
                    eval_targets.append(prefill_logits)
                if prefill_fastpath:
                    projected_segments = []
                    for retain_start, retain_end in chunk_retained_ranges:
                        local_start = retain_start - chunk_start
                        local_end = retain_end - chunk_start
                        feat = extract_context_feature_range_from_dict(
                            prefill_hidden_states,
                            target_layer_id_list,
                            start=local_start - capture_offset,
                            end=local_end - capture_offset,
                        )
                        projected = _project_target_feature_for_draft(
                            draft_model,
                            feat,
                        )
                        retained_context_segments.append((projected, retain_start))
                        projected_segments.append(projected)
                    if projected_segments:
                        eval_targets.extend(projected_segments)
                elif prefill_defer_context:
                    deferred_segments = []
                    for retain_start, retain_end in chunk_retained_ranges:
                        local_start = retain_start - chunk_start
                        local_end = retain_end - chunk_start
                        segment = mx.contiguous(
                            extract_context_feature_range_from_dict(
                                prefill_hidden_states,
                                target_layer_id_list,
                                start=local_start - capture_offset,
                                end=local_end - capture_offset,
                            )
                        )
                        deferred_context_segments.append((segment, retain_start))
                        deferred_segments.append(segment)
                    if deferred_segments:
                        eval_targets.extend(deferred_segments)
                else:
                    feat = extract_context_feature_from_dict(
                        prefill_hidden_states,
                        target_layer_id_list,
                    )
                    target_hidden_chunks.append(mx.contiguous(feat))
                    eval_targets.append(target_hidden_chunks[-1])
                if eval_targets:
                    mx.eval(*eval_targets)
                del prefill_hidden_states
        if prefill_fastpath:
            _materialize_projected_draft_context(
                draft_model=draft_model,
                draft_cache=draft_cache,
                projected_hidden_segments=retained_context_segments,
                total_context_len=prompt_len,
            )
            target_hidden = _empty_projected_target_hidden(draft_model)
            target_hidden_is_projected = True
        elif prefill_defer_context:
            target_hidden = _empty_projected_target_hidden(draft_model)
            target_hidden_is_projected = True
        elif target_hidden is None:
            if target_hidden_chunks:
                with _dflash_stream_context():
                    target_hidden = mx.concatenate(target_hidden_chunks, axis=1)
                    mx.eval(target_hidden)
            else:
                target_hidden = mx.zeros(
                    (1, 0, int(draft_model.args.hidden_size)),
                    dtype=_draft_hidden_dtype(draft_model),
                )
        if _clear_cache_after_prefill_enabled() and hasattr(mx, "clear_cache"):
            mx.clear_cache()
        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        draft_block_size = int(draft_model.block_size)
        requested_block_tokens = draft_block_size if block_tokens is None else int(block_tokens)
        effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))

        if cache_only_prefill:
            acceptance_position_attempts = [0] * max(0, effective_block_tokens - 1)
            acceptance_position_accepts = [0] * max(0, effective_block_tokens - 1)
            result = {
                "elapsed_us": (time.perf_counter_ns() - start_ns) / 1_000.0,
                "prompt_token_count": prompt_len,
                "generated_token_ids": [],
                "generation_tokens": 0,
                "accepted_from_draft": 0,
                "acceptance_ratio": 0.0,
                "draft_tokens_attempted": 0,
                "draft_acceptance_ratio": 0.0,
                "block_tokens": effective_block_tokens,
                "cycles_completed": 0,
                "phase_timings_us": {
                    "prefill": prefill_ns / 1_000.0,
                    "draft": 0.0,
                    "draft_prefill": 0.0,
                    "draft_incremental": 0.0,
                    "verify": 0.0,
                    "replay": 0.0,
                    "commit": 0.0,
                },
                "speculative_linear_cache": use_speculative_linear_cache,
                "prefill_cache_fastpath": bool(prefill_fastpath),
                "prefill_defer_draft_context": bool(prefill_defer_context),
                "prefill_skip_capture": bool(skip_prefill_capture),
                "prefill_context_tokens": int(retained_context_tokens),
                "prompt_cache_checkpoint_tokens": int(prompt_cache_checkpoint_tokens),
                "prompt_cache_checkpoints": int(checkpoint_count),
                "prompt_cache_checkpoint_us": checkpoint_ns_total / 1_000.0,
                "cache_only_prefill": True,
                "verify_chunk_tokens": int(verify_chunk_tokens) if verify_chunk_tokens else None,
                "verify_len_cap": int(_resolve_verify_len_cap(target_model, effective_block_tokens)),
                "quantize_kv_cache": bool(quantize_kv_cache),
                "kv_cache_bits": int(_resolve_kv_cache_bits(kv_cache_bits)),
                "kv_cache_group_size": int(_resolve_kv_cache_group_size(kv_cache_group_size)),
                "prefill_step_size": int(prefill_step_size),
                "tokens_per_cycle": 0.0,
                "acceptance_history": [],
                "acceptance_position_attempts": acceptance_position_attempts,
                "acceptance_position_accepts": acceptance_position_accepts,
                "acceptance_position_rates": _acceptance_position_rates(
                    acceptance_position_attempts,
                    acceptance_position_accepts,
                ),
                "acceptance_first_20_avg": 0.0,
                "acceptance_last_20_avg": 0.0,
                "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
            }
            if _profile_dflash_cycles_enabled():
                result["cycle_profile_us"] = []
                result["cycle_profile_totals_us"] = {
                    "draft": 0.0,
                    "verify": 0.0,
                    "acceptance": 0.0,
                    "hidden_extraction": 0.0,
                    "rollback": 0.0,
                    "other": 0.0,
                    "cycle_total": 0.0,
                }
            return result

        if prefill_logits is None:
            raise RuntimeError("DFlash prefill did not produce logits for decode")

        verify_install_start_ns = time.perf_counter_ns()
        verify_linear_swapped = ensure_verify_linears_for_decode(target_model)
        verify_install_ns = time.perf_counter_ns() - verify_install_start_ns
        verify_linear_install_ns_total = verify_install_ns

        with _dflash_stream_context():
            suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
            staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)

        generated_token_buffer = mx.full((max_new_tokens,), mask_token_id, dtype=mx.uint32)
        block_token_buffer = mx.full((effective_block_tokens,), mask_token_id, dtype=mx.uint32)
        mask_token_tail = mx.full(
            (max(0, effective_block_tokens - 1),),
            mask_token_id,
            dtype=mx.uint32,
        )
        with _dflash_stream_context():
            mask_embedding_tail = (
                _target_embed_tokens(target_model)(mask_token_tail[None])
                if int(mask_token_tail.shape[0]) > 0
                else None
            )
            if mask_embedding_tail is not None:
                mx.eval(mask_embedding_tail)
        generated_token_count = 0
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        acceptance_history: list[int] = []
        acceptance_position_attempts = [0] * max(0, effective_block_tokens - 1)
        acceptance_position_accepts = [0] * max(0, effective_block_tokens - 1)
        profile_cycles = _profile_dflash_cycles_enabled()
        cycle_profiles: list[dict[str, Any]] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }
        prefetched_draft: Optional[dict[str, Any]] = None
        if prefill_defer_context and max_new_tokens > 1:
            deferred_context_ns = _materialize_deferred_draft_context(
                draft_model=draft_model,
                draft_cache=draft_cache,
                target_hidden_segments=deferred_context_segments,
                total_context_len=prompt_len,
            )
            draft_ns_total += deferred_context_ns
            draft_prefill_ns += deferred_context_ns

        while generated_token_count < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - generated_token_count
            block_len = max(1, min(effective_block_tokens, remaining))
            block_token_buffer[:block_len] = mask_token_id
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]
            current_staged_first = staged_first
            drafted = None

            if block_len > 1:
                if profile_cycles:
                    draft_start_ns = time.perf_counter_ns()
                    with _dflash_stream_context():
                        drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=current_staged_first,
                            target_hidden=target_hidden,
                            target_hidden_is_projected=target_hidden_is_projected,
                            block_len=block_len,
                            mask_token_tail=mask_token_tail,
                            mask_embedding_tail=mask_embedding_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=False,
                        )
                        block_token_ids[1:block_len] = drafted
                    draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                else:
                    if (
                        prefetched_draft is not None
                        and int(prefetched_draft["block_len"]) == block_len
                    ):
                        drafted = prefetched_draft["drafted"]
                        current_staged_first = prefetched_draft["staged_first"]
                    else:
                        draft_start_ns = time.perf_counter_ns()
                        with _dflash_stream_context():
                            drafted = draft_backend.draft_greedy(
                                target_model=target_model,
                                draft_model=draft_model,
                                draft_cache=draft_cache,
                                staged_first=current_staged_first,
                                target_hidden=target_hidden,
                                target_hidden_is_projected=target_hidden_is_projected,
                                block_len=block_len,
                                mask_token_tail=mask_token_tail,
                                mask_embedding_tail=mask_embedding_tail,
                                suppress_token_mask=suppress_token_mask,
                                async_launch=True,
                            )
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    prefetched_draft = None
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_count = min(block_len, verify_len_cap)
            if profile_cycles or block_len <= 1:
                verify_token_ids = block_token_ids[:verify_token_count]
            elif verify_token_count <= 1:
                verify_token_ids = current_staged_first[:1]
            else:
                verify_token_ids = mx.concatenate(
                    [current_staged_first[:1], drafted[: verify_token_count - 1]],
                    axis=0,
                )
            verify_ids = verify_token_ids[None]
            if use_speculative_linear_cache:
                engine.arm_rollback(target_cache, prefix_len=start)
            verify_start_ns = time.perf_counter_ns()
            use_lm_head_argmax = (
                _lm_head_argmax_enabled() and suppress_token_mask is None
            )
            with _dflash_stream_context():
                verify_output, verify_hidden_states = engine.verify(
                    target_model=target_model,
                    verify_ids=verify_ids,
                    target_cache=target_cache,
                    verify_chunk_tokens=verify_chunk_tokens,
                    capture_layer_ids=capture_layer_ids,
                    return_normalized=use_lm_head_argmax,
                )
                if profile_cycles:
                    _eval_logits_and_captured(verify_output, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns

            acceptance_start_ns = time.perf_counter_ns()
            with _dflash_stream_context():
                if use_lm_head_argmax:
                    posterior = _lm_head_argmax(target_model, verify_output)[0]
                else:
                    posterior = greedy_tokens_with_mask(verify_output[0], suppress_token_mask)
                if not profile_cycles:
                    mx.async_eval(posterior, *verify_hidden_states.values())
                acceptance_len = int(
                    _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
                )
            acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            acceptance_history.append(acceptance_len)
            _record_acceptance_position_stats(
                acceptance_position_attempts,
                acceptance_position_accepts,
                drafted_count=max(0, verify_token_count - 1),
                acceptance_length=acceptance_len,
            )

            hidden_extract_start_ns = time.perf_counter_ns()
            with _dflash_stream_context():
                committed_hidden = extract_context_feature_from_dict(
                    verify_hidden_states,
                    target_layer_id_list,
                )[:, : (1 + acceptance_len), :]
                if target_hidden_is_projected:
                    committed_hidden = _project_target_feature_for_draft(
                        draft_model,
                        committed_hidden,
                    )
                if profile_cycles:
                    mx.eval(committed_hidden, posterior)
                else:
                    mx.async_eval(committed_hidden)
            hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            generated_token_buffer[generated_token_count : generated_token_count + commit_count] = committed_segment
            generated_token_count += commit_count
            accepted_from_draft += acceptance_len
            staged_first_next = posterior[acceptance_len : acceptance_len + 1]

            if not profile_cycles:
                next_remaining = max_new_tokens - generated_token_count
                next_block_len = max(1, min(effective_block_tokens, next_remaining))
                if next_remaining > 0 and next_block_len > 1:
                    draft_start_ns = time.perf_counter_ns()
                    with _dflash_stream_context():
                        next_drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=staged_first_next,
                            target_hidden=committed_hidden,
                            target_hidden_is_projected=target_hidden_is_projected,
                            block_len=next_block_len,
                            mask_token_tail=mask_token_tail,
                            mask_embedding_tail=mask_embedding_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=True,
                        )
                    launch_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += launch_ns
                    draft_incremental_ns += launch_ns
                    prefetched_draft = {
                        "block_len": next_block_len,
                        "staged_first": staged_first_next,
                        "drafted": next_drafted,
                    }
                else:
                    prefetched_draft = None

            commit_start_ns = time.perf_counter_ns()
            start += commit_count
            target_hidden = committed_hidden
            replay_cycle_ns = engine.rollback(
                target_cache,
                target_len=start,
                acceptance_len=acceptance_len,
                drafted_tokens=block_len - 1,
            )
            replay_ns_total += replay_cycle_ns
            cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            staged_first = staged_first_next

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profiles.append(
                    {
                        "cycle": cycles_completed,
                        "block_len": int(block_len),
                        "commit_count": int(commit_count),
                        "acceptance_len": int(acceptance_len),
                        "draft_us": _ns_to_us(draft_cycle_ns),
                        "verify_us": _ns_to_us(verify_cycle_ns),
                        "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                        "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                        "rollback_us": _ns_to_us(replay_cycle_ns),
                        "other_us": _ns_to_us(other_cycle_ns),
                        "cycle_total_us": _ns_to_us(cycle_total_ns),
                    }
                )
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        generated_token_ids = (
            generated_token_buffer[:generated_token_count].tolist()
            if generated_token_count > 0
            else []
        )
        draft_tokens_attempted = sum(acceptance_position_attempts)
        first_20 = acceptance_history[:20]
        last_20 = acceptance_history[-20:]
        result = {
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "draft_tokens_attempted": int(draft_tokens_attempted),
            "draft_acceptance_ratio": (
                accepted_from_draft / draft_tokens_attempted
                if draft_tokens_attempted
                else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
                "verify_linear_install": verify_linear_install_ns_total / 1_000.0,
            },
            "verify_linear_swapped": int(verify_linear_swapped),
            "speculative_linear_cache": use_speculative_linear_cache,
            "prefill_cache_fastpath": bool(prefill_fastpath),
            "prefill_defer_draft_context": bool(prefill_defer_context),
            "prefill_skip_capture": bool(skip_prefill_capture),
            "prefill_context_tokens": int(retained_context_tokens),
            "prompt_cache_checkpoint_tokens": int(prompt_cache_checkpoint_tokens),
            "prompt_cache_checkpoints": int(checkpoint_count),
            "prompt_cache_checkpoint_us": checkpoint_ns_total / 1_000.0,
            "cache_only_prefill": False,
            "verify_chunk_tokens": int(verify_chunk_tokens) if verify_chunk_tokens else None,
            "verify_len_cap": int(verify_len_cap),
            "quantize_kv_cache": bool(quantize_kv_cache),
            "kv_cache_bits": int(_resolve_kv_cache_bits(kv_cache_bits)),
            "kv_cache_group_size": int(_resolve_kv_cache_group_size(kv_cache_group_size)),
            "prefill_step_size": int(prefill_step_size),
            "tokens_per_cycle": (len(generated_token_ids) / cycles_completed) if cycles_completed > 0 else 0.0,
            "acceptance_history": list(acceptance_history),
            "acceptance_position_attempts": list(acceptance_position_attempts),
            "acceptance_position_accepts": list(acceptance_position_accepts),
            "acceptance_position_rates": _acceptance_position_rates(
                acceptance_position_attempts,
                acceptance_position_accepts,
            ),
            "acceptance_first_20_avg": (sum(first_20) / len(first_20)) if first_20 else 0.0,
            "acceptance_last_20_avg": (sum(last_20) / len(last_20)) if last_20 else 0.0,
            "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        }
        if profile_cycles:
            result["cycle_profile_us"] = cycle_profiles
            result["cycle_profile_totals_us"] = {key: _ns_to_us(value) for key, value in profile_totals_ns.items()}
        return result
    finally:
        _cleanup_generation_caches(target_cache, draft_cache)
        del draft_cache
        del target_cache
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()


def stream_dflash_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    verify_chunk_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    kv_cache_bits: int = _KV_CACHE_BITS_DEFAULT,
    kv_cache_group_size: int = _KV_CACHE_GROUP_SIZE_DEFAULT,
    prefill_step_size: int = 2048,
    prompt_cache: Optional[list[Any]] = None,
    prompt_cache_count: int = 0,
    return_prompt_cache: bool = False,
    prompt_cache_checkpoint_tokens: int = 0,
    prompt_cache_target_only_checkpoints: bool = False,
) -> Iterator[dict[str, Any]]:
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    _require_non_empty_prompt_tokens(prompt_tokens)
    fallback_ar = False
    fallback_reason: Optional[str] = None

    uncached_prompt_len = len(prompt_tokens)
    cached_prompt_len = max(0, int(prompt_cache_count)) if prompt_cache is not None else 0
    prompt_len = cached_prompt_len + uncached_prompt_len
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        if cached_prompt_len > 0:
            raise ValueError("DFlash cached fallback is not supported when DFLASH_MAX_CTX is exceeded")
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        yield from stream_baseline_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            kv_cache_bits=kv_cache_bits,
            kv_cache_group_size=kv_cache_group_size,
            fallback_reason=fallback_reason,
        )
        return
    if cached_prompt_len > 0 and uncached_prompt_len <= 0:
        raise ValueError("DFlash prompt cache reuse requires at least one uncached prompt token")
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    use_speculative_linear_cache = verify_chunk_tokens is None
    engine = detect_engine(target_model)
    draft_backend = make_draft_backend()
    if prompt_cache is not None:
        target_cache, draft_cache = _split_dflash_prompt_cache(target_model, prompt_cache)
    else:
        target_cache = make_target_cache(
            target_model,
            enable_speculative_linear_cache=use_speculative_linear_cache,
            quantize_kv_cache=quantize_kv_cache,
            kv_cache_bits=kv_cache_bits,
            kv_cache_group_size=kv_cache_group_size,
        )
        draft_cache = draft_backend.make_cache(
            draft_model=draft_model,
            sink_size=draft_sink_size,
            window_size=draft_window_size,
        )
    target_layer_id_list = list(draft_model.target_layer_ids)
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}
    profile_cycles = _profile_dflash_cycles_enabled()
    profile_prefill_chunks = _profile_prefill_chunks_enabled()
    prefill_chunk_profiles: list[dict[str, Any]] = []
    exported_prompt_cache = False

    try:
        start_ns = time.perf_counter_ns()
        _yield_pause_ns = 0
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = max(1, int(prefill_step_size))
        cache_only_prefill = max_new_tokens <= 0
        prefill_logits = None
        target_hidden: Optional[mx.array] = None
        target_hidden_chunks: list[mx.array] = []
        target_hidden_is_projected = False
        supports_draft_context_prefill = _supports_draft_context_prefill(draft_cache)
        prefill_defer_context = (
            _prefill_defer_draft_context_enabled()
            and supports_draft_context_prefill
        )
        prefill_fastpath = (
            _prefill_cache_fastpath_enabled()
            and supports_draft_context_prefill
            and not prefill_defer_context
        )
        skip_prefill_capture = prefill_defer_context or (
            prefill_fastpath and _prefill_skip_capture_enabled()
        )
        retained_context_ranges = _draft_context_retain_ranges(
            prompt_len,
            sink_size=draft_sink_size,
            window_size=draft_window_size,
        )
        retained_context_tokens = sum(
            end - start for start, end in retained_context_ranges
        )
        prefill_last_logits_only = _prefill_last_logits_only_enabled()
        prefill_slice_capture = _prefill_slice_capture_enabled()
        retained_context_segments: list[tuple[mx.array, int]] = []
        deferred_context_segments: list[tuple[mx.array, int]] = []
        prompt_cache_checkpoint_tokens = max(0, int(prompt_cache_checkpoint_tokens))
        prompt_cache_target_only_checkpoints = bool(
            prompt_cache_target_only_checkpoints
        )
        checkpoint_prefill_enabled = (
            cache_only_prefill
            and return_prompt_cache
            and prompt_cache_checkpoint_tokens > 0
            and supports_draft_context_prefill
            and (
                prompt_cache_target_only_checkpoints
                or prefill_defer_context
                or prefill_fastpath
            )
            # Checkpointing captures the draft window from the chunk ending at
            # the checkpoint. Keep it exact by only enabling the path when one
            # prefill chunk can contain the whole draft window; this matches the
            # M3 Max tuning we use with 2048-token prefill chunks and a 1024
            # token draft window.
            and (
                prompt_cache_target_only_checkpoints
                or prefill_step_size >= draft_window_size
            )
        )
        next_prompt_cache_checkpoint = 0
        if checkpoint_prefill_enabled:
            next_prompt_cache_checkpoint = (
                (cached_prompt_len // prompt_cache_checkpoint_tokens) + 1
            ) * prompt_cache_checkpoint_tokens
        checkpoint_seed_segments: list[tuple[mx.array, int]] = []
        checkpoint_count = 0
        checkpoint_ns_total = 0
        for chunk_start, chunk_end, chunk_abs_start, chunk_abs_end in _iter_uncached_prefill_chunks(
            cached_prompt_len=cached_prompt_len,
            uncached_prompt_len=uncached_prompt_len,
            prefill_step_size=prefill_step_size,
        ):
            chunk_profile_start_ns = time.perf_counter_ns()
            chunk_memory_before = (
                _mlx_memory_snapshot_gb() if profile_prefill_chunks else {}
            )
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            is_last_chunk = chunk_end >= uncached_prompt_len
            chunk_retained_ranges = _range_overlaps_chunk(
                retained_context_ranges,
                chunk_start=chunk_abs_start,
                chunk_end=chunk_abs_end,
            )
            checkpoint_token_count = 0
            checkpoint_chunk_ranges: list[tuple[int, int]] = []
            if (
                checkpoint_prefill_enabled
                and next_prompt_cache_checkpoint > 0
                and chunk_abs_end >= next_prompt_cache_checkpoint
                and chunk_abs_end < prompt_len
            ):
                checkpoint_token_count = int(chunk_abs_end)
                while next_prompt_cache_checkpoint <= chunk_abs_end:
                    next_prompt_cache_checkpoint += prompt_cache_checkpoint_tokens
                if not prompt_cache_target_only_checkpoints:
                    checkpoint_ranges = _draft_context_retain_ranges(
                        checkpoint_token_count,
                        sink_size=draft_sink_size,
                        window_size=draft_window_size,
                    )
                    checkpoint_chunk_ranges = _range_overlaps_chunk(
                        checkpoint_ranges,
                        chunk_start=chunk_abs_start,
                        chunk_end=chunk_abs_end,
                    )
            chunk_feature_ranges = _merge_context_ranges(
                chunk_retained_ranges + checkpoint_chunk_ranges
            )
            checkpoint_segments: list[tuple[mx.array, int]] = []
            needs_prefill_features = bool(chunk_feature_ranges) or (
                not skip_prefill_capture
                and not (prefill_fastpath or prefill_defer_context)
            )
            slice_candidate_ranges = (
                chunk_feature_ranges
                if (prefill_fastpath or prefill_defer_context)
                else []
            )
            capture_token_slice = (
                _local_capture_slice_for_ranges(
                    slice_candidate_ranges,
                    chunk_start=chunk_abs_start,
                )
                if prefill_slice_capture and slice_candidate_ranges
                else None
            )
            capture_offset = int(capture_token_slice[0]) if capture_token_slice else 0
            if (
                (cache_only_prefill or _prefill_middle_no_logits_enabled())
                and not needs_prefill_features
                and (cache_only_prefill or not is_last_chunk)
            ):
                chunk_path = "target_prefill_without_logits"
                with _dflash_stream_context():
                    h, h_dependencies = target_prefill_without_logits(
                        target_model,
                        input_ids=chunk_ids,
                        cache=target_cache,
                        skip_final_layer_mlp=_prefill_skip_final_mlp_enabled(),
                        skip_final_layer_attention=_prefill_skip_final_attention_enabled(),
                        return_dependencies=True,
                    )
                    if h_dependencies:
                        # Cache-only chunks discard the final hidden state. The
                        # direct K/V dependencies already force all preceding
                        # layer work on this M3 Max-tuned prefill path, so avoid
                        # making the discarded hidden tensor an eval root.
                        mx.eval(*h_dependencies)
                    else:
                        mx.eval(h)
                del h
                if (
                    checkpoint_token_count > 0
                    and checkpoint_prefill_enabled
                    and prompt_cache_target_only_checkpoints
                ):
                    checkpoint_count += 1
                    _pre_yield = time.perf_counter_ns()
                    yield {
                        "event": "prompt_cache_checkpoint",
                        "tokens_processed": checkpoint_token_count,
                        "tokens_total": prompt_len,
                        "prefill_step_size": int(prefill_step_size),
                        "prompt_cache_checkpoint_tokens": int(
                            prompt_cache_checkpoint_tokens
                        ),
                        "prompt_cache_target_only": True,
                        "prompt_cache": _target_only_dflash_prompt_cache(
                            target_cache,
                            draft_cache,
                            context_len=checkpoint_token_count,
                        ),
                    }
                    _yield_pause_ns += time.perf_counter_ns() - _pre_yield
                if profile_prefill_chunks:
                    prefill_chunk_profiles.append(
                        {
                            "chunk_start": int(chunk_abs_start),
                            "chunk_end": int(chunk_abs_end),
                            "tokens": int(chunk_abs_end - chunk_abs_start),
                            "elapsed_us": _ns_to_us(
                                time.perf_counter_ns() - chunk_profile_start_ns
                            ),
                            "path": chunk_path,
                            "needs_features": bool(needs_prefill_features),
                            "is_last_chunk": bool(is_last_chunk),
                            "memory_before": chunk_memory_before,
                            "memory_after": _mlx_memory_snapshot_gb(),
                        }
                    )
                _pre_yield = time.perf_counter_ns()
                yield {
                    "event": "prefill_progress",
                    "tokens_processed": chunk_abs_end,
                    "tokens_total": prompt_len,
                    "prefill_step_size": int(prefill_step_size),
                }
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield
                continue
            chunk_path = "target_forward_with_hidden_states"
            with _dflash_stream_context():
                prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
                    target_model,
                    input_ids=chunk_ids,
                    cache=target_cache,
                    capture_layer_ids=(
                        capture_layer_ids if needs_prefill_features else set()
                    ),
                    skip_logits=cache_only_prefill or not is_last_chunk,
                    force_hidden_state=(
                        not needs_prefill_features
                        and (cache_only_prefill or not is_last_chunk)
                    ),
                    last_logits_only=(
                        not cache_only_prefill
                        and is_last_chunk
                        and prefill_last_logits_only
                    ),
                    capture_token_slice=capture_token_slice,
                    skip_final_layer_mlp=_prefill_skip_final_mlp_enabled(),
                    skip_final_layer_attention=_prefill_skip_final_attention_enabled(),
                )
                if not needs_prefill_features:
                    if prefill_logits is not None:
                        _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
                    else:
                        _eval_hidden_state_container(prefill_hidden_states)
                    del prefill_hidden_states
                else:
                    eval_targets: list[mx.array] = []
                    if prefill_logits is not None:
                        eval_targets.append(prefill_logits)
                    if prefill_fastpath:
                        projected_segments = []
                        for retain_start, retain_end in chunk_retained_ranges:
                            local_start = retain_start - chunk_abs_start
                            local_end = retain_end - chunk_abs_start
                            feat = extract_context_feature_range_from_dict(
                                prefill_hidden_states,
                                target_layer_id_list,
                                start=local_start - capture_offset,
                                end=local_end - capture_offset,
                            )
                            projected = _project_target_feature_for_draft(
                                draft_model,
                                feat,
                            )
                            retained_context_segments.append((projected, retain_start))
                            projected_segments.append(projected)
                            if retain_start < draft_sink_size:
                                checkpoint_seed_segments.append(
                                    (projected, retain_start)
                                )
                        for retain_start, retain_end in checkpoint_chunk_ranges:
                            local_start = retain_start - chunk_abs_start
                            local_end = retain_end - chunk_abs_start
                            feat = extract_context_feature_range_from_dict(
                                prefill_hidden_states,
                                target_layer_id_list,
                                start=local_start - capture_offset,
                                end=local_end - capture_offset,
                            )
                            projected = _project_target_feature_for_draft(
                                draft_model,
                                feat,
                            )
                            checkpoint_segments.append((projected, retain_start))
                            projected_segments.append(projected)
                        if projected_segments:
                            eval_targets.extend(projected_segments)
                    elif prefill_defer_context:
                        deferred_segments = []
                        for retain_start, retain_end in chunk_retained_ranges:
                            local_start = retain_start - chunk_abs_start
                            local_end = retain_end - chunk_abs_start
                            segment = mx.contiguous(
                                extract_context_feature_range_from_dict(
                                    prefill_hidden_states,
                                    target_layer_id_list,
                                    start=local_start - capture_offset,
                                    end=local_end - capture_offset,
                                )
                            )
                            deferred_context_segments.append((segment, retain_start))
                            deferred_segments.append(segment)
                            if retain_start < draft_sink_size:
                                checkpoint_seed_segments.append(
                                    (segment, retain_start)
                                )
                        for retain_start, retain_end in checkpoint_chunk_ranges:
                            local_start = retain_start - chunk_abs_start
                            local_end = retain_end - chunk_abs_start
                            segment = mx.contiguous(
                                extract_context_feature_range_from_dict(
                                    prefill_hidden_states,
                                    target_layer_id_list,
                                    start=local_start - capture_offset,
                                    end=local_end - capture_offset,
                                )
                            )
                            checkpoint_segments.append((segment, retain_start))
                            deferred_segments.append(segment)
                        if deferred_segments:
                            eval_targets.extend(deferred_segments)
                    else:
                        feat = extract_context_feature_from_dict(
                            prefill_hidden_states,
                            target_layer_id_list,
                        )
                        target_hidden_chunks.append(mx.contiguous(feat))
                        eval_targets.append(target_hidden_chunks[-1])
                    if eval_targets:
                        mx.eval(*eval_targets)
                    del prefill_hidden_states
            if checkpoint_token_count > 0 and checkpoint_prefill_enabled:
                if prompt_cache_target_only_checkpoints:
                    prompt_cache_for_checkpoint = _target_only_dflash_prompt_cache(
                        target_cache,
                        draft_cache,
                        context_len=checkpoint_token_count,
                    )
                else:
                    checkpoint_context_segments = list(checkpoint_segments)
                    if _context_cache_offset(draft_cache) <= 0 and not any(
                        int(offset) < draft_sink_size
                        for _, offset in checkpoint_context_segments
                    ):
                        checkpoint_context_segments = (
                            list(checkpoint_seed_segments) + checkpoint_context_segments
                        )
                    checkpoint_context_segments = _context_segments_after_offset(
                        checkpoint_context_segments,
                        _context_cache_offset(draft_cache),
                    )
                    if prefill_fastpath:
                        checkpoint_ns_total += _materialize_projected_draft_context(
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            projected_hidden_segments=checkpoint_context_segments,
                            total_context_len=checkpoint_token_count,
                        )
                    else:
                        checkpoint_ns_total += _materialize_deferred_draft_context(
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            target_hidden_segments=checkpoint_context_segments,
                            total_context_len=checkpoint_token_count,
                        )
                    prompt_cache_for_checkpoint = _combined_dflash_prompt_cache(
                        target_cache,
                        draft_cache,
                    )
                checkpoint_count += 1
                _pre_yield = time.perf_counter_ns()
                yield {
                    "event": "prompt_cache_checkpoint",
                    "tokens_processed": checkpoint_token_count,
                    "tokens_total": prompt_len,
                    "prefill_step_size": int(prefill_step_size),
                    "prompt_cache_checkpoint_tokens": int(
                        prompt_cache_checkpoint_tokens
                    ),
                    "prompt_cache_target_only": bool(
                        prompt_cache_target_only_checkpoints
                    ),
                    "prompt_cache": prompt_cache_for_checkpoint,
                }
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield
            if profile_prefill_chunks:
                prefill_chunk_profiles.append(
                    {
                        "chunk_start": int(chunk_abs_start),
                        "chunk_end": int(chunk_abs_end),
                        "tokens": int(chunk_abs_end - chunk_abs_start),
                        "elapsed_us": _ns_to_us(
                            time.perf_counter_ns() - chunk_profile_start_ns
                        ),
                        "path": chunk_path,
                        "needs_features": bool(needs_prefill_features),
                        "is_last_chunk": bool(is_last_chunk),
                        "memory_before": chunk_memory_before,
                        "memory_after": _mlx_memory_snapshot_gb(),
                    }
                )
            if not needs_prefill_features:
                _pre_yield = time.perf_counter_ns()
                yield {
                    "event": "prefill_progress",
                    "tokens_processed": chunk_abs_end,
                    "tokens_total": prompt_len,
                    "prefill_step_size": int(prefill_step_size),
                }
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield
                continue
            _pre_yield = time.perf_counter_ns()
            yield {
                "event": "prefill_progress",
                "tokens_processed": chunk_abs_end,
                "tokens_total": prompt_len,
                "prefill_step_size": int(prefill_step_size),
            }
            _yield_pause_ns += time.perf_counter_ns() - _pre_yield
        if prefill_fastpath:
            _materialize_projected_draft_context(
                draft_model=draft_model,
                draft_cache=draft_cache,
                projected_hidden_segments=_context_segments_after_offset(
                    retained_context_segments,
                    _context_cache_offset(draft_cache),
                ),
                total_context_len=prompt_len,
            )
            target_hidden = _empty_projected_target_hidden(draft_model)
            target_hidden_is_projected = True
        elif prefill_defer_context:
            target_hidden = _empty_projected_target_hidden(draft_model)
            target_hidden_is_projected = True
        elif target_hidden is None:
            if target_hidden_chunks:
                with _dflash_stream_context():
                    target_hidden = mx.concatenate(target_hidden_chunks, axis=1)
                    mx.eval(target_hidden)
            else:
                target_hidden = mx.zeros(
                    (1, 0, int(draft_model.args.hidden_size)),
                    dtype=_draft_hidden_dtype(draft_model),
                )
        if _clear_cache_after_prefill_enabled() and hasattr(mx, "clear_cache"):
            mx.clear_cache()
        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        _pre_yield = time.perf_counter_ns()
        yield {
            "event": "prefill",
            "prefill_us": prefill_ns / 1_000.0,
            "prompt_token_count": prompt_len,
            "prefill_step_size": int(prefill_step_size),
        }
        _yield_pause_ns += time.perf_counter_ns() - _pre_yield

        draft_block_size = int(draft_model.block_size)
        requested_block_tokens = draft_block_size if block_tokens is None else int(block_tokens)
        effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))

        if cache_only_prefill:
            draft_ns_total = 0
            draft_prefill_ns = 0
            if prefill_defer_context and return_prompt_cache:
                deferred_context_ns = _materialize_deferred_draft_context(
                    draft_model=draft_model,
                    draft_cache=draft_cache,
                    target_hidden_segments=_context_segments_after_offset(
                        deferred_context_segments,
                        _context_cache_offset(draft_cache),
                    ),
                    total_context_len=prompt_len,
                )
                draft_ns_total += deferred_context_ns
                draft_prefill_ns += deferred_context_ns

            export_prompt_cache = bool(return_prompt_cache)
            if export_prompt_cache:
                _finalize_draft_context_cache(
                    draft_model=draft_model,
                    draft_cache=draft_cache,
                    target_hidden=target_hidden,
                    target_hidden_is_projected=target_hidden_is_projected,
                    total_context_len=prompt_len,
                )

            elapsed_us = (time.perf_counter_ns() - start_ns - _yield_pause_ns) / 1_000.0
            acceptance_position_attempts = [0] * max(0, effective_block_tokens - 1)
            acceptance_position_accepts = [0] * max(0, effective_block_tokens - 1)
            summary = {
                "event": "summary",
                "elapsed_us": elapsed_us,
                "prompt_token_count": prompt_len,
                "generated_token_ids": [],
                "generation_tokens": 0,
                "accepted_from_draft": 0,
                "acceptance_ratio": 0.0,
                "draft_tokens_attempted": 0,
                "draft_acceptance_ratio": 0.0,
                "block_tokens": effective_block_tokens,
                "adaptive_current_block_tokens": int(effective_block_tokens),
                "cycles_completed": 0,
                "phase_timings_us": {
                    "prefill": prefill_ns / 1_000.0,
                    "draft": draft_ns_total / 1_000.0,
                    "draft_prefill": draft_prefill_ns / 1_000.0,
                    "draft_incremental": 0.0,
                    "verify": 0.0,
                    "replay": 0.0,
                    "commit": 0.0,
                    "fallback_ar": 0.0,
                },
                "verify_len_cap": int(_resolve_verify_len_cap(target_model, effective_block_tokens)),
                "speculative_linear_cache": bool(use_speculative_linear_cache),
                "prefill_cache_fastpath": bool(prefill_fastpath),
                "prefill_defer_draft_context": bool(prefill_defer_context),
                "prefill_skip_capture": bool(skip_prefill_capture),
                "prefill_context_tokens": int(retained_context_tokens),
                "prompt_cache_checkpoint_tokens": int(prompt_cache_checkpoint_tokens),
                "prompt_cache_checkpoints": int(checkpoint_count),
                "prompt_cache_checkpoint_us": checkpoint_ns_total / 1_000.0,
                "prompt_cache_target_only_checkpoints": bool(
                    prompt_cache_target_only_checkpoints
                ),
                "cache_only_prefill": True,
                "verify_chunk_tokens": int(verify_chunk_tokens) if verify_chunk_tokens else None,
                "quantize_kv_cache": bool(quantize_kv_cache),
                "kv_cache_bits": int(_resolve_kv_cache_bits(kv_cache_bits)),
                "kv_cache_group_size": int(_resolve_kv_cache_group_size(kv_cache_group_size)),
                "prefill_step_size": int(prefill_step_size),
                "tokens_per_cycle": 0.0,
                "dflash_generation_tokens": 0,
                "fallback_ar_generation_tokens": 0,
                "adaptive_fallback_count": 0,
                "adaptive_reprobe_count": 0,
                "adaptive_block_tokens_history": [],
                "acceptance_history": [],
                "acceptance_position_attempts": acceptance_position_attempts,
                "acceptance_position_accepts": acceptance_position_accepts,
                "acceptance_position_rates": _acceptance_position_rates(
                    acceptance_position_attempts,
                    acceptance_position_accepts,
                ),
                "acceptance_first_20_avg": 0.0,
                "acceptance_last_20_avg": 0.0,
                "adaptive_fallback_ar": False,
                "adaptive_fallback_cycle": None,
                "adaptive_fallback_reason": None,
                "adaptive_fallback_recent_tokens_per_cycle": None,
                "adaptive_fallback_probe_cycles": None,
                "adaptive_fallback_window": None,
                "adaptive_fallback_min_tokens_per_cycle": None,
                "adaptive_fallback_cooldown_tokens": None,
                "adaptive_fallback_reprobe_block_tokens": None,
                "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
            }
            if profile_prefill_chunks:
                summary["prefill_chunk_profile_us"] = prefill_chunk_profiles
            if profile_cycles:
                summary["cycle_profile_us"] = []
                summary["cycle_profile_totals_us"] = {
                    "draft": 0.0,
                    "verify": 0.0,
                    "acceptance": 0.0,
                    "hidden_extraction": 0.0,
                    "rollback": 0.0,
                    "other": 0.0,
                    "cycle_total": 0.0,
                }
            if export_prompt_cache:
                summary["prompt_cache"] = _combined_dflash_prompt_cache(
                    target_cache,
                    draft_cache,
                )
                exported_prompt_cache = True
            yield summary
            return

        if prefill_logits is None:
            raise RuntimeError("DFlash prefill did not produce logits for decode")

        verify_install_start_ns = time.perf_counter_ns()
        verify_linear_swapped = ensure_verify_linears_for_decode(target_model)
        verify_install_ns = time.perf_counter_ns() - verify_install_start_ns

        with _dflash_stream_context():
            suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
            staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)

        first_token_yielded = False
        if max_new_tokens > 0:
            first_token_yielded = True
            _pre_yield = time.perf_counter_ns()
            yield {
                "event": "token",
                "token_id": int(staged_first.item()),
                "generated_tokens": 1,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
            }
            _yield_pause_ns += time.perf_counter_ns() - _pre_yield

        block_token_buffer = mx.full(
            (effective_block_tokens,),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        mask_token_tail = mx.full(
            (max(0, effective_block_tokens - 1),),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        with _dflash_stream_context():
            mask_embedding_tail = (
                _target_embed_tokens(target_model)(mask_token_tail[None])
                if int(mask_token_tail.shape[0]) > 0
                else None
            )
            if mask_embedding_tail is not None:
                mx.eval(mask_embedding_tail)
        generated_token_ids: list[int] = []
        dflash_generation_tokens = 0
        fallback_ar_generation_tokens = 0
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)
        current_effective_block_tokens = effective_block_tokens
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        fallback_ar_ns_total = 0
        verify_linear_install_ns_total = verify_install_ns
        seen_draft_cycle = False
        acceptance_history: list[int] = []
        acceptance_position_attempts = [0] * max(0, effective_block_tokens - 1)
        acceptance_position_accepts = [0] * max(0, effective_block_tokens - 1)
        adaptive_fallback_enabled = _adaptive_fallback_enabled()
        adaptive_fallback_probe_cycles = _adaptive_fallback_probe_cycles()
        adaptive_fallback_window = _adaptive_fallback_window()
        adaptive_fallback_min_tpc = _adaptive_fallback_min_tokens_per_cycle()
        adaptive_fallback_cooldown_tokens = _adaptive_fallback_cooldown_tokens()
        adaptive_fallback_reprobe_block_tokens = min(
            effective_block_tokens,
            _adaptive_fallback_reprobe_block_tokens(),
        )
        adaptive_fallback_triggered = False
        adaptive_fallback_cycle: Optional[int] = None
        adaptive_fallback_recent_tpc: Optional[float] = None
        adaptive_fallback_reason: Optional[str] = None
        adaptive_fallback_count = 0
        adaptive_reprobe_count = 0
        adaptive_probe_start = 0
        adaptive_block_tokens_history: list[int] = []
        cycle_profiles: list[dict[str, Any]] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }
        prefetched_draft: Optional[dict[str, Any]] = None
        if prefill_defer_context and (max_new_tokens > 1 or return_prompt_cache):
            deferred_context_ns = _materialize_deferred_draft_context(
                draft_model=draft_model,
                draft_cache=draft_cache,
                target_hidden_segments=deferred_context_segments,
                total_context_len=prompt_len,
            )
            draft_ns_total += deferred_context_ns
            draft_prefill_ns += deferred_context_ns

        while len(generated_token_ids) < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - len(generated_token_ids)
            block_len = max(1, min(current_effective_block_tokens, remaining))
            current_verify_len_cap = _resolve_verify_len_cap(
                target_model,
                current_effective_block_tokens,
            )
            block_token_buffer[:block_len] = int(draft_model.mask_token_id)
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]
            current_staged_first = staged_first
            drafted = None

            if block_len > 1:
                if profile_cycles:
                    draft_start_ns = time.perf_counter_ns()
                    with _dflash_stream_context():
                        drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=current_staged_first,
                            target_hidden=target_hidden,
                            target_hidden_is_projected=target_hidden_is_projected,
                            block_len=block_len,
                            mask_token_tail=mask_token_tail,
                            mask_embedding_tail=mask_embedding_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=False,
                        )
                        block_token_ids[1:block_len] = drafted
                    draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                else:
                    if (
                        prefetched_draft is not None
                        and int(prefetched_draft["block_len"]) == block_len
                    ):
                        drafted = prefetched_draft["drafted"]
                        current_staged_first = prefetched_draft["staged_first"]
                    else:
                        draft_start_ns = time.perf_counter_ns()
                        with _dflash_stream_context():
                            drafted = draft_backend.draft_greedy(
                                target_model=target_model,
                                draft_model=draft_model,
                                draft_cache=draft_cache,
                                staged_first=current_staged_first,
                                target_hidden=target_hidden,
                                target_hidden_is_projected=target_hidden_is_projected,
                                block_len=block_len,
                                mask_token_tail=mask_token_tail,
                                mask_embedding_tail=mask_embedding_tail,
                                suppress_token_mask=suppress_token_mask,
                                async_launch=True,
                            )
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    prefetched_draft = None
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_count = min(block_len, current_verify_len_cap)
            if profile_cycles or block_len <= 1:
                verify_token_ids = block_token_ids[:verify_token_count]
            elif verify_token_count <= 1:
                verify_token_ids = current_staged_first[:1]
            else:
                verify_token_ids = mx.concatenate(
                    [current_staged_first[:1], drafted[: verify_token_count - 1]],
                    axis=0,
                )
            verify_ids = verify_token_ids[None]
            engine.arm_rollback(target_cache, prefix_len=start)
            verify_start_ns = time.perf_counter_ns()
            use_lm_head_argmax = (
                _lm_head_argmax_enabled() and suppress_token_mask is None
            )
            with _dflash_stream_context():
                verify_output, verify_hidden_states = engine.verify(
                    target_model=target_model,
                    verify_ids=verify_ids,
                    target_cache=target_cache,
                    verify_chunk_tokens=verify_chunk_tokens,
                    capture_layer_ids=capture_layer_ids,
                    return_normalized=use_lm_head_argmax,
                )
                if profile_cycles:
                    _eval_logits_and_captured(verify_output, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns

            acceptance_start_ns = time.perf_counter_ns()
            with _dflash_stream_context():
                if use_lm_head_argmax:
                    posterior = _lm_head_argmax(target_model, verify_output)[0]
                else:
                    posterior = greedy_tokens_with_mask(verify_output[0], suppress_token_mask)
                if not profile_cycles:
                    mx.async_eval(posterior, *verify_hidden_states.values())
                acceptance_len = int(
                    _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
                )
            acceptance_history.append(acceptance_len)
            _record_acceptance_position_stats(
                acceptance_position_attempts,
                acceptance_position_accepts,
                drafted_count=max(0, verify_token_count - 1),
                acceptance_length=acceptance_len,
            )
            acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            hidden_extract_start_ns = time.perf_counter_ns()
            with _dflash_stream_context():
                committed_hidden = extract_context_feature_from_dict(
                    verify_hidden_states,
                    target_layer_id_list,
                )[:, : (1 + acceptance_len), :]
                if target_hidden_is_projected:
                    committed_hidden = _project_target_feature_for_draft(
                        draft_model,
                        committed_hidden,
                    )
                if profile_cycles:
                    mx.eval(committed_hidden, posterior)
                else:
                    mx.async_eval(committed_hidden)
            hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            commit_start_ns = time.perf_counter_ns()
            start += commit_count
            target_hidden = committed_hidden
            replay_cycle_ns = engine.rollback(
                target_cache,
                target_len=start,
                acceptance_len=acceptance_len,
                drafted_tokens=block_len - 1,
            )
            replay_ns_total += replay_cycle_ns
            cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            accepted_from_draft += acceptance_len
            staged_first_next = posterior[acceptance_len : acceptance_len + 1]
            should_fallback_now = False
            adaptive_block_tokens_history.append(int(block_len))
            if adaptive_fallback_enabled and current_effective_block_tokens > 1:
                adaptive_probe_history = acceptance_history[adaptive_probe_start:]
                should_fallback_now, adaptive_fallback_recent_tpc = (
                    _should_adaptive_fallback_to_ar(
                        adaptive_probe_history,
                        probe_cycles=adaptive_fallback_probe_cycles,
                        window=adaptive_fallback_window,
                        min_tokens_per_cycle=adaptive_fallback_min_tpc,
                    )
                )
                if should_fallback_now:
                    adaptive_fallback_triggered = True
                    adaptive_fallback_count += 1
                    if adaptive_fallback_cycle is None:
                        adaptive_fallback_cycle = cycles_completed
                    adaptive_fallback_reason = (
                        f"recent_tokens_per_cycle={adaptive_fallback_recent_tpc:.2f} "
                        f"< {adaptive_fallback_min_tpc:.2f}"
                    )
                elif (
                    len(adaptive_probe_history) >= adaptive_fallback_window
                    and current_effective_block_tokens < effective_block_tokens
                    and adaptive_fallback_recent_tpc is not None
                    and adaptive_fallback_recent_tpc >= adaptive_fallback_min_tpc
                ):
                    current_effective_block_tokens = min(
                        effective_block_tokens,
                        max(
                            current_effective_block_tokens + 1,
                            current_effective_block_tokens * 2,
                        ),
                    )
                    adaptive_reprobe_count += 1
                    adaptive_probe_start = len(acceptance_history)
                    prefetched_draft = None
            if not profile_cycles and not should_fallback_now:
                next_remaining = max_new_tokens - len(generated_token_ids) - commit_count
                next_block_len = max(
                    1,
                    min(current_effective_block_tokens, next_remaining),
                )
                if next_remaining > 0 and next_block_len > 1:
                    draft_start_ns = time.perf_counter_ns()
                    with _dflash_stream_context():
                        next_drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=staged_first_next,
                            target_hidden=committed_hidden,
                            target_hidden_is_projected=target_hidden_is_projected,
                            block_len=next_block_len,
                            mask_token_tail=mask_token_tail,
                            mask_embedding_tail=mask_embedding_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=True,
                        )
                    launch_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += launch_ns
                    draft_incremental_ns += launch_ns
                    prefetched_draft = {
                        "block_len": next_block_len,
                        "staged_first": staged_first_next,
                        "drafted": next_drafted,
                    }
                else:
                    prefetched_draft = None
            committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
            for token_id in committed_ids:
                if len(generated_token_ids) >= max_new_tokens:
                    break
                generated_token_ids.append(token_id)
                dflash_generation_tokens += 1
                if first_token_yielded:
                    first_token_yielded = False
                    continue
                _pre_yield = time.perf_counter_ns()
                yield {
                    "event": "token",
                    "token_id": token_id,
                    "generated_tokens": len(generated_token_ids),
                    "acceptance_ratio": (
                        accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
                    ),
                    "cycles_completed": cycles_completed,
                }
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            if should_fallback_now:
                _pre_yield = time.perf_counter_ns()
                yield {
                    "event": "adaptive_fallback",
                    "generated_tokens": len(generated_token_ids),
                    "cycles_completed": cycles_completed,
                    "recent_tokens_per_cycle": adaptive_fallback_recent_tpc,
                    "min_tokens_per_cycle": adaptive_fallback_min_tpc,
                    "cooldown_tokens": adaptive_fallback_cooldown_tokens,
                    "reprobe_block_tokens": adaptive_fallback_reprobe_block_tokens,
                    "reason": adaptive_fallback_reason,
                }
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield

                next_token = int(staged_first_next.item())
                cooldown_hidden_chunks: list[mx.array] = []
                cooldown_generated = 0
                cooldown_stop_hit = False
                cooldown_token_budget = min(
                    adaptive_fallback_cooldown_tokens,
                    max_new_tokens - len(generated_token_ids),
                )
                while (
                    len(generated_token_ids) < max_new_tokens
                    and cooldown_generated < cooldown_token_budget
                ):
                    generated_token_ids.append(next_token)
                    fallback_ar_generation_tokens += 1
                    cooldown_generated += 1
                    _pre_yield = time.perf_counter_ns()
                    yield {
                        "event": "token",
                        "token_id": next_token,
                        "generated_tokens": len(generated_token_ids),
                        "acceptance_ratio": (
                            accepted_from_draft / len(generated_token_ids)
                            if generated_token_ids
                            else 0.0
                        ),
                        "cycles_completed": cycles_completed,
                        "adaptive_fallback_ar": True,
                        "adaptive_fallback_reason": adaptive_fallback_reason,
                    }
                    _yield_pause_ns += time.perf_counter_ns() - _pre_yield
                    if next_token in stop_token_ids:
                        cooldown_stop_hit = True
                        break
                    if len(generated_token_ids) >= max_new_tokens:
                        break
                    fallback_ar_start_ns = time.perf_counter_ns()
                    with _dflash_stream_context():
                        token_array = mx.array([[next_token]], dtype=mx.uint32)
                        ar_output, ar_hidden_states = target_forward_with_hidden_states(
                            target_model,
                            input_ids=token_array,
                            cache=target_cache,
                            capture_layer_ids=capture_layer_ids,
                            return_normalized=use_lm_head_argmax,
                            last_logits_only=True,
                        )
                        if use_lm_head_argmax:
                            ar_posterior = _lm_head_argmax(target_model, ar_output)[0]
                        else:
                            ar_posterior = greedy_tokens_with_mask(
                                ar_output[:, -1, :],
                                suppress_token_mask,
                            )
                        ar_hidden = extract_context_feature_from_dict(
                            ar_hidden_states,
                            target_layer_id_list,
                        )
                        if target_hidden_is_projected:
                            ar_hidden = _project_target_feature_for_draft(
                                draft_model,
                                ar_hidden,
                            )
                        mx.eval(ar_posterior, ar_hidden)
                        cooldown_hidden_chunks.append(mx.contiguous(ar_hidden))
                        next_token = int(ar_posterior.item())
                    start += 1
                    fallback_ar_ns_total += time.perf_counter_ns() - fallback_ar_start_ns
                if cooldown_stop_hit or len(generated_token_ids) >= max_new_tokens:
                    resume_hidden_parts = [target_hidden]
                    resume_hidden_parts.extend(cooldown_hidden_chunks)
                    with _dflash_stream_context():
                        resume_hidden = mx.concatenate(resume_hidden_parts, axis=1)
                        mx.eval(resume_hidden)
                    _finalize_draft_context_cache(
                        draft_model=draft_model,
                        draft_cache=draft_cache,
                        target_hidden=resume_hidden,
                        target_hidden_is_projected=target_hidden_is_projected,
                        total_context_len=start,
                    )
                    target_hidden = _empty_projected_target_hidden(draft_model)
                    target_hidden_is_projected = True
                    break

                resume_hidden_parts = [target_hidden]
                resume_hidden_parts.extend(cooldown_hidden_chunks)
                with _dflash_stream_context():
                    resume_hidden = mx.concatenate(resume_hidden_parts, axis=1)
                    mx.eval(resume_hidden)
                _finalize_draft_context_cache(
                    draft_model=draft_model,
                    draft_cache=draft_cache,
                    target_hidden=resume_hidden,
                    target_hidden_is_projected=target_hidden_is_projected,
                    total_context_len=start,
                )
                target_hidden = _empty_projected_target_hidden(draft_model)
                target_hidden_is_projected = True
                staged_first = mx.array([next_token], dtype=mx.uint32)
                current_effective_block_tokens = adaptive_fallback_reprobe_block_tokens
                adaptive_probe_start = len(acceptance_history)
                adaptive_reprobe_count += 1
                prefetched_draft = None
                continue

            staged_first = staged_first_next

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profiles.append(
                    {
                        "cycle": cycles_completed,
                        "block_len": int(block_len),
                        "commit_count": int(commit_count),
                        "acceptance_len": int(acceptance_len),
                        "draft_us": _ns_to_us(draft_cycle_ns),
                        "verify_us": _ns_to_us(verify_cycle_ns),
                        "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                        "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                        "rollback_us": _ns_to_us(replay_cycle_ns),
                        "other_us": _ns_to_us(other_cycle_ns),
                        "cycle_total_us": _ns_to_us(cycle_total_ns),
                    }
                )
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        export_prompt_cache = bool(return_prompt_cache)
        if export_prompt_cache:
            total_context_len = prompt_len + len(generated_token_ids)
            if start < total_context_len:
                missing_start = max(0, start - prompt_len)
                missing_token_ids = generated_token_ids[missing_start:]
                if missing_token_ids:
                    cache_flush_start_ns = time.perf_counter_ns()
                    with _dflash_stream_context():
                        missing_ids = mx.array([missing_token_ids], dtype=mx.uint32)
                        _, missing_hidden_states = target_forward_with_hidden_states(
                            target_model,
                            input_ids=missing_ids,
                            cache=target_cache,
                            capture_layer_ids=capture_layer_ids,
                            skip_logits=True,
                            skip_final_layer_mlp=_prefill_skip_final_mlp_enabled(),
                            skip_final_layer_attention=_prefill_skip_final_attention_enabled(),
                        )
                        missing_hidden = extract_context_feature_from_dict(
                            missing_hidden_states,
                            target_layer_id_list,
                        )
                        if target_hidden_is_projected:
                            missing_hidden = _project_target_feature_for_draft(
                                draft_model,
                                missing_hidden,
                            )
                        mx.eval(missing_hidden)
                    commit_ns_total += time.perf_counter_ns() - cache_flush_start_ns
                    target_hidden = missing_hidden
                    start = total_context_len
            _finalize_draft_context_cache(
                draft_model=draft_model,
                draft_cache=draft_cache,
                target_hidden=target_hidden,
                target_hidden_is_projected=target_hidden_is_projected,
                total_context_len=prompt_len + len(generated_token_ids),
            )
        draft_tokens_attempted = sum(acceptance_position_attempts)
        first_20 = acceptance_history[:20]
        last_20 = acceptance_history[-20:]
        elapsed_us = (time.perf_counter_ns() - start_ns - _yield_pause_ns) / 1_000.0
        summary = {
            "event": "summary",
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "draft_tokens_attempted": int(draft_tokens_attempted),
            "draft_acceptance_ratio": (
                accepted_from_draft / draft_tokens_attempted
                if draft_tokens_attempted
                else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "adaptive_current_block_tokens": int(current_effective_block_tokens),
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
                "fallback_ar": fallback_ar_ns_total / 1_000.0,
                "verify_linear_install": verify_linear_install_ns_total / 1_000.0,
            },
            "verify_linear_swapped": int(verify_linear_swapped),
            "verify_len_cap": int(verify_len_cap),
            "speculative_linear_cache": bool(use_speculative_linear_cache),
            "prefill_cache_fastpath": bool(prefill_fastpath),
            "prefill_defer_draft_context": bool(prefill_defer_context),
            "prefill_skip_capture": bool(skip_prefill_capture),
            "prefill_context_tokens": int(retained_context_tokens),
            "prompt_cache_checkpoint_tokens": int(prompt_cache_checkpoint_tokens),
            "prompt_cache_checkpoints": int(checkpoint_count),
            "prompt_cache_checkpoint_us": checkpoint_ns_total / 1_000.0,
            "prompt_cache_target_only_checkpoints": bool(
                prompt_cache_target_only_checkpoints
            ),
            "cache_only_prefill": False,
            "verify_chunk_tokens": int(verify_chunk_tokens) if verify_chunk_tokens else None,
            "quantize_kv_cache": bool(quantize_kv_cache),
            "kv_cache_bits": int(_resolve_kv_cache_bits(kv_cache_bits)),
            "kv_cache_group_size": int(_resolve_kv_cache_group_size(kv_cache_group_size)),
            "prefill_step_size": int(prefill_step_size),
            "tokens_per_cycle": (dflash_generation_tokens / cycles_completed) if cycles_completed > 0 else 0.0,
            "dflash_generation_tokens": int(dflash_generation_tokens),
            "fallback_ar_generation_tokens": int(fallback_ar_generation_tokens),
            "adaptive_fallback_count": int(adaptive_fallback_count),
            "adaptive_reprobe_count": int(adaptive_reprobe_count),
            "adaptive_block_tokens_history": list(adaptive_block_tokens_history),
            "acceptance_history": list(acceptance_history),
            "acceptance_position_attempts": list(acceptance_position_attempts),
            "acceptance_position_accepts": list(acceptance_position_accepts),
            "acceptance_position_rates": _acceptance_position_rates(
                acceptance_position_attempts,
                acceptance_position_accepts,
            ),
            "acceptance_first_20_avg": (sum(first_20) / len(first_20)) if first_20 else 0.0,
            "acceptance_last_20_avg": (sum(last_20) / len(last_20)) if last_20 else 0.0,
            "adaptive_fallback_ar": bool(adaptive_fallback_triggered),
            "adaptive_fallback_cycle": adaptive_fallback_cycle,
            "adaptive_fallback_reason": adaptive_fallback_reason,
            "adaptive_fallback_recent_tokens_per_cycle": adaptive_fallback_recent_tpc,
            "adaptive_fallback_probe_cycles": (
                int(adaptive_fallback_probe_cycles) if adaptive_fallback_enabled else None
            ),
            "adaptive_fallback_window": (
                int(adaptive_fallback_window) if adaptive_fallback_enabled else None
            ),
            "adaptive_fallback_min_tokens_per_cycle": (
                float(adaptive_fallback_min_tpc) if adaptive_fallback_enabled else None
            ),
            "adaptive_fallback_cooldown_tokens": (
                int(adaptive_fallback_cooldown_tokens)
                if adaptive_fallback_enabled
                else None
            ),
            "adaptive_fallback_reprobe_block_tokens": (
                int(adaptive_fallback_reprobe_block_tokens)
                if adaptive_fallback_enabled
                else None
            ),
            "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        }
        if profile_prefill_chunks:
            summary["prefill_chunk_profile_us"] = prefill_chunk_profiles
        if profile_cycles:
            summary["cycle_profile_us"] = cycle_profiles
            summary["cycle_profile_totals_us"] = {
                key: _ns_to_us(value) for key, value in profile_totals_ns.items()
            }
        if export_prompt_cache:
            summary["prompt_cache"] = _combined_dflash_prompt_cache(
                target_cache,
                draft_cache,
            )
            exported_prompt_cache = True
        yield summary
    finally:
        if exported_prompt_cache:
            for cache_entry in target_cache:
                if hasattr(cache_entry, "clear_transients"):
                    cache_entry.clear_transients()
        else:
            _cleanup_generation_caches(target_cache, draft_cache)
            del draft_cache
            del target_cache
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
