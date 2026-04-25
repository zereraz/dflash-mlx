# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers <= 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (index * span) / (num_draft_layers - 1)))
        for index in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[mx.array],
    layer_ids: list[int],
) -> mx.array:
    selected = [hidden_states[layer_id + 1] for layer_id in layer_ids]
    return mx.concatenate(selected, axis=-1)


class ContextOnlyDraftKVCache:
    def __init__(self, sink_size: int = 64, window_size: int = 1024):
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        self.keys = None
        self.values = None
        self.offset = 0

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
    ) -> None:
        if context_keys is None or context_values is None or int(num_positions) <= 0:
            return
        if self.keys is None:
            self.keys = context_keys
            self.values = context_values
        else:
            self.keys = mx.concatenate([self.keys, context_keys], axis=2)
            self.values = mx.concatenate([self.values, context_values], axis=2)
        self.offset += int(num_positions)
        self._apply_window()

    def _apply_window(self) -> None:
        if self.keys is None or self.values is None:
            return
        cache_len = int(self.keys.shape[2])
        max_len = self.sink_size + self.window_size
        if cache_len <= max_len:
            return
        sink_k = self.keys[:, :, : self.sink_size, :]
        sink_v = self.values[:, :, : self.sink_size, :]
        window_k = self.keys[:, :, -self.window_size :, :]
        window_v = self.values[:, :, -self.window_size :, :]
        self.keys = mx.concatenate([sink_k, window_k], axis=2)
        self.values = mx.concatenate([sink_v, window_v], axis=2)

    def fetch(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        return self.keys, self.values

    def cache_length(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.shape[2])

    def empty(self) -> bool:
        return self.keys is None or self.values is None

    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        total = 0
        if self.keys is not None:
            total += int(self.keys.nbytes)
        if self.values is not None:
            total += int(self.values.nbytes)
        return total

    def set_context(
        self,
        keys: mx.array,
        values: mx.array,
        *,
        offset: int,
    ) -> None:
        self.keys = keys
        self.values = values
        self.offset = int(offset)


@dataclass
class DFlashDraftModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    num_target_layers: int
    block_size: int
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_scaling: Optional[dict[str, Any]] = None
    layer_types: tuple[str, ...] = ()
    dflash_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DFlashDraftModelArgs":
        data = dict(params)
        data["layer_types"] = tuple(data.get("layer_types") or ())
        data["dflash_config"] = dict(data.get("dflash_config") or {})
        return cls(
            **{key: value for key, value in data.items() if key in cls.__annotations__}
        )


class DFlashAttention(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def context_kv(
        self,
        target_hidden: mx.array,
        *,
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        batch, ctx_len, _ = target_hidden.shape
        context_keys = self.k_proj(target_hidden)
        context_keys = self.k_norm(
            context_keys.reshape(batch, ctx_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        context_values = self.v_proj(target_hidden).reshape(
            batch,
            ctx_len,
            self.n_kv_heads,
            -1,
        ).transpose(0, 2, 1, 3)
        return self.rope(context_keys, offset=int(offset)), context_values

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, block_len, _ = hidden_states.shape
        ctx_len = int(target_hidden.shape[1])

        queries = self.q_proj(hidden_states)
        queries = self.q_norm(queries.reshape(batch, block_len, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )

        noise_keys = self.k_proj(hidden_states)
        noise_keys = self.k_norm(
            noise_keys.reshape(batch, block_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        noise_values = self.v_proj(hidden_states).reshape(
            batch,
            block_len,
            self.n_kv_heads,
            -1,
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            if isinstance(cache, ContextOnlyDraftKVCache):
                cache_offset = int(cache.offset)
                query_offset = cache_offset + ctx_len
                queries = self.rope(queries, offset=query_offset)
                noise_keys = self.rope(noise_keys, offset=query_offset)

                if ctx_len > 0:
                    context_keys, context_values = self.context_kv(
                        target_hidden,
                        offset=cache_offset,
                    )
                    cache.append_context(context_keys, context_values, ctx_len)
                cached_keys, cached_values = cache.fetch()
                if cached_keys is None or cached_values is None:
                    keys = noise_keys
                    values = noise_values
                    output = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=None,
                        scale=self.scale,
                        mask=None,
                    )
                elif hasattr(mx.fast, "dflash_cross_attention"):
                    output = mx.fast.dflash_cross_attention(
                        queries,
                        cached_keys,
                        cached_values,
                        noise_keys,
                        noise_values,
                        scale=self.scale,
                    )
                else:
                    keys = mx.concatenate([cached_keys, noise_keys], axis=-2)
                    values = mx.concatenate([cached_values, noise_values], axis=-2)
                    output = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=None,
                        scale=self.scale,
                        mask=None,
                    )
            else:
                cache_offset = int(getattr(cache, "offset", 0) or 0)
                query_offset = cache_offset + ctx_len
                queries = self.rope(queries, offset=query_offset)
                context_keys, context_values = self.context_kv(
                    target_hidden,
                    offset=cache_offset,
                )
                noise_keys = self.rope(noise_keys, offset=query_offset)

                keys = mx.concatenate([context_keys, noise_keys], axis=-2)
                values = mx.concatenate([context_values, noise_values], axis=-2)
                keys, values = cache.update_and_fetch(keys, values)
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=None,
                )
        else:
            queries = self.rope(queries, offset=ctx_len)
            context_keys, context_values = self.context_kv(target_hidden, offset=0)
            noise_keys = self.rope(noise_keys, offset=ctx_len)
            if hasattr(mx.fast, "dflash_cross_attention"):
                output = mx.fast.dflash_cross_attention(
                    queries,
                    context_keys,
                    context_values,
                    noise_keys,
                    noise_values,
                    scale=self.scale,
                )
            else:
                keys = mx.concatenate([context_keys, noise_keys], axis=-2)
                values = mx.concatenate([context_values, noise_values], axis=-2)
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=self.scale,
                    mask=None,
                )

        output = output.transpose(0, 2, 1, 3).reshape(batch, block_len, -1)
        return self.o_proj(output)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__()
        self.self_attn = DFlashAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_hidden=target_hidden,
            cache=cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__()
        self.args = args
        self.model_type = "dflash_qwen3"
        self.layers = [DFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        target_layer_ids = list((args.dflash_config or {}).get("target_layer_ids") or ())
        self.target_layer_ids = target_layer_ids or build_target_layer_ids(
            args.num_target_layers,
            args.num_hidden_layers,
        )
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(len(self.target_layer_ids) * args.hidden_size, args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.block_size = int(args.block_size)
        self.mask_token_id = int((args.dflash_config or {}).get("mask_token_id", 0) or 0)

    def _project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(target_hidden))

    def prefill_context_cache(
        self,
        *,
        target_hidden_segments: list[tuple[mx.array, int]],
        cache: list[Any],
        total_context_len: int,
    ) -> None:
        if not target_hidden_segments:
            return
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            if not isinstance(layer_cache, ContextOnlyDraftKVCache):
                continue
            key_parts = []
            value_parts = []
            for target_hidden, offset in target_hidden_segments:
                if int(target_hidden.shape[1]) <= 0:
                    continue
                keys, values = layer.self_attn.context_kv(
                    target_hidden,
                    offset=int(offset),
                )
                key_parts.append(keys)
                value_parts.append(values)
            if not key_parts:
                continue
            keys = mx.concatenate(key_parts, axis=2)
            values = mx.concatenate(value_parts, axis=2)
            if layer_cache.empty():
                layer_cache.set_context(keys, values, offset=int(total_context_len))
            else:
                layer_cache.append_context(keys, values, int(keys.shape[2]))
                layer_cache.offset = int(total_context_len)

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[list[Any]] = None,
        target_hidden_is_projected: bool = False,
    ) -> mx.array:
        hidden_states = noise_embedding
        projected_hidden = (
            target_hidden
            if target_hidden_is_projected
            else self._project_target_hidden(target_hidden)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states = layer(
                hidden_states,
                target_hidden=projected_hidden,
                cache=layer_cache,
            )
        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return weights
