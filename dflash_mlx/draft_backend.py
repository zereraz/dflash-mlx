# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
)


class EagerDraftBackend:
    def make_cache(
        self,
        *,
        draft_model: DFlashDraftModel,
        sink_size: int,
        window_size: int,
    ) -> list[Any]:
        return [
            ContextOnlyDraftKVCache(
                sink_size=sink_size,
                window_size=window_size,
            )
            for _ in range(len(draft_model.layers))
        ]

    def draft_greedy(
        self,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        target_hidden: mx.array,
        target_hidden_is_projected: bool,
        block_len: int,
        mask_token_tail: mx.array,
        mask_embedding_tail: Optional[mx.array],
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
    ) -> mx.array:
        if int(block_len) <= 1:
            raise ValueError("draft_greedy requires block_len > 1")

        from dflash_mlx import runtime as runtime_mod

        block_tail_len = int(block_len) - 1
        first_embedding = runtime_mod._target_embed_tokens(target_model)(
            staged_first[:1][None]
        )
        if mask_embedding_tail is not None and block_tail_len > 0:
            noise_embedding = mx.concatenate(
                [first_embedding, mask_embedding_tail[:, :block_tail_len, :]],
                axis=1,
            )
        else:
            block_token_ids = mx.concatenate(
                [staged_first[:1], mask_token_tail[:block_tail_len]],
                axis=0,
            )
            noise_embedding = runtime_mod._target_embed_tokens(target_model)(
                block_token_ids[None]
            )
        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
            target_hidden_is_projected=target_hidden_is_projected,
        )
        drafted = runtime_mod._lm_head_argmax(
            target_model,
            draft_hidden[:, 1:, :],
            suppress_token_mask,
        ).squeeze(0)
        if async_launch:
            mx.async_eval(drafted)
        else:
            mx.eval(drafted)
        return drafted


def make_draft_backend() -> EagerDraftBackend:
    return EagerDraftBackend()
