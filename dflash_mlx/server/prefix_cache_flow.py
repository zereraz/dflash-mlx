# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.cache.codecs import build_snapshot, target_cache_is_serializable
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.policies import compute_stable_prefix_len, prefix_cache_enabled
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.server.prefix_cache_manager import (
    build_prefix_key,
    chat_template_marker_ids,
    format_stats_line,
    make_prefix_cache,
)


_DFLASH_PREFIX_CACHE_SINGLETON: Optional[DFlashPrefixCache] = None
_DFLASH_PREFIX_CACHE_LOCK = threading.Lock()


def get_dflash_prefix_cache() -> Optional[DFlashPrefixCache]:
    global _DFLASH_PREFIX_CACHE_SINGLETON
    if not prefix_cache_enabled():
        return None
    if _DFLASH_PREFIX_CACHE_SINGLETON is not None:
        return _DFLASH_PREFIX_CACHE_SINGLETON
    with _DFLASH_PREFIX_CACHE_LOCK:
        if _DFLASH_PREFIX_CACHE_SINGLETON is not None:
            return _DFLASH_PREFIX_CACHE_SINGLETON
        _DFLASH_PREFIX_CACHE_SINGLETON = make_prefix_cache()
    return _DFLASH_PREFIX_CACHE_SINGLETON


def log_prefix_cache_stats(label: str = "") -> None:
    cache = _DFLASH_PREFIX_CACHE_SINGLETON
    if cache is None:
        return
    format_stats_line(cache, label)


@dataclass
class PrefixCacheFlow:
    cache: Optional[DFlashPrefixCache]
    key: Optional[DFlashPrefixKey] = None
    stable_prefix_len: Optional[int] = None
    snapshot: Optional[DFlashPrefixSnapshot] = None
    lookup_ms: float = 0.0
    hit_tokens: int = 0
    insert_ms: float = 0.0

    @classmethod
    def for_request(
        cls,
        *,
        model_provider: Any,
        draft_model: Any,
        tokenizer: Any,
        prompt: list[int],
    ) -> "PrefixCacheFlow":
        cache = get_dflash_prefix_cache()
        if cache is None:
            return cls(cache=None)

        key = build_prefix_key(model_provider, draft_model)
        im_start_id, assistant_id = chat_template_marker_ids(tokenizer)
        stable_prefix_len = compute_stable_prefix_len(
            prompt,
            im_start_id=im_start_id,
            assistant_id=assistant_id,
        )
        lookup_tokens = prompt[:stable_prefix_len]
        lookup_t0 = time.perf_counter_ns()
        matched_len, snapshot = cache.lookup(lookup_tokens, key)
        lookup_ms = (time.perf_counter_ns() - lookup_t0) / 1e6
        hit_tokens = int(matched_len)
        if matched_len > 0:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache hit "
                f"{hit_tokens}/{len(prompt)} tokens (stable prefix {stable_prefix_len})\n"
            )
            sys.stderr.flush()
        log_prefix_cache_stats(label="lookup")
        return cls(
            cache=cache,
            key=key,
            stable_prefix_len=stable_prefix_len,
            snapshot=snapshot,
            lookup_ms=lookup_ms,
            hit_tokens=hit_tokens,
        )

    def handle_prefill_snapshot(self, event: dict[str, Any]) -> None:
        self._insert_snapshot(event, kind="prefill", require_logits=True)

    def handle_generation_snapshot(self, event: dict[str, Any]) -> None:
        self._insert_snapshot(event, kind="generation", require_logits=False)

    def _insert_snapshot(
        self,
        event: dict[str, Any],
        *,
        kind: str,
        require_logits: bool,
    ) -> None:
        if self.cache is None or self.key is None:
            return
        try:
            target_cache = event.get("target_cache")
            target_hidden = event.get("target_hidden")
            last_logits = event.get("last_logits")
            token_ids = event.get("token_ids") or []
            if (
                target_cache is None
                or target_hidden is None
                or (require_logits and last_logits is None)
                or not target_cache_is_serializable(target_cache)
            ):
                return
            snap = build_snapshot(
                token_ids=list(token_ids),
                target_cache=target_cache,
                target_hidden=target_hidden,
                last_logits=last_logits,
                key=self.key,
                kind=kind,
            )
            insert_t0 = time.perf_counter_ns()
            self.cache.insert(snap)
            self.insert_ms += (time.perf_counter_ns() - insert_t0) / 1e6
            if kind == "generation":
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"[dflash] end-of-request snapshot saved ({len(token_ids)} tokens)\n"
                )
                sys.stderr.flush()
        except Exception as cache_err:
            if kind == "prefill":
                msg = f"[dflash] prefix cache insert failed: {cache_err}"
            else:
                msg = f"[dflash] end-of-request snapshot failed: {cache_err}"
            sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
            sys.stderr.flush()
