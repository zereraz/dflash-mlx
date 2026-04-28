"""Measure synchronous vs async DiskBackedPromptCache insert blocking time."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from dflash_mlx.model import ContextOnlyDraftKVCache
from dflash_mlx.prompt_disk_cache import DiskBackedPromptCache, _cache_arrays
from dflash_mlx.runtime import configure_mlx_memory_limits


class _MemoryStore:
    def __init__(self) -> None:
        self.entries = {}

    def fetch_nearest_cache(self, model_key: Any, tokens: list[int]):
        return None, tokens

    def insert_cache(self, model_key: Any, tokens: list[int], prompt_cache: list[Any], *, cache_type: str = "assistant") -> None:
        self.entries[(model_key, tuple(tokens))] = prompt_cache

    def trim_to(self, **kwargs) -> None:
        del kwargs

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def nbytes(self) -> int:
        return 0


def _make_cache(*, layers: int, heads: int, tokens: int, dim: int) -> list[Any]:
    caches: list[Any] = []
    base_k = mx.ones((1, heads, tokens, dim), dtype=mx.bfloat16)
    base_v = mx.zeros((1, heads, tokens, dim), dtype=mx.bfloat16)
    mx.eval(base_k, base_v)
    for _ in range(layers):
        cache = ContextOnlyDraftKVCache()
        cache.set_context(base_k, base_v, offset=tokens)
        caches.append(cache)
    return caches


def _time_insert(store: DiskBackedPromptCache, cache: list[Any], tokens: list[int]) -> dict[str, float]:
    start = time.perf_counter_ns()
    store.insert_cache(("target", None, "draft"), tokens, cache, cache_type="user")
    insert_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    start_wait = time.perf_counter_ns()
    store.wait_for_pending_writes()
    wait_ms = (time.perf_counter_ns() - start_wait) / 1_000_000.0
    return {"insert_return_ms": insert_ms, "wait_ms": wait_ms, "total_ms": insert_ms + wait_ms}


def _time_fetch(directory: str | Path, tokens: list[int]) -> dict[str, float | int | bool]:
    store = DiskBackedPromptCache(
        _MemoryStore(),
        directory=directory,
        ttl_seconds=60,
        async_writes=False,
    )
    start = time.perf_counter_ns()
    prompt_cache, rest = store.fetch_nearest_cache(("target", None, "draft"), tokens)
    fetch_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    arrays = _cache_arrays(prompt_cache or [])
    start_eval = time.perf_counter_ns()
    if arrays:
        mx.eval(*arrays)
    eval_ms = (time.perf_counter_ns() - start_eval) / 1_000_000.0
    return {
        "hit": prompt_cache is not None,
        "rest_tokens": len(rest),
        "arrays": len(arrays),
        "fetch_return_ms": fetch_ms,
        "eval_ms": eval_ms,
        "total_ms": fetch_ms + eval_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    memory_config = configure_mlx_memory_limits()
    prompt_cache = _make_cache(
        layers=int(args.layers),
        heads=int(args.heads),
        tokens=int(args.tokens),
        dim=int(args.dim),
    )
    token_ids = list(range(int(args.tokens)))
    nbytes = sum(int(cache.nbytes) for cache in prompt_cache)

    with tempfile.TemporaryDirectory(prefix="dflash-sync-cache-") as sync_dir:
        sync_store = DiskBackedPromptCache(
            _MemoryStore(),
            directory=sync_dir,
            ttl_seconds=60,
            async_writes=False,
        )
        sync = _time_insert(sync_store, prompt_cache, token_ids)
        sync_fetch = _time_fetch(sync_dir, token_ids + [int(args.tokens)])

    with tempfile.TemporaryDirectory(prefix="dflash-async-cache-") as async_dir:
        async_store = DiskBackedPromptCache(
            _MemoryStore(),
            directory=async_dir,
            ttl_seconds=60,
            async_writes=True,
        )
        async_result = _time_insert(async_store, prompt_cache, token_ids)
        async_fetch = _time_fetch(async_dir, token_ids + [int(args.tokens)])

    result = {
        "memory_config": memory_config,
        "shape": {
            "layers": int(args.layers),
            "heads": int(args.heads),
            "tokens": int(args.tokens),
            "dim": int(args.dim),
            "nbytes": nbytes,
        },
        "sync": sync,
        "sync_fetch": sync_fetch,
        "async": async_result,
        "async_fetch": async_fetch,
        "request_thread_blocking_reduction_ms": sync["insert_return_ms"] - async_result["insert_return_ms"],
    }
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
