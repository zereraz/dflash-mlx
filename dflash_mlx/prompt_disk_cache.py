from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import time
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


_CACHE_VERSION = 1


def _stable_json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _cache_class_name(cache: Any) -> str:
    cls = type(cache)
    return f"{cls.__module__}:{cls.__qualname__}"


def _load_cache_class(name: str):
    module_name, _, qualname = name.partition(":")
    if not module_name or not qualname:
        raise ValueError(f"invalid cache class metadata: {name!r}")
    module = importlib.import_module(module_name)
    cls = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    return cls


def _cache_nbytes(prompt_cache: list[Any]) -> int:
    return int(sum(int(getattr(cache, "nbytes", 0) or 0) for cache in prompt_cache))


def _cache_arrays(prompt_cache: list[Any]) -> list[mx.array]:
    arrays: list[mx.array] = []
    for cache in prompt_cache:
        for _, value in tree_flatten(cache.state):
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def save_dflash_prompt_cache(
    file_name: str | Path,
    prompt_cache: list[Any],
    metadata: dict[str, str],
) -> None:
    cache_data = [cache.state for cache in prompt_cache]
    cache_info = [cache.meta_state for cache in prompt_cache]
    cache_classes = [_cache_class_name(cache) for cache in prompt_cache]
    arrays = dict(tree_flatten(cache_data))
    cache_metadata = dict(
        tree_flatten(
            [
                cache_info,
                {str(key): str(value) for key, value in metadata.items()},
                cache_classes,
            ]
        )
    )
    mx.save_safetensors(str(file_name), arrays, cache_metadata)


def load_dflash_prompt_cache(
    file_name: str | Path,
) -> tuple[list[Any], dict[str, str]]:
    arrays, cache_metadata = mx.load(str(file_name), return_metadata=True)
    cache_data = tree_unflatten(list(arrays.items()))
    cache_info, metadata, cache_classes = tree_unflatten(list(cache_metadata.items()))
    prompt_cache = [
        _load_cache_class(class_name).from_state(state, meta_state)
        for state, meta_state, class_name in zip(
            cache_data,
            cache_info,
            cache_classes,
            strict=True,
        )
    ]
    return prompt_cache, dict(metadata)


class DiskBackedPromptCache:
    def __init__(
        self,
        memory_cache: Any,
        *,
        directory: str | Path,
        ttl_seconds: float = 7 * 24 * 60 * 60,
        max_bytes: Optional[int] = None,
        async_writes: bool = False,
    ) -> None:
        self.memory_cache = memory_cache
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_bytes = int(max_bytes) if max_bytes and int(max_bytes) > 0 else None
        self.async_writes = bool(async_writes)
        self._disk_lock = RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._write_futures: list[Future] = []
        if self.async_writes:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="dflash-prompt-cache",
            )
        self.prune_stale()

    @property
    def nbytes(self) -> int:
        return int(getattr(self.memory_cache, "nbytes", 0) or 0)

    def __len__(self) -> int:
        return len(self.memory_cache)

    def trim_to(self, **kwargs) -> None:
        return self.memory_cache.trim_to(**kwargs)

    def stats_by_type(self):
        if hasattr(self.memory_cache, "stats_by_type"):
            return self.memory_cache.stats_by_type()
        return {}

    def _model_hash(self, model_key: Any) -> str:
        return _hash_text(_stable_json(model_key))

    def _paths_for(self, *, model_hash: str, tokens: list[int]) -> tuple[Path, Path]:
        digest = _token_hash(tokens)
        stem = f"{model_hash}-{len(tokens)}-{digest[:24]}"
        return self.directory / f"{stem}.safetensors", self.directory / f"{stem}.json"

    def _iter_metadata(self) -> list[tuple[Path, Path, dict[str, Any]]]:
        entries: list[tuple[Path, Path, dict[str, Any]]] = []
        for json_path in self.directory.glob("*.json"):
            cache_path = json_path.with_suffix(".safetensors")
            if not cache_path.exists():
                self._unlink_pair(cache_path, json_path)
                continue
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._unlink_pair(cache_path, json_path)
                continue
            if int(meta.get("version", 0) or 0) != _CACHE_VERSION:
                self._unlink_pair(cache_path, json_path)
                continue
            entries.append((cache_path, json_path, meta))
        return entries

    def _unlink_pair(self, cache_path: Path, json_path: Path) -> None:
        for path in (cache_path, json_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def prune_stale(self) -> None:
        with self._disk_lock:
            now = time.time()
            entries = self._iter_metadata()
            if self.ttl_seconds > 0:
                for cache_path, json_path, meta in entries:
                    accessed_at = float(
                        meta.get("accessed_at", meta.get("created_at", 0)) or 0
                    )
                    if now - accessed_at > self.ttl_seconds:
                        self._unlink_pair(cache_path, json_path)
                entries = self._iter_metadata()

            if self.max_bytes is None:
                return

            total_bytes = sum(
                int(meta.get("nbytes", 0) or 0)
                for _, _, meta in entries
            )
            if total_bytes <= self.max_bytes:
                return

            ordered = sorted(
                entries,
                key=lambda item: float(
                    item[2].get("accessed_at", item[2].get("created_at", 0)) or 0
                ),
            )
            for cache_path, json_path, meta in ordered:
                if total_bytes <= self.max_bytes:
                    break
                total_bytes -= int(meta.get("nbytes", 0) or 0)
                self._unlink_pair(cache_path, json_path)

    def fetch_nearest_cache(
        self,
        model_key: Any,
        tokens: list[int],
    ) -> tuple[Optional[list[Any]], list[int]]:
        cache, rest = self.memory_cache.fetch_nearest_cache(model_key, tokens)
        if cache is not None:
            return cache, rest

        model_hash = self._model_hash(model_key)
        best: tuple[int, Path, Path, dict[str, Any]] | None = None
        with self._disk_lock:
            for cache_path, json_path, meta in self._iter_metadata():
                if meta.get("model_hash") != model_hash:
                    continue
                cached_tokens = [int(token) for token in meta.get("tokens", [])]
                cached_len = len(cached_tokens)
                if cached_len <= 0 or cached_len > len(tokens):
                    continue
                if tokens[:cached_len] != cached_tokens:
                    continue
                if best is None or cached_len > best[0]:
                    best = (cached_len, cache_path, json_path, meta)

        if best is None:
            return None, tokens

        cached_len, cache_path, json_path, meta = best
        with self._disk_lock:
            try:
                prompt_cache, _ = load_dflash_prompt_cache(cache_path)
            except Exception:
                self._unlink_pair(cache_path, json_path)
                return None, tokens
            meta["accessed_at"] = time.time()
            try:
                json_path.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
            except OSError:
                pass
        self.memory_cache.insert_cache(
            model_key,
            list(meta["tokens"]),
            prompt_cache,
            cache_type=str(meta.get("cache_type", "assistant")),
        )
        return copy.deepcopy(prompt_cache), tokens[cached_len:]

    def insert_cache(
        self,
        model_key: Any,
        tokens: list[int],
        prompt_cache: list[Any],
        *,
        cache_type: str = "assistant",
    ) -> None:
        self.memory_cache.insert_cache(
            model_key,
            tokens,
            prompt_cache,
            cache_type=cache_type,
        )
        if self.async_writes and self._executor is not None:
            self._collect_finished_writes()
            # The caller follows mlx-lm's LRUPromptCache contract: live caches
            # that will keep mutating are deep-copied before insertion. Reuse
            # that owned snapshot for disk too, avoiding a second full cache
            # copy on checkpoint-heavy M3 Max chat sessions.
            arrays = _cache_arrays(prompt_cache)
            if arrays:
                mx.eval(*arrays)
            future = self._executor.submit(
                self._safe_insert_disk,
                model_key,
                list(tokens),
                prompt_cache,
                cache_type,
            )
            self._write_futures.append(future)
        else:
            self._safe_insert_disk(model_key, tokens, prompt_cache, cache_type)

    def _collect_finished_writes(self) -> None:
        pending: list[Future] = []
        for future in self._write_futures:
            if not future.done():
                pending.append(future)
                continue
            try:
                future.result()
            except Exception as exc:
                sys.stderr.write(f"[dflash] disk prompt cache save failed: {exc}\n")
                sys.stderr.flush()
        self._write_futures = pending

    def wait_for_pending_writes(self) -> None:
        futures = list(self._write_futures)
        self._write_futures = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                sys.stderr.write(f"[dflash] disk prompt cache save failed: {exc}\n")
                sys.stderr.flush()

    def _safe_insert_disk(
        self,
        model_key: Any,
        tokens: list[int],
        prompt_cache: list[Any],
        cache_type: str,
    ) -> None:
        try:
            stream_context = nullcontext()
            if self.async_writes and mx.metal.is_available():
                # Background Python threads do not inherit MLX's thread-local
                # GPU stream. Saving MLX arrays needs an explicit stream here.
                stream_context = mx.stream(mx.new_stream(mx.gpu))
            with stream_context:
                with self._disk_lock:
                    self._insert_disk(model_key, tokens, prompt_cache, cache_type=cache_type)
        except Exception as exc:
            sys.stderr.write(f"[dflash] disk prompt cache save failed: {exc}\n")
            sys.stderr.flush()

    def _insert_disk(
        self,
        model_key: Any,
        tokens: list[int],
        prompt_cache: list[Any],
        *,
        cache_type: str,
    ) -> None:
        now = time.time()
        model_hash = self._model_hash(model_key)
        cache_path, json_path = self._paths_for(model_hash=model_hash, tokens=tokens)
        nbytes = _cache_nbytes(prompt_cache)
        metadata = {
            "version": str(_CACHE_VERSION),
            "model_hash": model_hash,
            "token_count": str(len(tokens)),
            "token_hash": _token_hash(tokens),
            "cache_type": str(cache_type),
            "created_at": str(now),
            "accessed_at": str(now),
            "nbytes": str(nbytes),
        }
        sidecar = {
            "version": _CACHE_VERSION,
            "model_hash": model_hash,
            "tokens": [int(token) for token in tokens],
            "token_hash": metadata["token_hash"],
            "cache_type": str(cache_type),
            "created_at": now,
            "accessed_at": now,
            "nbytes": nbytes,
        }
        tmp_cache_path = cache_path.with_name(f".{cache_path.stem}.tmp.safetensors")
        tmp_json_path = json_path.with_name(f".{json_path.stem}.tmp.json")
        try:
            save_dflash_prompt_cache(tmp_cache_path, prompt_cache, metadata)
            tmp_json_path.write_text(
                json.dumps(sidecar, separators=(",", ":")),
                encoding="utf-8",
            )
            tmp_cache_path.replace(cache_path)
            tmp_json_path.replace(json_path)
        finally:
            self._unlink_pair(tmp_cache_path, tmp_json_path)
        self.prune_stale()
