# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from dflash_mlx.model import ContextOnlyDraftKVCache
from dflash_mlx.prompt_disk_cache import (
    DiskBackedPromptCache,
    load_dflash_prompt_cache,
    save_dflash_prompt_cache,
)
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
from dflash_mlx.runtime import (
    _combined_dflash_prompt_cache,
    _context_cache_offset,
    _context_segments_after_offset,
    _finalize_draft_context_cache,
    _iter_uncached_prefill_chunks,
    _materialize_projected_draft_context,
    _require_non_empty_prompt_tokens,
    _split_dflash_prompt_cache,
    _target_only_dflash_prompt_cache,
    extract_context_feature_range_from_dict,
    make_target_cache,
    target_prefill_without_logits,
    target_forward_with_hidden_states,
)
from dflash_mlx.serve import (
    _append_dflash_metrics_event,
    _apply_prefill_cache_strategy_args,
    _build_dflash_metrics_record,
    _dflash_server_prompt_cache_enabled,
    _fetch_dflash_prompt_cache,
    _resolve_dflash_prefill_step_size,
    _select_dflash_stable_prompt_prefix,
    _stabilize_dflash_prompt_cache_chat_template_args,
    _state_machine_is_terminal,
    _use_dflash_prompt_cache,
)


class _Target:
    def __init__(self, n_layers: int):
        self.model = SimpleNamespace(layers=[object() for _ in range(n_layers)])


def test_make_target_cache_uses_configured_quantized_kv_bits():
    target = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace()]))

    caches = make_target_cache(
        target,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=True,
        kv_cache_bits=4,
        kv_cache_group_size=128,
    )

    assert len(caches) == 1
    assert isinstance(caches[0], QuantizedKVCache)
    assert caches[0].bits == 4
    assert caches[0].group_size == 128


def test_split_and_combine_dflash_prompt_cache_preserves_layout():
    target_cache = ["target0", "target1", "target2"]
    draft_cache = ["draft0", "draft1"]
    combined = _combined_dflash_prompt_cache(target_cache, draft_cache)

    split_target, split_draft = _split_dflash_prompt_cache(_Target(3), combined)

    assert split_target == target_cache
    assert split_draft == draft_cache


def test_target_only_dflash_prompt_cache_preserves_draft_offsets_without_arrays():
    target_cache = ["target0", "target1"]
    draft_cache = [ContextOnlyDraftKVCache(sink_size=8, window_size=32)]
    draft_cache[0].append_context(
        mx.ones((1, 1, 4, 2)),
        mx.ones((1, 1, 4, 2)),
        4,
    )

    combined = _target_only_dflash_prompt_cache(
        target_cache,
        draft_cache,
        context_len=128,
    )

    assert combined[:2] == target_cache
    empty_draft = combined[2]
    assert isinstance(empty_draft, ContextOnlyDraftKVCache)
    assert empty_draft.offset == 128
    assert empty_draft.sink_size == 8
    assert empty_draft.window_size == 32
    assert empty_draft.empty()


def test_split_dflash_prompt_cache_rejects_missing_target_layers():
    with pytest.raises(ValueError, match="expected at least 3"):
        _split_dflash_prompt_cache(_Target(3), ["target0", "target1"])


def test_context_only_draft_cache_reports_memory_and_non_trimmable_state():
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=4)
    assert cache.empty()
    assert cache.nbytes == 0
    assert not cache.is_trimmable()

    keys = mx.zeros((1, 2, 3, 4), dtype=mx.float16)
    values = mx.zeros((1, 2, 3, 4), dtype=mx.float16)
    cache.set_context(keys, values, offset=3)

    assert not cache.empty()
    assert cache.cache_length() == 3
    assert cache.nbytes == keys.nbytes + values.nbytes
    assert cache.offset == 3


def test_context_cache_offset_uses_minimum_layer_offset():
    caches = [
        ContextOnlyDraftKVCache(),
        ContextOnlyDraftKVCache(),
        object(),
    ]
    caches[0].offset = 11
    caches[1].offset = 9

    assert _context_cache_offset(caches) == 9


def test_dflash_prompt_cache_serializer_round_trips_context_cache(tmp_path):
    keys = mx.arange(24, dtype=mx.float16).reshape(1, 2, 3, 4)
    values = mx.arange(100, 124, dtype=mx.float16).reshape(1, 2, 3, 4)
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=8)
    cache.set_context(keys, values, offset=11)

    path = tmp_path / "cache.safetensors"
    save_dflash_prompt_cache(path, [cache], {"kind": "unit"})
    loaded, metadata = load_dflash_prompt_cache(path)

    assert metadata["kind"] == "unit"
    assert len(loaded) == 1
    loaded_cache = loaded[0]
    assert isinstance(loaded_cache, ContextOnlyDraftKVCache)
    assert loaded_cache.sink_size == 2
    assert loaded_cache.window_size == 8
    assert loaded_cache.offset == 11
    assert mx.array_equal(loaded_cache.keys, keys).item()
    assert mx.array_equal(loaded_cache.values, values).item()


def test_dflash_prompt_cache_serializer_round_trips_combined_cache(tmp_path):
    kv = KVCache()
    kv_keys = mx.ones((1, 1, 2, 2), dtype=mx.float16)
    kv_values = mx.zeros((1, 1, 2, 2), dtype=mx.float16)
    kv.update_and_fetch(kv_keys, kv_values)

    recurrent = RecurrentRollbackCache(size=2, conv_kernel_size=5)
    recurrent[0] = mx.ones((1, 4, 3), dtype=mx.float16)
    recurrent[1] = mx.zeros((1, 3), dtype=mx.float32)

    path = tmp_path / "combined.safetensors"
    save_dflash_prompt_cache(path, [kv, recurrent], {})
    loaded, _ = load_dflash_prompt_cache(path)

    assert isinstance(loaded[0], KVCache)
    assert loaded[0].offset == 2
    assert mx.array_equal(loaded[0].keys, kv_keys).item()
    assert isinstance(loaded[1], RecurrentRollbackCache)
    assert loaded[1].conv_kernel_size == 5
    assert mx.array_equal(loaded[1][0], recurrent[0]).item()
    assert mx.array_equal(loaded[1][1], recurrent[1]).item()


def test_iter_uncached_prefill_chunks_tracks_absolute_prompt_positions():
    chunks = list(
        _iter_uncached_prefill_chunks(
            cached_prompt_len=3,
            uncached_prompt_len=5,
            prefill_step_size=2,
        )
    )

    assert chunks == [
        (0, 2, 3, 5),
        (2, 4, 5, 7),
        (4, 5, 7, 8),
    ]


def test_require_non_empty_prompt_tokens_rejects_empty_prompts():
    with pytest.raises(ValueError, match="at least one prompt token"):
        _require_non_empty_prompt_tokens([])

    _require_non_empty_prompt_tokens([1])


def test_extract_context_feature_range_slices_before_concatenating_layers():
    captured = {
        2: mx.arange(12, dtype=mx.float32).reshape(1, 4, 3),
        4: mx.arange(100, 112, dtype=mx.float32).reshape(1, 4, 3),
    }

    feature = extract_context_feature_range_from_dict(
        captured,
        [1, 3],
        start=1,
        end=3,
    )

    expected = mx.concatenate(
        [captured[2][:, 1:3, :], captured[4][:, 1:3, :]],
        axis=-1,
    )
    assert feature.shape == (1, 2, 6)
    assert mx.array_equal(feature, expected).item()


class _RecordingDraft:
    def __init__(self):
        self.calls = []

    def prefill_context_cache(self, *, target_hidden_segments, cache, total_context_len):
        self.calls.append(
            {
                "segments": target_hidden_segments,
                "cache": cache,
                "total_context_len": total_context_len,
            }
        )


def test_context_segments_after_offset_trims_already_cached_prefix():
    segment = mx.arange(10, dtype=mx.float32).reshape(1, 5, 2)

    trimmed = _context_segments_after_offset(
        [
            (mx.ones((1, 2, 2), dtype=mx.float32), 0),
            (segment, 2),
        ],
        current_offset=4,
    )

    assert len(trimmed) == 1
    trimmed_segment, offset = trimmed[0]
    assert offset == 4
    assert trimmed_segment.shape == (1, 3, 2)
    assert mx.array_equal(trimmed_segment, segment[:, 2:, :]).item()


def test_materialize_projected_draft_context_filters_empty_segments():
    draft = _RecordingDraft()
    non_empty = mx.arange(4, dtype=mx.float32).reshape(1, 2, 2)

    _materialize_projected_draft_context(
        draft_model=draft,
        draft_cache=[],
        projected_hidden_segments=[
            (mx.zeros((1, 0, 2), dtype=mx.float32), 0),
            (non_empty, 4),
        ],
        total_context_len=6,
    )

    assert len(draft.calls) == 1
    call = draft.calls[0]
    assert call["total_context_len"] == 6
    assert call["cache"] == []
    assert len(call["segments"]) == 1
    segment, offset = call["segments"][0]
    assert offset == 4
    assert mx.array_equal(segment, non_empty).item()


def test_finalize_draft_context_cache_appends_only_missing_tail():
    draft = _RecordingDraft()
    cache = ContextOnlyDraftKVCache()
    cache.offset = 5
    target_hidden = mx.arange(6, dtype=mx.float32).reshape(1, 3, 2)

    _finalize_draft_context_cache(
        draft_model=draft,
        draft_cache=[cache],
        target_hidden=target_hidden,
        target_hidden_is_projected=True,
        total_context_len=8,
    )

    assert len(draft.calls) == 1
    call = draft.calls[0]
    assert call["total_context_len"] == 8
    segment, offset = call["segments"][0]
    assert offset == 5
    assert segment.shape == (1, 3, 2)
    assert mx.array_equal(segment, target_hidden).item()


def test_finalize_draft_context_cache_skips_when_cache_is_current():
    draft = _RecordingDraft()
    cache = ContextOnlyDraftKVCache()
    cache.offset = 8

    _finalize_draft_context_cache(
        draft_model=draft,
        draft_cache=[cache],
        target_hidden=mx.zeros((1, 2, 2), dtype=mx.float32),
        target_hidden_is_projected=True,
        total_context_len=8,
    )

    assert draft.calls == []


class _FakePromptCacheStore:
    def __init__(self, cache, rest):
        self.cache = cache
        self.rest = rest
        self.requests = []

    def fetch_nearest_cache(self, model_key, prompt):
        self.requests.append((model_key, prompt))
        return self.cache, self.rest


def test_fetch_dflash_prompt_cache_uses_uncached_suffix():
    cache = [object()]
    store = _FakePromptCacheStore(cache, [4, 5])

    prompt_cache, prompt_rest, prompt_cache_count = _fetch_dflash_prompt_cache(
        store,
        ("model", None, "draft"),
        [1, 2, 3, 4, 5],
    )

    assert prompt_cache is cache
    assert prompt_rest == [4, 5]
    assert prompt_cache_count == 3
    assert store.requests == [(("model", None, "draft"), [1, 2, 3, 4, 5])]


def test_fetch_dflash_prompt_cache_disables_exact_hit_without_suffix():
    cache = [object()]
    store = _FakePromptCacheStore(cache, [])
    prompt = [1, 2, 3]

    prompt_cache, prompt_rest, prompt_cache_count = _fetch_dflash_prompt_cache(
        store,
        ("model", None, "draft"),
        prompt,
    )

    assert prompt_cache is None
    assert prompt_rest is prompt
    assert prompt_cache_count == 0


def test_disk_backed_prompt_cache_reuses_longest_saved_prefix(tmp_path):
    model_key = ("target", None, "draft")
    keys = mx.ones((1, 1, 2, 2), dtype=mx.float16)
    values = mx.zeros((1, 1, 2, 2), dtype=mx.float16)
    cache = ContextOnlyDraftKVCache()
    cache.set_context(keys, values, offset=3)

    writer_memory = _RecordingPromptCacheStore()
    writer = DiskBackedPromptCache(
        writer_memory,
        directory=tmp_path,
        ttl_seconds=60,
    )
    writer.insert_cache(model_key, [1, 2, 3], [cache], cache_type="user")

    reader_memory = _RecordingPromptCacheStore()
    reader = DiskBackedPromptCache(
        reader_memory,
        directory=tmp_path,
        ttl_seconds=60,
    )
    prompt_cache, rest = reader.fetch_nearest_cache(model_key, [1, 2, 3, 4, 5])

    assert rest == [4, 5]
    assert prompt_cache is not None
    assert isinstance(prompt_cache[0], ContextOnlyDraftKVCache)
    assert prompt_cache[0] is not cache
    assert mx.array_equal(prompt_cache[0].keys, keys).item()
    assert reader_memory.inserts[-1][0] == model_key
    assert reader_memory.inserts[-1][1] == [1, 2, 3]
    assert reader_memory.inserts[-1][3] == "user"


def test_disk_backed_prompt_cache_async_write_round_trips(tmp_path):
    model_key = ("target", None, "draft")
    keys = mx.ones((1, 1, 2, 2), dtype=mx.float16)
    values = mx.zeros((1, 1, 2, 2), dtype=mx.float16)
    cache = ContextOnlyDraftKVCache()
    cache.set_context(keys, values, offset=3)

    writer = DiskBackedPromptCache(
        _RecordingPromptCacheStore(),
        directory=tmp_path,
        ttl_seconds=60,
        async_writes=True,
    )
    writer.insert_cache(model_key, [1, 2, 3], [cache], cache_type="user")
    writer.wait_for_pending_writes()

    reader = DiskBackedPromptCache(
        _RecordingPromptCacheStore(),
        directory=tmp_path,
        ttl_seconds=60,
    )
    prompt_cache, rest = reader.fetch_nearest_cache(model_key, [1, 2, 3, 4])

    assert rest == [4]
    assert prompt_cache is not None
    assert mx.array_equal(prompt_cache[0].keys, keys).item()


def test_disk_backed_prompt_cache_deletes_stale_entries(tmp_path):
    model_key = ("target", None, "draft")
    cache = ContextOnlyDraftKVCache()
    cache.set_context(
        mx.ones((1, 1, 1, 1), dtype=mx.float16),
        mx.ones((1, 1, 1, 1), dtype=mx.float16),
        offset=1,
    )
    store = DiskBackedPromptCache(
        _RecordingPromptCacheStore(),
        directory=tmp_path,
        ttl_seconds=60,
    )
    store.insert_cache(model_key, [1], [cache], cache_type="assistant")

    sidecar = next(tmp_path.glob("*.json"))
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["created_at"] = 0
    metadata["accessed_at"] = 0
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    DiskBackedPromptCache(
        _RecordingPromptCacheStore(),
        directory=tmp_path,
        ttl_seconds=1,
    )

    assert list(tmp_path.glob("*")) == []


def test_select_dflash_stable_prompt_prefix_excludes_assistant_tail():
    stable, tail = _select_dflash_stable_prompt_prefix(
        [1, 2, 3, 4, 5, 6],
        [[1, 2, 3, 4], [5, 6]],
        ["user", "assistant"],
    )

    assert stable == [1, 2, 3, 4]
    assert tail == [5, 6]


def test_select_dflash_stable_prompt_prefix_keeps_uncached_token_fallback():
    stable, tail = _select_dflash_stable_prompt_prefix(
        [1, 2, 3],
        [[1, 2, 3]],
        ["user"],
    )

    assert stable == [1, 2]
    assert tail == [3]


class _RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _RecordingPromptCacheStore:
    def __init__(self):
        self.entries = {}
        self.inserts = []

    def fetch_nearest_cache(self, model_key, prompt):
        best_key = None
        for key_model, key_tokens in self.entries:
            if key_model == model_key and prompt[: len(key_tokens)] == list(key_tokens):
                if best_key is None or len(key_tokens) > len(best_key[1]):
                    best_key = (key_model, key_tokens)
        if best_key is None:
            return None, prompt
        cache = list(self.entries[best_key])
        return cache, prompt[len(best_key[1]) :]

    def insert_cache(self, model_key, tokens, prompt_cache, *, cache_type="assistant"):
        key = (model_key, tuple(tokens))
        self.entries[key] = list(prompt_cache)
        self.inserts.append((model_key, list(tokens), list(prompt_cache), cache_type))


class _FakeDetokenizer:
    last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = str(token)

    def finalize(self):
        pass


def test_serve_single_builds_stable_prompt_cache_before_tail(monkeypatch):
    from dflash_mlx import serve as serve_mod

    calls = []

    def fake_stream_dflash_generate(**kwargs):
        calls.append(kwargs)
        if kwargs["max_new_tokens"] == 0:
            yield {
                "event": "prompt_cache_checkpoint",
                "tokens_processed": 2,
                "prompt_cache": [["checkpoint-cache"]],
            }
            yield {
                "event": "summary",
                "prompt_cache": ["stable-cache"],
                "generated_token_ids": [],
                "generation_tokens": 0,
                "elapsed_us": 1.0,
                "phase_timings_us": {"prefill": 1.0},
            }
            return
        yield {
            "event": "prompt_cache_checkpoint",
            "tokens_processed": 5,
            "prompt_cache": ["main-checkpoint-cache"],
        }
        yield {
            "event": "summary",
            "prompt_cache": ["mutated-main-cache"],
            "generated_token_ids": [9],
            "generation_tokens": 1,
            "elapsed_us": 2.0,
            "phase_timings_us": {"prefill": 1.0},
            "acceptance_ratio": 0.0,
        }

    monkeypatch.setattr(serve_mod, "stream_dflash_generate", fake_stream_dflash_generate)
    monkeypatch.setattr(serve_mod, "get_stop_token_ids", lambda tokenizer: [])
    monkeypatch.setattr(
        serve_mod.DFlashResponseGenerator,
        "_build_generation_context",
        staticmethod(
            lambda tokenizer, prompt, stop_words=None, sequences=None: SimpleNamespace(
                prompt_cache_count=0,
                _should_stop=False,
            )
        ),
    )
    monkeypatch.setattr(
        serve_mod.DFlashResponseGenerator,
        "_make_state_machine",
        lambda self, model_key, tokenizer, stop_words, initial_state="normal": (
            SimpleNamespace(make_state=lambda: ("normal", object(), {"normal": object()})),
            {},
        ),
    )

    response_generator = serve_mod.DFlashResponseGenerator.__new__(
        serve_mod.DFlashResponseGenerator
    )
    response_generator.model_provider = SimpleNamespace(
        model=object(),
        tokenizer=SimpleNamespace(detokenizer=_FakeDetokenizer(), eos_token_ids=[]),
        draft_model=object(),
        model_key=("target", None, "draft"),
        cli_args=SimpleNamespace(
            dflash_prompt_cache=True,
            quantize_kv_cache=False,
            prefill_step_size=2048,
            block_tokens=None,
            dflash_prompt_cache_checkpoint_tokens=2,
        ),
    )
    response_generator.prompt_cache = _RecordingPromptCacheStore()
    response_generator._tokenize = lambda tokenizer, request, args: (
        [1, 2, 3, 4, 5, 6],
        [[1, 2, 3, 4], [5, 6]],
        ["user", "assistant"],
        "normal",
    )

    queue = _RecordingQueue()
    args = SimpleNamespace(max_tokens=300, seed=None, stop_words=[])

    response_generator._serve_single((queue, object(), args))

    assert not isinstance(queue.items[-1], Exception), queue.items
    assert len(calls) == 2
    assert calls[0]["max_new_tokens"] == 0
    assert calls[0]["prompt_tokens_override"] == [1, 2, 3, 4]
    assert calls[0]["return_prompt_cache"]
    assert calls[0]["prompt_cache_checkpoint_tokens"] == 2
    assert calls[1]["prompt_tokens_override"] == [5, 6]
    assert calls[1]["prompt_cache"] == ["stable-cache"]
    assert calls[1]["prompt_cache_count"] == 4
    assert calls[1]["return_prompt_cache"]
    assert response_generator.prompt_cache.inserts == [
        (("target", None, "draft"), [1, 2], [["checkpoint-cache"]], "user"),
        (("target", None, "draft"), [1, 2, 3, 4], ["stable-cache"], "user"),
        (
            ("target", None, "draft"),
            [1, 2, 3, 4, 5],
            ["main-checkpoint-cache"],
            "user",
        ),
        (
            ("target", None, "draft"),
            [1, 2, 3, 4, 5, 6, 9],
            ["mutated-main-cache"],
            "user",
        ),
    ]
    assert queue.items[-1] is None


def test_dflash_server_prompt_cache_is_opt_in(monkeypatch):
    monkeypatch.delenv("DFLASH_SERVER_PROMPT_CACHE", raising=False)
    assert not _dflash_server_prompt_cache_enabled()

    monkeypatch.setenv("DFLASH_SERVER_PROMPT_CACHE", "1")
    assert _dflash_server_prompt_cache_enabled()

    monkeypatch.setenv("DFLASH_SERVER_PROMPT_CACHE", "false")
    assert not _dflash_server_prompt_cache_enabled()


def test_dflash_prompt_cache_can_be_enabled_by_cli_flag(monkeypatch):
    monkeypatch.delenv("DFLASH_SERVER_PROMPT_CACHE", raising=False)
    assert _use_dflash_prompt_cache(SimpleNamespace(dflash_prompt_cache=True))
    assert not _use_dflash_prompt_cache(SimpleNamespace(dflash_prompt_cache=False))


def test_auto_prefill_step_size_keeps_measured_default_for_long_context():
    assert _resolve_dflash_prefill_step_size(
        SimpleNamespace(prefill_step_size=0),
        10_000,
    ) == 2048
    assert _resolve_dflash_prefill_step_size(
        SimpleNamespace(prefill_step_size=0),
        1_000,
    ) == 2048
    assert _resolve_dflash_prefill_step_size(
        SimpleNamespace(prefill_step_size=2048),
        10_000,
    ) == 2048


def test_dflash_prompt_cache_stabilizes_qwen_thinking_template(monkeypatch):
    monkeypatch.delenv("DFLASH_SERVER_PROMPT_CACHE", raising=False)
    args = SimpleNamespace(
        dflash_prompt_cache=True,
        chat_template_args={"enable_thinking": False},
    )

    _stabilize_dflash_prompt_cache_chat_template_args(args)

    assert args.chat_template_args == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }


def test_dflash_prompt_cache_respects_explicit_preserve_thinking(monkeypatch):
    monkeypatch.delenv("DFLASH_SERVER_PROMPT_CACHE", raising=False)
    args = SimpleNamespace(
        dflash_prompt_cache=True,
        chat_template_args={
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    )

    _stabilize_dflash_prompt_cache_chat_template_args(args)

    assert args.chat_template_args == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }


def test_prefill_cache_fastpath_cli_sets_opposite_strategy_env(monkeypatch):
    monkeypatch.delenv("DFLASH_PREFILL_CACHE_FASTPATH", raising=False)
    monkeypatch.delenv("DFLASH_PREFILL_DEFER_DRAFT_CONTEXT", raising=False)

    _apply_prefill_cache_strategy_args(SimpleNamespace(prefill_cache_fastpath=True))

    assert os.environ["DFLASH_PREFILL_CACHE_FASTPATH"] == "1"
    assert os.environ["DFLASH_PREFILL_DEFER_DRAFT_CONTEXT"] == "0"

    _apply_prefill_cache_strategy_args(SimpleNamespace(prefill_cache_fastpath=False))

    assert os.environ["DFLASH_PREFILL_CACHE_FASTPATH"] == "0"
    assert os.environ["DFLASH_PREFILL_DEFER_DRAFT_CONTEXT"] == "1"


def test_prefill_cache_strategy_cli_default_preserves_env(monkeypatch):
    monkeypatch.setenv("DFLASH_PREFILL_CACHE_FASTPATH", "custom")
    monkeypatch.setenv("DFLASH_PREFILL_DEFER_DRAFT_CONTEXT", "custom")

    _apply_prefill_cache_strategy_args(SimpleNamespace(prefill_cache_fastpath=None))

    assert os.environ["DFLASH_PREFILL_CACHE_FASTPATH"] == "custom"
    assert os.environ["DFLASH_PREFILL_DEFER_DRAFT_CONTEXT"] == "custom"


def test_state_machine_terminal_detection_handles_stopped_state():
    assert _state_machine_is_terminal((None, None, {"normal": object()}))
    assert not _state_machine_is_terminal(("normal", object(), {"normal": object()}))
    assert not _state_machine_is_terminal(None)


def test_build_dflash_metrics_record_adjusts_stable_cache_time():
    record = _build_dflash_metrics_record(
        request_id="req",
        summary_event={
            "elapsed_us": 10_000.0,
            "prompt_token_count": 10,
            "generation_tokens": 4,
            "accepted_from_draft": 3,
            "acceptance_ratio": 0.75,
            "draft_tokens_attempted": 8,
            "draft_acceptance_ratio": 0.375,
            "cycles_completed": 2,
            "tokens_per_cycle": 2.0,
            "phase_timings_us": {"prefill": 4_000.0, "draft": 1_000.0},
            "acceptance_position_rates": [1.0, 0.5],
            "adaptive_fallback_ar": True,
            "adaptive_fallback_cycle": 2,
            "adaptive_fallback_recent_tokens_per_cycle": 2.5,
            "adaptive_fallback_count": 1,
            "adaptive_reprobe_count": 1,
            "adaptive_current_block_tokens": 4,
            "adaptive_fallback_cooldown_tokens": 64,
            "adaptive_fallback_reprobe_block_tokens": 4,
            "dflash_generation_tokens": 3,
            "fallback_ar_generation_tokens": 1,
        },
        prompt_len=10,
        finish_reason="stop",
        prompt_cache_count=6,
        stable_cache_build_us=2_000.0,
        using_stable_prompt_cache=True,
        timestamp_s=1.0,
    )

    assert record["event"] == "summary"
    assert record["request_id"] == "req"
    assert record["prefill_ms"] == 6.0
    assert record["elapsed_ms"] == 12.0
    assert record["decode_ms"] == 6.0
    assert record["decode_tps"] == pytest.approx(4 / 0.006)
    assert record["cached_prompt_tokens"] == 6
    assert record["uncached_prompt_tokens"] == 4
    assert record["draft_acceptance_ratio"] == 0.375
    assert record["acceptance_position_rates"] == [1.0, 0.5]
    assert record["phase_timings_ms"]["prefill"] == 6.0
    assert record["phase_timings_ms"]["stable_cache_build"] == 2.0
    assert record["adaptive_fallback_ar"]
    assert record["adaptive_fallback_cycle"] == 2
    assert record["adaptive_fallback_recent_tokens_per_cycle"] == 2.5
    assert record["adaptive_fallback_count"] == 1
    assert record["adaptive_reprobe_count"] == 1
    assert record["adaptive_current_block_tokens"] == 4
    assert record["adaptive_fallback_cooldown_tokens"] == 64
    assert record["adaptive_fallback_reprobe_block_tokens"] == 4
    assert record["dflash_generation_tokens"] == 3
    assert record["fallback_ar_generation_tokens"] == 1


def test_append_dflash_metrics_event_writes_jsonl(tmp_path):
    metrics_path = tmp_path / "metrics" / "session.jsonl"
    args = SimpleNamespace(dflash_metrics_log=str(metrics_path))

    _append_dflash_metrics_event(
        args,
        {
            "event": "request_start",
            "request_id": "req",
            "prompt_tokens": 12,
        },
    )

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert rows == [
        {
            "event": "request_start",
            "prompt_tokens": 12,
            "request_id": "req",
            "schema": "dflash_session_metrics_v1",
            "timestamp": rows[0]["timestamp"],
            "timestamp_s": rows[0]["timestamp_s"],
        }
    ]


class _FakeEmbed:
    def __call__(self, input_ids):
        return mx.broadcast_to(
            input_ids[..., None].astype(mx.float32),
            (*input_ids.shape, 3),
        )


class _FakeLayer:
    def __call__(self, hidden_states, mask=None, cache=None):
        del mask, cache
        return hidden_states + 1


class _FakeNorm:
    def __call__(self, hidden_states):
        return hidden_states


class _FakeHead:
    def __init__(self):
        self.input_shapes = []

    def __call__(self, hidden_states):
        self.input_shapes.append(tuple(hidden_states.shape))
        return mx.zeros((*hidden_states.shape[:2], 7), dtype=mx.float32)


class _FakeTarget:
    def __init__(self):
        self.args = SimpleNamespace(tie_word_embeddings=False)
        self.lm_head = _FakeHead()
        self.model = SimpleNamespace(
            embed_tokens=_FakeEmbed(),
            layers=[_FakeLayer()],
            norm=_FakeNorm(),
        )


class _CountingNorm:
    def __call__(self, hidden_states):
        return hidden_states


class _CountingAttention:
    def __init__(self):
        self.calls = 0

    def __call__(self, hidden_states, mask=None, cache=None):
        del mask, cache
        self.calls += 1
        return hidden_states + 10


class _CacheOnlyAttention(_CountingAttention):
    num_key_value_heads = 1

    def k_proj(self, hidden_states):
        return hidden_states

    def v_proj(self, hidden_states):
        return hidden_states

    def k_norm(self, hidden_states):
        return hidden_states

    def rope(self, keys, offset=0):
        del offset
        return keys


class _RecordingKVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        self.keys = keys
        self.values = values
        self.offset += int(keys.shape[2])
        return keys, values


class _CountingMLP:
    def __init__(self):
        self.calls = 0

    def __call__(self, hidden_states):
        self.calls += 1
        return hidden_states + 100


class _StructuredLayer:
    is_linear = False

    def __init__(self):
        self.input_layernorm = _CountingNorm()
        self.post_attention_layernorm = _CountingNorm()
        self.self_attn = _CountingAttention()
        self.mlp = _CountingMLP()

    def __call__(self, hidden_states, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(hidden_states), mask, cache)
        h = hidden_states + r
        return h + self.mlp(self.post_attention_layernorm(h))


class _StructuredTarget:
    def __init__(self, n_layers: int):
        self.model = SimpleNamespace(
            embed_tokens=_FakeEmbed(),
            layers=[_StructuredLayer() for _ in range(n_layers)],
            fa_idx=0,
            ssm_idx=0,
        )


def test_target_forward_can_project_only_final_prefill_token_logits():
    target = _FakeTarget()
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.uint32)

    logits, captured = target_forward_with_hidden_states(
        target,
        input_ids=input_ids,
        cache=[None],
        capture_layer_ids={1},
        last_logits_only=True,
    )

    assert logits.shape == (1, 1, 7)
    assert target.lm_head.input_shapes == [(1, 1, 3)]
    assert captured[1].shape == (1, 4, 3)


def test_target_forward_default_still_projects_all_token_logits():
    target = _FakeTarget()
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.uint32)

    logits, _ = target_forward_with_hidden_states(
        target,
        input_ids=input_ids,
        cache=[None],
        capture_layer_ids={1},
    )

    assert logits.shape == (1, 4, 7)
    assert target.lm_head.input_shapes == [(1, 4, 3)]


def test_target_prefill_without_logits_skips_lm_head_projection():
    target = _FakeTarget()
    input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

    hidden = target_prefill_without_logits(
        target,
        input_ids=input_ids,
        cache=[None],
    )

    assert hidden.shape == (1, 3, 3)
    assert target.lm_head.input_shapes == []


def test_target_prefill_without_logits_can_skip_final_layer_mlp_only():
    target = _StructuredTarget(n_layers=2)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

    hidden = target_prefill_without_logits(
        target,
        input_ids=input_ids,
        cache=[None, None],
        skip_final_layer_mlp=True,
    )

    assert hidden.shape == (1, 3, 3)
    assert target.model.layers[0].self_attn.calls == 1
    assert target.model.layers[0].mlp.calls == 1
    assert target.model.layers[1].self_attn.calls == 1
    assert target.model.layers[1].mlp.calls == 0


def test_target_prefill_without_logits_can_update_final_attention_cache_only():
    target = _StructuredTarget(n_layers=2)
    target.model.layers[1].self_attn = _CacheOnlyAttention()
    final_cache = _RecordingKVCache()
    input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

    hidden = target_prefill_without_logits(
        target,
        input_ids=input_ids,
        cache=[None, final_cache],
        skip_final_layer_attention=True,
    )
    mx.eval(hidden, final_cache.keys, final_cache.values)

    assert hidden.shape == (1, 3, 3)
    assert target.model.layers[0].self_attn.calls == 1
    assert target.model.layers[0].mlp.calls == 1
    assert target.model.layers[1].self_attn.calls == 0
    assert target.model.layers[1].mlp.calls == 0
    assert final_cache.offset == 3
    assert final_cache.keys.shape == (1, 1, 3, 3)
    assert final_cache.values.shape == (1, 1, 3, 3)


def test_target_prefill_without_logits_can_return_cache_dependencies_directly():
    target = _StructuredTarget(n_layers=2)
    target.model.layers[1].self_attn = _CacheOnlyAttention()
    final_cache = _RecordingKVCache()
    input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

    hidden, dependencies = target_prefill_without_logits(
        target,
        input_ids=input_ids,
        cache=[None, final_cache],
        skip_final_layer_attention=True,
        return_dependencies=True,
    )
    mx.eval(hidden, *dependencies)

    assert hidden.shape == (1, 3, 3)
    assert len(dependencies) == 2
    assert target.model.layers[1].self_attn.calls == 0
    assert target.model.layers[1].mlp.calls == 0
    assert final_cache.offset == 3
    assert mx.array_equal(dependencies[0], final_cache.keys).item()
    assert mx.array_equal(dependencies[1], final_cache.values).item()


def test_target_prefill_without_logits_dependency_mode_falls_back_without_cache():
    target = _StructuredTarget(n_layers=2)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

    hidden, dependencies = target_prefill_without_logits(
        target,
        input_ids=input_ids,
        cache=[None, None],
        skip_final_layer_attention=True,
        return_dependencies=True,
    )

    assert hidden.shape == (1, 3, 3)
    assert dependencies == []
    assert target.model.layers[1].self_attn.calls == 1
    assert target.model.layers[1].mlp.calls == 1


def test_target_forward_with_hidden_states_can_skip_uncaptured_final_mlp():
    target = _StructuredTarget(n_layers=2)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

    logits, captured = target_forward_with_hidden_states(
        target,
        input_ids=input_ids,
        cache=[None, None],
        capture_layer_ids={1},
        skip_logits=True,
        skip_final_layer_mlp=True,
    )

    assert logits is None
    assert captured[1].shape == (1, 3, 3)
    assert target.model.layers[0].self_attn.calls == 1
    assert target.model.layers[0].mlp.calls == 1
    assert target.model.layers[1].self_attn.calls == 1
    assert target.model.layers[1].mlp.calls == 0


def test_target_forward_with_hidden_states_can_update_uncaptured_final_attention_cache_only():
    target = _StructuredTarget(n_layers=2)
    target.model.layers[1].self_attn = _CacheOnlyAttention()
    final_cache = _RecordingKVCache()
    input_ids = mx.array([[1, 2, 3]], dtype=mx.uint32)

    logits, captured = target_forward_with_hidden_states(
        target,
        input_ids=input_ids,
        cache=[None, final_cache],
        capture_layer_ids={1},
        skip_logits=True,
        skip_final_layer_attention=True,
    )
    mx.eval(captured[1], *captured[-1])

    assert logits is None
    assert captured[1].shape == (1, 3, 3)
    assert -1 in captured
    assert len(captured[-1]) == 2
    assert target.model.layers[0].self_attn.calls == 1
    assert target.model.layers[0].mlp.calls == 1
    assert target.model.layers[1].self_attn.calls == 0
    assert target.model.layers[1].mlp.calls == 0
    assert final_cache.offset == 3
    assert final_cache.keys.shape == (1, 1, 3, 3)
    assert final_cache.values.shape == (1, 1, 3, 3)


def test_stream_cache_only_prefill_skips_final_lm_head_projection(monkeypatch):
    from dflash_mlx import runtime as runtime_mod

    class _FakeDraftBackend:
        def make_cache(self, **kwargs):
            del kwargs
            return [None]

    monkeypatch.setattr(runtime_mod, "make_draft_backend", lambda: _FakeDraftBackend())
    monkeypatch.setattr(runtime_mod, "make_target_cache", lambda *args, **kwargs: [None])
    monkeypatch.setattr(runtime_mod, "detect_engine", lambda target_model: object())
    monkeypatch.setattr(runtime_mod, "_clear_cache_after_prefill_enabled", lambda: False)

    target = _FakeTarget()
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
        args=SimpleNamespace(hidden_size=3),
    )

    events = list(
        runtime_mod.stream_dflash_generate(
            target_model=target,
            tokenizer=None,
            draft_model=draft_model,
            prompt="",
            max_new_tokens=0,
            prompt_tokens_override=[1, 2, 3],
            prefill_step_size=2,
        )
    )

    assert target.lm_head.input_shapes == []
    assert [event["event"] for event in events] == [
        "prefill_progress",
        "prefill_progress",
        "prefill",
        "summary",
    ]
    assert events[-1]["generation_tokens"] == 0
    assert events[-1]["cache_only_prefill"]
