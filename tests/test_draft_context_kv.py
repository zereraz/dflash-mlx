import mlx.core as mx

from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    DFlashDraftModelArgs,
)


def _small_args() -> DFlashDraftModelArgs:
    return DFlashDraftModelArgs(
        model_type="dflash_qwen3",
        hidden_size=32,
        num_hidden_layers=3,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=1000000.0,
        head_dim=8,
        tie_word_embeddings=True,
        num_target_layers=6,
        block_size=8,
        dflash_config={"target_layer_ids": [1, 3, 5], "mask_token_id": 0},
    )


def test_fused_context_kv_matches_layerwise_path(monkeypatch):
    model = DFlashDraftModel(_small_args())
    target_hidden = mx.random.normal((1, 9, 32), dtype=mx.float32)

    monkeypatch.setenv("DFLASH_FUSED_DRAFT_CONTEXT_KV", "0")
    layerwise_cache = [ContextOnlyDraftKVCache() for _ in model.layers]
    model.prefill_context_cache(
        target_hidden_segments=[
            (target_hidden[:, :4, :], 7),
            (target_hidden[:, 4:, :], 11),
        ],
        cache=layerwise_cache,
        total_context_len=16,
    )
    mx.eval(
        *[entry.keys for entry in layerwise_cache],
        *[entry.values for entry in layerwise_cache],
    )

    monkeypatch.setenv("DFLASH_FUSED_DRAFT_CONTEXT_KV", "1")
    fused_cache = [ContextOnlyDraftKVCache() for _ in model.layers]
    model.prefill_context_cache(
        target_hidden_segments=[
            (target_hidden[:, :4, :], 7),
            (target_hidden[:, 4:, :], 11),
        ],
        cache=fused_cache,
        total_context_len=16,
    )
    mx.eval(
        *[entry.keys for entry in fused_cache],
        *[entry.values for entry in fused_cache],
    )

    for expected, actual in zip(layerwise_cache, fused_cache, strict=True):
        assert expected.offset == actual.offset
        assert bool(
            mx.allclose(expected.keys, actual.keys, rtol=1e-5, atol=1e-5).item()
        )
        assert bool(
            mx.allclose(expected.values, actual.values, rtol=1e-5, atol=1e-5).item()
        )
