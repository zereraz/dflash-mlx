from dflash_mlx.kernels import _compute_sdpa_2pass_blocks


def test_g15s_gqa_heavy_decode_uses_64_blocks_through_8k(monkeypatch):
    monkeypatch.delenv("DFLASH_SDPA_2PASS_BLOCKS", raising=False)

    assert _compute_sdpa_2pass_blocks(6, 2048, "applegpu_g15s") == 64
    assert _compute_sdpa_2pass_blocks(6, 4096, "applegpu_g15s") == 64
    assert _compute_sdpa_2pass_blocks(6, 8192, "applegpu_g15s") == 64
    assert _compute_sdpa_2pass_blocks(6, 16384, "applegpu_g15s") == 256


def test_other_s_class_gpus_keep_existing_heuristic(monkeypatch):
    monkeypatch.delenv("DFLASH_SDPA_2PASS_BLOCKS", raising=False)

    assert _compute_sdpa_2pass_blocks(6, 4096, "applegpu_g14s") == 128


def test_sdpa_block_override_still_wins(monkeypatch):
    monkeypatch.setenv("DFLASH_SDPA_2PASS_BLOCKS", "128")

    assert _compute_sdpa_2pass_blocks(6, 4096, "applegpu_g15s") == 128
