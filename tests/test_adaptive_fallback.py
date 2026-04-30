from __future__ import annotations

from dflash_mlx.engine.adaptive_fallback import (
    AdaptiveFallbackConfig,
    AdaptiveFallbackState,
    resolve_adaptive_fallback_config,
)


def test_adaptive_fallback_disabled_by_default():
    cfg = resolve_adaptive_fallback_config({})
    state = AdaptiveFallbackState(config=cfg, initial_block_tokens=8)

    assert cfg.enabled is False
    assert state.block_tokens_for_cycle(32) == 8
    assert state.record_cycle(block_len=8, commit_count=1) is None
    assert state.summary_fields()["adaptive_fallback_enabled"] is False


def test_low_tokens_per_cycle_enters_target_ar_cooldown():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=3.0,
            probe_cycles=2,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    assert state.record_cycle(block_len=8, commit_count=2) is None
    decision = state.record_cycle(block_len=8, commit_count=1)

    assert decision is not None
    assert decision.action == "fallback"
    assert decision.tokens_per_cycle == 1.5
    assert decision.cooldown_tokens == 4
    assert decision.block_tokens == 8
    assert decision.next_block_tokens == 4
    assert state.in_cooldown is True
    assert state.block_tokens_for_cycle(32) == 1
    assert state.summary_fields()["adaptive_fallback_triggered"] is True


def test_cooldown_completion_starts_reprobe_with_smaller_block():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=3.0,
            probe_cycles=1,
            cooldown_tokens=2,
        ),
        initial_block_tokens=8,
    )

    fallback = state.record_cycle(block_len=8, commit_count=1)
    assert fallback is not None
    assert fallback.action == "fallback"

    assert state.record_cycle(block_len=1, commit_count=1) is None
    reprobe = state.record_cycle(block_len=1, commit_count=1)

    assert reprobe is not None
    assert reprobe.action == "reprobe"
    assert reprobe.block_tokens == 8
    assert reprobe.next_block_tokens == 4
    assert state.in_cooldown is False
    assert state.block_tokens_for_cycle(32) == 4
    summary = state.summary_fields()
    assert summary["adaptive_fallback_tokens"] == 2
    assert summary["adaptive_reprobe_count"] == 1


def test_configured_reprobe_block_is_used_and_clamped():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=9.0,
            probe_cycles=1,
            cooldown_tokens=1,
            reprobe_block_tokens=6,
        ),
        initial_block_tokens=16,
    )

    fallback = state.record_cycle(block_len=16, commit_count=1)
    assert fallback is not None
    assert fallback.next_block_tokens == 6
    reprobe = state.record_cycle(block_len=1, commit_count=1)
    assert reprobe is not None
    assert state.block_tokens_for_cycle(32) == 6

    clamped_cfg = AdaptiveFallbackConfig(
        enabled=True,
        min_tokens_per_cycle=9.0,
        probe_cycles=1,
        cooldown_tokens=1,
        reprobe_block_tokens=64,
    )
    clamped = AdaptiveFallbackState(config=clamped_cfg, initial_block_tokens=16)
    clamped_fallback = clamped.record_cycle(block_len=16, commit_count=1)
    assert clamped_fallback is not None
    assert clamped_fallback.next_block_tokens == 16


def test_probe_window_pass_resets_without_fallback():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=2,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    assert state.record_cycle(block_len=8, commit_count=2) is None
    assert state.record_cycle(block_len=8, commit_count=2) is None

    summary = state.summary_fields()
    assert summary["adaptive_fallback_triggered"] is False
    assert summary["adaptive_last_probe_tokens_per_cycle"] == 2.0
    assert summary["adaptive_pending_probe_cycles"] == 0
    assert state.block_tokens_for_cycle(32) == 8


def test_final_probe_cycle_does_not_enter_fallback():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=9.0,
            probe_cycles=1,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    decision = state.record_cycle(
        block_len=8,
        commit_count=1,
        can_continue=False,
    )

    assert decision is None
    assert state.in_cooldown is False
    summary = state.summary_fields()
    assert summary["adaptive_fallback_triggered"] is False
    assert summary["adaptive_fallback_count"] == 0
    assert summary["adaptive_last_probe_tokens_per_cycle"] == 1.0


def test_final_cooldown_cycle_does_not_record_reprobe():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=9.0,
            probe_cycles=1,
            cooldown_tokens=1,
        ),
        initial_block_tokens=8,
    )

    fallback = state.record_cycle(block_len=8, commit_count=1)
    assert fallback is not None
    assert fallback.action == "fallback"

    decision = state.record_cycle(
        block_len=1,
        commit_count=1,
        can_continue=False,
    )

    assert decision is None
    summary = state.summary_fields()
    assert summary["adaptive_fallback_triggered"] is True
    assert summary["adaptive_reprobe_count"] == 0
    assert summary["adaptive_fallback_tokens"] == 1


def test_env_resolver_requires_opt_in():
    disabled = resolve_adaptive_fallback_config({"DFLASH_ADAPTIVE_FALLBACK": "0"})
    enabled = resolve_adaptive_fallback_config(
        {
            "DFLASH_ADAPTIVE_FALLBACK": "1",
            "DFLASH_ADAPTIVE_MIN_TOKENS_PER_CYCLE": "2.5",
            "DFLASH_ADAPTIVE_PROBE_CYCLES": "3",
            "DFLASH_ADAPTIVE_TARGET_AR_COOLDOWN": "5",
            "DFLASH_ADAPTIVE_REPROBE_BLOCK_TOKENS": "7",
        }
    )

    assert disabled.enabled is False
    assert enabled.enabled is True
    assert enabled.min_tokens_per_cycle == 2.5
    assert enabled.probe_cycles == 3
    assert enabled.cooldown_tokens == 5
    assert enabled.reprobe_block_tokens == 7
