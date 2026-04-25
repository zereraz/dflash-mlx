from dflash_mlx.runtime import (
    _acceptance_position_rates,
    _adaptive_fallback_recent_tokens_per_cycle,
    _record_acceptance_position_stats,
    _should_adaptive_fallback_to_ar,
)


def test_record_acceptance_position_stats_counts_prefix_accepts():
    attempts = [0, 0, 0, 0]
    accepts = [0, 0, 0, 0]

    _record_acceptance_position_stats(
        attempts,
        accepts,
        drafted_count=4,
        acceptance_length=2,
    )

    assert attempts == [1, 1, 1, 1]
    assert accepts == [1, 1, 0, 0]


def test_record_acceptance_position_stats_clamps_to_buffer_length():
    attempts = [0, 0]
    accepts = [0, 0]

    _record_acceptance_position_stats(
        attempts,
        accepts,
        drafted_count=4,
        acceptance_length=4,
    )

    assert attempts == [1, 1]
    assert accepts == [1, 1]


def test_acceptance_position_rates_handles_unattempted_positions():
    assert _acceptance_position_rates([2, 1, 0], [1, 1, 0]) == [0.5, 1.0, 0.0]


def test_adaptive_fallback_waits_for_probe_cycles():
    should_fallback, recent_tpc = _should_adaptive_fallback_to_ar(
        [0, 0, 0],
        probe_cycles=4,
        window=8,
        min_tokens_per_cycle=3.0,
    )

    assert not should_fallback
    assert recent_tpc == 1.0


def test_adaptive_fallback_uses_recent_tokens_per_cycle_window():
    assert _adaptive_fallback_recent_tokens_per_cycle([7, 7, 0, 0], window=2) == 1.0

    should_fallback, recent_tpc = _should_adaptive_fallback_to_ar(
        [7, 7, 0, 0],
        probe_cycles=4,
        window=2,
        min_tokens_per_cycle=3.0,
    )

    assert should_fallback
    assert recent_tpc == 1.0


def test_adaptive_fallback_keeps_good_recent_cycles():
    should_fallback, recent_tpc = _should_adaptive_fallback_to_ar(
        [0, 0, 6, 6],
        probe_cycles=4,
        window=2,
        min_tokens_per_cycle=3.0,
    )

    assert not should_fallback
    assert recent_tpc == 7.0
