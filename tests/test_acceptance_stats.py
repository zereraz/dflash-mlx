from dflash_mlx.runtime import (
    _acceptance_position_rates,
    _record_acceptance_position_stats,
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
