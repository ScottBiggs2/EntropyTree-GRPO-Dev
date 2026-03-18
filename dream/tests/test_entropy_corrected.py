import math

from dream.src.entropy import EntropyComputer


def test_analytic_entropy_weight_bounded():
    """Analytic w_ent should be in [0, 1] for entropy in [0, log(V)]."""
    vocab_size = 50_000
    log_v = math.log(vocab_size)

    w_max = EntropyComputer.compute_entropy_weight(
        measured_masked_mean=log_v,
        vocab_size=vocab_size,
        mode="analytic",
    )
    assert abs(w_max - 1.0) < 1e-6

    w_half = EntropyComputer.compute_entropy_weight(
        measured_masked_mean=log_v / 2,
        vocab_size=vocab_size,
        mode="analytic",
    )
    assert abs(w_half - 0.5) < 1e-6

    w_zero = EntropyComputer.compute_entropy_weight(
        measured_masked_mean=0.0,
        vocab_size=vocab_size,
        mode="analytic",
    )
    assert abs(w_zero) < 1e-6


def test_analytic_weight_independent_of_masking_ratio():
    """w_ent should not depend on masking_ratio for analytic mode."""
    vocab_size = 50_000
    h = 5.0
    w_high_r = EntropyComputer.compute_entropy_weight(
        measured_masked_mean=h,
        vocab_size=vocab_size,
        masking_ratio=0.9,
        mode="analytic",
    )
    w_low_r = EntropyComputer.compute_entropy_weight(
        measured_masked_mean=h,
        vocab_size=vocab_size,
        masking_ratio=0.1,
        mode="analytic",
    )
    assert abs(w_high_r - w_low_r) < 1e-6

