"""Phase 2: TimeWeighter tests."""

import pytest
from src.time_weight import TimeWeighter


def test_weights_sum_to_one():
    tw = TimeWeighter(total_steps=256)
    total = sum(tw.get_weight(i) for i in range(256))
    assert abs(total - 1.0) < 1e-5


def test_weight_decreases_with_step():
    tw = TimeWeighter(total_steps=256)
    w0 = tw.get_weight(0)
    w128 = tw.get_weight(128)
    w255 = tw.get_weight(255)
    assert w0 > w128 > w255
    assert w255 < 0.01  # final step weight near 0


def test_weight_out_of_range():
    tw = TimeWeighter(total_steps=256)
    assert tw.get_weight(-1) == 0.0
    assert tw.get_weight(256) == 0.0
    assert tw.get_weight(300) == 0.0
