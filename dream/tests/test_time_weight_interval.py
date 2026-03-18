from dream.src.time_weight import TimeWeighter


def test_mean_to_one_normalization():
    """Under mean_to_one, average weight should be 1.0."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    avg = float(tw.weights.mean().item())
    assert abs(avg - 1.0) < 1e-5


def test_sum_to_one_normalization():
    """Under sum_to_one, weights should sum to 1.0 (backward-compatible)."""
    tw = TimeWeighter(256, norm_mode="sum_to_one")
    s = float(tw.weights.sum().item())
    assert abs(s - 1.0) < 1e-5


def test_interval_weight_matches_point_sum():
    """Interval weight should equal sum of point weights."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    interval_w = tw.get_interval_weight(10, 42)
    point_sum = sum(tw.get_weight(t) for t in range(10, 42))
    assert abs(interval_w - point_sum) < 1e-5


def test_interval_weight_longer_edge_gets_more_weight():
    """A longer edge should receive more time weight than a shorter one."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    short = tw.get_interval_weight(10, 18)  # 8 steps
    long = tw.get_interval_weight(10, 50)   # 40 steps
    assert long > short * 3


def test_time_weights_are_order_one():
    """Time weights under mean_to_one should be O(1), not O(1/T)."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    w_early = tw.get_weight(0)
    w_mid = tw.get_weight(128)
    assert w_early > 0.5
    assert w_mid > 0.1

