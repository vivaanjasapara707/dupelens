"""Buy-now-or-wait advice on made-up price histories (the rule, not real prices)."""
from src.tracker import advice


def days(prices):
    return [(f"2026-11-{k + 1:02d}", p) for k, p in enumerate(prices)]


def test_needs_three_days():
    assert advice(days([999]))["kind"] == "collecting"


def test_lowest_price_means_buy():
    assert advice(days([1200, 1150, 1100, 999]))["kind"] == "buy"


def test_falling_price_means_wait():
    out = advice(days([1400, 1350, 1300, 1250, 1200, 1150, 1100, 1160]))
    assert out["kind"] == "wait" and out["slope_per_day"] < 0 and "projection_7d" in out


def test_above_low_without_trend():
    assert advice(days([1000, 1200, 1210, 1190, 1205]))["kind"] == "hold"
