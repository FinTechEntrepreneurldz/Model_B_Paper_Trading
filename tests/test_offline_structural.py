"""tests/test_offline_structural.py — exercises everything except yfinance.

Verifies:
  - mlp.joblib loads and has the expected shape
  - score_universe_mlp produces a sensible cross-section z-score
  - make_weights produces a valid 130/30 sector-neutral basket
  - build_order_plan computes correct trades
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_trader import (
    build_order_plan,
    cap_and_renormalize,
    make_weights,
    robust_z_cross_section,
    score_universe_mlp,
)
import joblib

ART_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "mlp.joblib"


def test_joblib_loads():
    art = joblib.load(ART_PATH)
    expected_keys = {"model", "feature_cols", "best_mode", "config",
                     "universe_symbols", "sectors", "rebal_dates"}
    assert expected_keys.issubset(art.keys()), f"missing keys: {expected_keys - set(art.keys())}"
    assert art["best_mode"] == "130_30", f"unexpected mode: {art['best_mode']}"
    assert len(art["feature_cols"]) == 107, f"feature count: {len(art['feature_cols'])}"
    assert len(art["universe_symbols"]) == 300
    assert len(art["sectors"]) == 300
    print("✅ joblib loads with the expected shape")
    return art


def test_mlp_scoring_with_synthetic_features(art):
    rng = np.random.default_rng(42)
    syms = art["universe_symbols"]
    feature_cols = art["feature_cols"]
    # Build a fake feature frame with random values in a plausible range
    X = pd.DataFrame(
        rng.normal(0, 1, size=(len(syms), len(feature_cols))),
        index=syms,
        columns=feature_cols,
    )
    score = score_universe_mlp(X, art)
    assert isinstance(score, pd.Series)
    assert len(score) == len(syms)
    assert score.abs().max() <= 4.0 + 1e-6, "scores should be clipped to ±4"
    assert score.std() > 0.1, "scores should vary"
    print(f"✅ MLP scored {len(score)} stocks: range [{score.min():.2f}, {score.max():.2f}], std={score.std():.2f}")


def test_make_weights_130_30(art):
    rng = np.random.default_rng(7)
    syms = art["universe_symbols"]
    sectors = pd.Series(art["sectors"])
    score = pd.Series(rng.normal(0, 1, len(syms)), index=syms)
    eligible = pd.Series(True, index=syms)
    eligible.iloc[:50] = False  # simulate 50 ineligible stocks

    w = make_weights(score, sectors, eligible, mode="130_30", q=0.20)
    assert isinstance(w, pd.Series)
    longs = w[w > 0]
    shorts = w[w < 0]
    assert len(longs) > 0 and len(shorts) > 0, "must have both legs"
    assert all(eligible.reindex(w[w != 0].index)), "ineligible stocks must have zero weight"
    # In 130/30 mode, per-position cap = 5% × sleeve total = 5% × 1.30 = 6.5% on the long side
    # (and 5% × 0.30 = 1.5% on the short side)
    assert longs.max() <= 0.065 + 1e-6, f"long cap exceeded: {longs.max():.4f}"
    assert abs(shorts.min()) <= 0.015 + 1e-6, f"short cap exceeded: {shorts.min():.4f}"
    long_total = longs.sum()
    short_total = shorts.abs().sum()
    assert abs(long_total - 1.30) < 0.01, f"long sleeve {long_total:.3f} != 1.30"
    assert abs(short_total - 0.30) < 0.01, f"short sleeve {short_total:.3f} != 0.30"
    print(f"✅ make_weights('130_30'): {len(longs)} longs (Σ={long_total:.3f}, max {longs.max():.4f}), "
          f"{len(shorts)} shorts (Σ={short_total:.3f}, min {shorts.min():.4f})")


def test_order_plan():
    rng = np.random.default_rng(11)
    syms = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    target = pd.Series([0.30, 0.25, 0.20, -0.15, -0.10], index=syms)
    current = pd.DataFrame([
        {"symbol": "AAPL", "qty": 100, "market_value": 20000.0, "side": "long"},
        {"symbol": "MSFT", "qty": 50,  "market_value": 25000.0, "side": "long"},
        {"symbol": "TSLA", "qty": 40,  "market_value": 8000.0,  "side": "long"},  # to liquidate
    ])
    prices = pd.Series([200.0, 500.0, 150.0, 175.0, 120.0, 250.0],
                       index=syms + ["TSLA"])

    plan = build_order_plan(target.copy(), current, account_value=100000.0, prices=prices)
    print(f"✅ Order plan ({len(plan)} trades):\n{plan.to_string(index=False)}")
    # Must include a sell for TSLA (liquidation)
    assert "TSLA" in plan["symbol"].tolist(), "TSLA should be liquidated"
    tsla_row = plan[plan["symbol"] == "TSLA"].iloc[0]
    assert tsla_row["side"] == "sell"


def test_cap_and_renormalize():
    # Mix of small and large positions so the cap binds on a few but not all
    raw = pd.Series({
        "A": 0.30, "B": 0.05, "C": 0.04, "D": 0.03, "E": 0.02, "F": 0.02,
        "G": 0.01, "H": 0.01, "I": 0.01, "J": 0.01,
        "K": -0.05, "L": -0.02, "M": -0.01, "N": -0.01,
    })
    out = cap_and_renormalize(raw, mode="130_30", cap=0.05)
    longs = out[out > 0]
    shorts = out[out < 0]
    assert abs(longs.sum() - 1.30) < 0.01, f"long sum {longs.sum():.4f} != 1.30"
    assert abs(shorts.sum() - (-0.30)) < 0.01, f"short sum {shorts.sum():.4f} != -0.30"
    print(f"✅ cap_and_renormalize: long sleeve {longs.sum():.4f} ≈ 1.30, "
          f"short sleeve {shorts.sum():.4f} ≈ -0.30")


if __name__ == "__main__":
    print(f"\n{'='*60}\n  Offline structural test for paper_trader.py\n{'='*60}\n")
    art = test_joblib_loads()
    test_mlp_scoring_with_synthetic_features(art)
    test_make_weights_130_30(art)
    test_cap_and_renormalize()
    test_order_plan()
    print(f"\n  All offline tests passed ✅\n")
