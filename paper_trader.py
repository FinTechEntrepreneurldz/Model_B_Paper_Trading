"""paper_trader.py — Model B (MLP_alpha__130_30) live paper trader

Strategy: every rebalance day (every 21 trading days),
  1. Download S&P 500 universe + benchmarks via yfinance
  2. Build per-stock features (chart-summary + Ichimoku + trailing)
  3. Score each stock with the cached MLP
  4. Build a 130/30 sector-neutral basket via make_weights()
  5. Submit Alpaca orders to rebalance to the target basket

Source pipeline: model_0002__1_.py (FinTechEntrepreneurldz / quarterly_pipeline).
Winning variant from BEST_MODEL/metadata.json: MLP_alpha__130_30
  test Sharpe 1.89, ann return 34.0%, max DD -8.7% (period 2025-10-01 → live)

The MLP is loaded from artifacts/mlp.joblib which contains:
  - model:            sklearn Pipeline (imputer + scaler + MLPRegressor)
  - feature_cols:     107 features the model was trained on
  - best_mode:        "130_30"
  - config:           MIN_PRICE, MIN_ADV, TAIL_Q, etc.
  - universe_symbols: 300 tickers fixed at training time
  - sectors:          symbol → sector mapping
  - rebal_dates:      historical rebalance schedule (the model was retrained
                      through this date)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# ════════════════════════════════════════════════════════════════════════
# Paths and config
# ════════════════════════════════════════════════════════════════════════

REPO_ROOT     = Path(__file__).parent.resolve()
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "mlp.joblib"
LOG_DIR       = REPO_ROOT / "logs"
for sub in ["decisions", "orders", "positions", "portfolio", "target_weights", "health"]:
    (LOG_DIR / sub).mkdir(parents=True, exist_ok=True)

# Alpaca paper endpoint (always paper for this repo)
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Trading params (env-overridable for dry runs)
SUBMIT_ORDERS         = os.environ.get("SUBMIT_ORDERS", "true").lower() == "true"
DEFAULT_ACCOUNT_VALUE = float(os.environ.get("DEFAULT_ACCOUNT_VALUE", "100000"))
REBALANCE_THRESHOLD   = float(os.environ.get("REBALANCE_THRESHOLD",   "0.005"))   # 0.5% drift
MIN_TRADE_DOLLARS     = float(os.environ.get("MIN_TRADE_DOLLARS",     "100"))     # $100 min trade
DATA_PERIOD           = os.environ.get("DATA_PERIOD", "3y")  # yfinance lookback for fresh download

# Ichimoku params (must match model_0002__1_.py CONFIG defaults)
ICHI_TENKAN = 9
ICHI_KIJUN  = 26
ICHI_SPAN_B = 52

# Trailing windows used by the MLP feature panel
TRAILING_WINDOWS = [21, 63, 126]

# ════════════════════════════════════════════════════════════════════════
# 1. Universe + price download
# ════════════════════════════════════════════════════════════════════════

def download_universe_prices(symbols: list[str], period: str = DATA_PERIOD) -> dict:
    """Bulk-download adjusted close + volume for the universe + benchmarks.

    Returns dict with: close, volume, high, low, dollar_vol, benchmarks (dict of
    daily returns), close_all (raw close incl benchmarks), top_bench_key.
    """
    BENCHES = ["SPY", "QQQ", "VTI", "RSP", "BIL"]
    all_syms = list(dict.fromkeys(symbols + BENCHES))

    print(f"  Downloading {len(all_syms)} tickers via yfinance (period={period})…")
    data = yf.download(
        tickers=" ".join(all_syms),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns a multiindex DataFrame when group_by='ticker'
    close_all = pd.DataFrame({s: data[s]["Close"] for s in all_syms if s in data.columns.levels[0]})
    high_all  = pd.DataFrame({s: data[s]["High"]  for s in all_syms if s in data.columns.levels[0]})
    low_all   = pd.DataFrame({s: data[s]["Low"]   for s in all_syms if s in data.columns.levels[0]})
    vol_all   = pd.DataFrame({s: data[s]["Volume"] for s in all_syms if s in data.columns.levels[0]})

    close_all = close_all.sort_index().ffill(limit=2)

    if close_all.shape[0] == 0:
        raise RuntimeError(
            "yfinance returned no price data. Common causes: rate-limiting / "
            "network blocking Yahoo Finance hosts. Wait a few minutes and retry, "
            "or run from a different network."
        )

    n_dropped = sum(1 for s in symbols if s not in close_all.columns)
    if n_dropped:
        print(f"  WARNING: {n_dropped} symbols had no yfinance data; will be excluded.")
    universe_syms = [s for s in symbols if s in close_all.columns]

    close  = close_all[universe_syms]
    high   = high_all.reindex(columns=universe_syms)
    low    = low_all.reindex(columns=universe_syms)
    volume = vol_all.reindex(columns=universe_syms)

    dollar_vol = (close * volume).rolling(60, min_periods=20).mean()

    # Benchmark daily returns
    benchmarks = {}
    for b in BENCHES:
        if b in close_all.columns:
            benchmarks[f"{b}_buy_hold"] = close_all[b].pct_change().fillna(0.0)
    # Equal-weight Top300-ADV benchmark (computed each day from the universe)
    if len(universe_syms) > 0:
        top_ret = close[universe_syms].pct_change().mean(axis=1).fillna(0.0)
        benchmarks["Top300_EW_daily"] = top_ret
        benchmarks["CurrentSP500_EW_daily"] = top_ret  # close enough for live use

    return {
        "symbols":        universe_syms,
        "close":          close,
        "close_all":      close_all,
        "high":           high,
        "low":            low,
        "volume":         volume,
        "dollar_vol":     dollar_vol,
        "benchmarks":     benchmarks,
        "top_bench_key":  "Top300_EW_daily",
    }


# ════════════════════════════════════════════════════════════════════════
# 2. Feature builders (port of Cells 4-6 of model_0002__1_.py)
# ════════════════════════════════════════════════════════════════════════

# ── Cell 4: Cheap chart features (8 numerics × 3 views = 24 features) ────

def chart_summary_features(px_window: pd.Series) -> dict:
    """Eight scalar features summarising a price window."""
    x = pd.Series(px_window).astype(float).dropna()
    if len(x) < 10:
        return {f"f{i}": np.nan for i in range(8)}
    norm = x / x.iloc[0] - 1.0
    t    = np.arange(len(norm))
    slope = np.polyfit(t, norm.values, 1)[0]
    quad  = np.polyfit(t, norm.values, 2)[0]
    eq    = (1 + x.pct_change().fillna(0)).cumprod()
    dd    = (eq / eq.cummax() - 1).min()
    dur   = float(((eq / eq.cummax() - 1) < -0.02).sum()) / len(eq)
    total = (x.diff().abs()).sum()
    net   = x.iloc[-1] - x.iloc[0]
    eff   = float(net / total) if total > 0 else 0.0
    vol   = float(x.pct_change().std() * np.sqrt(252)) if len(x) > 5 else 0.0
    rets  = x.pct_change().dropna()
    resid = float(rets.tail(5).mean() - rets.mean()) if len(rets) > 5 else 0.0
    if len(norm) > 12:
        third = len(norm) // 3
        a1 = np.polyfit(np.arange(third), norm.values[:third], 1)[0]
        a2 = np.polyfit(np.arange(third), norm.values[-third:], 1)[0]
        accel = float(a2 - a1)
    else:
        accel = 0.0
    return dict(f0=float(slope), f1=float(quad), f2=float(dd), f3=float(dur),
                f4=float(eff),    f5=float(vol),  f6=float(resid), f7=float(accel))


def build_chart_features_today(univ: dict, lookback: int = 63) -> pd.DataFrame:
    """Per-stock chart features at the latest date; 24 columns per symbol."""
    C   = univ["close"]
    SPY = univ["close_all"]["SPY"].reindex(C.index).ffill() if "SPY" in univ["close_all"].columns else None
    TOP = (1 + univ["benchmarks"][univ["top_bench_key"]]).cumprod().reindex(C.index).ffill()

    win = C.index[-lookback:]
    rows = []
    for sym in univ["symbols"]:
        px = C[sym].reindex(win)
        feats = {f"chart_price_{k}": v for k, v in chart_summary_features(px).items()}
        if SPY is not None:
            ratio_spy = (C[sym] / SPY).replace([np.inf, -np.inf], np.nan)
            feats.update({f"chart_ratio_spy_{k}": v
                          for k, v in chart_summary_features(ratio_spy.reindex(win)).items()})
        ratio_top = (C[sym] / TOP).replace([np.inf, -np.inf], np.nan)
        feats.update({f"chart_ratio_top_{k}": v
                      for k, v in chart_summary_features(ratio_top.reindex(win)).items()})
        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


# ── Cell 5: Ichimoku features (daily + weekly + RS variants) ─────────────

def compute_ichimoku_table(close_s: pd.Series, high_s: pd.Series | None = None,
                            low_s: pd.Series | None = None,
                            conv: int = ICHI_TENKAN, base: int = ICHI_KIJUN,
                            span_b: int = ICHI_SPAN_B) -> pd.DataFrame:
    close_s = pd.Series(close_s).astype(float)
    high_s  = close_s if high_s is None else pd.Series(high_s).astype(float)
    low_s   = close_s if low_s  is None else pd.Series(low_s).astype(float)
    tenkan = (high_s.rolling(conv,   min_periods=max(3, conv//2)).max() +
              low_s .rolling(conv,   min_periods=max(3, conv//2)).min()) / 2
    kijun  = (high_s.rolling(base,   min_periods=max(5, base//2)).max() +
              low_s .rolling(base,   min_periods=max(5, base//2)).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b_line = (high_s.rolling(span_b, min_periods=max(10, span_b//2)).max() +
                   low_s .rolling(span_b, min_periods=max(10, span_b//2)).min()) / 2
    out = pd.DataFrame({
        "close": close_s, "tenkan": tenkan, "kijun": kijun,
        "span_a": span_a, "span_b": span_b_line,
    })
    out["cloud_top"] = out[["span_a", "span_b"]].max(axis=1)
    out["cloud_bot"] = out[["span_a", "span_b"]].min(axis=1)
    out["cloud_mid"] = (out["cloud_top"] + out["cloud_bot"]) / 2
    return out


def _ichi_block_today(prefix: str, close_s: pd.Series, ichi: pd.DataFrame) -> dict:
    """Ichimoku features at the latest index of the series (12 features per prefix)."""
    cs = close_s.iloc[-1]
    out = {}
    out[f"{prefix}_vs_cloud_mid"] = (
        float(cs / ichi["cloud_mid"].iloc[-1] - 1.0)
        if pd.notna(ichi["cloud_mid"].iloc[-1]) and ichi["cloud_mid"].iloc[-1] else np.nan)
    out[f"{prefix}_above_cloud"]   = float(cs > ichi["cloud_top"].iloc[-1]) if pd.notna(ichi["cloud_top"].iloc[-1]) else np.nan
    out[f"{prefix}_below_cloud"]   = float(cs < ichi["cloud_bot"].iloc[-1]) if pd.notna(ichi["cloud_bot"].iloc[-1]) else np.nan
    out[f"{prefix}_cloud_state"]   = 1.0 if (pd.notna(ichi["span_a"].iloc[-1]) and pd.notna(ichi["span_b"].iloc[-1])
                                              and ichi["span_a"].iloc[-1] > ichi["span_b"].iloc[-1]) else -1.0
    out[f"{prefix}_tenkan_kijun_cross"] = 1.0 if (pd.notna(ichi["tenkan"].iloc[-1]) and pd.notna(ichi["kijun"].iloc[-1])
                                                    and ichi["tenkan"].iloc[-1] > ichi["kijun"].iloc[-1]) else -1.0
    tk = ichi["tenkan"].iloc[-1] - ichi["kijun"].iloc[-1] if pd.notna(ichi["tenkan"].iloc[-1]) and pd.notna(ichi["kijun"].iloc[-1]) else np.nan
    out[f"{prefix}_tenkan_minus_kijun"] = float(tk / cs) if cs and pd.notna(tk) else np.nan
    ct = ichi["cloud_top"].iloc[-1]; cb = ichi["cloud_bot"].iloc[-1]
    out[f"{prefix}_cloud_thickness"] = float(abs(ct - cb) / cs) if cs and pd.notna(ct) and pd.notna(cb) else np.nan
    sa = ichi["span_a"].iloc[-1]; sb = ichi["span_b"].iloc[-1]
    out[f"{prefix}_dist_to_span_a"] = float((cs - sa) / cs) if cs and pd.notna(sa) else np.nan
    out[f"{prefix}_dist_to_span_b"] = float((cs - sb) / cs) if cs and pd.notna(sb) else np.nan
    if len(ichi) > 5 and pd.notna(ichi["tenkan"].iloc[-6]) and ichi["tenkan"].iloc[-6]:
        out[f"{prefix}_tenkan_slope_5"] = float(ichi["tenkan"].iloc[-1] / ichi["tenkan"].iloc[-6] - 1.0)
    else:
        out[f"{prefix}_tenkan_slope_5"] = np.nan
    if len(ichi) > 5 and pd.notna(ichi["kijun"].iloc[-6]) and ichi["kijun"].iloc[-6]:
        out[f"{prefix}_kijun_slope_5"] = float(ichi["kijun"].iloc[-1] / ichi["kijun"].iloc[-6] - 1.0)
    else:
        out[f"{prefix}_kijun_slope_5"]  = np.nan
    return out


def build_ichimoku_features_today(univ: dict) -> pd.DataFrame:
    """Per-stock ichi_d_* (daily) and ichi_w_* (weekly) features for today."""
    C  = univ["close"]; H = univ["high"]; L = univ["low"]
    SPY = univ["close_all"]["SPY"].reindex(C.index).ffill() if "SPY" in univ["close_all"].columns else None
    TOP = (1 + univ["benchmarks"][univ["top_bench_key"]]).cumprod().reindex(C.index).ffill()

    rows = []
    for sym in univ["symbols"]:
        cs = C[sym]; hs = H[sym] if sym in H.columns else cs; ls = L[sym] if sym in L.columns else cs
        feats = {}

        # Daily price ichi
        ichi_d_price = compute_ichimoku_table(cs, hs, ls)
        feats.update(_ichi_block_today("ichi_d_price", cs, ichi_d_price))

        # Daily RS-vs-SPY ichi
        if SPY is not None:
            rs_spy = (cs / SPY).replace([np.inf, -np.inf], np.nan)
            ichi_d_spy = compute_ichimoku_table(rs_spy)
            feats.update(_ichi_block_today("ichi_d_rs_spy", rs_spy, ichi_d_spy))

        # Daily RS-vs-TOP ichi
        rs_top = (cs / TOP).replace([np.inf, -np.inf], np.nan)
        ichi_d_top = compute_ichimoku_table(rs_top)
        feats.update(_ichi_block_today("ichi_d_rs_top", rs_top, ichi_d_top))

        # Weekly variants (resample to W-FRI)
        cs_w = cs.resample("W-FRI").last()
        hs_w = hs.resample("W-FRI").max()
        ls_w = ls.resample("W-FRI").min()
        ichi_w_price = compute_ichimoku_table(cs_w, hs_w, ls_w)
        feats.update(_ichi_block_today("ichi_w_price", cs_w, ichi_w_price))

        if SPY is not None:
            rs_spy_w = rs_spy.resample("W-FRI").last()
            ichi_w_spy = compute_ichimoku_table(rs_spy_w)
            feats.update(_ichi_block_today("ichi_w_rs_spy", rs_spy_w, ichi_w_spy))

        rs_top_w = rs_top.resample("W-FRI").last()
        ichi_w_top = compute_ichimoku_table(rs_top_w)
        feats.update(_ichi_block_today("ichi_w_rs_top", rs_top_w, ichi_w_top))

        feats["symbol"] = sym
        rows.append(feats)

    return pd.DataFrame(rows).set_index("symbol")


# ── Cell 6: Trailing returns / vol / drawdown features ──────────────────

def build_trailing_features_today(univ: dict) -> pd.DataFrame:
    """Per-stock trailing return, vol, drawdown over [21, 63, 126] days."""
    C   = univ["close"]
    rows = []
    for sym in univ["symbols"]:
        px = C[sym].dropna()
        feats = {}
        for w in TRAILING_WINDOWS:
            tail = px.tail(w)
            if len(tail) >= max(5, w // 4):
                ret = float(tail.iloc[-1] / tail.iloc[0] - 1.0) if tail.iloc[0] > 0 else np.nan
                rets_d = tail.pct_change().dropna()
                vol = float(rets_d.std() * np.sqrt(252)) if len(rets_d) > 5 else np.nan
                eq  = (1 + rets_d).cumprod()
                dd  = float((eq / eq.cummax() - 1).min()) if len(eq) else np.nan
            else:
                ret = vol = dd = np.nan
            feats[f"ret_{w}"]  = ret
            feats[f"vol_{w}"]  = vol
            feats[f"dd_{w}"]   = dd
        # Recent vs longer residual
        rets126 = px.tail(126).pct_change().dropna()
        if len(rets126) > 30:
            feats["resid_recent"] = float(rets126.tail(5).mean() - rets126.mean())
            feats["resid_long"]   = float(rets126.tail(21).mean() - rets126.mean())
        else:
            feats["resid_recent"] = feats["resid_long"] = np.nan
        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


def build_feature_frame_today(univ: dict) -> pd.DataFrame:
    """Combine chart + ichi + trailing features. One row per symbol."""
    chart = build_chart_features_today(univ)
    ichi  = build_ichimoku_features_today(univ)
    trail = build_trailing_features_today(univ)
    df = chart.join(ichi, how="outer").join(trail, how="outer")
    return df


# ════════════════════════════════════════════════════════════════════════
# 3. MLP scoring
# ════════════════════════════════════════════════════════════════════════

def robust_z_cross_section(x: pd.Series, clip: float = 4.0) -> pd.Series:
    """Cross-section robust z-score (median/MAD), clipped."""
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    med = x.median()
    mad = (x - med).abs().median()
    if mad <= 1e-12 or pd.isna(mad):
        std = x.std()
        if std <= 1e-12 or pd.isna(std):
            return pd.Series(0.0, index=x.index)
        z = (x - x.mean()) / std
    else:
        z = (x - med) / (1.4826 * mad)
    return z.clip(-clip, clip).fillna(0.0)


def score_universe_mlp(features_df: pd.DataFrame, mlp_artifact: dict) -> pd.Series:
    """Run the MLP on per-stock features and return cross-section z-scored predictions."""
    feature_cols = mlp_artifact["feature_cols"]
    model        = mlp_artifact["model"]

    # Make sure all expected columns exist (fill missing with NaN; the imputer handles it)
    X = pd.DataFrame(index=features_df.index, columns=feature_cols, dtype=float)
    common = [c for c in feature_cols if c in features_df.columns]
    X[common] = features_df[common].astype(float)
    X = X[feature_cols]  # enforce exact ordering

    n_missing = len(feature_cols) - len(common)
    if n_missing > 0:
        print(f"  WARNING: {n_missing}/{len(feature_cols)} MLP features missing from live frame; imputed.")

    pred = model.predict(X.values)
    score = pd.Series(pred, index=features_df.index, name="mlp_score")
    return robust_z_cross_section(score)


# ════════════════════════════════════════════════════════════════════════
# 4. Basket builder (port of make_weights from model_0002__1_.py)
# ════════════════════════════════════════════════════════════════════════

def cap_and_renormalize(raw: pd.Series, mode: str, cap: float = 0.05) -> pd.Series:
    """Cap individual position weights and renormalize per-side to keep gross exposure intact."""
    raw = pd.Series(raw, dtype=float).fillna(0.0)
    pos_mask = raw > 0
    neg_mask = raw < 0

    pos = raw[pos_mask]
    neg = raw[neg_mask].abs()

    pos_t = pos.sum() if mode in {"long_top20", "long_exclude_bottom20"} else (1.30 if mode == "130_30" else pos.sum())
    neg_t = (0.30 if mode == "130_30" else neg.sum())

    def _cap(amts: pd.Series, total: float, capv: float) -> pd.Series:
        if amts.empty or total <= 0: return amts
        amts = amts.clip(upper=capv * total)
        return amts * (total / amts.sum()) if amts.sum() > 0 else amts

    out = pd.Series(0.0, index=raw.index, dtype=float)
    if pos_mask.any(): out.loc[pos.index] =  _cap(pos.abs(), pos_t, cap)
    if neg_mask.any(): out.loc[neg.index] = -_cap(neg,        neg_t, cap)
    return out.fillna(0.0)


def make_weights(score: pd.Series, sectors: pd.Series, eligible: pd.Series,
                 mode: str = "130_30", q: float = 0.20) -> pd.Series:
    """Sector-neutral basket from cross-sectional score.

    Ports model_0002__1_.py make_weights() exactly. Eligible = bool series of
    symbols meeting MIN_PRICE / MIN_ADV gates.
    """
    score = pd.Series(score).where(eligible).replace([np.inf, -np.inf], np.nan)
    raw = pd.Series(0.0, index=score.index, dtype=float)
    n_total = max(1, score.dropna().shape[0])

    for sec, idx in sectors.groupby(sectors).groups.items():
        cols = [c for c in idx if c in score.index]
        x = score.reindex(cols).dropna()
        if len(x) < 5 or x.nunique() < 2 or x.std() <= 1e-12:
            continue
        n = max(1, int(np.floor(len(x) * q)))
        top = x.nlargest(n).index
        bot = x.nsmallest(n).index
        sector_w = len(x) / n_total

        if mode == "ls":
            raw.loc[top] += sector_w / n
            raw.loc[bot] -= sector_w / n
        elif mode == "long_top20":
            raw.loc[top] += sector_w / n
        elif mode == "long_exclude_bottom20":
            keep = x.drop(index=bot).index
            if len(keep): raw.loc[keep] += sector_w / len(keep)
        elif mode == "130_30":
            raw.loc[top] += 1.30 * sector_w / n
            raw.loc[bot] -= 0.30 * sector_w / n
        else:
            raise ValueError(f"Unknown weight mode: {mode}")

    return cap_and_renormalize(raw, mode)


# ════════════════════════════════════════════════════════════════════════
# 5. Alpaca integration
# ════════════════════════════════════════════════════════════════════════

def get_alpaca_client():
    from alpaca.trading.client import TradingClient
    api_key    = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in env.")
    return TradingClient(api_key, api_secret, paper=True)


def get_account_info(client) -> dict:
    acct = client.get_account()
    return {
        "equity":         float(acct.equity),
        "cash":           float(acct.cash),
        "buying_power":   float(acct.buying_power),
        "long_value":     float(acct.long_market_value),
        "short_value":    float(acct.short_market_value),
        "status":         str(acct.status),
    }


def get_current_positions(client) -> pd.DataFrame:
    positions = client.get_all_positions()
    if not positions:
        return pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])
    rows = []
    for p in positions:
        rows.append({
            "symbol":       p.symbol,
            "qty":          float(p.qty),
            "market_value": float(p.market_value),
            "side":         "long" if float(p.qty) > 0 else "short",
        })
    return pd.DataFrame(rows)


def build_order_plan(target_weights: pd.Series, current_positions: pd.DataFrame,
                     account_value: float, prices: pd.Series) -> pd.DataFrame:
    """Compute the orders required to move from current_positions to target_weights.

    target_weights:   target portfolio weight per symbol (signed)
    current_positions: DataFrame with [symbol, qty, market_value, side]
    account_value:    total equity to size positions against
    prices:           latest close per symbol
    """
    cur_w = pd.Series(0.0, index=target_weights.index)
    if not current_positions.empty:
        for _, row in current_positions.iterrows():
            if row["symbol"] in cur_w.index:
                cur_w.loc[row["symbol"]] = row["market_value"] / account_value

    # Include any currently-held symbols that aren't in the new basket (need to liquidate)
    extra = [s for s in current_positions["symbol"].tolist() if s not in target_weights.index] if not current_positions.empty else []
    if extra:
        for s in extra:
            target_weights.loc[s] = 0.0
            mv = current_positions.loc[current_positions["symbol"] == s, "market_value"].iloc[0]
            cur_w.loc[s] = mv / account_value

    diff_w = target_weights - cur_w.reindex(target_weights.index).fillna(0.0)
    diff_dollars = diff_w * account_value

    rows = []
    for sym, dw in diff_w.items():
        dollars = float(dw * account_value)
        if abs(dollars) < MIN_TRADE_DOLLARS or abs(dw) < REBALANCE_THRESHOLD:
            continue
        px = prices.get(sym)
        if pd.isna(px) or px <= 0:
            print(f"  Skipping {sym}: no price")
            continue
        qty = dollars / float(px)
        side = "buy" if qty > 0 else "sell"
        rows.append({
            "symbol":       sym,
            "side":         side,
            "qty":          abs(round(qty, 4)),
            "notional":     round(abs(dollars), 2),
            "target_weight": round(float(target_weights[sym]), 4),
            "current_weight": round(float(cur_w.get(sym, 0.0)), 4),
            "weight_diff":   round(float(dw), 4),
            "price":         round(float(px), 2),
        })
    return pd.DataFrame(rows).sort_values("notional", ascending=False).reset_index(drop=True)


def submit_orders(order_plan: pd.DataFrame, client) -> pd.DataFrame:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    submitted = []
    for _, row in order_plan.iterrows():
        try:
            req = MarketOrderRequest(
                symbol=row["symbol"],
                notional=row["notional"],
                side=OrderSide.BUY if row["side"] == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            o = client.submit_order(req)
            submitted.append({
                "symbol":   row["symbol"],
                "side":     row["side"],
                "notional": row["notional"],
                "order_id": o.id,
                "status":   str(o.status),
            })
            print(f"    ✅ {row['side'].upper()} ${row['notional']:>8.2f} {row['symbol']}")
        except Exception as e:
            print(f"    ❌ {row['symbol']} ({row['side']} ${row['notional']:.2f}): {e}")
            submitted.append({
                "symbol": row["symbol"], "side": row["side"], "notional": row["notional"],
                "order_id": None, "status": f"ERROR: {e}",
            })
    return pd.DataFrame(submitted)


# ════════════════════════════════════════════════════════════════════════
# 6. Logging
# ════════════════════════════════════════════════════════════════════════

def _append_csv(path: Path, row_df: pd.DataFrame):
    """Append rows to a CSV (read-merge-write). Creates the file if missing."""
    if path.exists() and path.stat().st_size > 0:
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, row_df], ignore_index=True)
        except Exception:
            out = row_df
    else:
        out = row_df
    out.to_csv(path, index=False)


def log_outputs(decision: dict, target_weights: pd.Series, positions: pd.DataFrame,
                order_plan: pd.DataFrame, submitted: pd.DataFrame, account_info: dict):
    """Write logs in the canonical schema the central Streamlit dashboard expects.

    Mirrors the file/column layout used by Base_Model_BR_PPO/logs/<model>/, so the
    dashboard can read both models with one code path.
    """
    ts_iso   = datetime.now(timezone.utc).isoformat()                       # "2026-04-30T23:24:32.388+00:00"
    ts_file  = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")       # filename-safe
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── decisions/ ───────────────────────────────────────────────────────
    # Schema (matches model_a):
    #   market_date, variant, action, action_idx, last_action, submit_orders,
    #   account_status, account_value, n_target_positions, n_orders_planned,
    #   n_orders_submitted, model_path, metadata_path, timestamp_utc
    dec_row = {
        "market_date":         date_str,
        "variant":             "MLP_alpha__130_30",
        "action":              decision.get("action", "rebalance"),         # "rebalance" or "hold"
        "action_idx":          0,                                             # not a discrete action space
        "last_action":         "",
        "submit_orders":       bool(decision.get("submit_orders", False)),
        "account_status":      account_info.get("status", "unknown"),
        "account_value":       float(account_info.get("equity", 0.0)),
        "n_target_positions":  int(decision.get("n_target_positions", 0)),
        "n_orders_planned":    int(decision.get("n_planned_orders", 0)),
        "n_orders_submitted":  int(decision.get("n_submitted_orders", 0)),
        "model_path":          "artifacts/mlp.joblib",
        "metadata_path":       "",
        "timestamp_utc":       ts_iso,
    }
    dec_df = pd.DataFrame([dec_row])
    dec_df.to_csv(LOG_DIR / "decisions" / "latest_decision.csv", index=False)
    _append_csv(LOG_DIR / "decisions" / "decisions.csv", dec_df)

    # ── target_weights/ ──────────────────────────────────────────────────
    tw_df = target_weights.reset_index()
    tw_df.columns = ["symbol", "weight"]
    tw_df = tw_df[tw_df["weight"].abs() > 1e-6].sort_values("weight", ascending=False).reset_index(drop=True)
    tw_df["timestamp_utc"] = ts_iso
    tw_df.to_csv(LOG_DIR / "target_weights" / "latest_target_weights.csv", index=False)
    _append_csv(LOG_DIR / "target_weights" / "target_weights.csv", tw_df)

    # ── orders/ ──────────────────────────────────────────────────────────
    # Two latest snapshots + one cumulative history
    if order_plan is not None and not order_plan.empty:
        op = order_plan.copy()
        op["timestamp_utc"] = ts_iso
        op.to_csv(LOG_DIR / "orders" / "latest_planned_orders.csv", index=False)
    else:
        # Write empty file so the dashboard knows the run had no planned orders
        pd.DataFrame(columns=["symbol", "side", "qty", "notional", "timestamp_utc"]).to_csv(
            LOG_DIR / "orders" / "latest_planned_orders.csv", index=False)

    if submitted is not None and not submitted.empty:
        sub = submitted.copy()
        sub["timestamp_utc"] = ts_iso
        sub.to_csv(LOG_DIR / "orders" / "latest_submitted_orders.csv", index=False)
        _append_csv(LOG_DIR / "orders" / "submitted_orders.csv", sub)
    else:
        pd.DataFrame(columns=["symbol", "side", "notional", "order_id", "status", "timestamp_utc"]).to_csv(
            LOG_DIR / "orders" / "latest_submitted_orders.csv", index=False)

    # ── positions/ ───────────────────────────────────────────────────────
    if positions is not None and not positions.empty:
        pos = positions.copy()
        pos["timestamp_utc"] = ts_iso
        pos.to_csv(LOG_DIR / "positions" / "latest_positions.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "qty", "market_value", "side", "timestamp_utc"]).to_csv(
            LOG_DIR / "positions" / "latest_positions.csv", index=False)

    # ── portfolio/portfolio.csv ──────────────────────────────────────────
    # Schema: timestamp_utc, portfolio_value, action, submit_orders
    portfolio_row = pd.DataFrame([{
        "timestamp_utc":   ts_iso,
        "portfolio_value": float(account_info.get("equity", 0.0)),
        "action":          decision.get("action", "rebalance"),
        "submit_orders":   bool(decision.get("submit_orders", False)),
    }])
    _append_csv(LOG_DIR / "portfolio" / "portfolio.csv", portfolio_row)

    # ── health/ ──────────────────────────────────────────────────────────
    # health_status.json — lightweight summary the dashboard reads
    health_status = {
        "computed_at":           ts_iso,
        "lookback_days":         63,
        "overall_status":        decision.get("status", "ok"),
        "alerts":                [] if decision.get("status") == "ok" else [decision.get("status", "")],
        "metrics":               {},
        "action_counts":         {decision.get("action", "rebalance"): 1},
        "action_entropy":        None,
        "portfolio_sharpe_30d":  None,
        "spy_sharpe_30d":        None,
        "portfolio_return_30d":  None,
        "spy_return_30d":        None,
        "n_decisions":           1,
        "n_unique_actions":      1,
        "top_action":            decision.get("action", "rebalance"),
        "top_action_pct":        100.0,
        "days_since_last_run":   0,
        "training_recommended":  False,
    }
    with open(LOG_DIR / "health" / "health_status.json", "w") as f:
        json.dump(health_status, f, indent=2)

    # signal_history.csv — append per-run health row (matches signal_history shape)
    sig_row = pd.DataFrame([{
        "timestamp_utc":      ts_iso,
        "date":               date_str,
        "status":             decision.get("status", "ok"),
        "n_target_positions": int(decision.get("n_target_positions", 0)),
        "n_planned_orders":   int(decision.get("n_planned_orders", 0)),
        "n_submitted_orders": int(decision.get("n_submitted_orders", 0)),
        "account_value":      float(account_info.get("equity", 0.0)),
        "submit_orders":      bool(decision.get("submit_orders", False)),
    }])
    _append_csv(LOG_DIR / "health" / "signal_history.csv", sig_row)


# ════════════════════════════════════════════════════════════════════════
# 7. Main trading cycle
# ════════════════════════════════════════════════════════════════════════

def is_rebalance_day(today: pd.Timestamp, rebal_freq_days: int = 21,
                     last_rebal_date: pd.Timestamp | None = None) -> bool:
    """Heuristic: rebalance if it's been >= rebal_freq_days trading days since last rebal,
    OR if no prior rebalance exists. We approximate by counting calendar days * 5/7."""
    if last_rebal_date is None:
        return True
    days = (today - last_rebal_date).days
    return days * 5 / 7 >= rebal_freq_days


def get_last_rebalance_date() -> pd.Timestamp | None:
    """Return the most recent rebalance date from the decision log, or None if no history."""
    latest = LOG_DIR / "decisions" / "latest_decision.csv"
    if not latest.exists():
        return None
    try:
        df = pd.read_csv(latest)
        if df.empty: return None
        ts = df["timestamp_utc"].iloc[-1]
        return pd.Timestamp(ts.replace("_", " ").replace(" ", "T")[:10])
    except Exception:
        return None


def run_trading_cycle(force_rebalance: bool = False, dry_run: bool = False):
    print(f"\n{'='*70}\n  MODEL B — MLP_alpha__130_30 LIVE PAPER TRADER\n{'='*70}")
    print(f"  Time (UTC):   {datetime.now(timezone.utc).isoformat()}")
    print(f"  Submit live:  {SUBMIT_ORDERS and not dry_run}")
    print(f"  Force rebal:  {force_rebalance}")

    # 0. Load MLP artifact
    print(f"\n[0/6] Loading MLP artifact: {ARTIFACT_PATH}")
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"MLP artifact not found at {ARTIFACT_PATH}")
    art = joblib.load(ARTIFACT_PATH)
    print(f"      features={len(art['feature_cols'])}  universe={len(art['universe_symbols'])}  mode={art['best_mode']}")
    print(f"      saved={art.get('saved_at_utc')}  sklearn={art.get('sklearn_version')}")

    universe_symbols = art["universe_symbols"]
    sectors          = pd.Series(art["sectors"])
    cfg              = art["config"]
    mode             = art["best_mode"]

    # 0b. Decide whether today is a rebalance day
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    last_rebal = get_last_rebalance_date()
    do_rebal = force_rebalance or is_rebalance_day(today, rebal_freq_days=cfg.get("PATH_HORIZON_DAYS", 21),
                                                     last_rebal_date=last_rebal)
    print(f"\n      Today: {today.date()}  Last rebal: {last_rebal}  Rebalance today: {do_rebal}")

    # 1. Download universe prices
    print(f"\n[1/6] Downloading universe + benchmark prices…")
    univ = download_universe_prices(universe_symbols, period=DATA_PERIOD)
    if not univ["symbols"]:
        raise RuntimeError("Universe is empty after yfinance download.")
    print(f"      Got {len(univ['symbols'])} symbols, {univ['close'].shape[0]} trading days.")

    # 2. Build features
    print(f"\n[2/6] Building per-stock features…")
    features = build_feature_frame_today(univ)
    print(f"      Feature frame: {features.shape}")

    # 3. Score with MLP
    print(f"\n[3/6] Scoring universe with MLP…")
    score = score_universe_mlp(features, art)
    print(f"      Scores: top 5 → {score.nlargest(5).round(2).to_dict()}")
    print(f"              bot 5 → {score.nsmallest(5).round(2).to_dict()}")

    # 4. Build basket
    print(f"\n[4/6] Building 130/30 basket…")
    today_close = univ["close"].iloc[-1]
    today_dvol  = univ["dollar_vol"].iloc[-1]
    eligible    = (today_close > cfg["MIN_PRICE"]) & (today_dvol > cfg["MIN_ADV"])
    eligible    = eligible.reindex(score.index).fillna(False)
    print(f"      Eligible (price>${cfg['MIN_PRICE']}, ADV>${cfg['MIN_ADV']:,}): "
          f"{int(eligible.sum())}/{len(eligible)}")
    sectors_aligned = sectors.reindex(score.index).fillna("Unknown")
    target = make_weights(score, sectors_aligned, eligible, mode=mode, q=cfg["TAIL_Q"])
    target = target[target.abs() > 1e-6]
    print(f"      Basket: {len(target)} positions  (long={int((target>0).sum())}, "
          f"short={int((target<0).sum())}, gross={target.abs().sum():.2f})")

    # 5. Connect to Alpaca + plan orders
    print(f"\n[5/6] Connecting to Alpaca and planning orders…")
    if dry_run:
        print(f"      DRY RUN — skipping Alpaca, using defaults.")
        client = None
        account_info = {"equity": DEFAULT_ACCOUNT_VALUE, "cash": DEFAULT_ACCOUNT_VALUE,
                        "buying_power": DEFAULT_ACCOUNT_VALUE, "long_value": 0.0,
                        "short_value": 0.0, "status": "DRY_RUN"}
        positions = pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])
    else:
        try:
            client = get_alpaca_client()
            account_info = get_account_info(client)
            positions = get_current_positions(client)
            print(f"      Account equity: ${account_info['equity']:,.2f}  "
                  f"cash: ${account_info['cash']:,.2f}  status: {account_info['status']}")
            print(f"      Current positions: {len(positions)}")
        except Exception as e:
            print(f"      ❌ Alpaca connection failed: {e}")
            print(f"      Falling back to dry-run defaults.")
            client = None
            account_info = {"equity": DEFAULT_ACCOUNT_VALUE, "cash": DEFAULT_ACCOUNT_VALUE,
                            "buying_power": DEFAULT_ACCOUNT_VALUE, "long_value": 0.0,
                            "short_value": 0.0, "status": "ALPACA_ERROR"}
            positions = pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])

    if not do_rebal:
        print(f"\n      Not a rebalance day. Logging snapshot and exiting (no orders).")
        decision = {
            "action":            "hold",
            "do_rebalance":      False,
            "submit_orders":     False,
            "n_target_positions": 0,
            "n_planned_orders":   0,
            "n_submitted_orders": 0,
            "status":            "hold",
        }
        log_outputs(decision, pd.Series(dtype=float), positions, pd.DataFrame(), pd.DataFrame(), account_info)
        return decision

    order_plan = build_order_plan(target.copy(), positions, account_info["equity"], today_close)
    print(f"      Planned orders: {len(order_plan)}")
    if not order_plan.empty:
        print(f"\n{order_plan.head(15).to_string(index=False)}\n")

    # 6. Submit (or skip)
    submitted = pd.DataFrame()
    if SUBMIT_ORDERS and client is not None and not order_plan.empty and not dry_run:
        print(f"\n[6/6] Submitting {len(order_plan)} orders to Alpaca paper…")
        submitted = submit_orders(order_plan, client)
    else:
        print(f"\n[6/6] Skipping order submission "
              f"(SUBMIT_ORDERS={SUBMIT_ORDERS}, dry_run={dry_run}, n_orders={len(order_plan)}).")

    # Log
    decision = {
        "action":            "rebalance",
        "do_rebalance":      True,
        "submit_orders":     SUBMIT_ORDERS and client is not None and not dry_run,
        "n_target_positions": int(len(target)),
        "n_planned_orders":   int(len(order_plan)),
        "n_submitted_orders": int(len(submitted)),
        "gross_exposure":     float(target.abs().sum()),
        "long_exposure":      float(target[target > 0].sum()),
        "short_exposure":     float(target[target < 0].sum()),
        "status":             "ok",
    }
    log_outputs(decision, target, positions, order_plan, submitted, account_info)

    print(f"\n  ✅ Cycle complete. Logs → {LOG_DIR}")
    return decision


# ════════════════════════════════════════════════════════════════════════
# CLI entrypoint
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Model B paper trader (MLP_alpha__130_30)")
    parser.add_argument("--force-rebalance", action="store_true",
                        help="Force a rebalance regardless of schedule.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't connect to Alpaca; use defaults instead.")
    args = parser.parse_args()
    run_trading_cycle(force_rebalance=args.force_rebalance, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
