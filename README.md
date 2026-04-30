# Model B — MLP_alpha__130_30 paper trader

Live paper trader for the **MLP_alpha** strategy in `130/30` mode, deployed against an Alpaca paper account.

## Strategy (in one paragraph)

Every 21 trading days, score 300 large-cap US stocks with a multilayer perceptron trained to predict 21-day forward excess return over the strongest of {SPY, top-300 EW, RSP, VTI, QQQ}. Within each sector, take the top 20% long (1.30× sector weight) and bottom 20% short (0.30× sector weight); cap individual positions at 5% per side; hold for 21 trading days and rebalance.

Performance summary (from training notebook, validated through 2026-04-14 rebalance):

|              | Sharpe | Annual return | Max drawdown |
|--------------|-------:|--------------:|-------------:|
| Test (2025-10-01 →) | **1.89** | **34.0%** | **−8.7%** |
| Full period (2015–2026) | 2.60   | (cum 396%)    | −31%         |

## Repo layout

```
model-b-paper-trading/
├─ paper_trader.py            ← single-file engine
├─ artifacts/mlp.joblib       ← fitted MLP + universe + sectors + config
├─ requirements.txt
├─ .github/workflows/daily_trader.yml   ← runs every weekday at 14:00 UTC
└─ logs/                      ← committed back by the workflow each run
   ├─ decisions/              latest_decision.csv + per-run timestamps
   ├─ orders/                 order plan + submitted (per run)
   ├─ positions/              snapshot from Alpaca
   ├─ portfolio/              equity_history.csv (one row per run)
   ├─ target_weights/         the basket the engine wanted
   └─ health/                 health_history.csv
```

## Setup

You must have an **Alpaca paper account separate from any other paper trader you run** — the engine will rebalance the entire account to its target basket each cycle.

1. Create the account: https://app.alpaca.markets/paper/dashboard/overview → "Generate New Keys"
2. Copy the API key and secret.
3. In this repo on GitHub: `Settings → Secrets and variables → Actions → New repository secret`. Add:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
4. (Optional) Trigger the workflow manually under the Actions tab → `Daily paper trader (Model B)` → `Run workflow` to verify everything works end-to-end before relying on the daily cron.

## Local development / smoke test

```bash
pip install -r requirements.txt
python paper_trader.py --dry-run
```

Dry-run skips Alpaca and uses a $100k notional account so you can verify feature building and basket construction without credentials.

To force a real rebalance (e.g. on first deployment when there's no decision history), run with `--force-rebalance`.

## How rebalancing is decided

`paper_trader.py` reads `logs/decisions/latest_decision.csv` to find the last rebalance date. If at least 21 trading days (~30 calendar days) have passed, it rebalances; otherwise it logs a snapshot and exits. The first run always rebalances.

## Updating the model

The MLP in `artifacts/mlp.joblib` was saved by Cell 7 of the training notebook (`model_0002__1_.py`) at the end of the latest training cycle. To deploy a refresh:

1. Re-run the notebook through Cell 7 (the cell saves a fresh `mlp.joblib` plus historical returns CSV to your `BEST_MODEL/` folder on Drive).
2. Replace `artifacts/mlp.joblib` in this repo with the new one.
3. Commit and push. The next workflow run picks it up.

## Tuning knobs

All controlled by environment variables (set in the workflow file or your local shell):

| Variable | Default | Meaning |
|---|---|---|
| `SUBMIT_ORDERS` | `true` | If `false`, the engine plans + logs but doesn't submit orders. |
| `DEFAULT_ACCOUNT_VALUE` | `100000` | Used only for dry-runs and Alpaca-failure fallback. |
| `REBALANCE_THRESHOLD` | `0.005` | Minimum |target − current weight| to trigger a trade (0.5%). |
| `MIN_TRADE_DOLLARS` | `100` | Drop trades smaller than this. |
| `DATA_PERIOD` | `3y` | yfinance history lookback. The Ichimoku weekly features need ≳ 1 year. |

## What this is *not*

- This is **not** the BR-PPO V10 strategy — that's `model_a` deployed in `Base_Model_BR_PPO`. This is a separate, simpler **score-map** strategy that beat PPO on the test split (Sharpe 1.89 vs 1.69 in the same period).
- This is **paper trading only.** The Alpaca client is hard-coded to `paper=True`.
- This is research code. Past performance does not predict future returns. Don't run it on a live account.
