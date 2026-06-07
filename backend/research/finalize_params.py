"""
Confirm and persist the robust 15-min breakout config (true band cross 1.0/0.0,
no junk filters, target/stop from the validated plateau). Prints full-period +
monthly, then writes optimized_params.json.
"""
import sys, json
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")
import pandas as pd
from indicators import compute_all, resample_ohlc
from backtester.engine import run_backtest
from backtester.walk_forward import walk_forward, summarise_walk_forward
from config import OPTIMIZED_PARAMS_PATH

PARAMS = {
    "strategy": "momentum_breakout",
    "timeframe_min": 15,
    "bb_oversold": 0.0,      # downside break: close at/below lower band
    "bb_overbought": 1.0,    # upside break:  close at/above upper band
    "bb_exit": 3.0,          # target = 3.0 x ATR (let momentum run)
    "sl_buffer": 1.5,        # stop   = 1.5 x ATR (2:1 reward:risk)
    "rsi_min": 0,
    "rsi_max": 100,
    "min_atr_pct": 0.0,
}

raw = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                  index_col=0, parse_dates=True)
df15 = compute_all(resample_ohlc(raw, 15))

trades, daily, m = run_backtest(df15, params=PARAMS)
daily.index = pd.to_datetime(daily.index)
monthly = daily.groupby(daily.index.to_period("M")).sum()

print("=== FULL PERIOD (15-min breakout, cross 1.0/0.0, tgt 3.0 / stop 1.5) ===")
print(f"  Trades   : {m['total_trades']}   Trades/wk: {m['trades_per_week']:.1f}")
print(f"  Win rate : {m['win_rate']:.1%}    Profit factor: {m['profit_factor']:.2f}")
print(f"  Total P&L: ₹{m['total_pnl']:,.0f}   Sharpe: {m['sharpe']:.2f}")
print(f"  Max DD   : ₹{m['max_drawdown_inr']:,.0f}")
print(f"  Avg win  : ₹{m['avg_winner']:,.0f}   Avg loss: ₹{m['avg_loser']:,.0f}")
print(f"\n  Monthly P&L (delta-proxy):")
for per, val in monthly.items():
    print(f"    {per}: ₹{val:,.0f}")
print(f"  Months positive: {(monthly>0).mean():.0%} ({int((monthly>0).sum())}/{len(monthly)})")

# Walk-forward with the (now profit-based) objective summary for the record
wins, _ = walk_forward(df15, params=PARAMS)
s = summarise_walk_forward(wins)
print(f"\n  Walk-forward OOS: total ₹{s['total_oos_pnl']:,.0f}  "
      f"profitable_windows={s['pct_profitable_windows']:.0%}  win={s['avg_oos_win_rate']:.0%}")

out = dict(PARAMS)
out["_meta"] = {
    "strategy": "momentum_breakout",
    "timeframe_min": 15,
    "source": "robustness_grid (research/pick_breakout_config.py) — chosen for "
              "consistency across target/stop plateau at the true band cross; "
              "optimizer overfits this small (~140-event) dataset.",
    "full_period": {"trades": int(m["total_trades"]), "win_rate": round(m["win_rate"], 3),
                    "total_pnl": round(m["total_pnl"], 0), "sharpe": round(m["sharpe"], 2),
                    "profit_factor": round(m["profit_factor"], 2),
                    "max_dd": round(m["max_drawdown_inr"], 0),
                    "months_positive": round(float((monthly > 0).mean()), 2)},
    "walk_forward": {k: s[k] for k in ("total_oos_pnl", "pct_profitable_windows", "avg_oos_win_rate")},
    "note": "Delta-proxy P&L on the underlying (no theta). Live now buys the "
            "nearest-ATM strike that fits CAPITAL_PER_TRADE (default ₹18k → ~1 ATM "
            "lot) at 4-12 DTE. ATM chosen because the breakout edge survives a "
            "realistic ATM bid-ask (PF ~1.18, research/revalidate_model.py) whereas "
            "far-OTM under a ₹5k cap does not (PF ~0.95). Large vega: P&L is "
            "sensitive to intraday IV drift.",
}
with open(OPTIMIZED_PARAMS_PATH, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n✓ Wrote {OPTIMIZED_PARAMS_PATH}")
