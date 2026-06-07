"""
Pick a ROBUST 15-min breakout config by monthly consistency, not a noisy
5-window Sharpe average. Cross levels fixed at the true band (1.0 / 0.0);
no RSI/ATR junk filters. Only target/stop (and optionally a mild ATR gate)
vary. Robustness = total P&L + fraction of profitable months.
"""
import sys
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")
import numpy as np
import pandas as pd
from indicators import compute_all, resample_ohlc
from backtester.engine import run_backtest

raw = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                  index_col=0, parse_dates=True)
df15 = compute_all(resample_ohlc(raw, 15))

BASE = {"strategy": "momentum_breakout", "timeframe_min": 15,
        "bb_oversold": 0.0, "bb_overbought": 1.0,
        "rsi_min": 0, "rsi_max": 100}

def evaluate(tgt, stp, min_atr_pct=0.0):
    p = {**BASE, "bb_exit": tgt, "sl_buffer": stp, "min_atr_pct": min_atr_pct}
    trades, daily, m = run_backtest(df15, params=p)
    if trades.empty:
        return None
    daily.index = pd.to_datetime(daily.index)
    monthly = daily.groupby(daily.index.to_period("M")).sum()
    return dict(
        tgt=tgt, stp=stp, atr=min_atr_pct, n=m["total_trades"],
        win=m["win_rate"], pnl=m["total_pnl"], sharpe=m["sharpe"],
        pf=m["profit_factor"], maxdd=m["max_drawdown_inr"],
        mom_pos=(monthly > 0).mean(), n_months=len(monthly),
    )

rows = []
for tgt, stp in [(1.5,1.0),(2.0,1.0),(2.5,1.0),(3.0,1.0),(2.0,1.5),(3.0,1.5),(2.5,2.0),(4.0,1.5)]:
    for atr in [0.0, 50.0]:
        r = evaluate(tgt, stp, atr)
        if r:
            rows.append(r)

res = pd.DataFrame(rows)
res = res.sort_values(["mom_pos", "pnl"], ascending=False)
pd.set_option("display.width", 160)
print("\n=== 15-min breakout configs (cross 1.0/0.0, no RSI filter) ===")
print("ranked by % profitable months, then total P&L\n")
show = res.copy()
show["win"] = (show["win"]*100).round(0)
show["mom_pos"] = (show["mom_pos"]*100).round(0)
show["pnl"] = show["pnl"].round(0)
show["sharpe"] = show["sharpe"].round(2)
show["pf"] = show["pf"].round(2)
show["maxdd"] = show["maxdd"].round(0)
print(show[["tgt","stp","atr","n","win","pnl","sharpe","pf","maxdd","mom_pos"]].to_string(index=False))

# Best by consistency
best = res.iloc[0]
print(f"\nMost consistent: tgt={best['tgt']} stp={best['stp']} atr_gate={best['atr']}  "
      f"P&L=₹{best['pnl']:,.0f}  win={best['win']:.0%}  months_pos={best['mom_pos']:.0%}  n={best['n']}")
