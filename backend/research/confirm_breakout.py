"""
Confirm the 15-min Bollinger BAND-BREAKOUT momentum edge for buying options.

1) Robustness: expectancy by calendar month (is it persistent or one lucky run?).
2) Target/stop/horizon grid.
3) Theta haircut: re-price the edge as a long ATM option, charging realistic
   per-minute decay, to see if the UNDERLYING edge survives as OPTION P&L.
"""
import sys
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")
import numpy as np
import pandas as pd
from indicators import compute_all

raw = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                  index_col=0, parse_dates=True)
d = raw.resample("15min").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()
d = compute_all(d)
d = d.dropna(subset=["percent_b", "atr"]).copy()
pb = d["percent_b"].values; close = d["close"].values; atr = d["atr"].values
pb_prev = np.roll(pb, 1); pb_prev[0] = pb[0]
idx = d.index
n = len(d)

# Band-breakout signals: %b crosses above 1.0 -> long(+1); below 0 -> short(-1)
up = np.where((pb_prev <= 1.0) & (pb > 1.0))[0]
dn = np.where((pb_prev >= 0.0) & (pb < 0.0))[0]
sig = np.concatenate([up, dn])
dirn = np.concatenate([np.ones(len(up)), -np.ones(len(dn))]).astype(int)
order = np.argsort(sig); sig = sig[order]; dirn = dirn[order]
print(f"15-min bars={n}  breakout signals={len(sig)} (up={len(up)} dn={len(dn)})")

COST = 0.15

def run(tgt, stp, hor, theta_per_bar=0.0):
    """Return list of (timestamp, outcome_atr_net) for each signal.
    theta_per_bar: ATR-units charged per bar held (option decay proxy)."""
    res = []
    for i, d0 in zip(sig, dirn):
        if i + hor >= n:
            continue
        entry = close[i]; a = atr[i]
        if a <= 0:
            continue
        T = entry + d0 * tgt * a; S = entry - d0 * stp * a
        out = None; held = hor
        for j in range(i + 1, i + 1 + hor):
            if d0 == 1:
                if close[j] >= T: out = tgt; held = j - i; break
                if close[j] <= S: out = -stp; held = j - i; break
            else:
                if close[j] <= T: out = tgt; held = j - i; break
                if close[j] >= S: out = -stp; held = j - i; break
        if out is None:
            out = d0 * (close[i + hor] - entry) / a
        res.append((idx[i], out - COST - theta_per_bar * held))
    return res

# ── 1) Robustness by month (base target/stop/horizon) ────────────────────────
TGT, STP, HOR = 2.0, 1.0, 8
res = run(TGT, STP, HOR)
s = pd.Series({t: v for t, v in res})
by_month = s.groupby(s.index.to_period("M")).agg(["count", "mean"])
by_month["mean"] = by_month["mean"].round(3)
print(f"\n=== Monthly expectancy (ATR units, net cost), tgt={TGT} stp={STP} hor={HOR} ===")
print(by_month.to_string())
_win = (s > 0).mean()
_mpos = (by_month["mean"] > 0).mean()
print(f"  OVERALL: n={len(s)} mean_exp={s.mean():.3f}  win={_win:.3f}  months_positive={_mpos:.0%}")

# ── 2) Target/stop/horizon grid ──────────────────────────────────────────────
print("\n=== Grid: mean expectancy (ATR net), n in parens ===")
print(f"{'tgt/stp':>10} | " + " ".join(f"hor={h:<2}" for h in [4, 6, 8, 12]))
for tgt, stp in [(1.5, 1.0), (2.0, 1.0), (2.5, 1.0), (2.0, 1.5), (3.0, 1.5), (2.5, 2.0)]:
    cells = []
    for h in [4, 6, 8, 12]:
        r = run(tgt, stp, h)
        v = np.mean([x[1] for x in r]) if r else float("nan")
        cells.append(f"{v:+.3f}")
    print(f"{tgt:>4}/{stp:<4} | " + "   ".join(cells))

# ── 3) Theta haircut: does it survive as a long ATM option? ──────────────────
# ATM 0-DTE/weekly theta proxy: a bar held costs ~theta. Express in ATR units.
# Rough: ATM weekly option theta burns a meaningful fraction of premium/day.
# We sweep theta_per_bar to find the breakeven decay the edge can tolerate.
print("\n=== Theta tolerance (tgt=2.0 stp=1.0 hor=8) ===")
print("theta_per_bar(ATR) | net_exp(ATR) | win")
for theta in [0.0, 0.05, 0.10, 0.15, 0.20]:
    r = run(TGT, STP, HOR, theta_per_bar=theta)
    vals = np.array([x[1] for x in r])
    print(f"   {theta:>5.2f}          |   {vals.mean():+.3f}    | {(vals>0).mean():.3f}")
