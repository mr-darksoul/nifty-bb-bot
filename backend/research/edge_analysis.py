"""
Edge research for a LONG-OPTION (buy CE/PE) BB %b strategy.

Question 1: Do %b extremes predict forward NIFTY moves at all?
Question 2: Mean-reversion or momentum? (fade the extreme vs ride it)
Question 3: What holding horizon and target/stop (in ATR) gives positive
            expectancy on the UNDERLYING, conditioned on trend/time/vol?

We measure on the underlying (spot). A long-option buyer needs a fast, large
favorable move; so we look at signed forward drift and the MFE/MAE profile.
"""
import sys
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")

import numpy as np
import pandas as pd
from indicators import compute_all

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                 index_col=0, parse_dates=True)
df = compute_all(df)
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
df["trend_up"] = df["close"] > df["ema50"]
df["minute_of_day"] = df.index.hour * 60 + df.index.minute
df = df.dropna(subset=["percent_b", "atr", "rsi"]).copy()

n = len(df)
close = df["close"].values
atr = df["atr"].values
pb = df["percent_b"].values
print(f"Bars: {n}  range {df.index[0]} .. {df.index[-1]}")

# ── Q1+Q2: forward drift conditioned on %b bucket, at several horizons ────────
HORIZONS = [5, 10, 15, 30]
print("\n=== Forward SIGNED return (in ATR units) by %b bucket ===")
print("(positive = price rose. For CE-buy edge we want the oversold buckets to rise;")
print(" for momentum we'd want overbought buckets to keep rising.)")
buckets = [(-0.01, 0.05), (0.05, 0.20), (0.20, 0.80), (0.80, 0.95), (0.95, 1.01)]
rows = []
for lo, hi in buckets:
    mask = (pb >= lo) & (pb < hi)
    idxs = np.where(mask)[0]
    rec = {"pb_bucket": f"[{lo:.2f},{hi:.2f})", "n": len(idxs)}
    for h in HORIZONS:
        valid = idxs[idxs + h < n]
        if len(valid) == 0:
            rec[f"fwd{h}"] = np.nan
            continue
        fwd = (close[valid + h] - close[valid]) / atr[valid]
        rec[f"fwd{h}"] = round(float(np.mean(fwd)), 3)
    rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))

# ── Q3: MFE / MAE profile for the two extreme buckets, horizon 30 ────────────
print("\n=== MFE / MAE over next 30 min (in ATR units), by extreme bucket ===")
H = 30
def excursion_stats(idxs):
    mfe, mae = [], []
    for i in idxs:
        if i + H >= n:
            continue
        path = close[i + 1: i + 1 + H] - close[i]
        mfe.append(path.max() / atr[i])
        mae.append(path.min() / atr[i])
    return np.array(mfe), np.array(mae)

for label, (lo, hi) in [("OVERSOLD %b<0.05", (-0.01, 0.05)),
                        ("OVERBOUGHT %b>=0.95", (0.95, 1.01))]:
    idxs = np.where((pb >= lo) & (pb < hi))[0]
    mfe, mae = excursion_stats(idxs)
    print(f"\n{label}  (n={len(mfe)})")
    print(f"  Up-move  (MFE): mean={mfe.mean():.2f}  median={np.median(mfe):.2f}  p75={np.percentile(mfe,75):.2f}")
    print(f"  Down-move(MAE): mean={mae.mean():.2f}  median={np.median(mae):.2f}  p25={np.percentile(mae,25):.2f}")
    # symmetric: is favorable excursion bigger for CE (up) or PE (down)?
    print(f"  |up|-|down| favorable skew: {mfe.mean()+mae.mean():.2f}  (>0 => upward bias)")

# ── Q3b: simple expectancy test — fade vs ride, with ATR target/stop ─────────
print("\n=== Expectancy per signal (ATR units, net of ~0.15 ATR round-trip cost) ===")
print("Compares MEAN-REVERSION (fade) vs MOMENTUM (ride) for each extreme,")
print("first-touch target/stop over 30 min. Cost ~0.15 ATR ~ slippage+brokerage.")
COST = 0.15
def first_touch_expectancy(idxs, direction, target_atr, stop_atr):
    """direction +1 = bet price up (buy CE), -1 = bet down (buy PE)."""
    wins = pnl = trades = 0
    for i in idxs:
        if i + H >= n:
            continue
        entry = close[i]; a = atr[i]
        tgt = entry + direction * target_atr * a
        stp = entry - direction * stop_atr * a
        outcome = -stop_atr  # default: time-stop near stop side (approx)
        for j in range(i + 1, i + 1 + H):
            if direction == 1:
                if close[j] >= tgt: outcome = target_atr; break
                if close[j] <= stp: outcome = -stop_atr; break
            else:
                if close[j] <= tgt: outcome = target_atr; break
                if close[j] >= stp: outcome = -stop_atr; break
        else:
            outcome = direction * (close[i + H] - entry) / a  # close at horizon
        pnl += outcome - COST
        wins += outcome > 0
        trades += 1
    if trades == 0:
        return None
    return dict(trades=trades, win_rate=round(wins/trades, 3),
                exp_atr=round(pnl/trades, 3))

TGT, STP = 1.5, 1.0
os_idx = np.where(pb < 0.05)[0]
ob_idx = np.where(pb >= 0.95)[0]
print(f"\ntarget={TGT} ATR  stop={STP} ATR")
print(f"OVERSOLD  -> MEAN-REV buy CE (bet up):   {first_touch_expectancy(os_idx, +1, TGT, STP)}")
print(f"OVERSOLD  -> MOMENTUM buy PE (bet down):  {first_touch_expectancy(os_idx, -1, TGT, STP)}")
print(f"OVERBOUGHT-> MEAN-REV buy PE (bet down):  {first_touch_expectancy(ob_idx, -1, TGT, STP)}")
print(f"OVERBOUGHT-> MOMENTUM buy CE (bet up):    {first_touch_expectancy(ob_idx, +1, TGT, STP)}")

# ── Q4: does TREND ALIGNMENT help? (only fade with trend, etc.) ──────────────
print("\n=== Mean-reversion expectancy conditioned on EMA50 trend ===")
trend_up = df["trend_up"].values
for label, idx, direction in [("OVERSOLD buy CE", os_idx, +1),
                              ("OVERBOUGHT buy PE", ob_idx, -1)]:
    with_trend = idx[trend_up[idx] == (direction == 1)]   # CE in uptrend / PE in downtrend
    against    = idx[trend_up[idx] != (direction == 1)]
    print(f"\n{label}")
    print(f"  WITH trend:    {first_touch_expectancy(with_trend, direction, TGT, STP)}")
    print(f"  AGAINST trend: {first_touch_expectancy(against, direction, TGT, STP)}")
