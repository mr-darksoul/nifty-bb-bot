"""
Edge hunt v2 — where (if anywhere) does a LONG-OPTION %b signal have edge?

Tests, for buying CE/PE with an ATR target/stop over a fixed horizon:
  A) Timeframe: 1, 3, 5, 15-min resampled bars.
  B) Signal type:
       - AT_EXTREME      : %b beyond band (current strategy)
       - BAND_BREAKOUT   : %b crosses band outward (momentum)
       - REVERSAL_CONFIRM: %b crosses back inward from extreme (bounce)
  C) Time-of-day slice (first hour vs rest).
Expectancy is in ATR units net of a 0.15-ATR round-trip cost floor.
"""
import sys
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")
import numpy as np
import pandas as pd
from indicators import compute_all

COST = 0.15
raw = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                  index_col=0, parse_dates=True)

def resample(df1, rule):
    if rule == "1min":
        return df1.copy()
    o = df1.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                "close": "last", "volume": "sum"}).dropna()
    return o

def expectancy(close, atr, signal_idx, direction_arr, horizon, tgt, stp):
    n = len(close); pnl = wins = trades = 0
    for k, i in enumerate(signal_idx):
        if i + horizon >= n:
            continue
        d = direction_arr[k]; entry = close[i]; a = atr[i]
        if a <= 0:
            continue
        T = entry + d * tgt * a; S = entry - d * stp * a
        out = None
        for j in range(i + 1, i + 1 + horizon):
            if d == 1:
                if close[j] >= T: out = tgt; break
                if close[j] <= S: out = -stp; break
            else:
                if close[j] <= T: out = tgt; break
                if close[j] >= S: out = -stp; break
        if out is None:
            out = d * (close[i + horizon] - entry) / a
        pnl += out - COST; wins += out > 0; trades += 1
    if trades < 30:
        return None
    return dict(n=trades, win=round(wins/trades, 3), exp=round(pnl/trades, 3))

TGT, STP = 1.5, 1.0
HOR = {"1min": 30, "3min": 15, "5min": 12, "15min": 8}   # ~30-90 min real time

for rule in ["1min", "3min", "5min", "15min"]:
    d = compute_all(resample(raw, rule))
    d["mins"] = d.index.hour * 60 + d.index.minute
    d = d.dropna(subset=["percent_b", "atr"]).copy()
    pb = d["percent_b"].values; close = d["close"].values; atr = d["atr"].values
    mins = d["mins"].values
    pb_prev = np.roll(pb, 1); pb_prev[0] = pb[0]
    h = HOR[rule]
    print(f"\n################  TIMEFRAME {rule}  (horizon={h} bars)  n={len(d)}  ################")

    # AT_EXTREME (fade): oversold->CE, overbought->PE
    os_ = np.where(pb < 0.05)[0]; ob_ = np.where(pb >= 0.95)[0]
    sig = np.concatenate([os_, ob_]); dirn = np.concatenate([np.ones(len(os_)), -np.ones(len(ob_))])
    print(f"  AT_EXTREME  fade  : {expectancy(close, atr, sig, dirn, h, TGT, STP)}")
    print(f"  AT_EXTREME  ride  : {expectancy(close, atr, sig, -dirn, h, TGT, STP)}")

    # BAND_BREAKOUT (momentum): %b crosses ABOVE 1.0 -> CE ; below 0.0 -> PE
    up_break = np.where((pb_prev <= 1.0) & (pb > 1.0))[0]
    dn_break = np.where((pb_prev >= 0.0) & (pb < 0.0))[0]
    sig = np.concatenate([up_break, dn_break]); dirn = np.concatenate([np.ones(len(up_break)), -np.ones(len(dn_break))])
    print(f"  BAND_BREAKOUT ride: {expectancy(close, atr, sig, dirn, h, TGT, STP)}  (up={len(up_break)} dn={len(dn_break)})")

    # REVERSAL_CONFIRM (bounce): %b crosses back ABOVE 0.05 -> CE ; back below 0.95 -> PE
    up_rev = np.where((pb_prev < 0.05) & (pb >= 0.05))[0]
    dn_rev = np.where((pb_prev > 0.95) & (pb <= 0.95))[0]
    sig = np.concatenate([up_rev, dn_rev]); dirn = np.concatenate([np.ones(len(up_rev)), -np.ones(len(dn_rev))])
    print(f"  REVERSAL_CONFIRM  : {expectancy(close, atr, sig, dirn, h, TGT, STP)}  (up={len(up_rev)} dn={len(dn_rev)})")

    # Time-of-day: AT_EXTREME fade, first hour vs rest
    first_hr = (mins >= 9*60+15) & (mins <= 10*60+15)
    os_f = os_[first_hr[os_]]; ob_f = ob_[first_hr[ob_]]
    sigf = np.concatenate([os_f, ob_f]); dirf = np.concatenate([np.ones(len(os_f)), -np.ones(len(ob_f))])
    os_r = os_[~first_hr[os_]]; ob_r = ob_[~first_hr[ob_]]
    sigr = np.concatenate([os_r, ob_r]); dirr = np.concatenate([np.ones(len(os_r)), -np.ones(len(ob_r))])
    print(f"  AT_EXTREME fade  first-hour : {expectancy(close, atr, sigf, dirf, h, TGT, STP)}")
    print(f"  AT_EXTREME fade  rest-of-day: {expectancy(close, atr, sigr, dirr, h, TGT, STP)}")
