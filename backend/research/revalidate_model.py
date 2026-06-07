"""
Full-history, theta-aware re-validation of the momentum_breakout strategy on the
ACTUAL instrument the bot trades (nearest-OTM strike under the ~Rs5k premium cap).

Kite only serves history for currently-listed contracts, so a real-premium
backtest (research/revalidate_otm.py) covers ~6 trades — too few to tune on.
Here we instead:
  1. Calibrate a Black-Scholes implied vol to the handful of REAL priced trades
     (so the model reproduces observed weekly-option premiums), then
  2. Re-price the FULL signal history as premium-capped OTM long options with
     real theta (T shrinks over the hold), and
  3. Sweep target/stop and DTE to see whether ANY config has positive option
     expectancy at this budget — reporting IV sensitivity, since IV drives it.

This is the honest answer to "does the edge survive option costs at Rs5k?".
Run:  python research/revalidate_model.py
"""
import sys
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")

import json
from math import log, sqrt, erf

import numpy as np
import pandas as pd

from indicators import compute_all, resample_ohlc
from config import (
    BROKERAGE_PER_ORDER, CAPITAL_PER_TRADE, LOT_SIZE, SLIPPAGE_PCT,
    OPTIMIZED_PARAMS_PATH,
)

YEAR_MIN = 365 * 24 * 60.0   # minutes per year (calendar basis, matches DTE)


def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S, K, T, sigma, cp):
    """Black-Scholes (r=0). T in years. cp in {'C','P'}."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if cp == "C":
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def implied_vol(target, S, K, T, cp):
    """Bisection IV. None if unpriceable."""
    if target <= 0 or T <= 0:
        return None
    lo, hi = 0.01, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, cp) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ── Load data + signals ──────────────────────────────────────────────────────
raw = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                  index_col=0, parse_dates=True)
df = compute_all(resample_ohlc(raw, 15)).dropna(subset=["percent_b", "atr"])
df.index = pd.to_datetime(df.index)
idx = df.index
pb = df["percent_b"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
atr = df["atr"].values
pbp = np.roll(pb, 1); pbp[0] = pb[0]
n = len(df)

DTE = 6          # calendar days to expiry the bot targets (within 4-12 window)


def signals_and_exits(tgt_mult, stp_mult):
    """Replay engine logic (intrabar exits, 15:10 force) -> list of trades."""
    trades = []
    in_t = False
    tr = None
    for i in range(n):
        ts = idx[i]; hr, mn = ts.hour, ts.minute
        if in_t and ((hr == 15 and mn >= 10) or hr > 15):
            tr.update(exit_i=i, exit_px=c[i], reason="FORCE"); trades.append(tr); in_t = False; continue
        if in_t:
            d = tr["d"]
            if d == 1:
                if l[i] <= tr["stp"]: tr.update(exit_i=i, exit_px=tr["stp"], reason="SL"); trades.append(tr); in_t = False; continue
                if h[i] >= tr["tgt"]: tr.update(exit_i=i, exit_px=tr["tgt"], reason="TGT"); trades.append(tr); in_t = False; continue
            else:
                if h[i] >= tr["stp"]: tr.update(exit_i=i, exit_px=tr["stp"], reason="SL"); trades.append(tr); in_t = False; continue
                if l[i] <= tr["tgt"]: tr.update(exit_i=i, exit_px=tr["tgt"], reason="TGT"); trades.append(tr); in_t = False; continue
        if not in_t:
            if hr < 9 or (hr == 9 and mn < 35) or (hr == 15 and mn >= 10) or hr > 15:
                continue
            d = 0
            if pbp[i] <= 1.0 and pb[i] > 1.0: d = 1
            elif pbp[i] >= 0.0 and pb[i] < 0.0: d = -1
            if d != 0:
                ep = c[i]
                tr = dict(d=d, ei=i, ep=ep, tgt=ep + d * tgt_mult * atr[i],
                          stp=ep - d * stp_mult * atr[i])
                in_t = True
    if in_t:
        tr.update(exit_i=n - 1, exit_px=c[n - 1], reason="EOD"); trades.append(tr)
    return trades


def option_pnl(trades, sigma, dte=DTE):
    """Price each trade as a premium-capped OTM long option with real theta."""
    out = []
    for tr in trades:
        cp = "C" if tr["d"] == 1 else "P"
        S0 = tr["ep"]; S1 = tr["exit_px"]
        T0 = dte / 365.0
        held_min = (tr["exit_i"] - tr["ei"]) * 15
        T1 = max(T0 - held_min / YEAR_MIN, 1e-6)
        atm = round(S0 / 50) * 50
        # nearest OTM strike whose 1-lot premium fits the cap
        step = 50 if cp == "C" else -50
        K = atm
        e = None
        for _ in range(60):
            prem = bs_price(S0, K, T0, sigma, cp)
            if prem > 0 and prem * LOT_SIZE * (1 + SLIPPAGE_PCT) <= CAPITAL_PER_TRADE:
                e = prem; break
            K += step
        if e is None or e <= 0:
            continue
        x = bs_price(S1, K, T1, sigma, cp)
        entry_fill = e * (1 + SLIPPAGE_PCT)
        exit_fill = x * (1 - SLIPPAGE_PCT)
        qty = int(CAPITAL_PER_TRADE / entry_fill / LOT_SIZE) * LOT_SIZE
        if qty <= 0:
            continue
        pnl = (exit_fill - entry_fill) * qty - 2 * BROKERAGE_PER_ORDER
        out.append(pnl)
    return np.array(out)


def summarize(pnl):
    if len(pnl) == 0:
        return "  (no priceable trades)"
    gw = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return (f"n={len(pnl):3d}  win={ (pnl>0).mean():5.1%}  "
            f"total=Rs{pnl.sum():>9,.0f}  avg=Rs{pnl.mean():>6,.0f}  PF={pf:.2f}")


# ── 1. Calibrate IV to the real priced trades ────────────────────────────────
print("=== IV calibration against real Kite premiums ===")
real = [   # from research/revalidate_otm.py output (spot recovered below)
    ("2026-05-29 12:00:00", "PE", 23500, 11, 69.00),
    ("2026-06-01 14:00:00", "PE", 23050, 8, 71.30),
    ("2026-06-02 12:45:00", "CE", 23800, 7, 71.10),
    ("2026-06-03 14:00:00", "CE", 23800, 6, 68.80),
    ("2026-06-04 13:00:00", "PE", 23150, 5, 64.25),
    ("2026-06-05 13:00:00", "PE", 23200, 4, 68.40),
]
ivs = []
spot_at = df["close"]
for tstr, opt, K, dte, prem in real:
    ts = pd.Timestamp(tstr)
    near = spot_at.index[spot_at.index.get_indexer([ts.tz_localize(spot_at.index.tz)], method="nearest")[0]]
    S = float(spot_at.loc[near])
    iv = implied_vol(prem, S, K, dte / 365.0, "C" if opt == "C" or opt == "CE" else "P")
    if iv:
        ivs.append(iv)
        print(f"  {tstr}  {opt} K={K} S={S:.0f} DTE={dte} prem=Rs{prem}  -> IV={iv:.1%}")
iv_cal = float(np.median(ivs)) if ivs else 0.13
print(f"  Calibrated IV (median): {iv_cal:.1%}")

# ── 2. Full-history option P&L at the live config (tgt 3.0 / stop 1.5) ────────
params = json.load(open(OPTIMIZED_PARAMS_PATH))
base = signals_and_exits(params["bb_exit"], params["sl_buffer"])
print(f"\n=== Full-history OPTION P&L (premium-capped OTM, theta, DTE={DTE}) ===")
print(f"  live config tgt={params['bb_exit']} stop={params['sl_buffer']}:")
for iv in (iv_cal - 0.03, iv_cal, iv_cal + 0.03):
    print(f"    IV={iv:5.1%}  " + summarize(option_pnl(base, iv)))

# ── 3. Target/stop sweep at calibrated IV ────────────────────────────────────
print(f"\n=== Target/stop sweep on OPTION P&L (IV={iv_cal:.1%}, DTE={DTE}) ===")
print(f"  {'tgt/stop':>10} | option P&L")
for tgt, stp in [(1.5, 1.0), (2.0, 1.0), (2.5, 1.0), (3.0, 1.5),
                 (2.0, 1.5), (4.0, 2.0), (2.5, 2.0)]:
    trs = signals_and_exits(tgt, stp)
    print(f"  {tgt:>4}/{stp:<4} | " + summarize(option_pnl(trs, iv_cal)))
