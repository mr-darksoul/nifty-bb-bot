"""
Validate the 15-min Bollinger BAND-BREAKOUT momentum signal against REAL ATM
option premiums from Kite, comparing current-week (low DTE, high theta) vs
next-week (higher DTE, gentler theta) expiries.

Entry  = breakout 15-min bar. Direction = breakout side (CE up / PE down).
Exit   = first 15-min bar to touch target/stop on the UNDERLYING, else horizon.
Option = ATM strike at entry; priced from real 1-min option history at the
         entry and exit timestamps. P&L net of slippage + 2x brokerage.
"""
import sys, time
sys.path.insert(0, "/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend")
import numpy as np
import pandas as pd
from kiteconnect import KiteConnect
from indicators import compute_all
from config import (
    LOT_SIZE, BROKERAGE_PER_ORDER, SLIPPAGE_PCT,
    KITE_API_KEY, KITE_ACCESS_TOKEN,
)

TGT, STP, HOR = 2.5, 1.0, 8         # 15-min bars
RATE_SLEEP = 0.35
TOL = pd.Timedelta(minutes=20)

kite = KiteConnect(api_key=KITE_API_KEY); kite.set_access_token(KITE_ACCESS_TOKEN)
inst = pd.DataFrame(kite.instruments("NFO"))
nifty = inst[inst["name"] == "NIFTY"].copy()
nifty["expiry"] = pd.to_datetime(nifty["expiry"]).dt.date
listed_expiries = sorted(nifty["expiry"].unique())

# ── Underlying 15-min breakout signals ───────────────────────────────────────
raw = pd.read_csv("/Users/manavbansal/Documents/Alag/nifty-bb-bot/backend/nifty_1min.csv",
                  index_col=0, parse_dates=True)
d = raw.resample("15min").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()
d = compute_all(d).dropna(subset=["percent_b", "atr"])
pb = d["percent_b"].values; close = d["close"].values; atr = d["atr"].values
pb_prev = np.roll(pb, 1); pb_prev[0] = pb[0]; idx = d.index; n = len(d)
up = np.where((pb_prev <= 1.0) & (pb > 1.0))[0]
dn = np.where((pb_prev >= 0.0) & (pb < 0.0))[0]
sig = sorted([(i, +1) for i in up] + [(i, -1) for i in dn])

def weeklies_for(entry_date):
    """(current_week, next_week) listed expiries on/after entry_date."""
    fut = [e for e in listed_expiries if e >= entry_date]
    cur = fut[0] if fut else None
    nxt = fut[1] if len(fut) > 1 else None
    return cur, nxt

def atm_token(strike, opt_type, expiry):
    m = nifty[(nifty["strike"] == strike) & (nifty["instrument_type"] == opt_type)
              & (nifty["expiry"] == expiry)]
    if m.empty:
        return None, None
    return int(m.iloc[0]["instrument_token"]), str(m.iloc[0]["tradingsymbol"])

# Build trade list (entry/exit on underlying) for priceable recent signals
trades = []
for i, dirn in sig:
    if i + HOR >= n:
        continue
    entry_ts = idx[i]; entry_date = entry_ts.date()
    cur, nxt = weeklies_for(entry_date)
    if cur is None:
        continue
    # only attempt recent signals where the current-week contract is listed
    if (cur - entry_date).days > 9:   # contract too far → likely already expired window
        pass
    entry_spot = close[i]; a = atr[i]
    strike = int(round(entry_spot / 50) * 50)
    tgt = entry_spot + dirn * TGT * a; stp = entry_spot - dirn * STP * a
    exit_i = i + HOR
    for j in range(i + 1, i + 1 + HOR):
        if dirn == 1 and (close[j] >= tgt or close[j] <= stp): exit_i = j; break
        if dirn == -1 and (close[j] <= tgt or close[j] >= stp): exit_i = j; break
    trades.append(dict(entry_ts=entry_ts, exit_ts=idx[exit_i], dirn=dirn,
                       opt="CE" if dirn == 1 else "PE", strike=strike,
                       cur=cur, nxt=nxt))

tdf = pd.DataFrame(trades)
# Keep only signals whose current-week expiry is among the most recent listed
# (older weeklies are delisted → unpriceable). Restrict to last ~25 calendar days.
cutoff = d.index[-1].date() - pd.Timedelta(days=25).to_pytimedelta()
tdf = tdf[tdf["entry_ts"].dt.date >= cutoff].reset_index(drop=True)
print(f"Priceable-window signals: {len(tdf)}  ({cutoff} .. {d.index[-1].date()})")

opt_cache = {}
def fetch_opt(token, t0, t1):
    if token in opt_cache:
        return opt_cache[token]
    try:
        c = kite.historical_data(token, (t0 - pd.Timedelta(minutes=10)).to_pydatetime(),
                                 (t1 + pd.Timedelta(minutes=10)).to_pydatetime(), "minute")
        time.sleep(RATE_SLEEP)
        if not c:
            opt_cache[token] = None; return None
        o = pd.DataFrame(c); o["date"] = pd.to_datetime(o["date"]).dt.tz_localize(None)
        o = o.set_index("date").sort_index()
        opt_cache[token] = o; return o
    except Exception as e:
        print("  fetch fail", token, e); opt_cache[token] = None; return None

def price_at(o, ts):
    if o is None or o.empty:
        return None
    ts = ts.tz_localize(None) if ts.tzinfo else ts
    past = o[o.index <= ts]
    if not past.empty and (ts - past.index[-1]) <= TOL:
        return float(past.iloc[-1]["close"])
    fut = o[o.index >= ts]
    if not fut.empty and (fut.index[0] - ts) <= TOL:
        return float(fut.iloc[0]["close"])
    return None

def pnl_for(row, which):
    expiry = row[which]
    if expiry is None:
        return None
    token, sym = atm_token(row["strike"], row["opt"], expiry)
    if token is None:
        return None
    o = fetch_opt(token, row["entry_ts"], row["exit_ts"])
    e = price_at(o, row["entry_ts"]); x = price_at(o, row["exit_ts"])
    if not e or not x or e <= 0 or x <= 0:
        return None
    entry_fill = e * (1 + SLIPPAGE_PCT); exit_fill = x * (1 - SLIPPAGE_PCT)
    pnl = (exit_fill - entry_fill) * LOT_SIZE - 2 * BROKERAGE_PER_ORDER
    return dict(entry_prem=round(e, 2), exit_prem=round(x, 2),
                dte=(expiry - row["entry_ts"].date()).days,
                pnl=round(pnl, 2), ret_pct=round((exit_fill/entry_fill - 1) * 100, 2))

# ── Price each signal across ALL listed expiries; bucket by ACTUAL DTE ───────
records = []
for _, row in tdf.iterrows():
    for expiry in listed_expiries:
        dte = (expiry - row["entry_ts"].date()).days
        if dte < 0 or dte > 30:
            continue
        token, sym = atm_token(row["strike"], row["opt"], expiry)
        if token is None:
            continue
        o = fetch_opt(token, row["entry_ts"], row["exit_ts"])
        e = price_at(o, row["entry_ts"]); x = price_at(o, row["exit_ts"])
        if not e or not x or e <= 0 or x <= 0:
            continue
        entry_fill = e * (1 + SLIPPAGE_PCT); exit_fill = x * (1 - SLIPPAGE_PCT)
        pnl = (exit_fill - entry_fill) * LOT_SIZE - 2 * BROKERAGE_PER_ORDER
        records.append(dict(entry_ts=row["entry_ts"], opt=row["opt"], dte=dte,
                            entry_prem=e, pnl=pnl,
                            ret_pct=(exit_fill / entry_fill - 1) * 100))

rdf = pd.DataFrame(records)
print(f"\nTotal (signal x expiry) priced observations: {len(rdf)}")
if not rdf.empty:
    rdf["dte_bucket"] = pd.cut(rdf["dte"], [-1, 2, 5, 9, 16, 31],
                               labels=["0-2 DTE", "3-5 DTE", "6-9 DTE", "10-16 DTE", "17-30 DTE"])
    print("\n=== Real ATM-option P&L by ACTUAL days-to-expiry (theta gradient) ===")
    g = rdf.groupby("dte_bucket", observed=True).agg(
        n=("pnl", "size"), win=("pnl", lambda s: (s > 0).mean()),
        avg_pnl=("pnl", "mean"), med_ret=("ret_pct", "median"),
        avg_entry_prem=("entry_prem", "mean"))
    g["win"] = (g["win"] * 100).round(1); g["avg_pnl"] = g["avg_pnl"].round(0)
    g["med_ret"] = g["med_ret"].round(1); g["avg_entry_prem"] = g["avg_entry_prem"].round(1)
    print(g.to_string())
