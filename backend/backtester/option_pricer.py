"""
Prices backtest trades with REAL option premiums from Kite historical data.

The engine simulates entry/exit bars on NIFTY spot. This module looks up the
actual option premium (1-min close) at those timestamps from Kite Connect.

IMPORTANT — Kite limitation: kite.instruments("NFO") lists only currently
listed contracts. Expired weekly options and their historical tokens are not
retrievable. So real option data exists only for contracts still listed today
(roughly the current/next couple of weekly expiries). Trades whose correct
weekly contract is no longer listed — or that fall before the contract began
trading — cannot be priced and are dropped from a "real Kite data only"
backtest (drop_unpriced=True). The surviving trades define the real-data window.
"""

import logging
import time
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from config import BROKERAGE_PER_ORDER, CAPITAL_PER_TRADE, LOT_SIZE, MIN_DAYS_TO_EXPIRY, SLIPPAGE_PCT

logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 0.35    # seconds between historical_data calls (under Kite's 3/sec)
_BAR_TOLERANCE_MIN = 15     # a usable price must have an option bar within this many minutes


# ── Expiry resolution ─────────────────────────────────────────────────────────

def _nearest_expiry_on_or_after(entry_date: date, instruments_df: pd.DataFrame) -> Optional[date]:
    """Earliest listed expiry with at least MIN_DAYS_TO_EXPIRY remaining from entry_date."""
    candidates = sorted(
        d for d in instruments_df["expiry"].unique()
        if (d - entry_date).days >= MIN_DAYS_TO_EXPIRY
    )
    return candidates[0] if candidates else None


# ── Option LTP lookup ─────────────────────────────────────────────────────────

def _close_near(opt_df: pd.DataFrame, target: pd.Timestamp) -> Optional[float]:
    """Close at/just-before target. None if no bar within _BAR_TOLERANCE_MIN.

    Returning None means the contract was not actively trading at that time —
    so the trade cannot be priced from real data and must be dropped.
    """
    tol = pd.Timedelta(minutes=_BAR_TOLERANCE_MIN)
    past = opt_df[opt_df.index <= target]
    if not past.empty and (target - past.index[-1]) <= tol:
        return float(past.iloc[-1]["close"])
    # No prior bar in range → check a near-future bar (handles a missing exact minute)
    future = opt_df[opt_df.index >= target]
    if not future.empty and (future.index[0] - target) <= tol:
        return float(future.iloc[0]["close"])
    return None


# ── Main enrichment ───────────────────────────────────────────────────────────

def enrich_with_real_option_prices(
    kite,
    trades_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
    drop_unpriced: bool = True,
) -> pd.DataFrame:
    """
    Replace simulated P&L with actual Kite option premiums.

    Args:
        kite:            Authenticated KiteConnect instance.
        trades_df:       Output of backtester.engine.run_backtest().
        instruments_df:  NFO instruments (name == "NIFTY"), "expiry" as date objects.
        drop_unpriced:   When True (real-only mode), trades that cannot be priced
                         from real Kite data are removed from the result.

    Returns:
        Trades DataFrame. entry_price/exit_price hold real premiums; extra columns
        option_symbol, expiry, dte, real_entry_ltp, real_exit_ltp are added.
    """
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])

    # ── Step 1: Resolve token + expiry for every trade ────────────────────────
    tokens:   List[Optional[int]]  = []
    symbols:  List[Optional[str]]  = []
    expiries: List[Optional[date]] = []

    for _, row in trades.iterrows():
        atm_strike = int(row.get("atm_strike", 0))
        direction  = str(row["direction"])            # "CE" or "PE"
        entry_date = row["entry_time"].date()

        if atm_strike <= 0:
            tokens.append(None); symbols.append(None); expiries.append(None)
            continue

        expiry = _nearest_expiry_on_or_after(entry_date, instruments_df)
        if expiry is None:
            tokens.append(None); symbols.append(None); expiries.append(None)
            continue

        mask = (
            (instruments_df["strike"] == atm_strike)
            & (instruments_df["instrument_type"] == direction)
            & (instruments_df["expiry"] == expiry)
        )
        matches = instruments_df[mask]
        if matches.empty:
            for delta in [50, -50, 100, -100]:   # nearest available strike
                adj = atm_strike + delta
                matches = instruments_df[
                    (instruments_df["strike"] == adj)
                    & (instruments_df["instrument_type"] == direction)
                    & (instruments_df["expiry"] == expiry)
                ]
                if not matches.empty:
                    break

        if matches.empty:
            tokens.append(None); symbols.append(None); expiries.append(None)
        else:
            r = matches.iloc[0]
            tokens.append(int(r["instrument_token"]))
            symbols.append(str(r["tradingsymbol"]))
            expiries.append(expiry)

    trades["_token"]        = tokens
    trades["option_symbol"] = symbols
    trades["expiry"]        = expiries
    trades["dte"] = [
        (e - t.date()).days if e is not None else None
        for e, t in zip(expiries, trades["entry_time"])
    ]

    # ── Step 2: Fetch historical data per unique token (one call each) ────────
    unique_tokens = [int(t) for t in trades["_token"].dropna().unique()]
    logger.info(f'"Real option pricing: probing {len(unique_tokens)} unique instruments"')

    opt_cache: Dict[int, pd.DataFrame] = {}
    for token in unique_tokens:
        token_rows = trades[trades["_token"] == token]
        from_dt = (token_rows["entry_time"].min() - pd.Timedelta(minutes=5)).to_pydatetime()
        to_dt   = (token_rows["exit_time"].max()  + pd.Timedelta(minutes=5)).to_pydatetime()
        try:
            candles = kite.historical_data(
                instrument_token=token, from_date=from_dt, to_date=to_dt, interval="minute",
            )
            if candles:
                odf = pd.DataFrame(candles)
                odf["date"] = pd.to_datetime(odf["date"])
                odf = odf.set_index("date").sort_index()
                # Kite returns tz-aware timestamps; normalise to naive for comparison
                if odf.index.tz is not None:
                    odf.index = odf.index.tz_localize(None)
                opt_cache[token] = odf
                logger.info(f'"Option data: token={token} bars={len(odf)} '
                            f'range={odf.index.min()}..{odf.index.max()}"')
            else:
                logger.info(f'"No candles for token={token} (contract not listed in range)"')
        except Exception as exc:
            logger.warning(f'"historical_data failed token={token}: {exc}"')
        time.sleep(_RATE_LIMIT_SLEEP)

    # ── Step 3: Price each trade; drop those without real data ────────────────
    real_entry_ltps: List[Optional[float]] = []
    real_exit_ltps:  List[Optional[float]] = []

    # Strip any tz on trade timestamps to match option index
    if getattr(trades["entry_time"].dt, "tz", None) is not None:
        trades["entry_time"] = trades["entry_time"].dt.tz_localize(None)
        trades["exit_time"]  = trades["exit_time"].dt.tz_localize(None)

    for idx, row in trades.iterrows():
        token = row.get("_token")
        odf = opt_cache.get(int(token)) if token is not None and not pd.isna(token) else None

        entry_ltp = _close_near(odf, row["entry_time"]) if odf is not None else None
        exit_ltp  = _close_near(odf, row["exit_time"])  if odf is not None else None

        if not entry_ltp or not exit_ltp or entry_ltp <= 0 or exit_ltp <= 0:
            real_entry_ltps.append(None)
            real_exit_ltps.append(None)
            continue

        actual_entry = round(entry_ltp * (1 + SLIPPAGE_PCT), 2)
        actual_exit  = round(exit_ltp  * (1 - SLIPPAGE_PCT), 2)
        quantity     = int(CAPITAL_PER_TRADE / actual_entry / LOT_SIZE) * LOT_SIZE
        if quantity <= 0:
            quantity = LOT_SIZE
        pnl = round((actual_exit - actual_entry) * quantity - 2 * BROKERAGE_PER_ORDER, 2)

        trades.at[idx, "entry_price"] = actual_entry
        trades.at[idx, "exit_price"]  = actual_exit
        trades.at[idx, "pnl"]         = pnl
        trades.at[idx, "quantity"]    = quantity
        real_entry_ltps.append(entry_ltp)
        real_exit_ltps.append(exit_ltp)

    trades["real_entry_ltp"] = real_entry_ltps
    trades["real_exit_ltp"]  = real_exit_ltps

    priced = int(trades["real_entry_ltp"].notna().sum())
    total  = len(trades)
    logger.info(f'"Real option pricing: {priced}/{total} trades priced from Kite data"')

    trades = trades.drop(columns=["_token"], errors="ignore")

    if drop_unpriced:
        trades = trades[trades["real_entry_ltp"].notna()].reset_index(drop=True)

    return trades
