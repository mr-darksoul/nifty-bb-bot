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

from config import (
    BROKERAGE_PER_ORDER,
    CAPITAL_PER_TRADE,
    LOT_SIZE,
    MAX_DAYS_TO_EXPIRY,
    MIN_DAYS_TO_EXPIRY,
    SLIPPAGE_PCT,
)

logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 0.35    # seconds between historical_data calls (under Kite's 3/sec)
_BAR_TOLERANCE_MIN = 15     # a usable price must have an option bar within this many minutes


# ── Expiry resolution ─────────────────────────────────────────────────────────

def _nearest_expiry_on_or_after(entry_date: date, instruments_df: pd.DataFrame) -> Optional[date]:
    """Earliest listed expiry inside the configured current/next-week DTE window."""
    candidates = sorted(
        d for d in instruments_df["expiry"].unique()
        if MIN_DAYS_TO_EXPIRY <= (d - entry_date).days <= MAX_DAYS_TO_EXPIRY
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
    for col in ("entry_price", "exit_price", "pnl"):
        if col in trades.columns:
            trades[col] = trades[col].astype(float)

    # ── Step 1: Resolve token + expiry for every trade ────────────────────────
    candidate_lists: List[List[dict]] = []
    expiries: List[Optional[date]] = []

    for _, row in trades.iterrows():
        atm_strike = int(row.get("atm_strike", 0))
        direction  = str(row["direction"])            # "CE" or "PE"
        entry_date = row["entry_time"].date()

        if atm_strike <= 0:
            candidate_lists.append([]); expiries.append(None)
            continue

        expiry = _nearest_expiry_on_or_after(entry_date, instruments_df)
        if expiry is None:
            candidate_lists.append([]); expiries.append(None)
            continue

        strike_mask = (
            instruments_df["strike"] >= atm_strike
            if direction == "CE"
            else instruments_df["strike"] <= atm_strike
        )
        mask = (
            (instruments_df["instrument_type"] == direction)
            & (instruments_df["expiry"] == expiry)
            & strike_mask
        )
        matches = instruments_df[mask].copy()

        if matches.empty:
            candidate_lists.append([]); expiries.append(None)
            continue

        matches["distance"] = (matches["strike"] - atm_strike).abs()
        matches = matches.sort_values(
            ["distance", "strike"],
            ascending=[True, direction == "CE"],
        ).head(20)
        candidate_lists.append([
            {
                "token": int(r["instrument_token"]),
                "symbol": str(r["tradingsymbol"]),
                "strike": int(r["strike"]),
            }
            for _, r in matches.iterrows()
        ])
        expiries.append(expiry)

    trades["_candidates"]   = candidate_lists
    trades["expiry"]        = expiries
    trades["dte"] = [
        (e - t.date()).days if e is not None else None
        for e, t in zip(expiries, trades["entry_time"])
    ]

    # ── Step 2: Fetch historical data per unique token (one call each) ────────
    unique_tokens = sorted({
        int(c["token"])
        for candidates in trades["_candidates"]
        for c in candidates
    })
    logger.info(f'"Real option pricing: probing {len(unique_tokens)} unique instruments"')

    opt_cache: Dict[int, pd.DataFrame] = {}
    for token in unique_tokens:
        token_rows = trades[
            trades["_candidates"].apply(lambda candidates: any(c["token"] == token for c in candidates))
        ]
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
        selected = None
        for candidate in row["_candidates"]:
            odf = opt_cache.get(int(candidate["token"]))
            if odf is None:
                continue

            entry_ltp = _close_near(odf, row["entry_time"])
            exit_ltp  = _close_near(odf, row["exit_time"])
            if not entry_ltp or not exit_ltp or entry_ltp <= 0 or exit_ltp <= 0:
                continue

            actual_entry = round(entry_ltp * (1 + SLIPPAGE_PCT), 2)
            one_lot_value = actual_entry * LOT_SIZE
            if one_lot_value <= CAPITAL_PER_TRADE:
                selected = (candidate, entry_ltp, exit_ltp, actual_entry, one_lot_value)
                break

        if selected is None:
            logger.info(
                f'"Dropping untradeable backtest row: no priced {row["direction"]} '
                f'candidate under cap=₹{CAPITAL_PER_TRADE:.2f} '
                f'for atm={row.get("atm_strike")} expiry={row.get("expiry")}"'
            )
            real_entry_ltps.append(None)
            real_exit_ltps.append(None)
            continue

        candidate, entry_ltp, exit_ltp, actual_entry, one_lot_value = selected
        actual_exit  = round(exit_ltp  * (1 - SLIPPAGE_PCT), 2)
        quantity     = int(CAPITAL_PER_TRADE / actual_entry / LOT_SIZE) * LOT_SIZE
        if quantity <= 0:
            real_entry_ltps.append(None)
            real_exit_ltps.append(None)
            continue
        pnl = round((actual_exit - actual_entry) * quantity - 2 * BROKERAGE_PER_ORDER, 2)

        trades.at[idx, "entry_price"] = actual_entry
        trades.at[idx, "exit_price"]  = actual_exit
        trades.at[idx, "pnl"]         = pnl
        trades.at[idx, "quantity"]    = quantity
        trades.at[idx, "atm_strike"]  = candidate["strike"]
        trades.at[idx, "option_symbol"] = candidate["symbol"]
        trades.at[idx, "one_lot_value"] = round(one_lot_value, 2)
        trades.at[idx, "exit_outcome"] = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
        trades.at[idx, "exit_reason_is_profitable"] = bool(
            (row["exit_reason"] != "TARGET" or pnl > 0)
            and (row["exit_reason"] != "STOP_LOSS" or pnl < 0)
        )
        real_entry_ltps.append(entry_ltp)
        real_exit_ltps.append(exit_ltp)

    trades["real_entry_ltp"] = real_entry_ltps
    trades["real_exit_ltp"]  = real_exit_ltps

    priced = int(trades["real_entry_ltp"].notna().sum())
    total  = len(trades)
    logger.info(f'"Real option pricing: {priced}/{total} trades priced from Kite data"')

    trades = trades.drop(columns=["_candidates"], errors="ignore")

    if drop_unpriced:
        trades = trades[trades["real_entry_ltp"].notna()].reset_index(drop=True)

    return trades
