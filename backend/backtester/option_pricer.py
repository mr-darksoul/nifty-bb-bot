"""
Enriches backtest trades with real option P&L from Kite historical data.

The engine simulates on NIFTY spot prices using a fixed delta proxy (0.45).
This module replaces those approximated values with actual option premiums
fetched from Kite Connect — one historical_data call per unique instrument.

Fallback: if Kite returns no data for a trade's option (illiquid, too old,
instrument not found), the original delta-proxy P&L is kept so the backtest
still completes.
"""

import logging
import time
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from config import BROKERAGE_PER_ORDER, CAPITAL_PER_TRADE, LOT_SIZE, MIN_DAYS_TO_EXPIRY, SLIPPAGE_PCT

logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 0.35   # seconds between historical_data calls (stay under 3/sec Kite limit)


# ── Expiry resolution ─────────────────────────────────────────────────────────

def _nearest_expiry_on_or_after(entry_date: date, instruments_df: pd.DataFrame) -> Optional[date]:
    """Earliest expiry that has at least MIN_DAYS_TO_EXPIRY remaining from entry_date."""
    candidates = sorted(
        d for d in instruments_df["expiry"].unique()
        if (d - entry_date).days >= MIN_DAYS_TO_EXPIRY
    )
    return candidates[0] if candidates else None


# ── Option LTP lookup ─────────────────────────────────────────────────────────

def _close_at(opt_df: pd.DataFrame, target: pd.Timestamp) -> float:
    """Close price at or just before target from a 1-min option DataFrame."""
    past = opt_df[opt_df.index <= target]
    if not past.empty:
        return float(past.iloc[-1]["close"])
    future = opt_df[opt_df.index >= target]
    return float(future.iloc[0]["close"]) if not future.empty else 0.0


# ── Main enrichment ───────────────────────────────────────────────────────────

def enrich_with_real_option_prices(
    kite,
    trades_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace delta-proxy P&L in trades_df with actual Kite option prices.

    Algorithm:
      1. For each trade, resolve (atm_strike, direction) → nearest expiry →
         instrument token from the NFO instruments list.
      2. De-duplicate tokens so each unique option is fetched only once,
         covering the full time span of all trades that use it.
      3. For each trade whose token was fetched successfully, look up the
         1-min close at entry_time and exit_time, then compute real P&L.

    Args:
        kite:            Authenticated KiteConnect instance.
        trades_df:       Output of backtester.engine.run_backtest().
        instruments_df:  DataFrame of NFO instruments (name == "NIFTY"),
                         with "expiry" column as python date objects.

    Returns:
        Updated trades_df with extra columns: option_symbol, real_entry_ltp,
        real_exit_ltp.  P&L is real wherever possible; delta-proxy otherwise.
    """
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])

    # ── Step 1: Resolve token for every trade ─────────────────────────────────
    tokens:   List[Optional[int]]  = []
    symbols:  List[Optional[str]]  = []

    for _, row in trades.iterrows():
        atm_strike = int(row.get("atm_strike", 0))
        direction  = str(row["direction"])            # "CE" or "PE"
        entry_date = row["entry_time"].date()

        if atm_strike <= 0:
            tokens.append(None); symbols.append(None)
            continue

        expiry = _nearest_expiry_on_or_after(entry_date, instruments_df)
        if expiry is None:
            logger.warning(f'"No expiry found for {atm_strike} {direction} >= {entry_date}"')
            tokens.append(None); symbols.append(None)
            continue

        mask = (
            (instruments_df["strike"] == atm_strike)
            & (instruments_df["instrument_type"] == direction)
            & (instruments_df["expiry"] == expiry)
        )
        matches = instruments_df[mask]
        if matches.empty:
            # Try nearest available strike (±1 step) as fallback
            for delta in [50, -50, 100, -100]:
                adj = atm_strike + delta
                mask2 = (
                    (instruments_df["strike"] == adj)
                    & (instruments_df["instrument_type"] == direction)
                    & (instruments_df["expiry"] == expiry)
                )
                matches = instruments_df[mask2]
                if not matches.empty:
                    logger.debug(f'"Strike {atm_strike} not found, using {adj} for {direction} expiry={expiry}"')
                    break

        if matches.empty:
            logger.warning(f'"No instrument: strike={atm_strike} {direction} expiry={expiry}"')
            tokens.append(None); symbols.append(None)
        else:
            r = matches.iloc[0]
            tokens.append(int(r["instrument_token"]))
            symbols.append(str(r["tradingsymbol"]))

    trades["_token"]       = tokens
    trades["option_symbol"] = symbols

    # ── Step 2: Fetch historical data per unique token ────────────────────────
    unique_tokens = [t for t in trades["_token"].dropna().unique()]
    logger.info(f'"Fetching real option prices: {len(unique_tokens)} unique instruments"')

    opt_cache: Dict[int, pd.DataFrame] = {}

    for token in unique_tokens:
        token = int(token)
        token_rows = trades[trades["_token"] == token]
        # Cover all trades on this token in one request
        from_dt = (token_rows["entry_time"].min() - pd.Timedelta(minutes=5)).to_pydatetime()
        to_dt   = (token_rows["exit_time"].max()  + pd.Timedelta(minutes=5)).to_pydatetime()

        try:
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_dt,
                to_date=to_dt,
                interval="minute",
            )
            if candles:
                odf = pd.DataFrame(candles)
                odf["date"] = pd.to_datetime(odf["date"])
                odf = odf.set_index("date").sort_index()
                opt_cache[token] = odf
                logger.info(f'"Option data fetched: token={token} bars={len(odf)}"')
            else:
                logger.warning(f'"No candles returned: token={token}"')
        except Exception as exc:
            logger.warning(f'"Option historical_data failed token={token}: {exc}"')

        time.sleep(_RATE_LIMIT_SLEEP)   # stay within Kite rate limit

    # ── Step 3: Replace P&L with real option prices ───────────────────────────
    real_entry_ltps: List[Optional[float]] = []
    real_exit_ltps:  List[Optional[float]] = []

    for idx, row in trades.iterrows():
        token = row.get("_token")
        if token is None or int(token) not in opt_cache:
            real_entry_ltps.append(None)
            real_exit_ltps.append(None)
            continue

        odf       = opt_cache[int(token)]
        entry_ltp = _close_at(odf, row["entry_time"])
        exit_ltp  = _close_at(odf, row["exit_time"])

        if entry_ltp <= 0 or exit_ltp <= 0:
            logger.warning(f'"Zero option LTP for {row.get("option_symbol")} — keeping delta proxy"')
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
        real_entry_ltps.append(entry_ltp)
        real_exit_ltps.append(exit_ltp)

    trades["real_entry_ltp"] = real_entry_ltps
    trades["real_exit_ltp"]  = real_exit_ltps

    enriched = trades["real_entry_ltp"].notna().sum()
    total    = len(trades)
    logger.info(f'"Option enrichment complete: {enriched}/{total} trades used real prices"')

    return trades.drop(columns=["_token"], errors="ignore")
