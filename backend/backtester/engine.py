"""
Vectorized backtester for the Bollinger %b mean-reversion strategy on NIFTY options.

P&L is simulated using a delta proxy (ATM delta = 0.45) applied to the underlying
price move, scaled by lot size, with slippage and brokerage deducted per trade.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BROKERAGE_PER_ORDER,
    LOT_SIZE,
    MAX_TRADES_PER_DAY,
    SLIPPAGE_PCT,
    CAPITAL_PER_TRADE,
)
from backtester.metrics import compute_metrics, format_report

logger = logging.getLogger(__name__)

ATM_DELTA = 0.45    # option delta proxy
EXIT_REASON_TARGET = "TARGET"
EXIT_REASON_SL = "STOP_LOSS"
EXIT_REASON_FORCE = "FORCE_EXIT"
EXIT_REASON_EOD = "EOD"


def _estimate_option_premium(nifty_price: float, atr: float) -> float:
    """Rough ATM option premium estimate: ~1 ATR value as proxy LTP."""
    return max(atr * 1.0, 50.0)


def run_backtest(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Run the full backtest simulation.

    Args:
        df:     OHLCV DataFrame with indicator columns (output of compute_all).
                Must have a DatetimeIndex.
        params: Strategy parameters dict. Falls back to config defaults.

    Returns:
        trades:     DataFrame of individual trades.
        daily_pnl:  Series of daily P&L indexed by date.
        metrics:    Dict of performance metrics.
    """
    from config import DEFAULT_PARAMS
    p = {**DEFAULT_PARAMS, **(params or {})}

    bb_oversold: float = p["bb_oversold"]
    bb_overbought: float = p["bb_overbought"]
    bb_exit: float = p["bb_exit"]
    sl_buffer: float = p["sl_buffer"]

    required = ["percent_b", "rsi", "atr", "close", "high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for backtest: {missing}")

    pb = df["percent_b"]
    close = df["close"]
    atr_series = df["atr"]
    idx = df.index

    trades = []
    in_trade = False
    entry_iloc = -1
    entry_direction = 0   # +1 = CE, -1 = PE
    entry_pb = 0.0
    entry_price = 0.0
    sl_pb = 0.0
    entry_time = None
    trades_today = 0
    current_date = None

    for i in range(len(df)):
        ts = idx[i]
        bar_date = ts.date()

        # ── Day rollover ───────────────────────────────────────────────────────
        if bar_date != current_date:
            current_date = bar_date
            trades_today = 0

        hour = ts.hour
        minute = ts.minute

        # ── Force exit at 15:10 ────────────────────────────────────────────────
        if in_trade and ((hour == 15 and minute >= 10) or hour > 15):
            pnl = _calc_pnl(
                entry_direction, entry_price, close.iloc[i],
                atr_series.iloc[i], entry_iloc, i
            )
            trades.append(_make_trade(
                entry_time, ts, entry_direction, entry_price, close.iloc[i],
                pnl, EXIT_REASON_FORCE, i - entry_iloc
            ))
            in_trade = False
            continue

        pb_val = pb.iloc[i]
        if pd.isna(pb_val):
            continue

        # ── Exit checks (if in trade) ──────────────────────────────────────────
        if in_trade:
            exit_triggered = False
            reason = ""

            if entry_direction == 1:     # CE: exit when pb crosses up to bb_exit
                if pb_val >= bb_exit:
                    reason = EXIT_REASON_TARGET
                    exit_triggered = True
                elif pb_val <= sl_pb:
                    reason = EXIT_REASON_SL
                    exit_triggered = True
            else:                        # PE: exit when pb crosses down to bb_exit
                if pb_val <= bb_exit:
                    reason = EXIT_REASON_TARGET
                    exit_triggered = True
                elif pb_val >= sl_pb:
                    reason = EXIT_REASON_SL
                    exit_triggered = True

            if exit_triggered:
                pnl = _calc_pnl(
                    entry_direction, entry_price, close.iloc[i],
                    atr_series.iloc[i], entry_iloc, i
                )
                trades.append(_make_trade(
                    entry_time, ts, entry_direction, entry_price, close.iloc[i],
                    pnl, reason, i - entry_iloc
                ))
                in_trade = False
                continue

        # ── Entry signals (if not in trade + trade limits ok) ─────────────────
        if not in_trade and trades_today < MAX_TRADES_PER_DAY:
            # Market must be open (≥ 9:20 AM to allow at least one warm-up bar)
            if hour < 9 or (hour == 9 and minute < 20):
                continue
            if (hour == 15 and minute >= 10) or hour > 15:
                continue

            direction = 0
            if pb_val < bb_oversold:
                direction = 1   # CE
            elif pb_val > bb_overbought:
                direction = -1  # PE

            if direction != 0:
                in_trade = True
                entry_iloc = i
                entry_direction = direction
                entry_pb = pb_val
                entry_price = close.iloc[i] * (1 + direction * SLIPPAGE_PCT)  # slippage on entry
                entry_time = ts
                sl_pb = entry_pb - direction * sl_buffer
                trades_today += 1

    # ── Close any open trade at end of data ────────────────────────────────────
    if in_trade and len(df) > 0:
        i = len(df) - 1
        pnl = _calc_pnl(
            entry_direction, entry_price, close.iloc[i],
            atr_series.iloc[i], entry_iloc, i
        )
        trades.append(_make_trade(
            entry_time, idx[i], entry_direction, entry_price, close.iloc[i],
            pnl, EXIT_REASON_EOD, i - entry_iloc
        ))

    # ── Assemble results ───────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price",
            "exit_price", "pnl", "exit_reason", "duration_min"
        ])
        daily_pnl = pd.Series(dtype=float)
        metrics = compute_metrics(trades_df, daily_pnl)
        return trades_df, daily_pnl, metrics

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    trades_df["date"] = trades_df["entry_time"].dt.date

    daily_pnl = trades_df.groupby("date")["pnl"].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)

    metrics = compute_metrics(trades_df, daily_pnl)
    logger.info(f'"Backtest complete: {len(trades_df)} trades, Sharpe={metrics.get("sharpe", 0):.2f}"')
    return trades_df, daily_pnl, metrics


def _calc_pnl(
    direction: int,
    entry_price: float,
    exit_price: float,
    atr_val: float,
    entry_iloc: int,
    exit_iloc: int,
) -> float:
    """
    Estimate option P&L using delta proxy.
    P&L = delta * price_change * lot_size - 2 * brokerage - slippage_on_exit_value
    """
    exit_with_slip = exit_price * (1 - direction * SLIPPAGE_PCT)
    price_change = (exit_with_slip - entry_price) * direction   # positive = win for direction
    # Approximate option price move = delta * underlying move
    # Since entry_price here IS already the underlying (CE/PE share moves with delta)
    option_pnl_per_lot = ATM_DELTA * price_change * LOT_SIZE
    option_pnl_per_lot -= 2 * BROKERAGE_PER_ORDER
    return float(option_pnl_per_lot)


def _make_trade(
    entry_time,
    exit_time,
    direction: int,
    entry_price: float,
    exit_price: float,
    pnl: float,
    reason: str,
    duration_bars: int,
) -> dict:
    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": "CE" if direction == 1 else "PE",
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "pnl": round(pnl, 2),
        "exit_reason": reason,
        "duration_min": duration_bars * 5,
    }
