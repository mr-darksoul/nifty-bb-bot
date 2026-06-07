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
    CANDLE_INTERVAL_MINUTES,
    ENTRY_START_HOUR,
    ENTRY_START_MIN,
    LOT_SIZE,
    LOTS_PER_TRADE,
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

    strategy: str = p.get("strategy", "mean_reversion")
    bb_oversold: float = p["bb_oversold"]
    bb_overbought: float = p["bb_overbought"]
    bb_exit: float = p["bb_exit"]
    sl_buffer: float = p["sl_buffer"]
    rsi_min: float = float(p.get("rsi_min", 0))
    rsi_max: float = float(p.get("rsi_max", 100))
    # Volatility gate: only enter when ATR percentile >= this floor, so the
    # expected reversion move is large enough to clear fixed per-trade costs.
    # 0 disables the gate (back-compatible).
    min_atr_pct: float = float(p.get("min_atr_pct", 0.0))

    required = ["percent_b", "rsi", "atr", "close", "high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for backtest: {missing}")

    pb = df["percent_b"]
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr_series = df["atr"]
    # ATR percentile (rolling rank, 0–100). Absent in older frames → treat as
    # always-passing so the gate is a no-op.
    if "atr_pct" in df.columns:
        atr_pct_series = df["atr_pct"]
    else:
        atr_pct_series = pd.Series(100.0, index=df.index)
    rsi_series = df["rsi"] if "rsi" in df.columns else pd.Series(50.0, index=df.index)
    idx = df.index

    # Bar length in minutes, inferred from the index spacing. The engine is run
    # on whatever timeframe the strategy uses (15-min for momentum_breakout), so
    # reporting durations in real minutes requires the actual bar size — using a
    # hard-coded 1-min interval understated every duration by 15×.
    if len(idx) > 1:
        _deltas = pd.Series(idx).diff().dropna().dt.total_seconds().div(60).round()
        bar_minutes = int(_deltas.mode().iloc[0]) if not _deltas.empty else CANDLE_INTERVAL_MINUTES
    else:
        bar_minutes = CANDLE_INTERVAL_MINUTES
    bar_minutes = max(1, bar_minutes)

    trades = []
    in_trade = False
    entry_iloc = -1
    entry_direction = 0   # +1 = CE, -1 = PE
    entry_price = 0.0
    target_price = 0.0   # price-based target (ATR multiple from entry spot)
    sl_price = 0.0       # price-based stop (ATR multiple from entry spot)
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
                pnl, EXIT_REASON_FORCE, i - entry_iloc,
                int(round(entry_spot / 50) * 50), bar_minutes,
            ))
            in_trade = False
            continue

        pb_val = pb.iloc[i]
        if pd.isna(pb_val):
            continue

        # ── Exit checks (if in trade) — price-anchored targets/stops ──────────
        # bb_exit and sl_buffer are ATR multiples from the entry spot price.
        # This decouples P&L from BB drift so the R:R is locked in at entry.
        if in_trade:
            exit_triggered = False
            reason = ""
            fill_price = 0.0
            bar_high = high.iloc[i]
            bar_low = low.iloc[i]

            # Intrabar fills: a target/stop is hit when the bar's RANGE reaches it,
            # not only when the bar CLOSES through it. Testing close-only is
            # look-ahead-optimistic — it silently ignores stops pierced mid-bar
            # that the live bot (which samples spot every minute) would honour,
            # which inflated both win rate and P&L. When both levels sit inside one
            # bar we assume the STOP filled first (conservative worst-case order).
            if entry_direction == 1:     # CE: profit when price rises
                if bar_low <= sl_price:
                    reason = EXIT_REASON_SL; fill_price = sl_price; exit_triggered = True
                elif bar_high >= target_price:
                    reason = EXIT_REASON_TARGET; fill_price = target_price; exit_triggered = True
            else:                        # PE: profit when price falls
                if bar_high >= sl_price:
                    reason = EXIT_REASON_SL; fill_price = sl_price; exit_triggered = True
                elif bar_low <= target_price:
                    reason = EXIT_REASON_TARGET; fill_price = target_price; exit_triggered = True

            if exit_triggered:
                pnl = _calc_pnl(
                    entry_direction, entry_price, fill_price,
                    atr_series.iloc[i], entry_iloc, i
                )
                trades.append(_make_trade(
                    entry_time, ts, entry_direction, entry_price, fill_price,
                    pnl, reason, i - entry_iloc,
                    int(round(entry_spot / 50) * 50), bar_minutes,
                ))
                in_trade = False
                continue

        # ── Entry signals (if not in trade + trade limits ok) ─────────────────
        if not in_trade and trades_today < MAX_TRADES_PER_DAY:
            if hour < ENTRY_START_HOUR or (hour == ENTRY_START_HOUR and minute < ENTRY_START_MIN):
                continue
            if (hour == 15 and minute >= 10) or hour > 15:
                continue

            # Volatility gate: skip low-ATR bars where the move can't clear costs.
            atr_pct_val = atr_pct_series.iloc[i]
            if min_atr_pct > 0 and (pd.isna(atr_pct_val) or atr_pct_val < min_atr_pct):
                continue

            # RSI filter
            rsi_val = rsi_series.iloc[i]
            if not pd.isna(rsi_val) and not (rsi_min <= rsi_val <= rsi_max):
                continue

            direction = 0
            if strategy == "momentum_breakout":
                # Ride the band break: enter when %b crosses OUTWARD through the
                # band on this bar (needs the prior bar's %b).
                prev_pb = pb.iloc[i - 1] if i > 0 else pb_val
                if not pd.isna(prev_pb):
                    if prev_pb <= bb_overbought and pb_val > bb_overbought:
                        direction = 1    # upside break → CE
                    elif prev_pb >= bb_oversold and pb_val < bb_oversold:
                        direction = -1   # downside break → PE
            else:
                # mean_reversion: fade the extreme
                if pb_val < bb_oversold:
                    direction = 1   # CE
                elif pb_val > bb_overbought:
                    direction = -1  # PE

            if direction != 0:
                in_trade = True
                entry_iloc = i
                entry_direction = direction
                entry_spot = close.iloc[i]
                entry_price = entry_spot * (1 + direction * SLIPPAGE_PCT)
                entry_time = ts
                entry_atr = atr_series.iloc[i]
                # Lock in price-based target and stop at entry using ATR multiples.
                # bb_exit  = ATR multiples for the profit target
                # sl_buffer = ATR multiples for the stop loss
                target_price = entry_spot + direction * bb_exit * entry_atr
                sl_price     = entry_spot - direction * sl_buffer * entry_atr
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
            pnl, EXIT_REASON_EOD, i - entry_iloc,
            int(round(entry_spot / 50) * 50), bar_minutes,
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
    # Since entry_price here IS already the underlying (CE/PE share moves with delta).
    # Fixed-lot sizing (LOTS_PER_TRADE) so the proxy matches live/option_pricer.
    option_pnl = ATM_DELTA * price_change * LOT_SIZE * LOTS_PER_TRADE
    option_pnl -= 2 * BROKERAGE_PER_ORDER
    return float(option_pnl)


def _make_trade(
    entry_time,
    exit_time,
    direction: int,
    entry_price: float,
    exit_price: float,
    pnl: float,
    reason: str,
    duration_bars: int,
    atm_strike: int = 0,
    bar_minutes: int = CANDLE_INTERVAL_MINUTES,
) -> dict:
    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": "CE" if direction == 1 else "PE",
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "pnl": round(pnl, 2),
        "exit_reason": reason,
        "duration_min": duration_bars * bar_minutes,
        "atm_strike": atm_strike,
    }
