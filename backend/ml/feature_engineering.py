"""
Central feature engineering module.
All ML models import features exclusively from here to ensure consistency
between training and inference.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from indicators import compute_all

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "percent_b",
    "bb_width",
    "bb_width_pct",
    "rsi",
    "atr_norm",
    "atr_pct",
    "ema_cross",
    "price_vs_vwap",
    "pb_velocity",
    "pb_acceleration",
    "rsi_slope",
    "minutes_since_open_norm",
    "day_of_week",
    "is_first_30min",
    "is_last_30min",
    "volume_rank",
]

REGIME_FEATURE_COLUMNS: List[str] = [
    "log_return",
    "rolling_vol",
    "bb_width",
    "atr_norm",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an OHLCV DataFrame with a DatetimeIndex (IST), compute and return
    a feature matrix aligned to the same index.

    All NaN-producing rows (warm-up period) are retained but should be
    dropped by callers before feeding to models.
    """
    df = df.copy()
    df = compute_all(df)

    # ── Derived %b features ───────────────────────────────────────────────────
    df["pb_velocity"] = df["percent_b"].diff(3)
    df["pb_acceleration"] = df["percent_b"].diff().diff()
    df["rsi_slope"] = df["rsi"].diff(3)

    # ── ATR / BB width percentile ─────────────────────────────────────────────
    df["bb_width_pct"] = df["bb_width"].rolling(50).rank(pct=True) * 100.0

    # ── Temporal features ─────────────────────────────────────────────────────
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex (IST timezone-aware or naive)")

    market_open_minutes = 9 * 60 + 15   # 9:15 AM
    minutes_since_open = idx.hour * 60 + idx.minute - market_open_minutes
    trading_day_minutes = 375.0          # 9:15 – 15:30

    df["minutes_since_open_norm"] = np.clip(minutes_since_open / trading_day_minutes, 0.0, 1.0)
    df["day_of_week"] = idx.dayofweek.astype(float)
    df["is_first_30min"] = (minutes_since_open <= 30).astype(float)
    df["is_last_30min"] = (minutes_since_open >= (trading_day_minutes - 30)).astype(float)

    # ── Volume rank (always 0.5 for NIFTY index; placeholder for future use) ─
    df["volume_rank"] = 0.5

    return df


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the feature subset used by the HMM regime detector.
    Returns a DataFrame with REGIME_FEATURE_COLUMNS.
    """
    df = df.copy()
    df = compute_all(df)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol"] = df["log_return"].rolling(20).std()

    return df[REGIME_FEATURE_COLUMNS]


def get_signal_features_at_bar(df_features: pd.DataFrame, idx_loc: int) -> pd.Series:
    """
    Extract the signal-quality feature vector at a specific bar location.
    Used during live inference to score a signal before placing an order.
    """
    row = df_features.iloc[idx_loc]
    return row[FEATURE_COLUMNS]


def label_signals(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    direction: pd.Series,
    bb_exit: float,
    sl_buffer: float,
) -> pd.Series:
    """
    Generate binary labels for signal quality training.

    A signal bar is labelled 1 if the subsequent trade would reach bb_exit
    target before hitting the SL level, 0 otherwise.

    Args:
        df:          Feature DataFrame (must contain percent_b).
        entry_mask:  Boolean Series, True at signal bars.
        direction:   Series of +1 (CE/long) or -1 (PE/short) at signal bars.
        bb_exit:     %b exit target (e.g. 0.50).
        sl_buffer:   Offset beyond entry %b that constitutes stop-loss.

    Returns:
        labels: Series aligned to df.index, NaN except at entry bars (0 or 1).
    """
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    entry_indices = df.index[entry_mask]

    for entry_ts in entry_indices:
        iloc = df.index.get_loc(entry_ts)
        entry_pb = df.loc[entry_ts, "percent_b"]
        dir_val = direction.loc[entry_ts]

        if pd.isna(entry_pb):
            continue

        if dir_val == 1:                     # CE: bought when oversold
            sl_pb = entry_pb - sl_buffer
            target_pb = bb_exit
        else:                                # PE: bought when overbought
            sl_pb = entry_pb + sl_buffer
            target_pb = bb_exit

        label = 0
        for future_iloc in range(iloc + 1, min(iloc + 79, len(df))):  # max ~6.5h
            future_pb = df.iloc[future_iloc]["percent_b"]
            if pd.isna(future_pb):
                continue

            hour = df.index[future_iloc].hour
            minute = df.index[future_iloc].minute
            if (hour == 15 and minute >= 10) or hour > 15:
                break                        # force exit, call it SL

            if dir_val == 1:
                if future_pb >= target_pb:
                    label = 1
                    break
                if future_pb <= sl_pb:
                    label = 0
                    break
            else:
                if future_pb <= target_pb:
                    label = 1
                    break
                if future_pb >= sl_pb:
                    label = 0
                    break

        labels.loc[entry_ts] = label

    return labels
