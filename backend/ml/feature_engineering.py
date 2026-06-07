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
]
# NOTE: volume_rank was removed — NIFTY index always has zero volume so the
# column was a constant 0.5 in every row, adding no signal.  Retrain models
# after this change.

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

    If indicators (bb_upper etc.) are already present the compute_all step is
    skipped, avoiding a second full pass when the caller pre-computed them.

    All NaN-producing rows (warm-up period) are retained but should be
    dropped by callers before feeding to models.
    """
    df = df.copy()
    if "bb_upper" not in df.columns:
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

    return df


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the feature subset used by the HMM regime detector.
    Returns a DataFrame with REGIME_FEATURE_COLUMNS.

    Skips compute_all if indicators are already present.
    """
    df = df.copy()
    if "bb_upper" not in df.columns:
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

    A signal bar is labelled 1 if the subsequent trade would reach its
    price-anchored profit target before hitting the stop loss, 0 otherwise.
    Target/stop are ATR multiples from the entry spot, identical to the
    backtest engine and the live exit logic (so labels match what is traded).

    Args:
        df:          Feature DataFrame (must contain close, atr).
        entry_mask:  Boolean Series, True at signal bars.
        direction:   Series of +1 (CE/long) or -1 (PE/short) at signal bars.
        bb_exit:     Profit target in ATR multiples.
        sl_buffer:   Stop loss in ATR multiples.

    Returns:
        labels: Series aligned to df.index, NaN except at entry bars (0 or 1).
    """
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    entry_indices = df.index[entry_mask]
    close = df["close"]
    atr = df["atr"]

    for entry_ts in entry_indices:
        iloc = df.index.get_loc(entry_ts)
        entry_price = close.loc[entry_ts]
        entry_atr = atr.loc[entry_ts]
        dir_val = direction.loc[entry_ts]

        if pd.isna(entry_price) or pd.isna(entry_atr) or entry_atr <= 0:
            continue

        sign = 1.0 if dir_val == 1 else -1.0
        target_price = entry_price + sign * bb_exit * entry_atr
        sl_price = entry_price - sign * sl_buffer * entry_atr

        label = 0
        for future_iloc in range(iloc + 1, min(iloc + 240, len(df))):  # intraday horizon
            hour = df.index[future_iloc].hour
            minute = df.index[future_iloc].minute
            if (hour == 15 and minute >= 10) or hour > 15:
                break                        # force exit before target → not a win

            price = close.iloc[future_iloc]
            if pd.isna(price):
                continue

            if dir_val == 1:                 # CE
                if price >= target_price:
                    label = 1
                    break
                if price <= sl_price:
                    label = 0
                    break
            else:                            # PE
                if price <= target_price:
                    label = 1
                    break
                if price >= sl_price:
                    label = 0
                    break

        labels.loc[entry_ts] = label

    return labels
