"""
Technical indicator computations: Bollinger %b, RSI, ATR, EMA, VWAP proxy.
All functions accept and return pandas Series/DataFrames operating on OHLCV data.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands and %b.

    Returns:
        upper, middle, lower, percent_b
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    band_range = upper - lower
    percent_b = (close - lower) / band_range.replace(0, np.nan)
    return upper, middle, lower, percent_b


def bollinger_width(upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
    """Normalised Bollinger Band width: (upper - lower) / middle."""
    return (upper - lower) / middle.replace(0, np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def atr_normalised(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """ATR divided by closing price for cross-asset comparability."""
    return atr(high, low, close, period) / close.replace(0, np.nan)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def ema_crossover(close: pd.Series, fast: int = 9, slow: int = 21) -> pd.Series:
    """
    EMA crossover signal: +1 when fast > slow, -1 otherwise.
    Useful as a trend-direction feature.
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    return np.sign(fast_ema - slow_ema)


def vwap_proxy(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average used as an intraday VWAP proxy."""
    return close.rolling(window=period).mean()


def price_vs_vwap_proxy(close: pd.Series, period: int = 20) -> pd.Series:
    """Relative deviation of price from VWAP proxy: price/SMA - 1."""
    sma = vwap_proxy(close, period)
    return close / sma.replace(0, np.nan) - 1.0


def atr_percentile(atr_series: pd.Series, lookback: int = 50) -> pd.Series:
    """Rolling percentile rank of ATR over lookback bars (0–100)."""
    return atr_series.rolling(window=lookback).rank(pct=True) * 100.0


def resample_ohlc(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample a 1-min OHLCV frame to an N-minute frame.

    Used so the same engine/feature code can run on a higher timeframe (the
    momentum_breakout strategy trades 15-min bars, where the BB band-break has
    genuine follow-through; 1-min extremes are noise). minutes<=1 is a no-op.
    """
    if minutes is None or minutes <= 1:
        return df.copy()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(f"{minutes}min").agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out


def compute_all(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0) -> pd.DataFrame:
    """
    Compute and attach all indicators to an OHLCV DataFrame in-place.

    Required columns: open, high, low, close, volume (volume may be zeros for index).
    Returns the same DataFrame with indicator columns added.
    """
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain open, high, low, close columns")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    upper, middle, lower, pb = bollinger_bands(close, bb_period, bb_std)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["percent_b"] = pb
    df["bb_width"] = bollinger_width(upper, lower, middle)

    df["rsi"] = rsi(close)
    df["atr"] = atr(high, low, close)
    df["atr_norm"] = atr_normalised(high, low, close)
    df["atr_pct"] = atr_percentile(df["atr"])

    df["ema_fast"] = ema(close, 9)
    df["ema_slow"] = ema(close, 21)
    df["ema_cross"] = ema_crossover(close)

    df["vwap_proxy"] = vwap_proxy(close)
    df["price_vs_vwap"] = price_vs_vwap_proxy(close)

    logger.debug(f'"Indicators computed for {len(df)} rows"')
    return df
