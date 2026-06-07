"""
Multi-timeframe snapshot builder.

Resamples the rolling 1-min DataFrame (from CandleBuilder) into
5-min, 15-min, and 60-min bars, then runs indicators on each.

Call get_snapshot(df_1m) on every 1-min candle close.
Returns a MultiTFSnapshot with per-TF bias and an overall alignment score.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from indicators import compute_all, resample_ohlc

logger = logging.getLogger(__name__)

# Minimum completed bars required before a TF is considered valid
_MIN_BARS = {5: 22, 15: 22, 60: 10}


@dataclass
class TFBar:
    timeframe_min: int
    close: float
    percent_b: float
    rsi: float
    ema_cross: float        # +1.0 fast>slow, -1.0 fast<slow
    above_ema_slow: bool    # price > EMA-21 on this TF
    bars: int

    @property
    def bias(self) -> Optional[int]:
        """
        Directional bias: +1 bullish, -1 bearish, 0 neutral, None=no data.
        Majority vote across 4 sub-signals.
        """
        if self.bars < _MIN_BARS.get(self.timeframe_min, 10):
            return None
        bull = (
            int(self.percent_b > 0.55)
            + int(self.rsi > 52)
            + int(self.ema_cross > 0)
            + int(self.above_ema_slow)
        )
        bear = (
            int(self.percent_b < 0.45)
            + int(self.rsi < 48)
            + int(self.ema_cross < 0)
            + int(not self.above_ema_slow)
        )
        if bull >= 3:
            return 1
        if bear >= 3:
            return -1
        return 0


@dataclass
class MultiTFSnapshot:
    tf5: Optional[TFBar]
    tf15: Optional[TFBar]
    tf60: Optional[TFBar]
    alignment_score: int = 0          # net bullish timeframes: -3 to +3
    dominant_direction: Optional[str] = None   # "CE", "PE", or None

    def compute(self) -> None:
        frames = [self.tf5, self.tf15, self.tf60]
        biases = [f.bias for f in frames if f is not None and f.bias is not None]
        self.alignment_score = sum(biases)
        if self.alignment_score >= 2:
            self.dominant_direction = "CE"
        elif self.alignment_score <= -2:
            self.dominant_direction = "PE"
        else:
            self.dominant_direction = None

    def aligned_frames(self, direction: str) -> int:
        """Count how many timeframes agree with the given direction."""
        target_bias = 1 if direction == "CE" else -1
        frames = [self.tf5, self.tf15, self.tf60]
        return sum(
            1 for f in frames
            if f is not None and f.bias == target_bias
        )


def _build_tf_bar(df_1m: pd.DataFrame, tf_min: int) -> Optional[TFBar]:
    """Resample 1-min data to tf_min and return a TFBar from the latest closed bar."""
    try:
        df = resample_ohlc(df_1m, tf_min)
        n = len(df)
        if n < 2:
            return None
        df = compute_all(df.copy())
        # Use the last COMPLETED bar (drop in-progress bar at index -1)
        # For intraday data, the most recent bar may still be forming.
        # We use iloc[-2] when the last bar is only 1 period old; otherwise iloc[-1].
        last = df.iloc[-2] if n >= 3 else df.iloc[-1]
        close = float(last["close"])
        return TFBar(
            timeframe_min=tf_min,
            close=close,
            percent_b=float(last.get("percent_b", 0.5) or 0.5),
            rsi=float(last.get("rsi", 50) or 50),
            ema_cross=float(last.get("ema_cross", 0) or 0),
            above_ema_slow=close > float(last.get("ema_slow", close) or close),
            bars=n,
        )
    except Exception as exc:
        logger.debug(f'"MTF build failed for {tf_min}m: {exc}"')
        return None


def get_snapshot(df_1m: pd.DataFrame) -> MultiTFSnapshot:
    """
    Build a MultiTFSnapshot by resampling from the 1-min candle DataFrame.
    Safe to call on every candle close — purely computational, no I/O.
    """
    snap = MultiTFSnapshot(
        tf5=_build_tf_bar(df_1m, 5),
        tf15=_build_tf_bar(df_1m, 15),
        tf60=_build_tf_bar(df_1m, 60),
    )
    snap.compute()
    return snap
