"""
Market structure: VWAP, PDH/PDL, and swing-based support/resistance.

VWAP: cumulative from 9:15 AM each session using today's 1-min bars.
PDH/PDL: previous day high/low from Kite historical API (fetched once at startup).
Swing S/R: fractal highs/lows from the last ~250 1-min bars (resampled to 5-min).

Usage:
    warm_up_pdh_pdl()            # call once after Kite auth
    ms = get_market_structure(df_1m)   # call on every candle close
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SWING_STRENGTH: int = 2       # bars on each side that must be lower/higher
MAX_SWING_LEVELS: int = 5     # keep only the most recent swing highs/lows


@dataclass
class MarketStructure:
    vwap: float = 0.0
    vwap_upper1: float = 0.0   # VWAP + 1σ
    vwap_lower1: float = 0.0   # VWAP - 1σ
    pdh: float = 0.0           # previous day high
    pdl: float = 0.0           # previous day low
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)

    def price_above_vwap(self, price: float) -> bool:
        return price > self.vwap if self.vwap > 0 else True

    def _resistance_levels(self) -> List[float]:
        levels = []
        if self.pdh > 0:
            levels.append(self.pdh)
        if self.vwap_upper1 > 0:
            levels.append(self.vwap_upper1)
        levels.extend(self.swing_highs)
        return levels

    def _support_levels(self) -> List[float]:
        levels = []
        if self.pdl > 0:
            levels.append(self.pdl)
        if self.vwap_lower1 > 0:
            levels.append(self.vwap_lower1)
        levels.extend(self.swing_lows)
        return levels

    def nearest_resistance(self, price: float) -> Optional[float]:
        above = [l for l in self._resistance_levels() if l > price * 1.0005]
        return min(above) if above else None

    def nearest_support(self, price: float) -> Optional[float]:
        below = [l for l in self._support_levels() if l < price * 0.9995]
        return max(below) if below else None

    def distance_to_resistance_pct(self, price: float) -> float:
        r = self.nearest_resistance(price)
        if r is None or price <= 0:
            return 99.0
        return (r - price) / price * 100

    def distance_to_support_pct(self, price: float) -> float:
        s = self.nearest_support(price)
        if s is None or price <= 0:
            return 99.0
        return (price - s) / price * 100


def _compute_vwap(df_1m: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute session VWAP and ±1σ bands from today's 1-min bars.
    Returns (vwap, upper_1σ, lower_1σ). Returns (0, 0, 0) if no today bars.
    """
    today = date.today()
    today_bars = df_1m[df_1m.index.date == today] if not df_1m.empty else df_1m

    if today_bars.empty:
        return 0.0, 0.0, 0.0

    tp = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3.0
    vol = today_bars.get("volume", pd.Series(1, index=today_bars.index))
    vol = vol.where(vol > 0, 1)

    cum_pv = (tp * vol).cumsum()
    cum_v = vol.cumsum()
    vwap_s = cum_pv / cum_v

    # VWAP σ via running variance: Var(x) = E(x²) - E(x)²
    cum_p2v = (tp ** 2 * vol).cumsum()
    variance = (cum_p2v / cum_v) - (vwap_s ** 2)
    sigma = variance.clip(lower=0).apply(np.sqrt)

    vwap = float(vwap_s.iloc[-1])
    sig = float(sigma.iloc[-1])
    return vwap, vwap + sig, vwap - sig


def _detect_swings(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Fractal swing high/low detection.
    A swing high: bar where high > all adjacent SWING_STRENGTH bars on each side.
    """
    h = df["high"].values
    l = df["low"].values
    n = len(h)
    s = SWING_STRENGTH
    highs: List[float] = []
    lows: List[float] = []
    for i in range(s, n - s):
        if all(h[i] > h[i - j] for j in range(1, s + 1)) and \
           all(h[i] > h[i + j] for j in range(1, s + 1)):
            highs.append(float(h[i]))
        if all(l[i] < l[i - j] for j in range(1, s + 1)) and \
           all(l[i] < l[i + j] for j in range(1, s + 1)):
            lows.append(float(l[i]))
    return highs[-MAX_SWING_LEVELS:], lows[-MAX_SWING_LEVELS:]


# ── PDH/PDL module state ──────────────────────────────────────────────────────

_pdh: float = 0.0
_pdl: float = 0.0
_pdhl_date: Optional[date] = None
_pdhl_lock = threading.Lock()


def warm_up_pdh_pdl() -> None:
    """
    Fetch yesterday's OHLCV from Kite historical API to seed PDH/PDL.
    Call once after Kite authentication is confirmed. Non-fatal on failure.
    """
    global _pdh, _pdl, _pdhl_date
    try:
        from auth import get_kite
        from config import NIFTY_INDEX_TOKEN
        kite = get_kite()
        if not kite or not getattr(kite, "access_token", None):
            logger.warning('"PDH/PDL: Kite not authenticated — will retry on next startup"')
            return
        # Go back 5 calendar days to safely cover weekends and holidays
        from_d = date.today() - timedelta(days=5)
        to_d = date.today()
        bars = kite.historical_data(
            NIFTY_INDEX_TOKEN,
            from_date=from_d,
            to_date=to_d,
            interval="day",
        )
        if len(bars) >= 2:
            # bars[-1] is today's partial bar; bars[-2] is yesterday's complete bar
            prev = bars[-2]
            with _pdhl_lock:
                _pdh = float(prev["high"])
                _pdl = float(prev["low"])
                _pdhl_date = date.today()
            logger.info(f'"PDH/PDL loaded: H={_pdh:.2f} L={_pdl:.2f}"')
        else:
            logger.warning(f'"PDH/PDL: only {len(bars)} bars returned — skipping"')
    except Exception as exc:
        logger.warning(f'"PDH/PDL warm-up failed: {exc}"')


def get_market_structure(df_1m: pd.DataFrame) -> MarketStructure:
    """
    Build a full MarketStructure snapshot from the rolling 1-min DataFrame.
    Purely computational — safe to call on every candle close.
    """
    ms = MarketStructure()

    # 1. VWAP
    ms.vwap, ms.vwap_upper1, ms.vwap_lower1 = _compute_vwap(df_1m)

    # 2. PDH/PDL
    with _pdhl_lock:
        ms.pdh = _pdh
        ms.pdl = _pdl

    # 3. Swing S/R from last 250 1-min bars resampled to 5-min
    if len(df_1m) >= 50:
        try:
            from indicators import resample_ohlc
            df5 = resample_ohlc(df_1m.tail(250), 5)
            if len(df5) >= SWING_STRENGTH * 2 + 1:
                ms.swing_highs, ms.swing_lows = _detect_swings(df5)
        except Exception as exc:
            logger.debug(f'"Swing S/R computation skipped: {exc}"')

    return ms
