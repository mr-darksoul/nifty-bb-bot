"""
Options chain intelligence for NIFTY.

Fetches all CE/PE strikes for the current weekly expiry from Kite,
then computes:
  - Put-Call Ratio (PCR) by open interest
  - Max Pain strike (where option buyers lose the most)
  - Highest-OI call wall (resistance) and put wall (support)
  - Overall OI directional bias

Refreshes every OPTIONS_REFRESH_MIN minutes via a background thread.
Uses cached data between refreshes to avoid hitting Kite's rate limits.

Usage:
    start_oi_feed(spot_getter_fn)  # call once at startup
    oi = get_oi_reading()           # call from any thread
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from config import NFO_EXCHANGE

logger = logging.getLogger(__name__)

OPTIONS_REFRESH_MIN: int = 5
STRIKES_EACH_SIDE: int = 20    # fetch ATM ± 20 strikes = 40 total per type
NIFTY_STRIKE_STEP: int = 50


@dataclass
class OIReading:
    pcr: float = 1.0            # put OI / call OI
    max_pain: float = 0.0       # strike minimising writers' losses
    call_wall: float = 0.0      # strike with highest call OI (resistance)
    put_wall: float = 0.0       # strike with highest put OI (support)
    oi_bias: str = "NEUTRAL"    # "BULLISH", "BEARISH", "NEUTRAL"
    fetched_at: Optional[datetime] = None
    strikes_fetched: int = 0

    @property
    def age_minutes(self) -> float:
        if not self.fetched_at:
            return 999.0
        return (datetime.utcnow() - self.fetched_at).total_seconds() / 60

    @property
    def is_fresh(self) -> bool:
        return self.age_minutes < OPTIONS_REFRESH_MIN * 2

    def confirms_direction(self, direction: str) -> bool:
        """True if OI bias is aligned with the proposed trade direction."""
        if not self.is_fresh:
            return True   # stale data → don't penalise
        if direction == "CE":
            return self.oi_bias in ("BULLISH", "NEUTRAL")
        return self.oi_bias in ("BEARISH", "NEUTRAL")

    def distance_to_call_wall_pct(self, price: float) -> float:
        """% distance from current price to nearest call wall (resistance)."""
        if self.call_wall <= 0 or price <= 0:
            return 99.0
        return (self.call_wall - price) / price * 100

    def distance_to_put_wall_pct(self, price: float) -> float:
        """% distance from current price to nearest put wall (support)."""
        if self.put_wall <= 0 or price <= 0:
            return 99.0
        return (price - self.put_wall) / price * 100


def _fetch_oi(spot: float) -> OIReading:
    """Fetch full NIFTY options chain from Kite and compute OI metrics."""
    try:
        from auth import get_kite
        kite = get_kite()
        if not kite or not getattr(kite, "access_token", None):
            return OIReading()
    except Exception as exc:
        logger.warning(f'"OI: Kite not available: {exc}"')
        return OIReading()

    try:
        from options_selector import OptionsSelector
        sel = OptionsSelector()
        expiry = sel.get_weekly_expiry()
        instruments = sel._get_instruments()
    except Exception as exc:
        logger.warning(f'"OI: instrument load failed: {exc}"')
        return OIReading()

    atm = round(spot / NIFTY_STRIKE_STEP) * NIFTY_STRIKE_STEP
    strike_range = set(
        atm + i * NIFTY_STRIKE_STEP
        for i in range(-STRIKES_EACH_SIDE, STRIKES_EACH_SIDE + 1)
    )

    chain = instruments[
        (instruments["expiry"] == expiry) &
        (instruments["strike"].isin(strike_range))
    ].copy()

    if chain.empty:
        logger.warning('"OI: no instruments found for this expiry"')
        return OIReading()

    quote_keys = [f"{NFO_EXCHANGE}:{s}" for s in chain["tradingsymbol"].tolist()]
    # kite.quote() returns depth + OI; batch limit ~500 symbols
    try:
        raw_quotes = kite.quote(quote_keys[:400])
    except Exception as exc:
        logger.warning(f'"OI: kite.quote() failed: {exc}"')
        return OIReading()

    rows = []
    for key, data in raw_quotes.items():
        sym = key.split(":", 1)[-1]
        match = chain[chain["tradingsymbol"] == sym]
        if match.empty:
            continue
        opt_type = match.iloc[0]["instrument_type"]
        strike = float(match.iloc[0]["strike"])
        oi = data.get("oi", 0) or 0
        rows.append({"strike": strike, "type": opt_type, "oi": oi})

    if not rows:
        return OIReading()

    df = pd.DataFrame(rows)
    ce_oi = df[df["type"] == "CE"].groupby("strike")["oi"].sum()
    pe_oi = df[df["type"] == "PE"].groupby("strike")["oi"].sum()

    total_ce = float(ce_oi.sum())
    total_pe = float(pe_oi.sum())
    pcr = total_pe / total_ce if total_ce > 0 else 1.0

    # Max pain: strike that minimises total ITM value for option buyers
    all_strikes = sorted(set(ce_oi.index) | set(pe_oi.index))
    pain = {}
    for s in all_strikes:
        ce_pain = sum(max(0.0, s - k) * ce_oi.get(k, 0) for k in ce_oi.index)
        pe_pain = sum(max(0.0, k - s) * pe_oi.get(k, 0) for k in pe_oi.index)
        pain[s] = ce_pain + pe_pain
    max_pain_strike = float(min(pain, key=pain.get)) if pain else spot

    call_wall = float(ce_oi.idxmax()) if not ce_oi.empty else 0.0
    put_wall = float(pe_oi.idxmax()) if not pe_oi.empty else 0.0

    # PCR interpretation for intraday:
    #   PCR > 1.3 → heavy put writing → institutions expect stable/rising market → BULLISH
    #   PCR < 0.75 → heavy call writing → institutions capping upside → BEARISH
    if pcr > 1.3:
        bias = "BULLISH"
    elif pcr < 0.75:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    logger.info(
        f'"OI fetched: PCR={pcr:.3f} bias={bias} max_pain={max_pain_strike:.0f} '
        f'call_wall={call_wall:.0f} put_wall={put_wall:.0f} n={len(rows)}"'
    )
    return OIReading(
        pcr=round(pcr, 3),
        max_pain=max_pain_strike,
        call_wall=call_wall,
        put_wall=put_wall,
        oi_bias=bias,
        fetched_at=datetime.utcnow(),
        strikes_fetched=len(rows),
    )


# ── Module singleton ──────────────────────────────────────────────────────────

_reading: OIReading = OIReading()
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None


def get_oi_reading() -> OIReading:
    """Return the latest OI reading (thread-safe, non-blocking)."""
    with _lock:
        return _reading


def start_oi_feed(spot_getter: Callable[[], float]) -> None:
    """
    Start the background OI polling thread.
    spot_getter() is called each cycle to get the current NIFTY spot price.
    Idempotent — safe to call multiple times.
    """
    global _thread

    def _loop() -> None:
        global _reading
        while True:
            try:
                spot = spot_getter()
                if spot > 0:
                    result = _fetch_oi(spot)
                    with _lock:
                        _reading = result
            except Exception as exc:
                logger.error(f'"OI feed error: {exc}"')
            time.sleep(OPTIONS_REFRESH_MIN * 60)

    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="oi-feed")
    _thread.start()
    logger.info('"OI feed started"')
