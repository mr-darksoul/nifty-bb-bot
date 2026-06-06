"""
WebSocket tick handler and 1-minute candle builder for NIFTY index.

Subscribes to Kite WebSocket ticks for NIFTY 50 and aggregates them
into 1-minute OHLCV candles. Notifies registered callbacks on each close.
"""

import asyncio
import logging
import threading
from collections import deque
from datetime import datetime, time as dtime
from typing import Callable, Deque, Dict, List, Optional

import pandas as pd

from config import (
    CANDLE_INTERVAL_MINUTES,
    NIFTY_INDEX_TOKEN,
    NIFTY_SYMBOL,
)

logger = logging.getLogger(__name__)

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
MAX_CANDLES_IN_MEMORY = 500   # rolling window kept in RAM


class CandleBuilder:
    """
    Aggregates individual ticks into N-minute OHLCV candles.
    Thread-safe via a lock.
    """

    def __init__(self, interval_minutes: int = CANDLE_INTERVAL_MINUTES) -> None:
        self.interval = interval_minutes
        self._lock = threading.Lock()
        self._current: Optional[Dict] = None
        self._candles: Deque[Dict] = deque(maxlen=MAX_CANDLES_IN_MEMORY)
        self._callbacks: List[Callable[[pd.DataFrame], None]] = []
        self.last_price: float = 0.0

    def register_callback(self, fn: Callable[[pd.DataFrame], None]) -> None:
        """Register a function to be called with the completed candle DataFrame."""
        self._callbacks.append(fn)

    def on_tick(self, tick: Dict) -> None:
        """
        Process a single tick from the Kite WebSocket.

        Args:
            tick: dict with keys: instrument_token, last_price, timestamp (or exchange_timestamp).
        """
        if tick.get("instrument_token") != NIFTY_INDEX_TOKEN:
            return

        price: float = float(tick.get("last_price", 0))
        if price <= 0:
            return

        ts: datetime = tick.get("exchange_timestamp") or tick.get("timestamp") or datetime.now()
        if isinstance(ts, str):
            ts = pd.Timestamp(ts).to_pydatetime()

        self.last_price = price

        # Collect closed-candle data under lock, then fire callbacks outside lock
        # to avoid blocking tick processing while callbacks do I/O or compute.
        closed_df = None
        with self._lock:
            bucket = self._candle_bucket(ts)
            if self._current is None or self._current["bucket"] != bucket:
                if self._current is not None:
                    closed_df = self._close_candle_locked()
                self._current = {
                    "bucket": bucket,
                    "datetime": bucket,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0,
                    "tick_count": 1,
                }
            else:
                c = self._current
                c["high"] = max(c["high"], price)
                c["low"] = min(c["low"], price)
                c["close"] = price
                c["tick_count"] += 1

        if closed_df is not None:
            for cb in list(self._callbacks):
                try:
                    cb(closed_df)
                except Exception as exc:
                    logger.error(f'"Candle callback error: {exc}"')

    def _candle_bucket(self, ts: datetime) -> datetime:
        """Round timestamp down to the nearest N-minute interval."""
        total_minutes = ts.hour * 60 + ts.minute
        floored = (total_minutes // self.interval) * self.interval
        return ts.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)

    def _close_candle_locked(self) -> pd.DataFrame:
        """Finalise current candle and store it. Must be called under lock.

        Returns the completed candle DataFrame for callbacks to consume
        after the lock is released.
        """
        c = self._current.copy()
        c.pop("bucket", None)
        self._candles.append(c)
        logger.info(
            f'"Candle closed: {c["datetime"]} O={c["open"]:.2f} H={c["high"]:.2f} '
            f'L={c["low"]:.2f} C={c["close"]:.2f}"'
        )
        return self.get_dataframe()

    def get_dataframe(self) -> pd.DataFrame:
        """Return the full candle history as a DataFrame."""
        if not self._candles:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(list(self._candles))
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    def inject_candle(self, row: Dict) -> None:
        """Manually inject a historical candle (used for warm-up from CSV/API)."""
        with self._lock:
            self._candles.append(row)


class DataFeed:
    """
    Manages the Kite WebSocket connection and routes ticks to CandleBuilder.
    """

    def __init__(self) -> None:
        self.builder = CandleBuilder()
        self._ws = None
        self._running = False

    def register_candle_callback(self, fn: Callable[[pd.DataFrame], None]) -> None:
        self.builder.register_callback(fn)

    def start(self) -> None:
        """Start WebSocket in a background thread."""
        if self._running:
            logger.warning('"DataFeed already running"')
            return

        from auth import get_kite
        from kiteconnect import KiteTicker

        kite = get_kite()
        api_key = kite.api_key
        access_token = kite.access_token

        if not access_token:
            raise RuntimeError("No access token available — complete Kite login first")

        self._ws = KiteTicker(api_key, access_token)

        def on_ticks(ws, ticks):
            for tick in ticks:
                self.builder.on_tick(tick)

        def on_connect(ws, response):
            ws.subscribe([NIFTY_INDEX_TOKEN])
            ws.set_mode(ws.MODE_FULL, [NIFTY_INDEX_TOKEN])
            logger.info(f'"WebSocket connected, subscribed to {NIFTY_SYMBOL}"')

        def on_close(ws, code, reason):
            logger.warning(f'"WebSocket closed: code={code} reason={reason}"')
            self._running = False

        def on_error(ws, code, reason):
            logger.error(f'"WebSocket error: code={code} reason={reason}"')

        self._ws.on_ticks = on_ticks
        self._ws.on_connect = on_connect
        self._ws.on_close = on_close
        self._ws.on_error = on_error

        self._running = True
        thread = threading.Thread(
            target=self._ws.connect,
            kwargs={"threaded": True},
            daemon=True,
            name="kite-websocket",
        )
        thread.start()
        logger.info('"DataFeed WebSocket thread started"')

    def stop(self) -> None:
        """Close the WebSocket connection."""
        if self._ws and self._running:
            self._ws.stop()
            self._running = False
            logger.info('"DataFeed stopped"')

    def warm_up(self, df: pd.DataFrame) -> None:
        """
        Pre-populate CandleBuilder with historical candles so indicators
        have enough data on startup.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
        """
        for ts, row in df.iterrows():
            self.builder.inject_candle({
                "datetime": ts,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0),
            })
        logger.info(f'"DataFeed warmed up with {len(df)} historical candles"')

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_price(self) -> float:
        return self.builder.last_price
