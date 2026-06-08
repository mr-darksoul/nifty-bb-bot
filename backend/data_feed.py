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

    Lifecycle note (important): KiteTicker runs on Twisted's reactor, and the
    reactor cannot be restarted once stopped within the same process. So we
    deliberately never call ``KiteTicker.stop()`` (which calls ``reactor.stop()``).
    Instead the reactor/WebSocket thread is launched exactly once per process and
    the connection is kept alive across bot stop→start cycles. ``stop()`` pauses
    tick processing (``_running`` flag) rather than tearing down the reactor, and
    ``start()`` resumes it. This makes the Stop→Start dashboard flow safe; the old
    behaviour raised ``twisted.internet.error.ReactorNotRestartable`` and left a
    dead ticker thread reporting ``bot_running: True`` with zero ticks.
    """

    def __init__(self) -> None:
        self.builder = CandleBuilder()
        self._ws = None
        # `_running` is user intent: whether ticks should be fed to the builder.
        # It is NOT the WebSocket/reactor connection state — the connection is
        # kept alive while paused so it can be resumed without a reactor restart.
        self._running = False
        # `_thread_started` records that the reactor/WS thread has been launched
        # once. The reactor is process-global and single-shot, so we launch it
        # at most once and reuse it for every subsequent start.
        self._thread_started = False
        # The Kite access token baked into the current `_ws`'s socket URL.
        # Zerodha tokens expire daily and change on re-login, so on resume we
        # compare against the live token and rebuild the ticker if it rotated.
        self._ws_token = None

    def register_candle_callback(self, fn: Callable[[pd.DataFrame], None]) -> None:
        self.builder.register_callback(fn)

    # ── KiteTicker callbacks (bound methods so a rebuilt ticker can reuse them) ──

    def _on_ticks(self, ws, ticks) -> None:
        # Drop ticks while paused so a "stopped" bot builds no candles and fires
        # no callbacks, even though the socket stays connected.
        if not self._running:
            return
        for tick in ticks:
            self.builder.on_tick(tick)

    def _on_connect(self, ws, response) -> None:
        ws.subscribe([NIFTY_INDEX_TOKEN])
        ws.set_mode(ws.MODE_FULL, [NIFTY_INDEX_TOKEN])
        logger.info(f'"WebSocket connected, subscribed to {NIFTY_SYMBOL}"')

    def _on_close(self, ws, code, reason) -> None:
        # Do NOT flip `_running` here: a transient close is followed by
        # KiteTicker's auto-reconnect, and `_running` reflects user intent,
        # not socket state. Flipping it would falsely mark the feed stopped.
        logger.warning(f'"WebSocket closed: code={code} reason={reason}"')

    def _on_error(self, ws, code, reason) -> None:
        logger.error(f'"WebSocket error: code={code} reason={reason}"')

    def _build_ws(self, api_key: str, access_token: str):
        """Construct a KiteTicker wired to this feed's callbacks. The token is
        baked into the socket URL at construction, so a new token needs a new
        ticker."""
        from kiteconnect import KiteTicker
        ws = KiteTicker(api_key, access_token)
        ws.on_ticks = self._on_ticks
        ws.on_connect = self._on_connect
        ws.on_close = self._on_close
        ws.on_error = self._on_error
        return ws

    def start(self) -> None:
        """Start (or resume) the live feed.

        First call launches the KiteTicker WebSocket in a background thread.
        Subsequent calls resume tick processing on the still-running reactor
        without restarting it. Idempotent: a redundant start while already
        running is a no-op warning, so rapid double-starts cannot spawn a second
        ticker thread or crash the reactor.

        A valid Kite access token is required on every start (including resume):
        if the token rotated since the ticker was built (daily expiry / a
        re-login), the ticker is rebuilt with the fresh token and reconnected on
        the already-running reactor, so a re-login + restart actually recovers
        ticks instead of silently reconnecting with a dead token.
        """
        if self._running:
            logger.warning('"DataFeed already running"')
            return

        from auth import get_kite

        kite = get_kite()
        api_key = kite.api_key
        access_token = kite.access_token

        if not access_token:
            raise RuntimeError("No access token available — complete Kite login first")

        # Resume path: reactor/WS thread already launched earlier in this process.
        if self._thread_started:
            self._running = True
            if access_token != self._ws_token:
                # Token rotated since the ticker was built. The token lives in
                # KiteTicker's socket URL, so reusing the old ticker would
                # reconnect with a dead token. Rebuild with the fresh token and
                # connect on the ALREADY-running reactor (never restarting it).
                logger.info('"DataFeed resume: Kite token changed — rebuilding ticker"')
                self._reconnect_with_new_token(api_key, access_token)
            else:
                logger.info('"DataFeed resumed (reactor kept alive)"')
                self._ensure_connected()
            return

        # First start: build the ticker and launch the reactor thread once.
        self._ws = self._build_ws(api_key, access_token)
        self._ws_token = access_token
        self._running = True
        self._thread_started = True
        thread = threading.Thread(
            target=self._ws.connect,
            kwargs={"threaded": True},
            daemon=True,
            name="kite-websocket",
        )
        thread.start()
        logger.info('"DataFeed WebSocket thread started"')

    def _reconnect_with_new_token(self, api_key: str, access_token: str) -> None:
        """Swap in a freshly-built ticker carrying the new token and connect it
        on the already-running reactor.

        The old ticker's stale-token auto-reconnect is halted and its socket
        closed (``stop_retry`` + ``_close``) — but the reactor itself is NEVER
        stopped, preserving the single-reactor invariant. Never raises.
        """
        old = self._ws
        new = self._build_ws(api_key, access_token)
        self._ws = new
        self._ws_token = access_token

        def _swap():
            # Runs in the reactor thread (via callFromThread): safe to touch
            # Twisted protocol/factory state here.
            try:
                if old is not None:
                    old.stop_retry()          # stop stale-token reconnect attempts
                    old._close(reason="token rotated")
            except Exception as exc:
                logger.error(f'"Old ticker close failed: {exc}"')
            try:
                # Reactor is already running, so connect() just opens a fresh
                # WebSocket and does NOT attempt to start the reactor.
                new.connect(threaded=True)
            except Exception as exc:
                logger.error(f'"Ticker reconnect (new token) failed: {exc}"')

        try:
            from twisted.internet import reactor
            if reactor.running:
                reactor.callFromThread(_swap)
            else:
                # Anomaly: thread was started but the reactor isn't running.
                # Launch connect in a thread (connect() will start it) as a
                # best-effort recovery.
                threading.Thread(
                    target=new.connect, kwargs={"threaded": True},
                    daemon=True, name="kite-websocket",
                ).start()
        except Exception as exc:
            logger.error(f'"DataFeed token-rotation reconnect failed: {exc}"')

    def _ensure_connected(self) -> None:
        """Best-effort reconnect on resume if the socket dropped while paused.

        Auto-reconnect normally keeps the connection alive, but if it exhausted
        its retries during a long pause the socket may be closed. Re-issue
        ``connect()`` on the existing (running) reactor via ``callFromThread``;
        the reactor is already running so this opens a fresh WebSocket without
        attempting a reactor restart. Never raises — resume must not fail here.
        """
        try:
            if self._ws is None or self._ws.is_connected():
                return
            from twisted.internet import reactor
            if reactor.running:
                reactor.callFromThread(self._ws.connect, threaded=True)
                logger.info('"DataFeed reconnect requested on running reactor"')
        except Exception as exc:
            logger.error(f'"DataFeed reconnect attempt failed: {exc}"')

    def stop(self) -> None:
        """Pause the live feed.

        Stops feeding ticks to the candle builder but deliberately leaves the
        KiteTicker connection and Twisted reactor alive, so a later ``start()``
        can resume without triggering ReactorNotRestartable.
        """
        if self._running:
            self._running = False
            logger.info('"DataFeed paused (connection kept alive)"')

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
