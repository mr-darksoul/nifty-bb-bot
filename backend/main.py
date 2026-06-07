"""
Entry point: wires all components together and starts the FastAPI server.

On every 1-min candle close, executes the full signal pipeline:
  1. Compute indicators
  2. Detect market regime
  3. Compute base %b signal
  4. Score signal quality via ML
  5. Load optimized params
  6. Check trade limits
  7. Select ATM option
  8. Place order

Exit checks run on every candle close when a position is open.
"""

import asyncio
import logging
import sys
import threading
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

import uvicorn

# Add backend/ to sys.path so imports work from both Railway and locally
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import (
    CANDLE_INTERVAL_MINUTES,
    CHOPPY_REGIME_ID,
    DRY_RUN,
    ENTRY_START_HOUR,
    ENTRY_START_MIN,
    FORCE_EXIT_HOUR,
    FORCE_EXIT_MIN,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MIN,
    MAX_TRADES_PER_DAY,
    PORT,
    SIGNAL_QUALITY_THRESHOLD,
    TRENDING_DOWN_REGIME_ID,
    load_optimized_params,
    validate_secrets,
)
from indicators import compute_all
from data_feed import DataFeed
from options_selector import OptionsSelector
from order_manager import OrderManager
from api_server import app, state, on_startup, on_shutdown

logger = logging.getLogger(__name__)

# ML modules are optional: a missing/broken ml package disables the trading
# pipeline but must not stop the REST API (status, candles, backtest) from
# serving. When unavailable, process_candle short-circuits.
try:
    from ml.feature_engineering import (
        build_features,
        build_regime_features,
        get_signal_features_at_bar,
    )
    from ml.regime_detector import RegimeDetector, REGIME_NAMES
    from ml.signal_filter import SignalFilter
    _ML_AVAILABLE = True
except Exception as _ml_exc:
    logger.error(f'"ML modules unavailable: {_ml_exc} — trading pipeline disabled, API still serves"')
    _ML_AVAILABLE = False
    REGIME_NAMES = {0: "TRENDING_DOWN", 1: "CHOPPY", 2: "TRENDING_UP"}


# ── Runtime globals ───────────────────────────────────────────────────────────

data_feed = DataFeed()
order_manager = OrderManager()
options_selector = OptionsSelector()
regime_detector = RegimeDetector() if _ML_AVAILABLE else None
signal_filter = SignalFilter() if _ML_AVAILABLE else None

_bot_task: Optional[asyncio.Task] = None
_stop_event = asyncio.Event()
_candle_event = asyncio.Event()
_latest_df: Optional[pd.DataFrame] = None
_latest_df_lock = threading.Lock()   # guards _latest_df across the WS and async threads
_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _market_open() -> bool:
    now = datetime.now().time()
    return dtime(MARKET_OPEN_HOUR, MARKET_OPEN_MIN) <= now < dtime(15, 30)


def _is_force_exit_time() -> bool:
    now = datetime.now().time()
    return now >= dtime(FORCE_EXIT_HOUR, FORCE_EXIT_MIN)


def _is_entry_allowed() -> bool:
    now = datetime.now().time()
    return now >= dtime(ENTRY_START_HOUR, ENTRY_START_MIN)


def _refresh_state(df: pd.DataFrame) -> None:
    """Update the shared API state from the latest indicator row."""
    if df.empty:
        return
    last = df.iloc[-1]
    state.nifty_price = float(last["close"])
    state.percent_b = float(last.get("percent_b", 0) or 0)
    state.rsi = float(last.get("rsi", 0) or 0)
    state.atr = float(last.get("atr", 0) or 0)
    state.last_candle_time = str(df.index[-1])
    state.market_open = _market_open()

    if order_manager.has_open_position:
        state.active_trade = order_manager.active_trade
    else:
        state.active_trade = None

    state.trades_today = len(order_manager.today_trades())
    state.daily_pnl = order_manager.today_pnl()


# ── Candle close handler ──────────────────────────────────────────────────────

def on_candle_close(df: pd.DataFrame) -> None:
    """Called by DataFeed on every 1-min candle close. Runs in WebSocket thread."""
    global _latest_df
    with _latest_df_lock:
        _latest_df = df
    if _event_loop is None:
        logger.error('"Candle received before bot event loop was registered"')
        return
    _event_loop.call_soon_threadsafe(_candle_event.set)


async def process_candle(df: pd.DataFrame) -> None:
    """
    Full signal pipeline executed on each candle close.
    Runs in the async event loop.
    """
    if not _market_open():
        return
    if not _ML_AVAILABLE:
        return

    # ── Step 1: Compute indicators ────────────────────────────────────────────
    # Pass pre-computed df_ind into build_features so compute_all runs only once.
    try:
        df_ind = compute_all(df.copy())
        df_feat = build_features(df_ind)   # skips recompute since bb_upper present
    except Exception as exc:
        logger.error(f'"Indicator computation failed: {exc}"')
        return

    _refresh_state(df_feat)

    # ── Force exit at 15:10 ───────────────────────────────────────────────────
    if _is_force_exit_time() and order_manager.has_open_position:
        await _do_exit(df_feat, reason="FORCE_EXIT")
        _end_of_day_summary()
        return

    # ── Exit checks (position open) ───────────────────────────────────────────
    if order_manager.has_open_position:
        await _check_exit(df_feat)

    # ── Entry pipeline ────────────────────────────────────────────────────────
    if order_manager.has_open_position:
        return
    if len(order_manager.today_trades()) >= MAX_TRADES_PER_DAY:
        return
    if not _is_entry_allowed():
        return

    # Step 2: Regime filter
    try:
        reg_feat = build_regime_features(df_ind).tail(60)
        current_regime = regime_detector.predict_regime(reg_feat)
        state.regime = current_regime
        state.regime_name = REGIME_NAMES.get(current_regime, "UNKNOWN")
    except Exception as exc:
        logger.warning(f'"Regime detection failed: {exc} — blocking entry (TRENDING_DOWN)"')
        current_regime = TRENDING_DOWN_REGIME_ID  # blocks entry; safe failure
        state.regime = current_regime
        state.regime_name = "UNKNOWN"

    if current_regime != CHOPPY_REGIME_ID:
        pb = float(df_feat.iloc[-1].get("percent_b", 0) or 0)
        logger.info(f'"Signal skipped: regime={state.regime_name} pb={pb:.3f}"')
        state.signal = "FILTERED_REGIME"
        return

    # Step 3: Base %b signal
    last = df_feat.iloc[-1]
    pb = float(last.get("percent_b", 0.5) or 0.5)
    params = load_optimized_params()

    direction = None
    if pb < params["bb_oversold"]:
        direction = "CE"
    elif pb > params["bb_overbought"]:
        direction = "PE"

    if direction is None:
        state.signal = "NONE"
        return

    # RSI filter
    rsi_val = float(last.get("rsi", 50) or 50)
    if not (params["rsi_min"] <= rsi_val <= params["rsi_max"]):
        logger.info(f'"Signal skipped: RSI={rsi_val:.1f} outside [{params["rsi_min"]}, {params["rsi_max"]}]"')
        state.signal = "FILTERED_RSI"
        return

    # Step 4: ML signal quality score
    try:
        feat_vec = get_signal_features_at_bar(df_feat, -1)
        score = signal_filter.score(feat_vec)
    except Exception as exc:
        logger.error(f'"Signal scoring failed: {exc} — rejecting signal"')
        score = 0.0

    state.signal = direction
    state.signal_quality_score = score

    if score < SIGNAL_QUALITY_THRESHOLD:
        logger.info(f'"ML filter rejected: score={score:.3f} < {SIGNAL_QUALITY_THRESHOLD}"')
        return

    # Steps 7–10: Select option, place order, alert
    await _do_entry(direction, pb, current_regime, score, params)


async def _check_exit(df_feat: pd.DataFrame) -> None:
    """Evaluate exit conditions for the open position."""
    trade = order_manager.active_trade
    if trade is None:
        return

    params = load_optimized_params()
    last = df_feat.iloc[-1]
    pb = float(last.get("percent_b", 0.5) or 0.5)
    bb_exit = params["bb_exit"]
    sl_buffer = params["sl_buffer"]

    direction = trade.direction
    entry_pb = trade.entry_pb
    sl_pb = entry_pb - (sl_buffer if direction == "CE" else -sl_buffer)

    exit_reason = None
    if direction == "CE":
        if pb >= bb_exit:
            exit_reason = "TARGET"
        elif pb <= sl_pb:
            exit_reason = "STOP_LOSS"
    else:
        if pb <= bb_exit:
            exit_reason = "TARGET"
        elif pb >= sl_pb:
            exit_reason = "STOP_LOSS"

    if exit_reason:
        await _do_exit(df_feat, exit_reason)


async def _do_entry(
    direction: str,
    pb: float,
    regime: int,
    score: float,
    params: dict,
) -> None:
    """Resolve a premium-capped option, place entry order, send alert."""
    try:
        spot = state.nifty_price
        if spot <= 0:
            logger.error('"Cannot enter trade: spot price is zero"')
            return

        expiry = options_selector.get_weekly_expiry()
        symbol, strike, token, ltp = options_selector.get_premium_capped_instrument(
            spot_price=spot,
            option_type=direction,
            expiry=expiry,
        )
        state.strike_candidates = list(options_selector.last_candidates)

        trade = order_manager.enter_trade(
            direction=direction,
            symbol=symbol,
            strike=strike,
            ltp=ltp,
            percent_b=pb,
            regime=regime,
            signal_quality_score=score,
        )

        if trade is None:
            return

    except Exception as exc:
        logger.error(f'"Entry failed: {exc}"')


async def _do_exit(df_feat: pd.DataFrame, reason: str) -> None:
    """Fetch LTP, close position, send alert."""
    trade = order_manager.active_trade
    if trade is None:
        return

    try:
        ltp = options_selector.get_ltp(0, trade.symbol)
        if ltp <= 0:
            ltp = state.nifty_price * 0.01   # fallback

        pb = float(df_feat.iloc[-1].get("percent_b", 0.5) or 0.5)
        order_manager.exit_trade(ltp=ltp, percent_b=pb, reason=reason)
    except Exception as exc:
        logger.error(f'"Exit failed: {exc}"')


def _end_of_day_summary() -> None:
    today = order_manager.today_trades()
    n = len(today)
    pnl = order_manager.today_pnl()
    wins = sum(1 for t in today if t.pnl > 0)
    win_rate = wins / n if n > 0 else 0.0
    logger.info(f'"Daily summary: trades={n} win_rate={win_rate:.0%} pnl=₹{pnl:.2f}"')


# ── Bot loop ──────────────────────────────────────────────────────────────────

async def _bot_loop() -> None:
    """Main async trading loop: waits for candle events and runs the pipeline."""
    state.bot_running = True
    logger.info(f'"Bot loop started (DRY_RUN={DRY_RUN})"')

    try:
        while not _stop_event.is_set():
            _candle_event.clear()
            try:
                await asyncio.wait_for(
                    _candle_event.wait(),
                    timeout=CANDLE_INTERVAL_MINUTES * 60 + 30,
                )
            except asyncio.TimeoutError:
                logger.warning('"No completed candle received before timeout; continuing bot loop"')
                continue

            if _stop_event.is_set():
                break

            with _latest_df_lock:
                df = _latest_df
            if df is not None and len(df) >= 20:
                await process_candle(df)

    except asyncio.CancelledError:
        logger.info('"Bot loop cancelled"')
    except Exception as exc:
        logger.error(f'"Bot loop error: {exc}"')
    finally:
        state.bot_running = False
        logger.info('"Bot loop exited"')


async def _start_bot() -> None:
    global _bot_task, _stop_event, _event_loop
    _event_loop = asyncio.get_running_loop()
    _stop_event.clear()
    data_feed.start()
    _bot_task = asyncio.create_task(_bot_loop())
    logger.info('"Bot started"')


async def _stop_bot() -> None:
    global _bot_task
    _stop_event.set()

    # Force-exit any open position
    with _latest_df_lock:
        latest_df = _latest_df
    if order_manager.has_open_position and latest_df is not None and _ML_AVAILABLE:
        try:
            df_ind = compute_all(latest_df.copy())
            df_feat = build_features(df_ind)
            await _do_exit(df_feat, reason="BOT_STOP")
        except Exception as exc:
            logger.error(f'"Force exit on stop failed: {exc}"')

    data_feed.stop()
    if _bot_task and not _bot_task.done():
        _bot_task.cancel()
        try:
            await _bot_task
        except asyncio.CancelledError:
            pass

    logger.info('"Bot stopped"')


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@on_startup
async def startup_event() -> None:
    """Initialise models, register callbacks, wire state."""
    # Load ML models (degrade gracefully if missing)
    if _ML_AVAILABLE:
        regime_detector.load()
        signal_filter.load()

    # Load today's trades from CSV (restart recovery)
    order_manager.load_today_from_csv()

    # Register candle callback
    data_feed.register_candle_callback(on_candle_close)

    # Warm up with historical data if available
    from config import DATA_CACHE_PATH
    if DATA_CACHE_PATH.exists():
        try:
            df_hist = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
            data_feed.warm_up(df_hist.tail(500))
        except Exception as exc:
            logger.warning(f'"DataFeed warm-up failed: {exc}"')

    # Register bot control functions with API state
    state.order_manager = order_manager
    state.data_feed = data_feed
    state._start_fn = _start_bot
    state._stop_fn = _stop_bot

    logger.info('"API server startup complete"')


@on_shutdown
async def shutdown_event() -> None:
    if state.bot_running:
        await _stop_bot()
    logger.info('"API server shutdown complete"')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validate_secrets()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        reload=False,
    )
