"""
Entry point: wires all components together and starts the FastAPI server.

On every 5-min candle close, executes the full signal pipeline:
  1. Compute indicators
  2. Detect market regime
  3. Compute base %b signal
  4. Score signal quality via ML
  5. Load optimized params
  6. Check trade limits
  7. Select ATM option
  8. Place order
  9. Send Telegram alert

Exit checks run on every candle close when a position is open.
"""

import asyncio
import logging
import sys
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
    FORCE_EXIT_HOUR,
    FORCE_EXIT_MIN,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MIN,
    MAX_TRADES_PER_DAY,
    PORT,
    SIGNAL_QUALITY_THRESHOLD,
    load_optimized_params,
    validate_secrets,
)
from indicators import compute_all
from ml.feature_engineering import build_features, build_regime_features, get_signal_features_at_bar
from ml.regime_detector import RegimeDetector, REGIME_NAMES
from ml.signal_filter import SignalFilter
from data_feed import DataFeed
from options_selector import OptionsSelector
from order_manager import OrderManager
from telegram_notifier import (
    notify_bot_started,
    notify_bot_stopped,
    notify_entry,
    notify_exit,
    notify_ml_filter,
    notify_regime_filter,
    notify_daily_summary,
    notify_error,
)
from api_server import app, state

logger = logging.getLogger(__name__)


# ── Runtime globals ───────────────────────────────────────────────────────────

data_feed = DataFeed()
order_manager = OrderManager()
options_selector = OptionsSelector()
regime_detector = RegimeDetector()
signal_filter = SignalFilter()

_bot_task: Optional[asyncio.Task] = None
_stop_event = asyncio.Event()
_candle_event = asyncio.Event()
_latest_df: Optional[pd.DataFrame] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _market_open() -> bool:
    now = datetime.now().time()
    return dtime(MARKET_OPEN_HOUR, MARKET_OPEN_MIN) <= now < dtime(15, 30)


def _is_force_exit_time() -> bool:
    now = datetime.now().time()
    return now >= dtime(FORCE_EXIT_HOUR, FORCE_EXIT_MIN)


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
    """Called by DataFeed on every 5-min candle close. Runs in WebSocket thread."""
    global _latest_df
    _latest_df = df
    # Signal the async bot loop
    try:
        asyncio.get_event_loop().call_soon_threadsafe(_candle_event.set)
    except Exception:
        pass


async def process_candle(df: pd.DataFrame) -> None:
    """
    Full signal pipeline executed on each candle close.
    Runs in the async event loop.
    """
    if not _market_open():
        return

    # ── Step 1: Compute indicators ────────────────────────────────────────────
    try:
        df_ind = compute_all(df.copy())
        df_feat = build_features(df_ind)
    except Exception as exc:
        logger.error(f'"Indicator computation failed: {exc}"')
        notify_error("Indicator computation", str(exc))
        return

    _refresh_state(df_feat)

    # ── Exit checks (position open) ───────────────────────────────────────────
    if order_manager.has_open_position:
        await _check_exit(df_feat)

    # ── Force exit at 15:10 ───────────────────────────────────────────────────
    if _is_force_exit_time() and order_manager.has_open_position:
        await _do_exit(df_feat, reason="FORCE_EXIT")
        _end_of_day_summary()
        return

    # ── Entry pipeline ────────────────────────────────────────────────────────
    if order_manager.has_open_position:
        return
    if len(order_manager.today_trades()) >= MAX_TRADES_PER_DAY:
        return

    # Step 2: Regime filter
    try:
        reg_feat = build_regime_features(df_ind).tail(60)
        current_regime = regime_detector.predict_regime(reg_feat)
        state.regime = current_regime
        state.regime_name = REGIME_NAMES.get(current_regime, "UNKNOWN")
    except Exception as exc:
        logger.warning(f'"Regime detection failed: {exc} — assuming CHOPPY"')
        current_regime = CHOPPY_REGIME_ID
        state.regime = current_regime
        state.regime_name = "CHOPPY"

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

    # Step 4: ML signal quality score
    try:
        feat_vec = get_signal_features_at_bar(df_feat, -1)
        score = signal_filter.score(feat_vec)
    except Exception as exc:
        logger.warning(f'"Signal scoring failed: {exc} — using score=1.0"')
        score = 1.0

    state.signal = direction
    state.signal_quality_score = score

    if score < SIGNAL_QUALITY_THRESHOLD:
        logger.info(f'"ML filter rejected: score={score:.3f} < {SIGNAL_QUALITY_THRESHOLD}"')
        notify_ml_filter(pb, score, SIGNAL_QUALITY_THRESHOLD)
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
    """Resolve ATM option, place entry order, send alert."""
    try:
        spot = state.nifty_price
        if spot <= 0:
            logger.error('"Cannot enter trade: spot price is zero"')
            return

        expiry = options_selector.get_weekly_expiry()
        symbol, strike, token = options_selector.get_atm_instrument(spot, direction, expiry)
        ltp = options_selector.get_ltp(token, symbol) if token else spot * 0.01

        if ltp <= 0:
            ltp = spot * 0.01   # fallback estimate

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

        notify_entry(
            trade_id=trade.trade_id,
            direction=direction,
            symbol=symbol,
            strike=strike,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            percent_b=pb,
            signal_quality_score=score,
            regime_name=REGIME_NAMES.get(regime, "UNKNOWN"),
            dry_run=DRY_RUN,
        )

    except Exception as exc:
        logger.error(f'"Entry failed: {exc}"')
        notify_error("Entry execution", str(exc))


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
        closed = order_manager.exit_trade(ltp=ltp, percent_b=pb, reason=reason)

        if closed:
            notify_exit(
                trade_id=closed.trade_id,
                symbol=closed.symbol,
                exit_price=closed.exit_price,
                pnl=closed.pnl,
                exit_reason=reason,
                daily_pnl=order_manager.today_pnl(),
                dry_run=DRY_RUN,
            )
    except Exception as exc:
        logger.error(f'"Exit failed: {exc}"')
        notify_error("Exit execution", str(exc))


def _end_of_day_summary() -> None:
    today = order_manager.today_trades()
    n = len(today)
    pnl = order_manager.today_pnl()
    wins = sum(1 for t in today if t.pnl > 0)
    win_rate = wins / n if n > 0 else 0.0
    notify_daily_summary(n, pnl, win_rate, DRY_RUN)


# ── Bot loop ──────────────────────────────────────────────────────────────────

async def _bot_loop() -> None:
    """Main async trading loop: waits for candle events and runs the pipeline."""
    state.bot_running = True
    logger.info('"Bot loop started"')
    notify_bot_started(DRY_RUN)

    try:
        while not _stop_event.is_set():
            _candle_event.clear()
            await asyncio.wait_for(
                _candle_event.wait(),
                timeout=CANDLE_INTERVAL_MINUTES * 60 + 30,
            )

            if _stop_event.is_set():
                break

            df = _latest_df
            if df is not None and len(df) >= 20:
                await process_candle(df)

    except asyncio.CancelledError:
        logger.info('"Bot loop cancelled"')
    except Exception as exc:
        logger.error(f'"Bot loop error: {exc}"')
        notify_error("Bot loop", str(exc))
    finally:
        state.bot_running = False
        logger.info('"Bot loop exited"')


async def _start_bot() -> None:
    global _bot_task, _stop_event
    _stop_event.clear()
    data_feed.start()
    _bot_task = asyncio.create_task(_bot_loop())
    logger.info('"Bot started"')


async def _stop_bot() -> None:
    global _bot_task
    _stop_event.set()

    # Force-exit any open position
    if order_manager.has_open_position and _latest_df is not None:
        try:
            df_ind = compute_all(_latest_df.copy())
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

    notify_bot_stopped()
    logger.info('"Bot stopped"')


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Initialise models, register callbacks, wire state."""
    # Load ML models (degrade gracefully if missing)
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


@app.on_event("shutdown")
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
