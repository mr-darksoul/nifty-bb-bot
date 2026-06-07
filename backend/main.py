"""
Entry point: wires all components together and starts the FastAPI server.

On every 1-min candle close, executes the full signal pipeline:
  1. Compute indicators (1-min + multi-timeframe resampled)
  2. Detect market regime (HMM, optional)
  3. Compute base %b momentum signal (BB breakout on strategy TF)
  4. Apply ATR + RSI gates
  5. [NEW] Signal Fusion — confluence scoring across:
       • Multi-TF alignment (5m / 15m / 60m)
       • VWAP position
       • Options OI / PCR
       • News + govt press release sentiment
       • Swing S/R proximity
  6. Score signal quality via ML XGBoost filter (optional)
  7. Select premium-capped option, place order
  8. Lock ATR-based spot target/stop

Exit checks run on every candle close when a position is open.
Target: 70% win rate at 1:3 R:R.
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
    STRATEGY,
    STRATEGY_TIMEFRAME_MIN,
    TRENDING_DOWN_REGIME_ID,
    USE_ML_FILTER,
    USE_REGIME_FILTER,
    USE_SIGNAL_FUSION,
    load_optimized_params,
    validate_secrets,
)
from indicators import compute_all, resample_ohlc
from data_feed import DataFeed
from options_selector import OptionsSelector
from order_manager import OrderManager
from api_server import app, state, on_startup, on_shutdown

logger = logging.getLogger(__name__)

# ── Multi-source intelligence modules (opt-in, non-fatal if unavailable) ──────
try:
    from multi_tf_feed import get_snapshot as _mtf_snapshot
    from sentiment_feed import get_sentiment, start_sentiment_feed
    from options_intelligence import get_oi_reading, start_oi_feed
    from market_structure import get_market_structure, warm_up_pdh_pdl
    from signal_fusion import evaluate as _fusion_evaluate
    _FUSION_AVAILABLE = True
    logger.info('"Multi-source intelligence modules loaded"')
except Exception as _fusion_exc:
    logger.error(f'"Intelligence modules unavailable: {_fusion_exc} — fusion disabled"')
    _FUSION_AVAILABLE = False

# ── ML modules (optional, non-fatal) ──────────────────────────────────────────
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
# Timestamp of the last strategy-timeframe bar we already evaluated for entry,
# so a 15-min breakout signal fires once per completed bar, not every 1-min tick.
_last_signal_bar_ts = None


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

    # Update multi-source intelligence state fields (non-fatal)
    if _FUSION_AVAILABLE:
        try:
            sent = get_sentiment()
            oi = get_oi_reading()
            ms = get_market_structure(df)
            state.sentiment_score = sent.score
            state.sentiment_halt = sent.halt
            state.vwap = ms.vwap
            state.pcr = oi.pcr
            state.oi_bias = oi.oi_bias
            state.pdh = ms.pdh
            state.pdl = ms.pdl
        except Exception as exc:
            logger.debug(f'"State refresh (intelligence): {exc}"')


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
    Full signal pipeline executed on each 1-min candle close.

    Exits are evaluated every minute against the spot levels locked in at entry.
    Entries are evaluated once per COMPLETED strategy-timeframe bar (e.g. 15-min
    for momentum_breakout) so the signal isn't re-fired on every intrabar tick.
    """
    global _last_signal_bar_ts
    if not _market_open():
        return
    if not _ML_AVAILABLE:
        return

    # ── 1-min indicators (drive spot, state, and exit checks) ─────────────────
    try:
        df1 = compute_all(df.copy())
    except Exception as exc:
        logger.error(f'"Indicator computation failed: {exc}"')
        return

    _refresh_state(df1)
    spot = float(df1.iloc[-1].get("close", 0) or 0)

    # ── Force exit at 15:10 ───────────────────────────────────────────────────
    if _is_force_exit_time() and order_manager.has_open_position:
        await _do_exit(df1, reason="FORCE_EXIT")
        _end_of_day_summary()
        return

    # ── Exit checks every minute (responsive target/stop on spot) ─────────────
    if order_manager.has_open_position:
        await _check_exit(df1)

    # ── Entry gating ──────────────────────────────────────────────────────────
    if order_manager.has_open_position:
        return
    if len(order_manager.today_trades()) >= MAX_TRADES_PER_DAY:
        return
    if not _is_entry_allowed():
        return

    params = load_optimized_params()
    strategy = params.get("strategy", STRATEGY)
    tf = int(params.get("timeframe_min", STRATEGY_TIMEFRAME_MIN))

    # ── Build the strategy-timeframe frame and act on COMPLETED bars only ─────
    try:
        if tf > 1:
            dtf = compute_all(resample_ohlc(df, tf))
            completed = dtf.iloc[:-1]    # drop the still-forming current bar
        else:
            dtf = df1
            completed = dtf
    except Exception as exc:
        logger.error(f'"Timeframe resample/indicator failed: {exc}"')
        return

    if len(completed) < 21:
        return
    bar_ts = completed.index[-1]
    if bar_ts == _last_signal_bar_ts:
        return                            # already evaluated this bar
    _last_signal_bar_ts = bar_ts

    last = completed.iloc[-1]
    prev = completed.iloc[-2]
    pb = float(last.get("percent_b", 0.5) or 0.5)
    pb_prev = float(prev.get("percent_b", 0.5) or 0.5)
    atr_val = float(last.get("atr", 0) or 0)

    # ── Direction (must match backtester.engine exactly) ──────────────────────
    direction = None
    if strategy == "momentum_breakout":
        if pb_prev <= params["bb_overbought"] and pb > params["bb_overbought"]:
            direction = "CE"            # upside band break → ride up
        elif pb_prev >= params["bb_oversold"] and pb < params["bb_oversold"]:
            direction = "PE"            # downside band break → ride down
    else:  # mean_reversion
        if pb < params["bb_oversold"]:
            direction = "CE"
        elif pb > params["bb_overbought"]:
            direction = "PE"

    if direction is None:
        state.signal = "NONE"
        return

    # ── Volatility gate ───────────────────────────────────────────────────────
    min_atr_pct = float(params.get("min_atr_pct", 0.0))
    atr_pct_val = float(last.get("atr_pct", 100.0) or 100.0)
    if min_atr_pct > 0 and atr_pct_val < min_atr_pct:
        logger.info(f'"Signal skipped: ATR pct={atr_pct_val:.1f} < {min_atr_pct:.1f}"')
        state.signal = "FILTERED_ATR"
        return

    # ── RSI band filter ───────────────────────────────────────────────────────
    rsi_val = float(last.get("rsi", 50) or 50)
    if not (params["rsi_min"] <= rsi_val <= params["rsi_max"]):
        logger.info(f'"Signal skipped: RSI={rsi_val:.1f} outside [{params["rsi_min"]}, {params["rsi_max"]}]"')
        state.signal = "FILTERED_RSI"
        return

    # ── NEW: Multi-source Signal Fusion ───────────────────────────────────────
    # Combines MTF alignment, VWAP, options OI/PCR, news sentiment, and S/R
    # proximity into a single confidence score. Entry fires only when score ≥
    # threshold (default 60/100). This is the main quality filter targeting
    # the 70% win-rate goal.
    if USE_SIGNAL_FUSION and _FUSION_AVAILABLE:
        try:
            mtf_snap = _mtf_snapshot(df)
            ms = get_market_structure(df1)
            oi = get_oi_reading()
            sent = get_sentiment()

            fused = _fusion_evaluate(
                base_direction=direction,
                mtf_snap=mtf_snap,
                df_ind=df1,
                market_struct=ms,
                oi_reading=oi,
                sentiment=sent,
                params=params,
            )

            # Persist fusion state for API/dashboard visibility
            state.mtf_alignment = mtf_snap.alignment_score
            state.fusion_confidence = fused.confidence
            state.fusion_components = fused.components
            state.fusion_reasons = fused.reasons
            state.fusion_blocking = fused.blocking

            if not fused.approved:
                logger.info(
                    f'"Signal fusion BLOCKED dir={direction} '
                    f'conf={fused.confidence:.1f} blocking={fused.blocking}"'
                )
                state.signal = "FILTERED_FUSION"
                return

            logger.info(
                f'"Signal fusion APPROVED dir={direction} '
                f'conf={fused.confidence:.1f} reasons={fused.reasons}"'
            )
        except Exception as exc:
            # Fusion failure is non-fatal: fall through to existing pipeline
            logger.error(f'"Signal fusion error (skipped): {exc}"')

    # ── Optional risk overlays (off by default for backtest/live coherence) ───
    current_regime = CHOPPY_REGIME_ID
    if USE_REGIME_FILTER:
        try:
            reg_feat = build_regime_features(dtf).tail(60)
            current_regime = regime_detector.predict_regime(reg_feat)
            state.regime = current_regime
            state.regime_name = REGIME_NAMES.get(current_regime, "UNKNOWN")
        except Exception as exc:
            logger.warning(f'"Regime detection failed: {exc} — blocking entry"')
            current_regime = TRENDING_DOWN_REGIME_ID
            state.regime = current_regime
            state.regime_name = "UNKNOWN"
        if current_regime != CHOPPY_REGIME_ID:
            logger.info(f'"Signal skipped: regime={state.regime_name}"')
            state.signal = "FILTERED_REGIME"
            return

    score = 1.0
    if USE_ML_FILTER:
        try:
            feat_vec = get_signal_features_at_bar(build_features(dtf), -2 if tf > 1 else -1)
            score = signal_filter.score(feat_vec)
        except Exception as exc:
            logger.error(f'"Signal scoring failed: {exc} — rejecting signal"')
            score = 0.0
        if score < SIGNAL_QUALITY_THRESHOLD:
            logger.info(f'"ML filter rejected: score={score:.3f} < {SIGNAL_QUALITY_THRESHOLD}"')
            state.signal = "FILTERED_ML"
            return

    state.signal = direction
    state.signal_quality_score = score

    # Select option, place order, set spot-anchored exit levels (using tf ATR).
    await _do_entry(direction, pb, current_regime, score, params, spot, atr_val)


async def _check_exit(df_feat: pd.DataFrame) -> None:
    """Evaluate exit conditions for the open position.

    Exits are price-anchored on the NIFTY spot using target/stop levels locked in
    at entry (ATR multiples) — identical to the backtest engine, so live and
    simulated behaviour match.
    """
    trade = order_manager.active_trade
    if trade is None:
        return

    spot = float(df_feat.iloc[-1].get("close", 0) or 0)
    if spot <= 0:
        return

    exit_reason = None
    if trade.direction == "CE":          # long delta: profit when spot rises
        if trade.target_spot and spot >= trade.target_spot:
            exit_reason = "TARGET"
        elif trade.sl_spot and spot <= trade.sl_spot:
            exit_reason = "STOP_LOSS"
    else:                                # PE: profit when spot falls
        if trade.target_spot and spot <= trade.target_spot:
            exit_reason = "TARGET"
        elif trade.sl_spot and spot >= trade.sl_spot:
            exit_reason = "STOP_LOSS"

    if exit_reason:
        await _do_exit(df_feat, exit_reason)


async def _do_entry(
    direction: str,
    pb: float,
    regime: int,
    score: float,
    params: dict,
    spot: float,
    atr_val: float,
) -> None:
    """Resolve a premium-capped option, place entry order, lock spot exit levels.

    spot + atr_val come from the strategy-timeframe signal bar, so the target/stop
    distances match the bars the signal was generated on.
    """
    try:
        spot = float(spot) or state.nifty_price
        if spot <= 0:
            logger.error('"Cannot enter trade: spot price is zero"')
            return

        # Price-anchored exit levels on the spot, using ATR multiples locked in
        # at entry (bb_exit = target ATRs, sl_buffer = stop ATRs). Mirrors the
        # backtest engine exactly. Default: 2.5 ATR target, 1.0 ATR stop → 1:2.5 R:R.
        # With USE_SIGNAL_FUSION active, only high-confidence entries get here,
        # so the effective R:R (accounting for accuracy) is significantly better.
        sign = 1.0 if direction == "CE" else -1.0
        target_spot = spot + sign * float(params["bb_exit"]) * atr_val
        sl_spot = spot - sign * float(params["sl_buffer"]) * atr_val

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
            entry_spot=spot,
            target_spot=target_spot,
            sl_spot=sl_spot,
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
    logger.info(f'"Bot loop started (DRY_RUN={DRY_RUN} FUSION={USE_SIGNAL_FUSION})"')

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
    global _bot_task, _stop_event, _event_loop, _last_signal_bar_ts
    _event_loop = asyncio.get_running_loop()
    _last_signal_bar_ts = None
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

    # ── Start multi-source intelligence background threads ─────────────────────
    if _FUSION_AVAILABLE:
        # Sentiment feed: polls RSS + optional NewsAPI every 5 min
        try:
            start_sentiment_feed()
            logger.info('"Sentiment feed started"')
        except Exception as exc:
            logger.error(f'"Sentiment feed start failed: {exc}"')

        # OI feed: polls Kite options chain for PCR/max-pain every 5 min
        try:
            start_oi_feed(lambda: state.nifty_price)
            logger.info('"OI feed started"')
        except Exception as exc:
            logger.error(f'"OI feed start failed: {exc}"')

        # PDH/PDL: fetch yesterday's high/low once at startup (non-fatal)
        def _pdh_pdl_warmup():
            try:
                import time
                time.sleep(3)   # give auth a moment to settle
                warm_up_pdh_pdl()
            except Exception as exc:
                logger.warning(f'"PDH/PDL warm-up thread failed: {exc}"')

        threading.Thread(target=_pdh_pdl_warmup, daemon=True, name="pdh-pdl-warmup").start()

        logger.info(
            f'"Multi-source intelligence active: '
            f'USE_SIGNAL_FUSION={USE_SIGNAL_FUSION}"'
        )
    else:
        logger.warning('"Intelligence modules not available — running on BB-only strategy"')

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
