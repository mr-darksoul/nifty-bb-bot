"""
FastAPI REST + WebSocket server exposing bot state, indicators, trades,
backtest results, and bot control endpoints.
"""

import asyncio
import secrets
import json
import logging
import os
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from config import (
    API_AUTH_TOKEN,
    DRY_RUN,
    FRONTEND_ORIGIN,
    MODELS_DIR,
    OPTIMIZED_PARAMS_PATH,
    REGIME_MODEL_PATH,
    SIGNAL_FILTER_MODEL_PATH,
    TRADE_LOG_PATH,
    USE_SIGNAL_FUSION,
    load_optimized_params,
    save_params,
)

logger = logging.getLogger(__name__)

# Lifespan hooks populated by main.py before the server starts. Using the
# lifespan context manager instead of the deprecated @app.on_event handlers.
_startup_hooks: List[Callable[[], Awaitable[None]]] = []
_shutdown_hooks: List[Callable[[], Awaitable[None]]] = []


def on_startup(fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
    """Register an async startup hook (runs once when the server boots)."""
    _startup_hooks.append(fn)
    return fn


def on_shutdown(fn: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
    """Register an async shutdown hook (runs once when the server stops)."""
    _shutdown_hooks.append(fn)
    return fn


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for hook in _startup_hooks:
        await hook()
    yield
    for hook in _shutdown_hooks:
        await hook()


app = FastAPI(
    title="NIFTY BB Bot API",
    version="1.0.0",
    description="Algorithmic trading bot for NIFTY weekly options",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared bot state (injected by main.py) ────────────────────────────────────

class BotState:
    """Singleton holding live bot runtime state."""

    def __init__(self) -> None:
        self.bot_running: bool = False
        self.market_open: bool = False
        # Runtime toggle for the multi-source Signal Fusion overlay. Defaults to
        # the config/env value; flippable live via POST /config/fusion. When OFF
        # the live bot trades the raw BB %b strategy, matching the backtester.
        self.use_signal_fusion: bool = USE_SIGNAL_FUSION
        self.regime: int = -1
        self.regime_name: str = "UNKNOWN"
        self.active_trade: Optional[Dict] = None
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.nifty_price: float = 0.0
        self.percent_b: float = 0.0
        self.rsi: float = 0.0
        self.atr: float = 0.0
        self.signal: str = "NONE"
        self.signal_quality_score: float = 0.0
        self.last_candle_time: str = ""
        self.strike_candidates: list = []  # [{strike, symbol, ltp, status}] from last option selection
        self.order_manager: Optional[Any] = None
        self.data_feed: Optional[Any] = None
        self._start_fn: Optional[Any] = None
        self._stop_fn: Optional[Any] = None
        # ── Multi-source intelligence state ────────────────────────────────────
        self.fusion_confidence: float = 0.0
        self.fusion_components: dict = {}
        self.fusion_reasons: list = []
        self.fusion_blocking: list = []
        self.vwap: float = 0.0
        self.pcr: float = 1.0
        self.oi_bias: str = "NEUTRAL"
        self.sentiment_score: float = 0.0
        self.sentiment_halt: bool = False
        self.mtf_alignment: int = 0        # -3 to +3
        self.pdh: float = 0.0
        self.pdl: float = 0.0


state = BotState()
_ws_clients: List[WebSocket] = []
_ws_tickets: Dict[str, float] = {}  # ticket → expiry unix timestamp


# ── Helper ────────────────────────────────────────────────────────────────────

def _model_mtime(path) -> str:
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        return "N/A"


def require_api_token(
    authorization: Optional[str] = Header(default=None),
    x_api_token: Optional[str] = Header(default=None),
) -> None:
    """Require the shared dashboard API token for REST endpoints."""
    if not API_AUTH_TOKEN:
        raise HTTPException(status_code=503, detail="API_AUTH_TOKEN is not configured")

    token = x_api_token or ""
    if not token and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            token = value

    if not secrets.compare_digest(token, API_AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid API token")


async def _accept_authorized_ws(websocket: WebSocket, ticket: Optional[str]) -> bool:
    """Accept the WebSocket then validate the one-time ticket.

    Must accept before closing — closing an unaccepted WS raises in some
    Starlette versions.  Tickets are issued by POST /ws/ticket and expire
    after 30 s, so the real API token never appears in WS URLs or logs.
    """
    await websocket.accept()
    now = _time.time()
    # Purge stale tickets
    for k in [k for k, exp in list(_ws_tickets.items()) if exp < now]:
        del _ws_tickets[k]
    if not ticket or ticket not in _ws_tickets:
        await websocket.close(code=1008)
        return False
    del _ws_tickets[ticket]
    return True


def _trade_to_dict(trade) -> Dict:
    from order_manager import Trade
    if isinstance(trade, Trade):
        return {
            "trade_id": trade.trade_id,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "direction": trade.direction,
            "symbol": trade.symbol,
            "strike": trade.strike,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "pnl": trade.pnl,
            "exit_reason": trade.exit_reason,
            "signal_quality_score": trade.signal_quality_score,
            "entry_pb": trade.entry_pb,
            "exit_pb": trade.exit_pb,
            "regime": trade.regime,
            "is_open": trade.is_open,
        }
    return dict(trade)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/status", dependencies=[Depends(require_api_token)])
async def get_status() -> Dict:
    """Overall bot health and state."""
    active = None
    if state.active_trade:
        active = _trade_to_dict(state.active_trade)

    return {
        "bot_running": state.bot_running,
        "dry_run": DRY_RUN,
        "use_signal_fusion": state.use_signal_fusion,
        "market_open": state.market_open,
        "regime": state.regime,
        "regime_name": state.regime_name,
        "active_trade": active,
        "trades_today": state.trades_today,
        "daily_pnl": round(state.daily_pnl, 2),
        "strike_candidates": state.strike_candidates,
        "timestamp": datetime.now().isoformat(),
        # Multi-source intelligence snapshot
        "fusion_confidence": round(state.fusion_confidence, 1),
        "fusion_components": state.fusion_components,
        "fusion_blocking": state.fusion_blocking,
        "vwap": round(state.vwap, 2),
        "pcr": round(state.pcr, 3),
        "oi_bias": state.oi_bias,
        "sentiment_score": round(state.sentiment_score, 3),
        "sentiment_halt": state.sentiment_halt,
        "mtf_alignment": state.mtf_alignment,
        "pdh": round(state.pdh, 2),
        "pdl": round(state.pdl, 2),
    }


@app.get("/indicators", dependencies=[Depends(require_api_token)])
async def get_indicators() -> Dict:
    """Current indicator snapshot."""
    return {
        "timestamp": state.last_candle_time or datetime.now().isoformat(),
        "nifty_price": round(state.nifty_price, 2),
        "percent_b": round(state.percent_b, 4),
        "rsi": round(state.rsi, 2),
        "atr": round(state.atr, 2),
        "regime": state.regime,
        "regime_name": state.regime_name,
        "signal": state.signal,
        "signal_quality_score": round(state.signal_quality_score, 4),
        # Multi-source intelligence
        "mtf_alignment": state.mtf_alignment,
        "vwap": round(state.vwap, 2),
        "pcr": round(state.pcr, 3),
        "oi_bias": state.oi_bias,
        "sentiment_score": round(state.sentiment_score, 3),
        "fusion_confidence": round(state.fusion_confidence, 1),
        "fusion_reasons": state.fusion_reasons,
        "pdh": round(state.pdh, 2),
        "pdl": round(state.pdl, 2),
    }


@app.get("/candles", dependencies=[Depends(require_api_token)])
async def get_candles(count: int = 375) -> Dict:
    """
    Recent historical 1-minute NIFTY candles with Bollinger Bands, so the
    chart can render even when the market is closed / bot is stopped.
    Requires a valid Kite session (historical data API).
    """
    from datetime import timedelta
    import pandas as pd
    from auth import get_kite
    from config import NIFTY_INDEX_TOKEN, BB_PERIOD, BB_STD, KITE_HISTORICAL_INTERVAL
    import indicators

    try:
        kite = get_kite()
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Kite not authenticated: {e}")

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=8)
    try:
        records = kite.historical_data(
            instrument_token=NIFTY_INDEX_TOKEN,
            from_date=from_dt,
            to_date=to_dt,
            interval=KITE_HISTORICAL_INTERVAL,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"historical_data failed: {e}")

    if not records:
        return {"candles": [], "count": 0}

    df = pd.DataFrame(records)
    df = indicators.compute_all(df, bb_period=BB_PERIOD, bb_std=BB_STD)
    df = df.tail(max(1, int(count)))

    cols = ["open", "high", "low", "close", "bb_upper", "bb_middle", "bb_lower"]
    out = df[["date", *cols]].copy()
    # Epoch seconds — resolution- and tz-safe. Kite returns tz-aware timestamps
    # that pandas 2.x may hold at microsecond resolution, so astype("int64")//1e9
    # would be 1000× too small. Timedelta floor-division is resolution-proof.
    dt = pd.to_datetime(out["date"], utc=True)
    out["time"] = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    out[cols] = out[cols].round(2)
    # NaN → None for JSON; ints stay native through to_dict
    out = out.where(pd.notna(out), None)

    candles = [
        {
            "time": int(rec["time"]),
            "open": rec["open"],
            "high": rec["high"],
            "low": rec["low"],
            "close": rec["close"],
            "bb_upper": rec["bb_upper"],
            "bb_middle": rec["bb_middle"],
            "bb_lower": rec["bb_lower"],
        }
        for rec in out.to_dict(orient="records")
    ]
    return {"candles": candles, "count": len(candles)}


@app.get("/trades", dependencies=[Depends(require_api_token)])
async def get_trades_today() -> List[Dict]:
    """Today's closed trades."""
    if state.order_manager is None:
        return []
    return [_trade_to_dict(t) for t in state.order_manager.today_trades()]


@app.get("/trades/history", dependencies=[Depends(require_api_token)])
async def get_trade_history():
    """Return the full trade log CSV."""
    if TRADE_LOG_PATH.exists():
        return FileResponse(
            path=str(TRADE_LOG_PATH),
            media_type="text/csv",
            filename="trades.csv",
        )
    return JSONResponse(content=[], status_code=200)


@app.post("/backtest/run", dependencies=[Depends(require_api_token)])
async def run_backtest_endpoint() -> Dict:
    """Run backtester on 90 days of Kite historical candles (no local cache required)."""
    try:
        import pandas as pd
        from datetime import timedelta
        from auth import get_kite
        from config import NIFTY_INDEX_TOKEN, BB_PERIOD, BB_STD, KITE_HISTORICAL_INTERVAL, STRATEGY_TIMEFRAME_MIN
        from indicators import compute_all, resample_ohlc
        from backtester.engine import run_backtest

        try:
            kite = get_kite()
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Kite not authenticated: {e}")

        # Kite limits 1-min data to ~60 days per request; fetch two 45-day chunks.
        to_dt = datetime.now()
        mid_dt = to_dt - timedelta(days=45)
        from_dt = to_dt - timedelta(days=90)
        try:
            chunk1 = kite.historical_data(
                instrument_token=NIFTY_INDEX_TOKEN,
                from_date=from_dt,
                to_date=mid_dt,
                interval=KITE_HISTORICAL_INTERVAL,
            )
            chunk2 = kite.historical_data(
                instrument_token=NIFTY_INDEX_TOKEN,
                from_date=mid_dt,
                to_date=to_dt,
                interval=KITE_HISTORICAL_INTERVAL,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"historical_data failed: {e}")

        records = chunk1 + chunk2
        if not records:
            raise HTTPException(status_code=404, detail="No historical data returned from Kite")

        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["date"]).sort_values("date")
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        params = load_optimized_params()
        # Resample the 1-min Kite candles to the STRATEGY timeframe (15-min for
        # momentum_breakout) BEFORE computing indicators / signals, exactly like
        # the live pipeline (main.process_candle) and finalize_params. Without
        # this the backtest fires BB breakouts on 1-min noise — a different,
        # higher-frequency strategy than the bot actually trades, producing
        # ~16 trades/day instead of ~3-4/week and meaningless metrics.
        tf = int(params.get("timeframe_min", STRATEGY_TIMEFRAME_MIN))
        df = compute_all(resample_ohlc(df, tf), bb_period=BB_PERIOD, bb_std=BB_STD)

        trades_df, daily_pnl, metrics = run_backtest(df, params=params)

        # ── Real Kite option prices only ──────────────────────────────────────
        # Trades that cannot be priced from currently-listed contracts are
        # dropped; the survivors define the real-data window.
        from backtester.option_pricer import enrich_with_real_option_prices
        from backtester.metrics import compute_metrics

        candidate_trades = int(len(trades_df))
        data_window: Dict[str, Optional[str]] = {"start": None, "end": None}

        if not trades_df.empty:
            nfo_raw = kite.instruments(exchange="NFO")
            inst_df = pd.DataFrame(nfo_raw)
            inst_df = inst_df[inst_df["name"] == "NIFTY"].copy()
            inst_df["expiry"] = pd.to_datetime(inst_df["expiry"]).dt.date

            trades_df = enrich_with_real_option_prices(
                kite, trades_df, inst_df, drop_unpriced=True
            )

            if not trades_df.empty:
                trades_df["date"] = trades_df["entry_time"].dt.date
                daily_pnl = trades_df.groupby("date")["pnl"].sum()
                daily_pnl.index = pd.to_datetime(daily_pnl.index)
                metrics = compute_metrics(trades_df, daily_pnl)
                data_window = {
                    "start": trades_df["entry_time"].min().date().isoformat(),
                    "end":   trades_df["exit_time"].max().date().isoformat(),
                }
            else:
                # No trade fell inside the window where Kite has option data.
                metrics = compute_metrics(trades_df, pd.Series(dtype=float))

        priced_trades = int(len(trades_df))
        logger.info(
            f'"Backtest real-only: {priced_trades}/{candidate_trades} trades priced, '
            f'window={data_window}"'
        )

        trades_list = []
        if not trades_df.empty:
            trades_list = trades_df.to_dict(orient="records")
            for t in trades_list:
                for k in ("entry_time", "exit_time"):
                    if hasattr(t[k], "isoformat"):
                        t[k] = t[k].isoformat()
                if hasattr(t.get("expiry"), "isoformat"):
                    t["expiry"] = t["expiry"].isoformat()
                for k in ("real_entry_ltp", "real_exit_ltp"):
                    v = t.get(k)
                    t[k] = None if v is None or (isinstance(v, float) and v != v) else round(float(v), 2)

        return {
            "metrics": metrics,
            "trades": trades_list,
            "price_mode": "real_options",
            "candidate_trades": candidate_trades,
            "priced_trades": priced_trades,
            "data_window": data_window,
            "params_used": {k: v for k, v in params.items() if not k.startswith("_")},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f'"Backtest endpoint error: {exc}"', exc_info=True)
        raise HTTPException(status_code=500, detail="Backtest failed — check server logs")


@app.get("/model/status", dependencies=[Depends(require_api_token)])
async def get_model_status() -> Dict:
    """Model file metadata and current optimized parameter values."""
    params = load_optimized_params()
    meta = params.pop("_meta", {})

    return {
        "regime_model": {
            "path": str(REGIME_MODEL_PATH),
            "exists": REGIME_MODEL_PATH.exists(),
            "last_modified": _model_mtime(REGIME_MODEL_PATH),
        },
        "signal_filter_model": {
            "path": str(SIGNAL_FILTER_MODEL_PATH),
            "exists": SIGNAL_FILTER_MODEL_PATH.exists(),
            "last_modified": _model_mtime(SIGNAL_FILTER_MODEL_PATH),
        },
        "optimized_params": {
            "path": str(OPTIMIZED_PARAMS_PATH),
            "exists": OPTIMIZED_PARAMS_PATH.exists(),
            "last_modified": _model_mtime(OPTIMIZED_PARAMS_PATH),
            "values": params,
            "meta": meta,
        },
    }


@app.put("/params", dependencies=[Depends(require_api_token)])
async def update_params(body: Dict) -> Dict:
    """Update strategy parameters. Persists to optimized_params.json for live and backtest use."""
    # bb_exit / sl_buffer are ATR multiples (profit target / stop in ATR units);
    # min_atr_pct is the volatility-percentile gate. All optional except the core set.
    FLOAT_KEYS = {"bb_oversold", "bb_overbought", "bb_exit", "sl_buffer"}
    INT_KEYS = {"rsi_min", "rsi_max"}
    OPTIONAL_FLOAT_KEYS = {"min_atr_pct"}
    REQUIRED_KEYS = FLOAT_KEYS | INT_KEYS

    for k in REQUIRED_KEYS:
        if k not in body:
            raise HTTPException(status_code=400, detail=f"Missing parameter: {k}")

    try:
        params: Dict = {k: float(body[k]) for k in FLOAT_KEYS}
        params.update({k: int(body[k]) for k in INT_KEYS})
        for k in OPTIONAL_FLOAT_KEYS:
            if k in body:
                params[k] = float(body[k])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid parameter value: {exc}")

    # %b cross thresholds. For momentum_breakout these are the TRUE band edges
    # (bb_oversold=0.0 = close at/below the lower band, bb_overbought=1.0 = close
    # at/above the upper band) — the validated config. Bounds must be INCLUSIVE;
    # the old open-interval (0,1) check wrongly rejected 0.0/1.0 (a mean-reversion
    # assumption). Keep oversold < overbought to prevent inversion.
    if not (0 <= params["bb_oversold"] < params["bb_overbought"] <= 1):
        raise HTTPException(status_code=400, detail="bb_oversold must be < bb_overbought, both in [0, 1]")
    if not (0 < params["bb_exit"] <= 10):
        raise HTTPException(status_code=400, detail="bb_exit (ATR multiple) must be in (0, 10]")
    if not (0 < params["sl_buffer"] <= 10):
        raise HTTPException(status_code=400, detail="sl_buffer (ATR multiple) must be in (0, 10]")
    if not (0 <= params["rsi_min"] < params["rsi_max"] <= 100):
        raise HTTPException(status_code=400, detail="rsi_min must be < rsi_max and both in [0, 100]")
    if "min_atr_pct" in params and not (0 <= params["min_atr_pct"] <= 100):
        raise HTTPException(status_code=400, detail="min_atr_pct must be in [0, 100]")

    save_params(params)
    return {"status": "saved", "params": params}


@app.post("/ws/ticket", dependencies=[Depends(require_api_token)])
async def get_ws_ticket() -> Dict:
    """Issue a one-time WebSocket auth ticket valid for 30 seconds.

    The frontend exchanges its API token (in a header) for a short-lived
    ticket, then passes that ticket as a query param in the WS URL.  This
    keeps the durable API token out of access logs and browser history.
    """
    ticket = secrets.token_urlsafe(32)
    _ws_tickets[ticket] = _time.time() + 30
    return {"ticket": ticket}


@app.post("/bot/start", dependencies=[Depends(require_api_token)])
async def start_bot() -> Dict:
    """Start the trading loop as a background asyncio task."""
    if state.bot_running:
        return {"status": "already_running"}
    if state._start_fn is None:
        raise HTTPException(status_code=503, detail="Bot start function not registered")
    try:
        await state._start_fn()
        return {"status": "started", "dry_run": DRY_RUN}
    except Exception as exc:
        logger.error(f'"Bot start failed: {exc}"', exc_info=True)
        raise HTTPException(status_code=500, detail="Bot start failed — check server logs")


@app.post("/bot/stop", dependencies=[Depends(require_api_token)])
async def stop_bot() -> Dict:
    """Stop the bot and force-exit any open position."""
    if not state.bot_running:
        return {"status": "not_running"}
    if state._stop_fn is None:
        raise HTTPException(status_code=503, detail="Bot stop function not registered")
    try:
        await state._stop_fn()
        return {"status": "stopped"}
    except Exception as exc:
        logger.error(f'"Bot stop failed: {exc}"', exc_info=True)
        raise HTTPException(status_code=500, detail="Bot stop failed — check server logs")


@app.post("/config/fusion", dependencies=[Depends(require_api_token)])
async def set_fusion(body: Dict) -> Dict:
    """Toggle the Signal Fusion overlay live.

    Body: { "enabled": bool }. When OFF the bot trades the raw BB %b strategy
    (matches the backtester); when ON it additionally requires fusion approval.
    Takes effect on the next candle — no restart needed. Runtime-only: resets to
    the config/env default on container restart.
    """
    if "enabled" not in body:
        raise HTTPException(status_code=400, detail="Missing 'enabled' (bool)")
    enabled = bool(body["enabled"])
    state.use_signal_fusion = enabled
    logger.info(f'"Signal Fusion toggled {"ON" if enabled else "OFF"} via API"')
    return {"use_signal_fusion": state.use_signal_fusion}


@app.get("/auth/login-url", dependencies=[Depends(require_api_token)])
async def get_login_url() -> Dict:
    """Return the Kite OAuth URL for the user to open in their browser."""
    try:
        from auth import get_login_url as _get_url
        url = _get_url()
        return {"login_url": url}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/auth/login", dependencies=[Depends(require_api_token)])
async def auth_login(body: Dict) -> Dict:
    """
    Exchange a Kite request_token for an access_token.
    Body: { "request_token": "..." }
    """
    request_token = body.get("request_token", "").strip()
    if not request_token:
        raise HTTPException(status_code=400, detail="request_token is required")
    try:
        from auth import exchange_request_token
        access_token = exchange_request_token(request_token)
        return {"status": "ok", "access_token_set": True}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket, token: Optional[str] = Query(default=None)) -> None:
    """
    Streams live indicator + regime + signal snapshot every 5 seconds.
    Payload: { price, percent_b, rsi, atr, regime, regime_name, signal,
               signal_quality, active_trade, daily_pnl }
    """
    if not await _accept_authorized_ws(websocket, token):
        return
    _ws_clients.append(websocket)
    logger.info(f'"WebSocket client connected. Total clients: {len(_ws_clients)}"')

    try:
        while True:
            active = None
            if state.order_manager and state.order_manager.active_trade:
                active = _trade_to_dict(state.order_manager.active_trade)

            payload = {
                "price": round(state.nifty_price, 2),
                "percent_b": round(state.percent_b, 4),
                "rsi": round(state.rsi, 2),
                "atr": round(state.atr, 2),
                "regime": state.regime,
                "regime_name": state.regime_name,
                "signal": state.signal,
                "signal_quality": round(state.signal_quality_score, 4),
                "active_trade": active,
                "daily_pnl": round(state.daily_pnl, 2),
                "trades_today": state.trades_today,
                "strike_candidates": state.strike_candidates,
                "timestamp": datetime.now().isoformat(),
            }
            await websocket.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info('"WebSocket client disconnected"')
    except Exception as exc:
        logger.error(f'"WebSocket error: {exc}"')
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


async def broadcast_ws(payload: Dict) -> None:
    """Broadcast a payload to all connected WebSocket clients."""
    disconnected = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
