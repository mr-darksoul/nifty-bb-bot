"""
FastAPI REST + WebSocket server exposing bot state, indicators, trades,
backtest results, and bot control endpoints.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from config import (
    DRY_RUN,
    FRONTEND_ORIGIN,
    MODELS_DIR,
    OPTIMIZED_PARAMS_PATH,
    REGIME_MODEL_PATH,
    SIGNAL_FILTER_MODEL_PATH,
    TRADE_LOG_PATH,
    load_optimized_params,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NIFTY BB Bot API",
    version="1.0.0",
    description="Algorithmic trading bot for NIFTY weekly options",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000", "http://localhost:5500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared bot state (injected by main.py) ────────────────────────────────────

class BotState:
    """Singleton holding live bot runtime state."""

    bot_running: bool = False
    market_open: bool = False
    regime: int = -1
    regime_name: str = "UNKNOWN"
    active_trade: Optional[Dict] = None
    trades_today: int = 0
    daily_pnl: float = 0.0
    nifty_price: float = 0.0
    percent_b: float = 0.0
    rsi: float = 0.0
    atr: float = 0.0
    signal: str = "NONE"
    signal_quality_score: float = 0.0
    last_candle_time: str = ""
    order_manager: Optional[Any] = None
    data_feed: Optional[Any] = None
    _start_fn: Optional[Any] = None
    _stop_fn: Optional[Any] = None


state = BotState()
_ws_clients: List[WebSocket] = []


# ── Helper ────────────────────────────────────────────────────────────────────

def _model_mtime(path) -> str:
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        return "N/A"


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

@app.get("/status")
async def get_status() -> Dict:
    """Overall bot health and state."""
    active = None
    if state.active_trade:
        active = _trade_to_dict(state.active_trade)

    return {
        "bot_running": state.bot_running,
        "dry_run": DRY_RUN,
        "market_open": state.market_open,
        "regime": state.regime,
        "regime_name": state.regime_name,
        "active_trade": active,
        "trades_today": state.trades_today,
        "daily_pnl": round(state.daily_pnl, 2),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/indicators")
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
    }


@app.get("/trades")
async def get_trades_today() -> List[Dict]:
    """Today's closed trades."""
    if state.order_manager is None:
        return []
    return [_trade_to_dict(t) for t in state.order_manager.today_trades()]


@app.get("/trades/history")
async def get_trade_history():
    """Return the full trade log CSV."""
    if TRADE_LOG_PATH.exists():
        return FileResponse(
            path=str(TRADE_LOG_PATH),
            media_type="text/csv",
            filename="trades.csv",
        )
    return JSONResponse(content=[], status_code=200)


@app.get("/backtest/run")
async def run_backtest_endpoint() -> Dict:
    """Run backtester on the most recent cached data and return metrics."""
    try:
        import pandas as pd
        from config import DATA_CACHE_PATH
        from indicators import compute_all
        from backtester.engine import run_backtest
        from backtester.metrics import compute_metrics

        if not DATA_CACHE_PATH.exists():
            raise HTTPException(status_code=404, detail="No cached NIFTY data found")

        df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
        df = compute_all(df)

        # Last 90 days
        cutoff = df.index[-1] - pd.Timedelta(days=90)
        df = df[df.index >= cutoff]

        params = load_optimized_params()
        trades_df, daily_pnl, metrics = run_backtest(df, params=params)

        trades_list = []
        if not trades_df.empty:
            trades_list = trades_df.to_dict(orient="records")

        return {
            "metrics": metrics,
            "trades": trades_list,
            "params_used": {k: v for k, v in params.items() if not k.startswith("_")},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f'"Backtest endpoint error: {exc}"')
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model/status")
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


@app.post("/bot/start")
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
        logger.error(f'"Bot start failed: {exc}"')
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/bot/stop")
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
        logger.error(f'"Bot stop failed: {exc}"')
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/auth/login-url")
async def get_login_url() -> Dict:
    """Return the Kite OAuth URL for the user to open in their browser."""
    try:
        from auth import get_login_url as _get_url
        url = _get_url()
        return {"login_url": url}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/auth/login")
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
async def ws_live(websocket: WebSocket) -> None:
    """
    Streams live indicator + regime + signal snapshot every 5 seconds.
    Payload: { price, percent_b, rsi, atr, regime, regime_name, signal,
               signal_quality, active_trade, daily_pnl }
    """
    await websocket.accept()
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
