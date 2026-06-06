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
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        self.order_manager: Optional[Any] = None
        self.data_feed: Optional[Any] = None
        self._start_fn: Optional[Any] = None
        self._stop_fn: Optional[Any] = None


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
        "market_open": state.market_open,
        "regime": state.regime,
        "regime_name": state.regime_name,
        "active_trade": active,
        "trades_today": state.trades_today,
        "daily_pnl": round(state.daily_pnl, 2),
        "timestamp": datetime.now().isoformat(),
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

    def _num(v):
        return None if v is None or pd.isna(v) else round(float(v), 2)

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": int(pd.Timestamp(row["date"]).timestamp()),
            "open": _num(row["open"]),
            "high": _num(row["high"]),
            "low": _num(row["low"]),
            "close": _num(row["close"]),
            "bb_upper": _num(row.get("bb_upper")),
            "bb_middle": _num(row.get("bb_middle")),
            "bb_lower": _num(row.get("bb_lower")),
        })
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
        from config import NIFTY_INDEX_TOKEN, BB_PERIOD, BB_STD, KITE_HISTORICAL_INTERVAL
        from indicators import compute_all
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
        df = compute_all(df, bb_period=BB_PERIOD, bb_std=BB_STD)

        params = load_optimized_params()
        trades_df, daily_pnl, metrics = run_backtest(df, params=params)

        trades_list = []
        if not trades_df.empty:
            trades_list = trades_df.to_dict(orient="records")
            for t in trades_list:
                for k in ("entry_time", "exit_time"):
                    if hasattr(t[k], "isoformat"):
                        t[k] = t[k].isoformat()

        return {
            "metrics": metrics,
            "trades": trades_list,
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
