"""
Central configuration: loads environment variables and defines all constants.
Raises ValueError on startup if required secrets are missing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","msg":%(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Kite / Broker ─────────────────────────────────────────────────────────────

KITE_API_KEY: str = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET: str = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

# ── Telegram ──────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Server ────────────────────────────────────────────────────────────────────

PORT: int = int(os.getenv("PORT", "8000"))
FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
API_AUTH_TOKEN: str = os.getenv("API_AUTH_TOKEN", "")

# ── Trading mode ──────────────────────────────────────────────────────────────

DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

# ── Instrument ────────────────────────────────────────────────────────────────

NIFTY_INDEX_TOKEN: int = 256265          # NSE:NIFTY 50 instrument token
NIFTY_SYMBOL: str = "NIFTY 50"
NFO_EXCHANGE: str = "NFO"
NSE_EXCHANGE: str = "NSE"
LOT_SIZE: int = 65                       # NIFTY options lot size from NSE/FAOP/70616
CAPITAL_PER_TRADE: float = 5_000.0       # ₹ maximum premium value per trade
BROKERAGE_PER_ORDER: float = 20.0       # Zerodha flat ₹20
SLIPPAGE_PCT: float = 0.0005            # 0.05% slippage each leg
MAX_TRADES_PER_DAY: int = 2          # reduced from 3: fewer, higher-quality 1-min trades

# ── Market session ────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MIN: int = 15
ENTRY_START_HOUR: int = 9
ENTRY_START_MIN: int = 35   # earliest bar for new entries; allows warm-up + skips gap-open noise
FORCE_EXIT_HOUR: int = 15
FORCE_EXIT_MIN: int = 10
MIN_DAYS_TO_EXPIRY: int = 0  # 0 = trade expiry-day (0 DTE) options; raise to 1+ to roll to next week
MAX_DAYS_TO_EXPIRY: int = 7  # cap historical/live selection to current/next weekly options

# ── Candle interval ───────────────────────────────────────────────────────────

CANDLE_INTERVAL_MINUTES: int = 1
# Kite Connect historical-data interval string for the above.
# NOTE: Kite calls the 1-minute interval "minute" (not "1minute").
KITE_HISTORICAL_INTERVAL: str = "minute"

# ── Indicator defaults ────────────────────────────────────────────────────────

BB_PERIOD: int = 20
BB_STD: float = 2.0
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_FAST: int = 9
EMA_SLOW: int = 21

# ── Strategy parameter defaults (used when optimized_params.json missing) ────

DEFAULT_PARAMS: dict = {
    "bb_oversold": 0.05,
    "bb_overbought": 0.95,
    "bb_exit": 0.50,
    "sl_buffer": 0.10,
    "rsi_min": 35,
    "rsi_max": 65,
    "min_atr_pct": 60.0,   # only trade when ATR is in top ~40% (move clears costs)
}

# ── ML thresholds ─────────────────────────────────────────────────────────────

SIGNAL_QUALITY_THRESHOLD: float = 0.70   # raised from 0.60: stricter ML gate, fewer trades
CHOPPY_REGIME_ID: int = 1               # HMM state that means CHOPPY

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR: Path = Path(__file__).parent
MODELS_DIR: Path = BASE_DIR / "ml" / "models"
OPTIMIZED_PARAMS_PATH: Path = MODELS_DIR / "optimized_params.json"
REGIME_MODEL_PATH: Path = MODELS_DIR / "regime_model.joblib"
SIGNAL_FILTER_MODEL_PATH: Path = MODELS_DIR / "signal_filter_model.joblib"
TRADE_LOG_PATH: Path = BASE_DIR / "trades.csv"
DATA_CACHE_PATH: Path = BASE_DIR / "nifty_1min.csv"


_params_cache: Optional[dict] = None
_params_cache_mtime: float = 0.0


def save_params(params: dict) -> None:
    """Write strategy parameters to optimized_params.json and reset the in-process cache."""
    global _params_cache, _params_cache_mtime
    import json as _json
    from datetime import datetime as _dt
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if OPTIMIZED_PARAMS_PATH.exists():
        try:
            with open(OPTIMIZED_PARAMS_PATH) as f:
                existing = _json.load(f)
        except Exception:
            pass
    meta = existing.get("_meta", {})
    meta["updated_manually"] = _dt.utcnow().isoformat() + "Z"
    to_save = {k: v for k, v in params.items() if not k.startswith("_")}
    to_save["_meta"] = meta
    with open(OPTIMIZED_PARAMS_PATH, "w") as f:
        _json.dump(to_save, f, indent=2)
    _params_cache = None
    _params_cache_mtime = 0.0
    logger.info('"Saved user-defined strategy params to optimized_params.json"')


def load_optimized_params() -> dict:
    """Load strategy parameters from optimizer output, falling back to defaults.

    Result is cached in-process and only re-read when the file's mtime changes,
    avoiding a JSON parse on every candle close (375×/day).
    """
    global _params_cache, _params_cache_mtime
    if OPTIMIZED_PARAMS_PATH.exists():
        try:
            mtime = OPTIMIZED_PARAMS_PATH.stat().st_mtime
            if _params_cache is not None and mtime == _params_cache_mtime:
                return _params_cache.copy()
            with open(OPTIMIZED_PARAMS_PATH) as f:
                params = json.load(f)
            _params_cache = params
            _params_cache_mtime = mtime
            logger.info('"Loaded optimized params from file"')
            return params.copy()
        except Exception as exc:
            logger.warning(f'"Failed to load optimized params: {exc} — using defaults"')
    else:
        logger.warning('"optimized_params.json not found — using default strategy params"')
    return DEFAULT_PARAMS.copy()


def validate_secrets() -> None:
    """Raise ValueError if any required secret is absent."""
    required = {
        "KITE_API_KEY": KITE_API_KEY,
        "KITE_API_SECRET": KITE_API_SECRET,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
