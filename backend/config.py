"""
Central configuration: loads environment variables and defines all constants.
Raises ValueError on startup if required secrets are missing.
"""

import json
import logging
import os
import threading
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
# ₹ maximum premium value per trade. Set to fit ~1 ATM weekly lot (~₹150-280 ×
# 65 ≈ ₹10-18k). The selector buys the nearest strike to ATM whose one-lot
# premium fits this cap, so a cap >= one ATM lot makes it trade ATM rather than
# far-OTM. ATM is the only instrument the breakout edge survives on: at ~₹177
# premium a realistic ₹0.5-1.0/side bid-ask is a small % of premium and the edge
# clears it (PF ~1.18), whereas far-OTM (~₹70, cap ₹5k) is eaten by its wider
# relative spread (PF ~0.95). See research/revalidate_model.py.
CAPITAL_PER_TRADE: float = float(os.getenv("CAPITAL_PER_TRADE", "18000"))
BROKERAGE_PER_ORDER: float = 20.0       # Zerodha flat ₹20
SLIPPAGE_PCT: float = 0.00003           # applied to underlying price; ~₹40 round-trip option impact
# Per-day trade cap. Effectively unlimited by default — entries are throttled by
# the BB %b extreme + RSI band + ATR-percentile volatility gate, not an arbitrary
# count. Override with MAX_TRADES_PER_DAY env var to impose a hard ceiling.
MAX_TRADES_PER_DAY: int = int(os.getenv("MAX_TRADES_PER_DAY", "50"))

# ── Market session ────────────────────────────────────────────────────────────

MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MIN: int = 15
ENTRY_START_HOUR: int = 9
ENTRY_START_MIN: int = 35   # earliest bar for new entries; allows warm-up + skips gap-open noise
FORCE_EXIT_HOUR: int = 15
FORCE_EXIT_MIN: int = 10
# Days-to-expiry window for option selection. The momentum strategy BUYS options,
# so it avoids 0-DTE (lethal theta) and targets a few days out where the validated
# edge survives. Override via env. (Mean-reversion legacy used MIN=0.)
MIN_DAYS_TO_EXPIRY: int = int(os.getenv("MIN_DAYS_TO_EXPIRY", "4"))
MAX_DAYS_TO_EXPIRY: int = int(os.getenv("MAX_DAYS_TO_EXPIRY", "12"))

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

# ── Strategy selector ─────────────────────────────────────────────────────────
# "momentum_breakout" (default): on STRATEGY_TIMEFRAME_MIN bars, BUY in the
#   direction of a Bollinger band break (%b crosses out of the band) and ride it
#   with an ATR target/stop. Validated edge for long options on NIFTY.
# "mean_reversion" (legacy): fade %b extremes on 1-min bars. Kept for comparison;
#   research shows it has no edge.
STRATEGY: str = os.getenv("STRATEGY", "momentum_breakout")
STRATEGY_TIMEFRAME_MIN: int = int(os.getenv("STRATEGY_TIMEFRAME_MIN", "15"))

# ── Strategy parameter defaults (used when optimized_params.json missing) ────
# Interpretation depends on "strategy":
#   momentum_breakout: bb_overbought/bb_oversold are the OUTWARD cross levels
#     (~1.0 / ~0.0 = a true band break); entry fires when %b crosses through.
#   mean_reversion: bb_oversold/bb_overbought are the inward extremes to fade.
# bb_exit / sl_buffer are ATR multiples for the price-anchored target / stop.

DEFAULT_PARAMS: dict = {
    "strategy": STRATEGY,
    "timeframe_min": STRATEGY_TIMEFRAME_MIN,
    "bb_oversold": 0.0,
    "bb_overbought": 1.0,
    "bb_exit": 2.5,
    "sl_buffer": 1.0,
    "rsi_min": 0,
    "rsi_max": 100,
    "min_atr_pct": 0.0,
}

# Legacy mean-reversion defaults (1-min fade). Used only when strategy override
# selects mean_reversion without an optimized_params.json present.
MEAN_REVERSION_PARAMS: dict = {
    "strategy": "mean_reversion",
    "timeframe_min": 1,
    "bb_oversold": 0.05,
    "bb_overbought": 0.95,
    "bb_exit": 1.5,
    "sl_buffer": 0.75,
    "rsi_min": 35,
    "rsi_max": 65,
    "min_atr_pct": 60.0,
}

# ── Multi-source signal fusion (opt-in) ──────────────────────────────────────
# When enabled, every BB breakout signal must also pass the full confluence
# scoring engine (signal_fusion.py) which combines multi-timeframe alignment,
# VWAP position, options OI/PCR, news sentiment, and S/R proximity.
# Target: 70% win rate at 1:3 R:R. Trades will be fewer but higher quality.
# Disable to revert to pure BB %b momentum strategy.
USE_SIGNAL_FUSION: bool = os.getenv("USE_SIGNAL_FUSION", "true").lower() in ("true", "1", "yes")
SIGNAL_FUSION_THRESHOLD: float = float(os.getenv("SIGNAL_FUSION_THRESHOLD", "60.0"))

# Optional NewsAPI key for broader news sentiment coverage (free tier: 100 req/day)
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# ── ML overlay filters (opt-in) ──────────────────────────────────────────────
# The optimizer tunes the BASE strategy (BB %b + RSI + ATR-percentile gate +
# price-anchored ATR exits). The HMM regime classifier and XGBoost signal filter
# are optional risk overlays that can only REDUCE trades, never add them. They
# are OFF by default so live trading exactly reproduces the backtested strategy
# (coherence). Enable them knowingly via env to add conservative gating.

USE_REGIME_FILTER: bool = os.getenv("USE_REGIME_FILTER", "false").lower() in ("true", "1", "yes")
USE_ML_FILTER: bool = os.getenv("USE_ML_FILTER", "false").lower() in ("true", "1", "yes")

SIGNAL_QUALITY_THRESHOLD: float = float(os.getenv("SIGNAL_QUALITY_THRESHOLD", "0.55"))
CHOPPY_REGIME_ID: int = 1               # HMM state that means CHOPPY
TRENDING_DOWN_REGIME_ID: int = 0        # HMM state used as the safe fallback (blocks entry)

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
# Guards the cache against concurrent access: load_optimized_params runs on the
# candle pipeline while save_params runs from the /params REST endpoint.
_params_lock = threading.Lock()


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
    with _params_lock:
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
            with _params_lock:
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
