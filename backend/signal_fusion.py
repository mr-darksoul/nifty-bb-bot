"""
Signal fusion engine — the confidence gatekeeper.

Takes the raw BB-breakout direction (already validated by the existing momentum
strategy) and overlays it with multi-timeframe alignment, VWAP position,
RSI momentum, options OI bias, news sentiment, ATR gate, and S/R proximity.

Returns a FusedSignal that either confirms the trade (confidence ≥ threshold)
or blocks it with a reason list.

Target profile: ~70% win rate at 1:3 R:R. Entry fires only when the
confluence is strong — accepting fewer trades for higher quality.

Usage:
    fused = evaluate(base_direction, mtf_snap, df_ind, ms, oi, sent, params)
    if fused.approved:
        # place trade
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Confidence weights (sum = 100) ────────────────────────────────────────────
W_MTF           = 25   # timeframe alignment (3 TFs)
W_BB_STRENGTH   = 20   # how far %b has crossed beyond the band
W_VWAP          = 15   # price side of VWAP
W_RSI           = 15   # RSI in momentum zone
W_OPTIONS_OI    = 10   # PCR / OI bias
W_SENTIMENT     = 10   # news not adverse
W_ATR_GATE      = 5    # volatility in tradeable range

# Minimum confidence to approve entry. Sourced from config so the env override
# (SIGNAL_FUSION_THRESHOLD) actually takes effect — previously this was a
# hard-coded 60.0 that silently ignored the configured value.
try:
    from config import SIGNAL_FUSION_THRESHOLD as ENTRY_THRESHOLD
except Exception:
    ENTRY_THRESHOLD: float = 60.0

# ATR percentile gate (from indicators.atr_percentile, 0–100 scale)
ATR_PCT_MIN: float = 25.0
ATR_PCT_MAX: float = 88.0

# Minimum % distance from nearest S/R level before entry is penalised
MIN_SR_DISTANCE_PCT: float = 0.30

# Time gates — no new entries during these windows
_NO_TRADE_ZONES = [
    (dtime(9, 15), dtime(9, 35)),    # opening gap resolution
    (dtime(11, 30), dtime(12, 5)),   # midday chop / lunch
]


@dataclass
class FusedSignal:
    direction: str = "NONE"          # confirmed "CE" or "PE", else "NONE"
    confidence: float = 0.0          # 0-100 composite score
    reasons: List[str] = field(default_factory=list)    # passed checks
    blocking: List[str] = field(default_factory=list)   # failed / blocking checks
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.direction != "NONE" and self.confidence >= ENTRY_THRESHOLD


def _in_no_trade_zone() -> bool:
    now = datetime.now().time()
    for start, end in _NO_TRADE_ZONES:
        if start <= now <= end:
            return True
    return False


def evaluate(
    base_direction: str,
    mtf_snap,           # MultiTFSnapshot from multi_tf_feed.get_snapshot()
    df_ind,             # pd.DataFrame: indicator-enriched 1-min candles
    market_struct,      # MarketStructure from market_structure.get_market_structure()
    oi_reading,         # OIReading from options_intelligence.get_oi_reading()
    sentiment,          # SentimentReading from sentiment_feed.get_sentiment()
    params: dict,       # from config.load_optimized_params()
) -> FusedSignal:
    """
    Evaluate whether to approve an entry in base_direction.
    Returns FusedSignal.approved == True to proceed, False to skip.
    """
    result = FusedSignal()

    if base_direction not in ("CE", "PE"):
        result.blocking.append("invalid_direction")
        return result

    result.direction = base_direction
    is_long = base_direction == "CE"

    # ── Hard blockers (abort immediately) ────────────────────────────────────

    if _in_no_trade_zone():
        result.direction = "NONE"
        result.blocking.append("no_trade_zone")
        return result

    if sentiment.halt:
        result.direction = "NONE"
        result.blocking.append("sentiment_halt")
        return result

    if df_ind.empty:
        result.direction = "NONE"
        result.blocking.append("no_data")
        return result

    # ── Indicator snapshot ────────────────────────────────────────────────────

    import math
    last = df_ind.iloc[-1]
    close = float(last["close"])
    pb_1m = float(last.get("percent_b", 0.5) or 0.5)
    rsi_1m = float(last.get("rsi", 50) or 50)
    _atr_pct_raw = last.get("atr_pct", 50)
    atr_pct = float(_atr_pct_raw) if (_atr_pct_raw is not None and not math.isnan(float(_atr_pct_raw or 0))) else 50.0

    # Also grab 15-min %b for BB strength (already computed by multi_tf_feed)
    pb15 = mtf_snap.tf15.percent_b if mtf_snap.tf15 else pb_1m

    # ── 1. MTF alignment (25 pts) ─────────────────────────────────────────────
    aligned = mtf_snap.aligned_frames(base_direction)  # 0-3
    mtf_score = (aligned / 3.0) * W_MTF
    result.reasons.append(f"mtf:{aligned}/3")

    # MTF must have at least 2/3 frames aligned (soft requirement → penalise but not block)
    if aligned < 2:
        result.blocking.append(f"weak_mtf:{aligned}/3")
        mtf_score *= 0.3    # keep some score, let threshold decide

    # ── 2. BB breakout strength on 15-min TF (20 pts) ────────────────────────
    # momentum_breakout: CE fires when %b crosses ABOVE bb_overbought (1.0);
    # PE fires when %b crosses BELOW bb_oversold (0.0).
    # Strength = how far %b has gone past the band edge on the signal bar.
    bb_overbought = float(params.get("bb_overbought", 1.0))
    bb_oversold = float(params.get("bb_oversold", 0.0))
    if is_long:
        excess = max(0.0, pb15 - bb_overbought)   # how far above the upper band
    else:
        excess = max(0.0, bb_oversold - pb15)     # how far below the lower band
    # Full marks at 0.10 excess (strong breakout); partial at smaller values
    bb_score = min(1.0, excess / 0.10) * W_BB_STRENGTH
    result.reasons.append(f"bb_excess:{pb15:.3f}")

    # ── 3. VWAP position (15 pts) ─────────────────────────────────────────────
    vwap_aligned = (
        (is_long and market_struct.price_above_vwap(close)) or
        (not is_long and not market_struct.price_above_vwap(close))
    )
    vwap_score = W_VWAP if vwap_aligned else 0.0
    result.reasons.append(
        f"vwap:{'above' if market_struct.price_above_vwap(close) else 'below'}"
        f"({'ok' if vwap_aligned else 'against'})"
    )

    # ── 4. RSI momentum (15 pts) ──────────────────────────────────────────────
    rsi_score = 0.0
    if is_long:
        # RSI 40-65: momentum but not overbought
        if 40 <= rsi_1m <= 65:
            rsi_score = W_RSI
        elif 35 <= rsi_1m < 40 or 65 < rsi_1m <= 72:
            rsi_score = W_RSI * 0.5
    else:
        # RSI 35-60: momentum down, not oversold
        if 35 <= rsi_1m <= 60:
            rsi_score = W_RSI
        elif 28 <= rsi_1m < 35 or 60 < rsi_1m <= 65:
            rsi_score = W_RSI * 0.5
    result.reasons.append(f"rsi:{rsi_1m:.1f}")

    # ── 5. Options OI / PCR (10 pts) ─────────────────────────────────────────
    oi_score = 0.0
    if oi_reading.is_fresh:
        if oi_reading.confirms_direction(base_direction):
            oi_score = W_OPTIONS_OI if oi_reading.oi_bias != "NEUTRAL" else W_OPTIONS_OI * 0.6
        # Penalise only if price is within 0.5% of the OI wall — approaching
        # resistance (CE) or approaching support from above (PE).
        # A negative distance means price already broke through the wall,
        # which is NOT a reason to penalise (it's actually a confirmation).
        if is_long:
            d = oi_reading.distance_to_call_wall_pct(close)  # +ve = wall above price
        else:
            d = oi_reading.distance_to_put_wall_pct(close)   # +ve = price above wall
        if 0.0 <= d < 0.5:   # within 0.5% but not yet through the wall
            oi_score *= 0.5
            result.blocking.append(f"near_oi_wall:{d:.2f}%")
        result.reasons.append(f"pcr:{oi_reading.pcr:.2f}({oi_reading.oi_bias})")
    else:
        oi_score = W_OPTIONS_OI * 0.4   # partial credit for stale OI
        result.reasons.append("pcr:stale")

    # ── 6. News sentiment (10 pts) ────────────────────────────────────────────
    sent_score = 0.0
    if sentiment.is_fresh:
        if sentiment.direction_ok(base_direction):
            sent_score = W_SENTIMENT if abs(sentiment.score) > 0.1 else W_SENTIMENT * 0.7
        result.reasons.append(f"sentiment:{sentiment.score:+.2f}")
    else:
        sent_score = W_SENTIMENT * 0.5
        result.reasons.append("sentiment:stale")

    # ── 7. ATR percentile gate (5 pts) ────────────────────────────────────────
    atr_score = 0.0
    if ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX:
        atr_score = W_ATR_GATE
    result.reasons.append(f"atr_pct:{atr_pct:.0f}")

    # ── Total score ───────────────────────────────────────────────────────────
    total = mtf_score + bb_score + vwap_score + rsi_score + oi_score + sent_score + atr_score
    result.components = {
        "mtf": round(mtf_score, 1),
        "bb": round(bb_score, 1),
        "vwap": round(vwap_score, 1),
        "rsi": round(rsi_score, 1),
        "oi": round(oi_score, 1),
        "sentiment": round(sent_score, 1),
        "atr": round(atr_score, 1),
    }

    # ── S/R proximity penalty ─────────────────────────────────────────────────
    if is_long:
        sr_dist = market_struct.distance_to_resistance_pct(close)
    else:
        sr_dist = market_struct.distance_to_support_pct(close)

    if sr_dist < MIN_SR_DISTANCE_PCT:
        total *= 0.65
        result.blocking.append(f"near_sr:{sr_dist:.2f}%")

    result.confidence = round(min(100.0, total), 1)

    # ── Final verdict ─────────────────────────────────────────────────────────
    if result.confidence < ENTRY_THRESHOLD:
        result.direction = "NONE"
        result.blocking.append(f"low_confidence:{result.confidence:.1f}<{ENTRY_THRESHOLD}")

    logger.info(
        f'"SignalFusion dir={base_direction} conf={result.confidence:.1f} '
        f'approved={result.approved} blocking={result.blocking}"'
    )
    return result
