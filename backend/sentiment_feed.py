"""
News and government press release sentiment feed.

Polls a curated list of RSS feeds every POLL_INTERVAL_MIN minutes in a
background daemon thread. Each headline is scored by keyword matching.

No API key required — all sources are public RSS.
Optional: set NEWS_API_KEY env var to also query newsapi.org (100 req/day free).

Usage:
    start_sentiment_feed()   # call once at startup
    reading = get_sentiment()  # call from any thread
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

POLL_INTERVAL_MIN: int = 5       # refresh every 5 minutes
FEEDPARSER_TIMEOUT: int = 8      # seconds per feed
MAX_ENTRIES_PER_FEED: int = 20   # cap items read from each feed
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# ── RSS sources (all public, no auth required) ────────────────────────────────
# Ordered by reliability and relevance for NIFTY intraday moves.
RSS_FEEDS = [
    ("PIB",        "https://pib.gov.in/RssMain.aspx"),
    ("RBI",        "https://www.rbi.org.in/Scripts/rss.aspx"),
    ("ET_Markets", "https://economictimes.indiatimes.com/markets/stocks/rss.cms"),
    ("ET_Economy", "https://economictimes.indiatimes.com/news/economy/rss.cms"),
    ("BS_Markets", "https://www.business-standard.com/rss/markets-106.rss"),
    ("Mint",       "https://www.livemint.com/rss/markets"),
]

# ── Keyword dictionaries ──────────────────────────────────────────────────────
# All comparisons are lower-cased.

# If ANY of these appear → do not trade (circuit-level event)
HALT_KEYWORDS = [
    "circuit breaker", "market halt", "trading suspended", "trading halted",
    "market closed", "emergency session", "systemic risk", "black swan",
    "war declared", "nuclear",
]

# Phrases scored +1 (NIFTY-bullish)
BULLISH_KEYWORDS = [
    "rate cut", "repo cut", "easing", "stimulus", "fiscal boost",
    "gdp growth", "gdp beat", "record high", "record surplus",
    "trade surplus", "fii buying", "dii buying", "foreign inflow",
    "market rally", "rally", "bullish", "upgrade", "relief",
    "recovery", "expansion", "growth", "reform", "capex boost",
    "rate pause", "dovish", "lower inflation", "disinflation",
]

# Phrases scored -1 (NIFTY-bearish)
BEARISH_KEYWORDS = [
    "rate hike", "repo hike", "tightening", "ban", "default",
    "crisis", "crash", "recession", "penalty", "probe",
    "fii selling", "foreign outflow", "market fall", "bearish",
    "downgrade", "caution", "risk-off", "trade deficit",
    "current account deficit", "higher inflation", "inflation surge",
    "hawkish", "sanctions", "fine", "suspension", "sebi action",
    "tax hike", "tax increase", "npa", "npa rise",
]

# Terms that mark a headline as relevant to Indian equity markets
RELEVANCE_KEYWORDS = [
    "nifty", "sensex", "bse", "nse", "equity", "market", "stock",
    "index", "rbi", "sebi", "mpc", "repo", "fiscal", "gdp", "cpi",
    "inflation", "budget", "fii", "dii", "foreign investor",
    "rupee", "inr", "india", "indian", "economy",
]

_RELEVANCE_RE = re.compile("|".join(re.escape(k) for k in RELEVANCE_KEYWORDS), re.I)


def _is_relevant(text: str) -> bool:
    return bool(_RELEVANCE_RE.search(text))


def _score_text(text: str) -> int:
    """Return -99 (halt), or a signed score in [-5, 5] per article."""
    t = text.lower()
    if any(k in t for k in HALT_KEYWORDS):
        return -99
    score = 0
    for k in BULLISH_KEYWORDS:
        if k in t:
            score += 1
    for k in BEARISH_KEYWORDS:
        if k in t:
            score -= 1
    return max(-5, min(5, score))


@dataclass
class SentimentReading:
    score: float = 0.0           # -1.0 very bearish → +1.0 very bullish
    halt: bool = False           # True = breaking event, do not trade
    headlines: List[str] = field(default_factory=list)
    fetched_at: Optional[datetime] = None
    feeds_ok: int = 0

    @property
    def age_minutes(self) -> float:
        if not self.fetched_at:
            return 999.0
        return (datetime.now(timezone.utc) - self.fetched_at).total_seconds() / 60

    @property
    def is_fresh(self) -> bool:
        return self.age_minutes < POLL_INTERVAL_MIN * 2

    @property
    def bullish(self) -> bool:
        return self.score > 0.15

    @property
    def bearish(self) -> bool:
        return self.score < -0.15

    def direction_ok(self, trade_dir: str) -> bool:
        """True if sentiment does not actively contradict the trade direction."""
        if self.halt:
            return False
        if trade_dir == "CE":
            return not self.bearish
        if trade_dir == "PE":
            return not self.bullish
        return True


def _fetch_rss() -> SentimentReading:
    """Synchronously poll all RSS feeds. Blocks the caller."""
    try:
        import feedparser
    except ImportError:
        logger.warning('"feedparser not installed — sentiment feed inactive"')
        return SentimentReading(fetched_at=datetime.now(timezone.utc))

    total_score = 0
    n_scored = 0
    halt = False
    headlines: List[str] = []
    feeds_ok = 0

    for name, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"},
                                    )
            if feed.bozo and not feed.entries:
                continue
            feeds_ok += 1
            for entry in feed.entries[:MAX_ENTRIES_PER_FEED]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title} {summary}"
                if not _is_relevant(text):
                    continue
                s = _score_text(text)
                if s == -99:
                    halt = True
                    headlines.append(f"[HALT] {title[:100]}")
                    continue
                total_score += s
                n_scored += 1
                if len(headlines) < 10:
                    tag = "+" if s > 0 else ("-" if s < 0 else "~")
                    headlines.append(f"[{tag}/{name}] {title[:90]}")
        except Exception as exc:
            logger.debug(f'"RSS feed {name} failed: {exc}"')

    # Also query NewsAPI if key present (broader English coverage)
    if NEWS_API_KEY:
        try:
            import httpx
            url = (
                "https://newsapi.org/v2/everything"
                f"?q=NIFTY+OR+Sensex+OR+RBI+OR+BSE&language=en"
                f"&sortBy=publishedAt&pageSize=20&apiKey={NEWS_API_KEY}"
            )
            r = httpx.get(url, timeout=8)
            if r.status_code == 200:
                for art in r.json().get("articles", []):
                    text = f"{art.get('title', '')} {art.get('description', '')}"
                    if not _is_relevant(text):
                        continue
                    s = _score_text(text)
                    if s == -99:
                        halt = True
                    else:
                        total_score += s
                        n_scored += 1
        except Exception as exc:
            logger.debug(f'"NewsAPI fetch failed: {exc}"')

    norm = total_score / max(n_scored, 1)
    # Clamp to [-1, 1] with softer normalization (score of 3 → ≈1.0)
    norm_score = max(-1.0, min(1.0, norm / 3.0))
    return SentimentReading(
        score=round(norm_score, 3),
        halt=halt,
        headlines=headlines,
        fetched_at=datetime.now(timezone.utc),
        feeds_ok=feeds_ok,
    )


# ── Module-level singleton ────────────────────────────────────────────────────

_reading: SentimentReading = SentimentReading()
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None


def get_sentiment() -> SentimentReading:
    """Return the latest sentiment reading (thread-safe, non-blocking)."""
    with _lock:
        return _reading


def _poll_loop() -> None:
    global _reading
    while True:
        try:
            result = _fetch_rss()
            with _lock:
                _reading = result
            logger.info(
                f'"Sentiment: score={result.score:+.2f} halt={result.halt} '
                f'feeds_ok={result.feeds_ok}/{len(RSS_FEEDS)} '
                f'age={result.age_minutes:.0f}m"'
            )
        except Exception as exc:
            logger.error(f'"Sentiment poll error: {exc}"')
        time.sleep(POLL_INTERVAL_MIN * 60)


def start_sentiment_feed() -> None:
    """Start the background sentiment polling thread. Idempotent."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_poll_loop, daemon=True, name="sentiment-poll")
    _thread.start()
    logger.info('"Sentiment feed started"')
