"""
Telegram alert sender for trade events, errors, and daily summaries.
Uses the Telegram Bot API directly via requests (no extra dependency).
"""

import logging
from typing import Optional

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
SEND_TIMEOUT = 10   # seconds


def _send(text: str) -> bool:
    """
    Send a Telegram message. Returns True on success.
    Silently logs and returns False if credentials are missing or call fails.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug('"Telegram credentials not set — skipping notification"')
        return False

    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=SEND_TIMEOUT)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.error(f'"Telegram send failed: {exc}"')
        return False


def notify_entry(
    trade_id: str,
    direction: str,
    symbol: str,
    strike: int,
    entry_price: float,
    quantity: int,
    percent_b: float,
    signal_quality_score: float,
    regime_name: str,
    dry_run: bool,
) -> None:
    """Send trade entry alert."""
    mode = "[DRY RUN] " if dry_run else ""
    emoji = "🟢" if direction == "CE" else "🔴"
    text = (
        f"{emoji} <b>{mode}ENTRY — {direction}</b>\n"
        f"Symbol   : <code>{symbol}</code>\n"
        f"Strike   : {strike}\n"
        f"Entry ₹  : {entry_price:.2f}\n"
        f"Qty      : {quantity}\n"
        f"%%b       : {percent_b:.3f}\n"
        f"ML Score : {signal_quality_score:.2f}\n"
        f"Regime   : {regime_name}\n"
        f"Trade ID : {trade_id}"
    )
    _send(text)
    logger.info(f'"Telegram entry alert sent for {trade_id}"')


def notify_exit(
    trade_id: str,
    symbol: str,
    exit_price: float,
    pnl: float,
    exit_reason: str,
    daily_pnl: float,
    dry_run: bool,
) -> None:
    """Send trade exit alert."""
    mode = "[DRY RUN] " if dry_run else ""
    pnl_emoji = "✅" if pnl >= 0 else "❌"
    text = (
        f"{pnl_emoji} <b>{mode}EXIT — {exit_reason}</b>\n"
        f"Symbol   : <code>{symbol}</code>\n"
        f"Exit ₹   : {exit_price:.2f}\n"
        f"P&amp;L      : ₹{pnl:+.2f}\n"
        f"Day P&amp;L  : ₹{daily_pnl:+.2f}\n"
        f"Trade ID : {trade_id}"
    )
    _send(text)
    logger.info(f'"Telegram exit alert sent for {trade_id}: P&L=₹{pnl:.2f}"')


def notify_error(context: str, error: str) -> None:
    """Send error alert."""
    text = (
        f"⚠️ <b>ERROR</b>\n"
        f"Context : {context}\n"
        f"Error   : <code>{error[:300]}</code>"
    )
    _send(text)


def notify_regime_filter(regime_name: str, percent_b: float) -> None:
    """Alert when a signal was skipped due to non-CHOPPY regime."""
    text = (
        f"🟡 <b>Signal skipped — Regime filter</b>\n"
        f"Regime : {regime_name}\n"
        f"%%b     : {percent_b:.3f}"
    )
    _send(text)


def notify_ml_filter(percent_b: float, score: float, threshold: float) -> None:
    """Alert when a signal was rejected by the ML quality filter."""
    text = (
        f"🔵 <b>Signal skipped — ML filter</b>\n"
        f"%%b       : {percent_b:.3f}\n"
        f"Score    : {score:.3f}\n"
        f"Threshold: {threshold:.2f}"
    )
    _send(text)


def notify_daily_summary(
    trades_today: int,
    daily_pnl: float,
    win_rate: float,
    dry_run: bool,
) -> None:
    """End-of-day summary alert."""
    mode = "[DRY RUN] " if dry_run else ""
    emoji = "📊"
    text = (
        f"{emoji} <b>{mode}Daily Summary</b>\n"
        f"Trades : {trades_today}\n"
        f"Win %  : {win_rate:.0%}\n"
        f"P&amp;L   : ₹{daily_pnl:+.2f}"
    )
    _send(text)


def notify_bot_started(dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "LIVE"
    _send(f"🤖 <b>Bot started [{mode}]</b>")


def notify_bot_stopped() -> None:
    _send("🛑 <b>Bot stopped</b>")
