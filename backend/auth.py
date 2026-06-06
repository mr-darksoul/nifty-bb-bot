"""
Kite Connect OAuth flow and token management.

Handles the two-step auth:
  1. Redirect user to Kite login URL → they get a request_token
  2. Exchange request_token + API secret for a session access_token
  3. Persist access_token in environment for the current process lifetime
"""

import logging
import os
import stat
from pathlib import Path
from typing import Optional

from config import KITE_API_KEY, KITE_API_SECRET

logger = logging.getLogger(__name__)

_kite_instance = None
_TOKEN_FILE = Path(__file__).parent / ".kite_token"


def _save_token_to_file(token: str) -> None:
    try:
        _TOKEN_FILE.write_text(token)
        _TOKEN_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # owner read/write only
    except Exception as exc:
        logger.warning(f'"Could not persist access token to file: {exc}"')


def _load_token_from_file() -> str:
    try:
        if _TOKEN_FILE.exists():
            return _TOKEN_FILE.read_text().strip()
    except Exception:
        pass
    return ""


def get_kite():
    """Return the singleton KiteConnect instance, creating it if needed."""
    global _kite_instance
    if _kite_instance is None:
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            raise RuntimeError("kiteconnect package not installed")

        if not KITE_API_KEY:
            raise ValueError("KITE_API_KEY environment variable is not set")

        _kite_instance = KiteConnect(api_key=KITE_API_KEY)
        current_token = os.getenv("KITE_ACCESS_TOKEN", "") or _load_token_from_file()
        if current_token:
            _kite_instance.set_access_token(current_token)
            logger.info('"KiteConnect initialised with existing access token"')
        else:
            logger.warning('"No KITE_ACCESS_TOKEN set — Kite login required"')

    return _kite_instance


def get_login_url() -> str:
    """Return the Kite OAuth login URL for the user to open in their browser."""
    kite = get_kite()
    url = kite.login_url()
    logger.info(f'"Generated Kite login URL"')
    return url


def exchange_request_token(request_token: str) -> str:
    """
    Exchange a one-time request_token for a persistent access_token.

    Args:
        request_token: Token received from Kite redirect URL query param.

    Returns:
        access_token string.

    Raises:
        ValueError: If exchange fails.
    """
    if not KITE_API_SECRET:
        raise ValueError("KITE_API_SECRET is not set — cannot generate session")

    kite = get_kite()
    try:
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token: str = data["access_token"]
        kite.set_access_token(access_token)
        os.environ["KITE_ACCESS_TOKEN"] = access_token
        _save_token_to_file(access_token)
        logger.info('"Kite session generated successfully"')
        return access_token
    except Exception as exc:
        logger.error(f'"Kite session generation failed: {exc}"')
        raise ValueError(f"Token exchange failed: {exc}") from exc


def is_authenticated() -> bool:
    """Return True if the current KiteConnect instance has a valid access token."""
    token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not token:
        return False
    try:
        kite = get_kite()
        kite.profile()   # lightweight API call to verify token
        return True
    except Exception:
        return False


def set_access_token(token: str) -> None:
    """Manually set the access token (e.g. loaded from .env at startup)."""
    global _kite_instance
    os.environ["KITE_ACCESS_TOKEN"] = token
    if _kite_instance is not None:
        _kite_instance.set_access_token(token)
    logger.info('"Access token updated"')
