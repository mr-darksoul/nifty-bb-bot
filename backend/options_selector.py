"""
ATM strike and weekly expiry resolver for NIFTY options.

Queries the Kite instruments API to find:
  - The current week's expiry date
  - The ATM strike closest to the spot price
  - The full tradingsymbol (e.g. NIFTY2461222000CE)
"""

import logging
from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd

from config import CAPITAL_PER_TRADE, LOT_SIZE, MIN_DAYS_TO_EXPIRY, NFO_EXCHANGE, NIFTY_SYMBOL, SLIPPAGE_PCT

logger = logging.getLogger(__name__)

NIFTY_STRIKE_STEP = 50   # NIFTY strikes are in multiples of 50


def _round_to_strike(price: float, step: int = NIFTY_STRIKE_STEP) -> int:
    """Round price to nearest valid NIFTY strike."""
    return round(price / step) * step


def _next_expiry_weekday(from_date: Optional[date] = None) -> date:
    """Return the nearest upcoming NIFTY weekly expiry (Tuesday)."""
    d = from_date or date.today()
    days_ahead = 1 - d.weekday()   # Tuesday is weekday 1
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


def _expiry_suffix(expiry: date) -> str:
    """
    Kite uses a compact expiry format in the trading symbol.
    Example: 2024-06-12 → '2461'2 = year24 + month6(1-char) + day12
    Month encoding: Jan=1, Feb=2, ..., Sep=9, Oct=O, Nov=N, Dec=D
    """
    month_map = {
        1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
        7: "7", 8: "8", 9: "9", 10: "O", 11: "N", 12: "D",
    }
    year_2 = str(expiry.year)[-2:]
    mon = month_map[expiry.month]
    day = f"{expiry.day:02d}"
    return f"{year_2}{mon}{day}"


class OptionsSelector:
    """Resolves ATM NIFTY option instruments for a given spot price and date."""

    def __init__(self) -> None:
        self._instruments_cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[date] = None
        self.last_candidates: list = []  # [{strike, symbol, ltp, status}] from last get_premium_capped_instrument call

    def _get_instruments(self) -> pd.DataFrame:
        """Fetch NFO instrument list, cached for the trading day."""
        today = date.today()
        if self._instruments_cache is not None and self._cache_date == today:
            return self._instruments_cache

        from auth import get_kite
        kite = get_kite()
        try:
            instruments = kite.instruments(exchange=NFO_EXCHANGE)
            df = pd.DataFrame(instruments)
            df = df[df["name"] == "NIFTY"]
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
            self._instruments_cache = df
            self._cache_date = today
            logger.info(f'"Fetched {len(df)} NIFTY NFO instruments"')
        except Exception as exc:
            logger.error(f'"Failed to fetch instruments: {exc}"')
            if self._instruments_cache is not None:
                return self._instruments_cache
            raise

        return self._instruments_cache

    def get_weekly_expiry(self) -> date:
        """Return the nearest weekly expiry with at least MIN_DAYS_TO_EXPIRY remaining.

        On expiry day (0 DTE Tuesday) this rolls forward to next week, avoiding
        near-zero-time-value options with extreme gamma and poor liquidity.
        """
        try:
            df = self._get_instruments()
            today = date.today()
            future_expiries = sorted(df["expiry"].unique())
            upcoming = [e for e in future_expiries if (e - today).days >= MIN_DAYS_TO_EXPIRY]
            if upcoming:
                expiry = upcoming[0]
                days_left = (expiry - today).days
                logger.info(f'"Selected expiry: {expiry} ({days_left} DTE)"')
                return expiry
        except Exception as exc:
            logger.warning(f'"Could not fetch expiry from instruments: {exc} — estimating"')
        # Fallback: next expiry weekday (Tuesday), skipping today if it is expiry day
        fallback = _next_expiry_weekday()
        if (fallback - date.today()).days < MIN_DAYS_TO_EXPIRY:
            fallback = _next_expiry_weekday(fallback + timedelta(days=1))
        return fallback

    def get_atm_instrument(
        self,
        spot_price: float,
        option_type: str,         # "CE" or "PE"
        expiry: Optional[date] = None,
    ) -> Tuple[str, int, int]:
        """
        Find the ATM option instrument for the given spot and option type.

        Returns:
            (tradingsymbol, strike, instrument_token)
        """
        if option_type not in ("CE", "PE"):
            raise ValueError(f"option_type must be CE or PE, got: {option_type}")

        expiry = expiry or self.get_weekly_expiry()
        atm_strike = _round_to_strike(spot_price)

        try:
            df = self._get_instruments()
            mask = (
                (df["expiry"] == expiry)
                & (df["instrument_type"] == option_type)
                & (df["strike"] == atm_strike)
            )
            matches = df[mask]

            if matches.empty:
                # Try adjacent strikes
                for delta in [NIFTY_STRIKE_STEP, -NIFTY_STRIKE_STEP, 2 * NIFTY_STRIKE_STEP]:
                    alt_strike = atm_strike + delta
                    mask = (
                        (df["expiry"] == expiry)
                        & (df["instrument_type"] == option_type)
                        & (df["strike"] == alt_strike)
                    )
                    matches = df[mask]
                    if not matches.empty:
                        atm_strike = alt_strike
                        break

            if matches.empty:
                raise ValueError(
                    f"No {option_type} instrument found for strike {atm_strike}, expiry {expiry}"
                )

            row = matches.iloc[0]
            symbol: str = row["tradingsymbol"]
            token: int = int(row["instrument_token"])
            logger.info(
                f'"Selected ATM instrument: {symbol} strike={atm_strike} expiry={expiry}"'
            )
            return symbol, atm_strike, token

        except Exception as exc:
            # Fallback: construct symbol manually
            logger.warning(f'"Instrument lookup failed: {exc} — constructing symbol manually"')
            suffix = _expiry_suffix(expiry)
            symbol = f"NIFTY{suffix}{atm_strike}{option_type}"
            logger.info(f'"Fallback symbol: {symbol}"')
            return symbol, atm_strike, 0

    def get_premium_capped_instrument(
        self,
        spot_price: float,
        option_type: str,
        expiry: Optional[date] = None,
        max_trade_value: float = CAPITAL_PER_TRADE,
    ) -> Tuple[str, int, int, float]:
        """
        Find the nearest OTM option whose one-lot premium fits the trade cap.

        Returns:
            (tradingsymbol, strike, instrument_token, ltp)
        """
        if option_type not in ("CE", "PE"):
            raise ValueError(f"option_type must be CE or PE, got: {option_type}")

        expiry = expiry or self.get_weekly_expiry()
        atm_strike = _round_to_strike(spot_price)
        max_ltp = max_trade_value / (LOT_SIZE * (1 + SLIPPAGE_PCT))
        df = self._get_instruments()
        strike_mask = df["strike"] >= atm_strike if option_type == "CE" else df["strike"] <= atm_strike

        mask = (
            (df["expiry"] == expiry)
            & (df["instrument_type"] == option_type)
            & strike_mask
        )
        candidates = df[mask].copy()
        if candidates.empty:
            raise ValueError(f"No {option_type} instruments found for expiry {expiry}")

        candidates["distance"] = (candidates["strike"] - atm_strike).abs()
        candidates = candidates.sort_values(["distance", "strike"], ascending=[True, option_type == "CE"])

        top = candidates.head(20)
        quote_keys = [f"{NFO_EXCHANGE}:{row['tradingsymbol']}" for _, row in top.iterrows()]
        try:
            from auth import get_kite
            batch_quotes = get_kite().ltp(quote_keys)
        except Exception as exc:
            logger.warning(f'"Batch LTP fetch failed: {exc} — falling back to per-symbol"')
            batch_quotes = {}

        self.last_candidates = []
        selected: Optional[tuple] = None

        for _, row in top.iterrows():
            symbol = str(row["tradingsymbol"])
            strike = int(row["strike"])
            token = int(row["instrument_token"])
            quote_key = f"{NFO_EXCHANGE}:{symbol}"
            ltp_data = batch_quotes.get(quote_key, {})
            ltp = float(ltp_data.get("last_price", 0)) if ltp_data else 0.0
            if ltp <= 0:
                ltp = self.get_ltp(token, symbol)  # per-symbol fallback

            if ltp <= 0:
                status = "NO_LTP"
            elif ltp > max_ltp:
                status = "CAP_EXCEEDED"
            elif selected is None:
                status = "SELECTED"
                selected = (symbol, strike, token, ltp)
            else:
                status = "AVAILABLE"

            self.last_candidates.append({
                "strike": strike,
                "symbol": symbol,
                "ltp": round(ltp, 2),
                "status": status,
            })

        if selected:
            sym, stk, tok, ltp_val = selected
            logger.info(
                f'"Selected premium-capped option: {sym} strike={stk} '
                f'ltp={ltp_val:.2f} max_ltp={max_ltp:.2f} cap={max_trade_value:.2f}"'
            )
            return sym, stk, tok, ltp_val

        raise ValueError(
            f"No {option_type} option under premium cap ₹{max_trade_value:.2f} "
            f"for lot_size={LOT_SIZE}, max_ltp={max_ltp:.2f}"
        )

    def get_ltp(self, instrument_token: int, tradingsymbol: str) -> float:
        """Fetch last traded price for an option symbol."""
        from auth import get_kite
        kite = get_kite()
        try:
            quote_key = f"{NFO_EXCHANGE}:{tradingsymbol}"
            quotes = kite.ltp([quote_key])
            ltp = quotes[quote_key]["last_price"]
            logger.debug(f'"LTP for {tradingsymbol}: {ltp}"')
            return float(ltp)
        except Exception as exc:
            logger.error(f'"LTP fetch failed for {tradingsymbol}: {exc}"')
            return 0.0
