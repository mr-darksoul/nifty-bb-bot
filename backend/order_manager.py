"""
Order lifecycle management: entry, exit, position tracking, and CSV persistence.

In DRY_RUN mode all orders are logged but kite.place_order() is not called.
"""

import csv
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import (
    BROKERAGE_PER_ORDER,
    CAPITAL_PER_TRADE,
    DRY_RUN,
    LOT_SIZE,
    NFO_EXCHANGE,
    SLIPPAGE_PCT,
    TRADE_LOG_PATH,
)

logger = logging.getLogger(__name__)

CSV_FIELDNAMES = [
    "trade_id", "entry_time", "exit_time", "direction", "symbol", "strike",
    "entry_price", "exit_price", "quantity", "pnl", "exit_reason",
    "signal_quality_score", "entry_pb", "exit_pb", "regime",
    "entry_spot", "target_spot", "sl_spot",
]


@dataclass
class Trade:
    """Single option trade record."""
    trade_id: str
    entry_time: str
    exit_time: str = ""
    direction: str = ""            # "CE" or "PE"
    symbol: str = ""               # tradingsymbol e.g. NIFTY2461222000CE
    strike: int = 0
    entry_price: float = 0.0       # option LTP at entry
    exit_price: float = 0.0
    quantity: int = 0              # number of lots * LOT_SIZE
    pnl: float = 0.0
    exit_reason: str = ""
    signal_quality_score: float = 0.0
    entry_pb: float = 0.0
    exit_pb: float = 0.0
    regime: int = -1
    # Price-anchored exit levels on the NIFTY spot, locked in at entry (ATR
    # multiples). The position is closed when spot crosses target_spot/sl_spot.
    entry_spot: float = 0.0
    target_spot: float = 0.0
    sl_spot: float = 0.0
    kite_entry_order_id: str = ""
    kite_exit_order_id: str = ""
    is_open: bool = True


class OrderManager:
    """Manages order placement, position tracking, and trade persistence."""

    def __init__(self) -> None:
        self._active_trade: Optional[Trade] = None
        self._trade_history: List[Trade] = []
        self._trade_counter: int = 0
        self._ensure_csv_header()
        logger.info(f'"OrderManager initialised. DRY_RUN={DRY_RUN}"')

    # ── CSV persistence ───────────────────────────────────────────────────────

    def _ensure_csv_header(self) -> None:
        if not TRADE_LOG_PATH.exists():
            with open(TRADE_LOG_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()

    def _append_trade_to_csv(self, trade: Trade) -> None:
        row = {k: getattr(trade, k, "") for k in CSV_FIELDNAMES}
        with open(TRADE_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writerow(row)

    # ── Entry ─────────────────────────────────────────────────────────────────

    def enter_trade(
        self,
        direction: str,
        symbol: str,
        strike: int,
        ltp: float,
        percent_b: float,
        regime: int,
        signal_quality_score: float,
        entry_spot: float = 0.0,
        target_spot: float = 0.0,
        sl_spot: float = 0.0,
    ) -> Optional[Trade]:
        """
        Place a market buy order for the option and record the trade.

        Args:
            direction:            "CE" or "PE".
            symbol:               Kite tradingsymbol.
            strike:               Strike price.
            ltp:                  Option last traded price.
            percent_b:            %b value at entry.
            regime:               Current regime id.
            signal_quality_score: ML score for this signal.

        Returns:
            Trade object or None if rejected.
        """
        if self._active_trade is not None:
            logger.warning('"enter_trade called with an existing open trade — ignored"')
            return None

        if ltp <= 0:
            logger.error('"Cannot enter trade: LTP is zero or negative"')
            return None

        entry_price = ltp * (1 + SLIPPAGE_PCT)
        one_lot_value = entry_price * LOT_SIZE
        if one_lot_value > CAPITAL_PER_TRADE:
            logger.error(
                f'"Cannot enter trade: one lot value ₹{one_lot_value:.2f} exceeds '
                f'capital cap ₹{CAPITAL_PER_TRADE:.2f}"'
            )
            return None

        quantity = int(CAPITAL_PER_TRADE / entry_price / LOT_SIZE) * LOT_SIZE
        if quantity <= 0:
            logger.error(
                f'"Cannot enter trade: computed quantity=0 for {symbol} '
                f'entry_price={entry_price:.2f} CAPITAL_PER_TRADE={CAPITAL_PER_TRADE} LOT_SIZE={LOT_SIZE}"'
            )
            return None

        self._trade_counter += 1
        trade_id = f"T{datetime.now().strftime('%Y%m%d')}-{self._trade_counter:03d}"

        trade = Trade(
            trade_id=trade_id,
            entry_time=datetime.now().isoformat(),
            direction=direction,
            symbol=symbol,
            strike=strike,
            entry_price=round(entry_price, 2),
            quantity=quantity,
            signal_quality_score=round(signal_quality_score, 4),
            entry_pb=round(percent_b, 4),
            regime=regime,
            entry_spot=round(entry_spot, 2),
            target_spot=round(target_spot, 2),
            sl_spot=round(sl_spot, 2),
        )

        if DRY_RUN:
            logger.info(
                f'"[DRY RUN] BUY {quantity} {symbol} @ {entry_price:.2f} | '
                f'trade_id={trade_id} pb={percent_b:.3f} score={signal_quality_score:.2f}"'
            )
        else:
            order_id = self._place_kite_order(symbol, quantity, "BUY")
            if not order_id:
                logger.error(f'"Entry rejected: broker BUY order failed for {symbol} qty={quantity}"')
                return None
            trade.kite_entry_order_id = order_id or ""

        self._active_trade = trade
        logger.info(
            f'"Trade entered: {trade_id} {direction} {symbol} qty={quantity} '
            f'@ ₹{entry_price:.2f} | regime={regime} score={signal_quality_score:.2f}"'
        )
        return trade

    # ── Exit ──────────────────────────────────────────────────────────────────

    def exit_trade(
        self,
        ltp: float,
        percent_b: float,
        reason: str,
    ) -> Optional[Trade]:
        """
        Close the active trade position.

        Args:
            ltp:        Current option last traded price.
            percent_b:  %b value at exit.
            reason:     Exit reason string (TARGET / STOP_LOSS / FORCE_EXIT).

        Returns:
            Completed Trade object or None if no open trade.
        """
        if self._active_trade is None:
            logger.warning('"exit_trade called with no active trade"')
            return None

        trade = self._active_trade
        exit_price = ltp * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - trade.entry_price) * trade.quantity - 2 * BROKERAGE_PER_ORDER

        if DRY_RUN:
            logger.info(
                f'"[DRY RUN] SELL {trade.quantity} {trade.symbol} @ {exit_price:.2f} | '
                f'P&L=₹{pnl:.2f} reason={reason}"'
            )
        else:
            order_id = self._place_kite_order(trade.symbol, trade.quantity, "SELL")
            if not order_id:
                logger.error(
                    f'"Exit rejected: broker SELL order failed for {trade.symbol} '
                    f'qty={trade.quantity}; keeping trade open"'
                )
                return None
            trade.kite_exit_order_id = order_id or ""

        trade.exit_time = datetime.now().isoformat()
        trade.exit_price = round(exit_price, 2)
        trade.exit_pb = round(percent_b, 4)
        trade.exit_reason = reason
        trade.pnl = round(pnl, 2)
        trade.is_open = False

        self._active_trade = None
        self._trade_history.append(trade)
        self._append_trade_to_csv(trade)

        logger.info(
            f'"Trade closed: {trade.trade_id} {trade.symbol} @ ₹{exit_price:.2f} | '
            f'P&L=₹{pnl:.2f} reason={reason}"'
        )
        return trade

    # ── Kite order placement ──────────────────────────────────────────────────

    def _place_kite_order(self, symbol: str, quantity: int, transaction_type: str) -> Optional[str]:
        """Place a market order via Kite Connect. Returns order_id or None on failure."""
        from auth import get_kite
        from kiteconnect import KiteConnect

        kite = get_kite()
        try:
            order_id = kite.place_order(
                tradingsymbol=symbol,
                exchange=NFO_EXCHANGE,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=KiteConnect.ORDER_TYPE_MARKET,
                product=KiteConnect.PRODUCT_MIS,
                variety=KiteConnect.VARIETY_REGULAR,
            )
            logger.info(f'"Kite order placed: {transaction_type} {symbol} qty={quantity} order_id={order_id}"')
            return str(order_id)
        except Exception as exc:
            logger.error(f'"Kite order failed: {transaction_type} {symbol} qty={quantity}: {exc}"')
            return None

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def active_trade(self) -> Optional[Trade]:
        return self._active_trade

    @property
    def has_open_position(self) -> bool:
        return self._active_trade is not None

    def today_trades(self) -> List[Trade]:
        today = datetime.now().date().isoformat()
        closed = [t for t in self._trade_history if t.entry_time.startswith(today)]
        if self._active_trade and self._active_trade.entry_time.startswith(today):
            return closed + [self._active_trade]
        return closed

    def today_pnl(self) -> float:
        return sum(t.pnl for t in self.today_trades())

    def all_trades(self) -> List[Trade]:
        return list(self._trade_history)

    def load_today_from_csv(self) -> None:
        """Load today's trades from CSV into memory (used on restart)."""
        if not TRADE_LOG_PATH.exists():
            return
        today = datetime.now().date().isoformat()
        try:
            with open(TRADE_LOG_PATH, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("entry_time", "").startswith(today):
                        trade = Trade(
                            trade_id=row.get("trade_id", ""),
                            entry_time=row.get("entry_time", ""),
                            exit_time=row.get("exit_time", ""),
                            direction=row.get("direction", ""),
                            symbol=row.get("symbol", ""),
                            strike=int(row.get("strike", 0)),
                            entry_price=float(row.get("entry_price", 0)),
                            exit_price=float(row.get("exit_price", 0)),
                            quantity=int(row.get("quantity", 0)),
                            pnl=float(row.get("pnl", 0)),
                            exit_reason=row.get("exit_reason", ""),
                            signal_quality_score=float(row.get("signal_quality_score", 0)),
                            entry_pb=float(row.get("entry_pb", 0)),
                            exit_pb=float(row.get("exit_pb", 0)),
                            regime=int(row.get("regime", -1)),
                            entry_spot=float(row.get("entry_spot", 0) or 0),
                            target_spot=float(row.get("target_spot", 0) or 0),
                            sl_spot=float(row.get("sl_spot", 0) or 0),
                            is_open=False,
                        )
                        self._trade_history.append(trade)
            logger.info(f'"Loaded {len(self.today_trades())} today\'s trades from CSV"')
        except Exception as exc:
            logger.error(f'"Failed to load trades from CSV: {exc}"')
