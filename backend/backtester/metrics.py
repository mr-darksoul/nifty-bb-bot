"""
Performance metrics computed from a trade-level DataFrame and a daily P&L Series.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.065   # 6.5% annual


def compute_metrics(trades: pd.DataFrame, daily_pnl: pd.Series) -> Dict[str, Any]:
    """
    Compute a comprehensive set of performance metrics.

    Args:
        trades:     DataFrame with columns [entry_time, exit_time, pnl, duration_min].
        daily_pnl:  Series indexed by date, values = daily P&L in ₹.

    Returns:
        Dictionary of metric name → value.
    """
    metrics: Dict[str, Any] = {}

    if trades.empty or daily_pnl.empty:
        return _empty_metrics()

    # ── Basic counts ──────────────────────────────────────────────────────────
    n_trades = len(trades)
    winners = trades[trades["pnl"] > 0]
    losers = trades[trades["pnl"] <= 0]

    metrics["total_trades"] = n_trades
    metrics["win_rate"] = len(winners) / n_trades if n_trades > 0 else 0.0

    # ── P&L ───────────────────────────────────────────────────────────────────
    metrics["total_pnl"] = float(daily_pnl.sum())
    metrics["avg_winner"] = float(winners["pnl"].mean()) if not winners.empty else 0.0
    metrics["avg_loser"] = float(losers["pnl"].mean()) if not losers.empty else 0.0

    gross_profit = winners["pnl"].sum() if not winners.empty else 0.0
    gross_loss = abs(losers["pnl"].sum()) if not losers.empty else 0.0
    metrics["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # ── Annualised return (CAGR) ──────────────────────────────────────────────
    n_days = (daily_pnl.index[-1] - daily_pnl.index[0]).days
    if n_days > 0:
        initial_capital = 100_000.0   # assumed for CAGR normalisation
        final_value = initial_capital + metrics["total_pnl"]
        years = n_days / 365.25
        if years <= 0:
            metrics["cagr"] = 0.0
        elif final_value <= 0:
            # Lost >=100% of capital: CAGR is undefined (negative base to a
            # fractional power → complex). Floor it at -100%.
            metrics["cagr"] = -1.0
        else:
            metrics["cagr"] = (final_value / initial_capital) ** (1 / years) - 1
    else:
        metrics["cagr"] = 0.0

    # ── Sharpe Ratio ──────────────────────────────────────────────────────────
    daily_rf = RISK_FREE_RATE / 252
    excess = daily_pnl - daily_rf * 100_000.0 / 252   # scale rf to ₹ daily
    if daily_pnl.std() > 0:
        metrics["sharpe"] = float(excess.mean() / daily_pnl.std() * np.sqrt(252))
    else:
        metrics["sharpe"] = 0.0

    # ── Sortino Ratio ─────────────────────────────────────────────────────────
    downside = daily_pnl[daily_pnl < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    if downside_std > 0:
        metrics["sortino"] = float(daily_pnl.mean() / downside_std * np.sqrt(252))
    else:
        metrics["sortino"] = 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cum_pnl = daily_pnl.cumsum()
    rolling_max = cum_pnl.cummax()
    drawdown = cum_pnl - rolling_max
    metrics["max_drawdown_inr"] = float(drawdown.min())
    metrics["max_drawdown_pct"] = (
        float(drawdown.min() / (rolling_max.max() + 100_000.0)) * 100
        if (rolling_max.max() + 100_000.0) > 0
        else 0.0
    )

    # ── Trade duration ────────────────────────────────────────────────────────
    if "duration_min" in trades.columns:
        metrics["avg_duration_min"] = float(trades["duration_min"].mean())
    else:
        metrics["avg_duration_min"] = 0.0

    # ── Trades per week ───────────────────────────────────────────────────────
    if n_days > 0:
        metrics["trades_per_week"] = n_trades / (n_days / 7)
    else:
        metrics["trades_per_week"] = 0.0

    metrics["avg_winner_loser_ratio"] = (
        abs(metrics["avg_winner"] / metrics["avg_loser"])
        if metrics["avg_loser"] != 0
        else np.inf
    )

    return metrics


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_winner": 0.0,
        "avg_loser": 0.0,
        "profit_factor": 0.0,
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown_inr": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_duration_min": 0.0,
        "trades_per_week": 0.0,
        "avg_winner_loser_ratio": 0.0,
    }


def format_report(metrics: Dict[str, Any]) -> str:
    """Human-readable performance report string."""
    lines = [
        "─" * 50,
        "  PERFORMANCE REPORT",
        "─" * 50,
        f"  Total Trades       : {metrics.get('total_trades', 0)}",
        f"  Win Rate           : {metrics.get('win_rate', 0):.1%}",
        f"  Total P&L          : ₹{metrics.get('total_pnl', 0):,.2f}",
        f"  CAGR               : {metrics.get('cagr', 0):.1%}",
        f"  Sharpe Ratio       : {metrics.get('sharpe', 0):.2f}",
        f"  Sortino Ratio      : {metrics.get('sortino', 0):.2f}",
        f"  Profit Factor      : {metrics.get('profit_factor', 0):.2f}",
        f"  Max Drawdown       : ₹{metrics.get('max_drawdown_inr', 0):,.2f} "
        f"({metrics.get('max_drawdown_pct', 0):.1f}%)",
        f"  Avg Winner         : ₹{metrics.get('avg_winner', 0):,.2f}",
        f"  Avg Loser          : ₹{metrics.get('avg_loser', 0):,.2f}",
        f"  W/L Ratio          : {metrics.get('avg_winner_loser_ratio', 0):.2f}",
        f"  Avg Duration       : {metrics.get('avg_duration_min', 0):.0f} min",
        f"  Trades / Week      : {metrics.get('trades_per_week', 0):.1f}",
        "─" * 50,
    ]
    return "\n".join(lines)
