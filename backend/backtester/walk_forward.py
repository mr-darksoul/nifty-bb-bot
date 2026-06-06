"""
Rolling walk-forward validation: split data into overlapping in-sample /
out-of-sample windows, run backtest on each, and aggregate results.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, format_report

logger = logging.getLogger(__name__)


def walk_forward(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    in_sample_days: int = 90,
    out_sample_days: int = 30,
    step_days: int = 30,
) -> Tuple[List[Dict], pd.Series]:
    """
    Run rolling walk-forward validation.

    Args:
        df:               Full OHLCV + indicator DataFrame.
        params:           Strategy parameters (dict). None = use defaults.
        in_sample_days:   Length of each training window in calendar days.
        out_sample_days:  Length of each test window in calendar days.
        step_days:        Stride between windows in calendar days.

    Returns:
        window_results:   List of per-window metric dicts, each including
                          window_start, window_end, in/out_sample keys.
        combined_equity:  Daily P&L Series across all out-of-sample periods.
    """
    idx = df.index

    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    start_date = idx[0].normalize()
    end_date = idx[-1].normalize()

    window_start = start_date
    window_results: List[Dict] = []
    all_oos_pnl: List[pd.Series] = []

    window_num = 0
    while True:
        in_sample_end = window_start + pd.Timedelta(days=in_sample_days)
        out_sample_end = in_sample_end + pd.Timedelta(days=out_sample_days)

        if out_sample_end > end_date + pd.Timedelta(days=1):
            break

        df_in = df[(idx >= window_start) & (idx < in_sample_end)]
        df_out = df[(idx >= in_sample_end) & (idx < out_sample_end)]

        if len(df_in) < 100 or len(df_out) < 20:
            window_start += pd.Timedelta(days=step_days)
            continue

        window_num += 1

        # ── In-sample backtest ────────────────────────────────────────────────
        try:
            trades_in, pnl_in, metrics_in = run_backtest(df_in, params)
        except Exception as exc:
            logger.warning(f'"Walk-forward window {window_num} in-sample failed: {exc}"')
            window_start += pd.Timedelta(days=step_days)
            continue

        # ── Out-of-sample backtest ────────────────────────────────────────────
        try:
            trades_out, pnl_out, metrics_out = run_backtest(df_out, params)
        except Exception as exc:
            logger.warning(f'"Walk-forward window {window_num} out-sample failed: {exc}"')
            window_start += pd.Timedelta(days=step_days)
            continue

        result = {
            "window": window_num,
            "in_sample_start": window_start.date().isoformat(),
            "in_sample_end": in_sample_end.date().isoformat(),
            "out_sample_start": in_sample_end.date().isoformat(),
            "out_sample_end": out_sample_end.date().isoformat(),
            "in_sharpe": metrics_in.get("sharpe", 0.0),
            "out_sharpe": metrics_out.get("sharpe", 0.0),
            "out_win_rate": metrics_out.get("win_rate", 0.0),
            "out_total_pnl": metrics_out.get("total_pnl", 0.0),
            "out_trades": metrics_out.get("total_trades", 0),
            "out_max_dd": metrics_out.get("max_drawdown_inr", 0.0),
        }
        window_results.append(result)

        if not pnl_out.empty:
            all_oos_pnl.append(pnl_out)

        logger.info(
            f'"WF window {window_num}: '
            f'IS Sharpe={metrics_in.get("sharpe",0):.2f}, '
            f'OOS Sharpe={metrics_out.get("sharpe",0):.2f}, '
            f'OOS trades={metrics_out.get("total_trades",0)}"'
        )

        window_start += pd.Timedelta(days=step_days)

    if all_oos_pnl:
        combined_equity = pd.concat(all_oos_pnl).sort_index()
        combined_equity = combined_equity.groupby(combined_equity.index).sum()
    else:
        combined_equity = pd.Series(dtype=float)

    logger.info(f'"Walk-forward complete: {window_num} windows"')
    return window_results, combined_equity


def summarise_walk_forward(window_results: List[Dict]) -> Dict:
    """Aggregate walk-forward window results into summary statistics."""
    if not window_results:
        return {}

    oos_sharpes = [w["out_sharpe"] for w in window_results]
    oos_win_rates = [w["out_win_rate"] for w in window_results]
    oos_pnls = [w["out_total_pnl"] for w in window_results]

    import numpy as np
    return {
        "n_windows": len(window_results),
        "avg_oos_sharpe": float(np.mean(oos_sharpes)),
        "median_oos_sharpe": float(np.median(oos_sharpes)),
        "avg_oos_win_rate": float(np.mean(oos_win_rates)),
        "total_oos_pnl": float(sum(oos_pnls)),
        "pct_profitable_windows": float(sum(1 for p in oos_pnls if p > 0) / len(oos_pnls)),
        "windows": window_results,
    }
