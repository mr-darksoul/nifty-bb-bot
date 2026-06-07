"""
Bayesian walk-forward parameter optimizer using Optuna.

Searches for optimal Bollinger %b strategy parameters by maximising
the average out-of-sample Sharpe Ratio across rolling windows.
Saves the best parameters to ml/models/optimized_params.json.
"""

import json
import logging
from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd

from backtester.engine import run_backtest
from backtester.walk_forward import walk_forward, summarise_walk_forward
from config import OPTIMIZED_PARAMS_PATH

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# Per-strategy Optuna search spaces. bb_exit / sl_buffer are ATR multiples for
# the price-anchored target / stop in both strategies.
PARAM_SPACES = {
    "mean_reversion": {
        "bb_oversold":   (0.01, 0.15),
        "bb_overbought": (0.85, 0.99),
        "bb_exit":       (0.5, 4.0),
        "sl_buffer":     (0.2, 2.0),
        "rsi_min":       (25, 45),
        "rsi_max":       (55, 75),
        "min_atr_pct":   (40.0, 95.0),
    },
    "momentum_breakout": {
        # Deliberately LOW-dimensional: with only ~140 breakout events over 9
        # months, optimising 7 knobs overfits noise. The cross is pinned AT/BEYOND
        # the band — a real breakout means price closes OUTSIDE the band (%b>1.0
        # up / <0.0 down); crossing just *inside* (e.g. 0.999) is noise and
        # destroys the edge. RSI/ATR junk filters are disabled; only target/stop
        # are tuned, inside the validated plateau.
        "bb_oversold":   (-0.05, 0.0),  # downside break: close at/below lower band
        "bb_overbought": (1.0, 1.05),   # upside break:  close at/above upper band
        "bb_exit":       (2.0, 4.0),    # let momentum winners run
        "sl_buffer":     (1.0, 2.0),
        "rsi_min":       (0, 1),        # effectively off
        "rsi_max":       (99, 100),     # effectively off
        "min_atr_pct":   (0.0, 0.0),    # ATR gate hurts breakout — keep off
    },
}
# Back-compat alias for callers importing PARAM_SPACE.
PARAM_SPACE = PARAM_SPACES["mean_reversion"]

MIN_TRADES_IN_SAMPLE = 30


def _objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    in_sample_days: int,
    out_sample_days: int,
    step_days: int,
    strategy: str = "momentum_breakout",
    timeframe_min: int = 15,
) -> float:
    """
    Optuna objective: returns negative average OOS Sharpe (to minimise).
    Returns 0.0 (neutral) when constraints are violated (< MIN_TRADES_IN_SAMPLE).
    """
    space = PARAM_SPACES[strategy]
    params = {
        "strategy":      strategy,
        "timeframe_min": timeframe_min,
        "bb_oversold":   trial.suggest_float("bb_oversold",   *space["bb_oversold"]),
        "bb_overbought": trial.suggest_float("bb_overbought", *space["bb_overbought"]),
        "bb_exit":       trial.suggest_float("bb_exit",       *space["bb_exit"]),
        "sl_buffer":     trial.suggest_float("sl_buffer",     *space["sl_buffer"]),
        "rsi_min":       trial.suggest_int("rsi_min",         *space["rsi_min"]),
        "rsi_max":       trial.suggest_int("rsi_max",         *space["rsi_max"]),
        "min_atr_pct":   trial.suggest_float("min_atr_pct",   *space["min_atr_pct"]),
    }

    # Enforce logical constraints
    if params["bb_oversold"] >= params["bb_overbought"]:
        return 0.0
    if params["rsi_min"] >= params["rsi_max"]:
        return 0.0

    try:
        window_results, _ = walk_forward(
            df,
            params=params,
            in_sample_days=in_sample_days,
            out_sample_days=out_sample_days,
            step_days=step_days,
        )

        if not window_results:
            return 0.0

        # Robust objective: maximise TOTAL out-of-sample P&L, scaled by the
        # fraction of profitable windows so a single lucky window can't win.
        # Averaging per-window Sharpe over a handful of tiny-sample windows is
        # too noisy and overfits (it rewarded -₹35k configs); total profit with a
        # consistency multiplier and a trade-count floor is far more stable.
        total_trades = sum(w["out_trades"] for w in window_results)
        if total_trades < MIN_TRADES_IN_SAMPLE:
            return 0.0

        oos_pnls = [w["out_total_pnl"] for w in window_results]
        total_pnl = float(np.sum(oos_pnls))
        pct_profitable = float(np.mean([p > 0 for p in oos_pnls]))
        # Reward consistency: profitable total scaled up by hit-rate; losing
        # total scaled up by miss-rate (so it is penalised harder).
        score = total_pnl * (pct_profitable if total_pnl > 0 else (1.0 - pct_profitable) + 1.0)
        return -score   # Optuna minimises

    except Exception as exc:
        logger.debug(f'"Optuna trial failed: {exc}"')
        return 0.0


def optimize(
    df: pd.DataFrame,
    n_trials: int = 200,
    in_sample_days: int = 90,
    out_sample_days: int = 30,
    step_days: int = 30,
    seed: int = 42,
    strategy: str = "momentum_breakout",
    timeframe_min: int = 15,
) -> Dict:
    """
    Run Bayesian optimization to find best strategy parameters.

    Args:
        df:               OHLCV + indicator DataFrame ALREADY on the strategy
                          timeframe (caller resamples + compute_all first).
        n_trials:         Number of Optuna trials.
        in_sample_days:   In-sample window length (days).
        out_sample_days:  Out-of-sample window length (days).
        step_days:        Window step size (days).
        seed:             Random seed for reproducibility.
        strategy:         "momentum_breakout" or "mean_reversion".
        timeframe_min:    Bar timeframe the df is on (stored in params).

    Returns:
        best_params: Dict of optimized parameter values.
    """
    logger.info(f'"Starting Bayesian optimization: {n_trials} trials, strategy={strategy}"')

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"nifty_{strategy}_optimization",
    )

    study.optimize(
        lambda trial: _objective(trial, df, in_sample_days, out_sample_days,
                                 step_days, strategy, timeframe_min),
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1,
    )

    # Re-attach the fixed (non-suggested) keys the engine needs.
    best_params = {
        "strategy": strategy,
        "timeframe_min": timeframe_min,
        **study.best_params,
    }
    best_value = study.best_value
    logger.info(f'"Optimization complete. Best objective={-best_value:,.0f}, params={best_params}"')

    # Evaluate best params on full walk-forward for reporting
    window_results, _ = walk_forward(
        df,
        params=best_params,
        in_sample_days=in_sample_days,
        out_sample_days=out_sample_days,
        step_days=step_days,
    )
    summary = summarise_walk_forward(window_results)

    output = {
        **best_params,
        "_meta": {
            "n_trials": n_trials,
            "strategy": strategy,
            "timeframe_min": timeframe_min,
            "best_objective_score": -best_value,
            "walk_forward_summary": summary,
        },
    }

    return output


def save_params(params: Dict, path=OPTIMIZED_PARAMS_PATH) -> None:
    """Persist optimized parameters to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Strip _meta for the live config file (keep a full copy with meta)
    save_data = {k: v for k, v in params.items() if not k.startswith("_")}
    save_data["_meta"] = params.get("_meta", {})

    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info(f'"Optimized params saved to {path}"')
