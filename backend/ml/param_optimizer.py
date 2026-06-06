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

PARAM_SPACE = {
    "bb_oversold":   (0.01, 0.15),
    "bb_overbought": (0.85, 0.99),
    "bb_exit":       (0.40, 0.90),   # wider: let target ride more of the reversion
    "sl_buffer":     (0.05, 0.20),
    "rsi_min":       (25, 45),
    "rsi_max":       (55, 75),
    "min_atr_pct":   (40.0, 95.0),   # volatility floor: trade only big-enough moves
}

MIN_TRADES_IN_SAMPLE = 30


def _objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    in_sample_days: int,
    out_sample_days: int,
    step_days: int,
) -> float:
    """
    Optuna objective: returns negative average OOS Sharpe (to minimise).
    Returns 0.0 (neutral) when constraints are violated (< MIN_TRADES_IN_SAMPLE).
    """
    params = {
        "bb_oversold":   trial.suggest_float("bb_oversold",   *PARAM_SPACE["bb_oversold"]),
        "bb_overbought": trial.suggest_float("bb_overbought", *PARAM_SPACE["bb_overbought"]),
        "bb_exit":       trial.suggest_float("bb_exit",       *PARAM_SPACE["bb_exit"]),
        "sl_buffer":     trial.suggest_float("sl_buffer",     *PARAM_SPACE["sl_buffer"]),
        "rsi_min":       trial.suggest_int("rsi_min",         *PARAM_SPACE["rsi_min"]),
        "rsi_max":       trial.suggest_int("rsi_max",         *PARAM_SPACE["rsi_max"]),
        "min_atr_pct":   trial.suggest_float("min_atr_pct",   *PARAM_SPACE["min_atr_pct"]),
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

        # Filter windows where in-sample had enough trades
        valid_windows = [w for w in window_results if w["out_trades"] >= MIN_TRADES_IN_SAMPLE // 3]
        if not valid_windows:
            return 0.0

        avg_oos_sharpe = float(np.mean([w["out_sharpe"] for w in valid_windows]))
        return -avg_oos_sharpe   # Optuna minimises

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
) -> Dict:
    """
    Run Bayesian optimization to find best strategy parameters.

    Args:
        df:               Full OHLCV + indicator DataFrame.
        n_trials:         Number of Optuna trials.
        in_sample_days:   In-sample window length (days).
        out_sample_days:  Out-of-sample window length (days).
        step_days:        Window step size (days).
        seed:             Random seed for reproducibility.

    Returns:
        best_params: Dict of optimized parameter values.
    """
    logger.info(f'"Starting Bayesian optimization: {n_trials} trials"')

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="nifty_bb_optimization",
    )

    study.optimize(
        lambda trial: _objective(trial, df, in_sample_days, out_sample_days, step_days),
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1,
    )

    best_params = study.best_params
    best_value = study.best_value
    logger.info(f'"Optimization complete. Best OOS Sharpe={-best_value:.2f}, params={best_params}"')

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
            "best_oos_sharpe": -best_value,
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
