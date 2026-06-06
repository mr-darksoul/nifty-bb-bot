"""
Offline training pipeline: trains all ML models end-to-end and saves artifacts.

Usage:
    python ml/train.py --months 9 --trials 200

Steps:
    1. Load or fetch NIFTY 5-min OHLCV data
    2. Compute all indicators + features
    3. Train HMM regime detector
    4. Label signals for signal filter training
    5. Train XGBoost signal filter
    6. Run Bayesian walk-forward parameter optimization
    7. Print comprehensive performance report
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure backend/ is on path when run as python ml/train.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    DATA_CACHE_PATH,
    DEFAULT_PARAMS,
    KITE_API_KEY,
    KITE_ACCESS_TOKEN,
    MODELS_DIR,
    load_optimized_params,
)
from indicators import compute_all
from ml.feature_engineering import (
    FEATURE_COLUMNS,
    build_features,
    build_regime_features,
    label_signals,
)
from ml.regime_detector import RegimeDetector
from ml.signal_filter import SignalFilter
from ml.param_optimizer import optimize, save_params
from backtester.engine import run_backtest
from backtester.metrics import format_report, compute_metrics
from backtester.walk_forward import walk_forward, summarise_walk_forward

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _fetch_from_kite(months: int) -> pd.DataFrame:
    """Download historical NIFTY 5-min data from Kite Connect."""
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise RuntimeError("kiteconnect not installed — cannot fetch data")

    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        raise RuntimeError("KITE_API_KEY and KITE_ACCESS_TOKEN required for data fetch")

    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)

    from_date = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Fetching NIFTY 5-min data from {from_date} to {to_date}")

    records = kite.historical_data(
        instrument_token=256265,
        from_date=from_date,
        to_date=to_date,
        interval="5minute",
    )

    df = pd.DataFrame(records)
    df.rename(
        columns={"date": "datetime", "open": "open", "high": "high",
                 "low": "low", "close": "close", "volume": "volume"},
        inplace=True,
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def _load_or_fetch(months: int) -> pd.DataFrame:
    """Load data from CSV cache; fetch from Kite if absent or too short."""
    if DATA_CACHE_PATH.exists():
        df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        min_date = datetime.now() - timedelta(days=months * 31)
        if df.index[0] <= pd.Timestamp(min_date):
            logger.info(f"Loaded {len(df)} bars from cache: {DATA_CACHE_PATH}")
            return df
        logger.info("Cache too short — fetching fresh data")

    df = _fetch_from_kite(months)
    df.to_csv(DATA_CACHE_PATH)
    logger.info(f"Saved {len(df)} bars to {DATA_CACHE_PATH}")
    return df


def _generate_demo_data(months: int) -> pd.DataFrame:
    """Generate synthetic NIFTY-like data for offline testing when Kite unavailable."""
    logger.warning("Generating synthetic NIFTY data (Kite credentials not available)")
    n_bars = months * 21 * 75   # ~75 5-min bars per trading day
    dates = pd.bdate_range(
        end=datetime.now(),
        periods=months * 21,
        freq="B",
    )
    all_bars = []
    price = 22000.0
    for d in dates:
        for bar_num in range(75):
            minutes_offset = 9 * 60 + 15 + bar_num * 5
            ts = pd.Timestamp(d) + pd.Timedelta(minutes=minutes_offset)
            ret = np.random.normal(0, 0.0008)
            o = price
            h = o * (1 + abs(np.random.normal(0, 0.0005)))
            l = o * (1 - abs(np.random.normal(0, 0.0005)))
            c = o * (1 + ret)
            price = c
            all_bars.append({"open": o, "high": h, "low": l, "close": c, "volume": 0})
    df = pd.DataFrame(all_bars)
    df.index = pd.DatetimeIndex([
        d + pd.Timedelta(minutes=9 * 60 + 15 + i * 5)
        for i, d in enumerate(dates.repeat(75))
    ])
    return df


# ── Training steps ────────────────────────────────────────────────────────────

def step_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 2: Building feature matrix")
    df_feat = build_features(df)
    logger.info(f"Feature matrix shape: {df_feat.shape}")
    return df_feat


def step_train_regime(df: pd.DataFrame) -> RegimeDetector:
    logger.info("Step 3: Training HMM regime detector")
    regime_features = build_regime_features(df).dropna()
    detector = RegimeDetector()
    detector.train(regime_features)
    detector.save()
    logger.info("Regime model saved")
    return detector


def step_label_signals(df_feat: pd.DataFrame, params: dict) -> pd.Series:
    logger.info("Step 4: Labelling signals for signal filter training")
    pb = df_feat["percent_b"]
    entry_mask = (pb < params["bb_oversold"]) | (pb > params["bb_overbought"])
    direction = pd.Series(0, index=df_feat.index)
    direction[pb < params["bb_oversold"]] = 1
    direction[pb > params["bb_overbought"]] = -1
    labels = label_signals(
        df_feat, entry_mask, direction,
        params["bb_exit"], params["sl_buffer"]
    )
    n_labelled = labels.notna().sum()
    logger.info(f"Labelled {n_labelled} signals (positive rate: {labels.dropna().mean():.1%})")
    return labels


def step_train_signal_filter(df_feat: pd.DataFrame, labels: pd.Series) -> SignalFilter:
    logger.info("Step 5: Training XGBoost signal filter")
    labelled = labels.notna()
    X = df_feat.loc[labelled, FEATURE_COLUMNS].fillna(0.0)
    y = labels[labelled]

    # Walk-forward split: train on first 80%, validate on last 20%
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    sf = SignalFilter()
    sf.train(X_train, y_train, eval_set=[(X_val, y_val.astype(int))])
    sf.save()
    logger.info("Signal filter saved")
    return sf


def step_optimize(df_feat: pd.DataFrame, n_trials: int) -> dict:
    logger.info(f"Step 6: Bayesian walk-forward optimization ({n_trials} trials)")
    best = optimize(df_feat, n_trials=n_trials)
    save_params(best)
    logger.info(f"Optimized params: {best}")
    return best


def step_report(df_feat: pd.DataFrame, best_params: dict) -> None:
    logger.info("Step 7: Generating performance report")
    window_results, combined_equity = walk_forward(df_feat, params=best_params)
    summary = summarise_walk_forward(window_results)

    if not combined_equity.empty:
        trades_df, daily_pnl, metrics = run_backtest(df_feat, params=best_params)
        print(format_report(metrics))
        print(f"\nWalk-Forward Summary:")
        print(f"  Windows evaluated    : {summary.get('n_windows', 0)}")
        print(f"  Avg OOS Sharpe       : {summary.get('avg_oos_sharpe', 0):.2f}")
        print(f"  Median OOS Sharpe    : {summary.get('median_oos_sharpe', 0):.2f}")
        print(f"  Avg OOS Win Rate     : {summary.get('avg_oos_win_rate', 0):.1%}")
        print(f"  Total OOS P&L        : ₹{summary.get('total_oos_pnl', 0):,.2f}")
        print(f"  % Profitable Windows : {summary.get('pct_profitable_windows', 0):.1%}")

        # Regime distribution
        from ml.regime_detector import RegimeDetector, REGIME_NAMES
        detector = RegimeDetector()
        if detector.load():
            from ml.feature_engineering import build_regime_features
            reg_feat = build_regime_features(df_feat).dropna()
            import numpy as np
            X = reg_feat.values.astype(np.float64)
            raw_states = detector.model.predict(X)
            mapped = [detector._regime_map.get(int(s), 1) for s in raw_states]
            unique, counts = np.unique(mapped, return_counts=True)
            print(f"\nRegime Distribution (last {len(mapped)} bars):")
            for r, c in zip(unique, counts):
                print(f"  {REGIME_NAMES.get(r, r)}: {c/len(mapped):.1%}")
    else:
        logger.warning("No out-of-sample equity curve generated")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all NIFTY BB bot ML models")
    parser.add_argument("--months", type=int, default=9, help="Months of historical data")
    parser.add_argument("--trials", type=int, default=200, help="Optuna trials for optimizer")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no Kite needed)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    logger.info(f"Step 1: Loading {args.months} months of NIFTY 5-min data")
    if args.demo:
        df = _generate_demo_data(args.months)
    else:
        try:
            df = _load_or_fetch(args.months)
        except Exception as exc:
            logger.warning(f"Cannot fetch from Kite ({exc}) — using synthetic data")
            df = _generate_demo_data(args.months)

    logger.info(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    df_feat = step_features(df)
    step_train_regime(df)

    params = load_optimized_params()  # use current best as label reference
    labels = step_label_signals(df_feat, params)
    step_train_signal_filter(df_feat, labels)

    best_params = step_optimize(df_feat, n_trials=args.trials)
    step_report(df_feat, best_params)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
