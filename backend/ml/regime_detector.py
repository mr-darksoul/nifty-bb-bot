"""
HMM-based market regime classifier.

Regime IDs (post-training label assignment):
  0 = TRENDING_DOWN
  1 = CHOPPY          ← only regime where trades are allowed
  2 = TRENDING_UP
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import CHOPPY_REGIME_ID, REGIME_MODEL_PATH

logger = logging.getLogger(__name__)

REGIME_NAMES = {0: "TRENDING_DOWN", 1: "CHOPPY", 2: "TRENDING_UP"}
N_COMPONENTS = 3
LOOKBACK_FOR_PREDICT = 60   # bars fed to HMM at inference time


class RegimeDetector:
    """Wraps a GaussianHMM to classify current market regime."""

    def __init__(self) -> None:
        self.model: Optional[GaussianHMM] = None
        self._regime_map: dict = {}      # HMM state → canonical regime id
        self._loaded = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, features: pd.DataFrame) -> "RegimeDetector":
        """
        Train the HMM on a feature matrix produced by build_regime_features().

        Args:
            features: DataFrame with columns [log_return, rolling_vol, bb_width, atr_norm].
                      Must be clean (no NaN rows).
        """
        X = features.dropna().values.astype(np.float64)
        if len(X) < 200:
            raise ValueError(f"Need at least 200 clean bars to train HMM, got {len(X)}")

        model = GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            verbose=False,
        )
        model.fit(X)
        self.model = model
        self._regime_map = self._assign_regime_labels(model, features.dropna())
        self._loaded = True
        logger.info(f'"HMM trained on {len(X)} bars. Regime map: {self._regime_map}"')
        return self

    def _assign_regime_labels(self, model: GaussianHMM, features: pd.DataFrame) -> dict:
        """
        Map HMM states to canonical regime IDs by examining the mean log-return
        of each state. Most negative → TRENDING_DOWN(0), most positive → TRENDING_UP(2),
        middle → CHOPPY(1).
        """
        X = features.values.astype(np.float64)
        states = model.predict(X)
        log_ret_col = features.columns.get_loc("log_return")

        state_means = {}
        for s in range(N_COMPONENTS):
            mask = states == s
            if mask.sum() == 0:
                state_means[s] = 0.0
            else:
                state_means[s] = features.iloc[mask, log_ret_col].mean()

        sorted_states = sorted(state_means, key=lambda s: state_means[s])
        regime_map = {
            sorted_states[0]: 0,   # most negative mean → TRENDING_DOWN
            sorted_states[1]: 1,   # middle → CHOPPY
            sorted_states[2]: 2,   # most positive mean → TRENDING_UP
        }
        return regime_map

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = REGIME_MODEL_PATH) -> None:
        """Persist model + regime map to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "regime_map": self._regime_map}, path)
        logger.info(f'"Regime model saved to {path}"')

    def load(self, path: Path = REGIME_MODEL_PATH) -> bool:
        """Load model from disk. Returns True on success, False if file missing."""
        if not path.exists():
            logger.warning(f'"Regime model not found at {path} — will accept all signals"')
            return False
        try:
            bundle = joblib.load(path)
            self.model = bundle["model"]
            self._regime_map = bundle["regime_map"]
            self._loaded = True
            logger.info(f'"Regime model loaded from {path}"')
            return True
        except Exception as exc:
            logger.error(f'"Failed to load regime model: {exc}"')
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_regime(self, feature_window: pd.DataFrame) -> int:
        """
        Predict regime for the most recent bar given a rolling feature window.

        Args:
            feature_window: DataFrame of REGIME_FEATURE_COLUMNS, last LOOKBACK_FOR_PREDICT rows.

        Returns:
            Canonical regime id (0, 1, 2). Returns CHOPPY(1) as default if model unavailable.
        """
        if not self._loaded or self.model is None:
            # Default to TRENDING_DOWN (blocks entry) rather than CHOPPY (allows entry),
            # so a missing model is a safe failure — no unfiltered trades.
            logger.warning('"Regime model not loaded — defaulting to TRENDING_DOWN (entries blocked)"')
            return 0

        X = feature_window.dropna().values.astype(np.float64)
        if len(X) < 10:
            logger.warning(f'"Too few clean bars ({len(X)}) for regime prediction — defaulting CHOPPY"')
            return CHOPPY_REGIME_ID

        try:
            raw_state = self.model.predict(X)[-1]
            regime = self._regime_map.get(int(raw_state), CHOPPY_REGIME_ID)
            logger.info(
                f'"Regime prediction: raw_state={raw_state} → regime={regime} ({REGIME_NAMES[regime]})"'
            )
            return regime
        except Exception as exc:
            logger.error(f'"Regime prediction failed: {exc} — defaulting TRENDING_DOWN (entries blocked)"')
            return 0

    def is_choppy(self, feature_window: pd.DataFrame) -> bool:
        """Convenience: True when regime is CHOPPY (trades allowed)."""
        return self.predict_regime(feature_window) == CHOPPY_REGIME_ID

    def regime_name(self, regime_id: int) -> str:
        return REGIME_NAMES.get(regime_id, "UNKNOWN")
