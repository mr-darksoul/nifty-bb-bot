"""
XGBoost binary classifier that scores each %b signal as HIGH_QUALITY (1)
or LOW_QUALITY (0). Only signals with predicted_proba >= SIGNAL_QUALITY_THRESHOLD
are forwarded to order execution.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from config import SIGNAL_FILTER_MODEL_PATH, SIGNAL_QUALITY_THRESHOLD
from ml.feature_engineering import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


class SignalFilter:
    """Wraps XGBClassifier for signal quality scoring."""

    def __init__(self) -> None:
        self.model: Optional[object] = None
        self._loaded = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[list] = None
    ) -> "SignalFilter":
        """
        Train the signal quality classifier.

        Args:
            X: Feature DataFrame (FEATURE_COLUMNS).
            y: Binary labels (0 = low quality, 1 = high quality).
            eval_set: Optional [(X_val, y_val)] for early stopping logging.
        """
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost is not installed — cannot train SignalFilter")

        X_clean = X[FEATURE_COLUMNS].copy()
        y_clean = y.copy()

        # Drop rows where y is NaN
        valid = ~y_clean.isna()
        X_clean = X_clean[valid]
        y_clean = y_clean[valid].astype(int)

        if len(X_clean) < 50:
            raise ValueError(f"Need at least 50 labelled signals, got {len(X_clean)}")

        self.model = XGBClassifier(**XGB_PARAMS)

        fit_kwargs = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False

        self.model.fit(X_clean, y_clean, **fit_kwargs)
        self._loaded = True

        pos_rate = y_clean.mean()
        logger.info(
            f'"SignalFilter trained: {len(X_clean)} samples, positive rate={pos_rate:.2%}"'
        )
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = SIGNAL_FILTER_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f'"SignalFilter saved to {path}"')

    def load(self, path: Path = SIGNAL_FILTER_MODEL_PATH) -> bool:
        if not path.exists():
            logger.warning(f'"SignalFilter model not found at {path} — will reject live signals"')
            return False
        if not _XGB_AVAILABLE:
            logger.warning('"xgboost not installed — SignalFilter disabled"')
            return False
        try:
            self.model = joblib.load(path)
            self._loaded = True
            logger.info(f'"SignalFilter loaded from {path}"')
            return True
        except Exception as exc:
            logger.error(f'"Failed to load SignalFilter: {exc}"')
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, features: pd.Series) -> float:
        """
        Score a single signal bar's feature vector.

        Args:
            features: pd.Series with index == FEATURE_COLUMNS.

        Returns:
            Predicted probability of HIGH_QUALITY (0.0–1.0).
            Returns 0.0 (reject) if model unavailable.
        """
        if not self._loaded or self.model is None:
            logger.warning('"SignalFilter not loaded — returning default score 0.0"')
            return 0.0

        try:
            X = features[FEATURE_COLUMNS].values.reshape(1, -1).astype(np.float64)
            proba = float(self.model.predict_proba(X)[0][1])
            logger.info(
                f'"SignalFilter score: {proba:.4f} | features={features[FEATURE_COLUMNS].to_dict()}"'
            )
            return proba
        except Exception as exc:
            logger.error(f'"SignalFilter scoring failed: {exc} — returning 0.0"')
            return 0.0

    def is_high_quality(self, features: pd.Series, threshold: float = SIGNAL_QUALITY_THRESHOLD) -> bool:
        """True if signal score meets the quality threshold."""
        score = self.score(features)
        accepted = score >= threshold
        logger.info(f'"SignalFilter decision: score={score:.4f}, threshold={threshold}, accepted={accepted}"')
        return accepted
