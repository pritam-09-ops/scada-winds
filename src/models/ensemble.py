"""Ensemble model that combines XGBoost and LSTM predictions.

The ensemble uses a configurable weighted average of both models.  If only
one model has been trained, it falls back to that single model's predictions.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score

from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted-average ensemble of XGBoost and LSTM regressors."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        xgb_weight: Optional[float] = None,
        lstm_weight: Optional[float] = None,
    ) -> None:
        """Initialize ensemble with model weights from *config.yaml* or
        explicit keyword arguments.

        Args:
            config_path: Path to the YAML configuration file.
            xgb_weight:  Weight for XGBoost predictions.  Overrides config.
            lstm_weight: Weight for LSTM predictions.  Overrides config.
        """
        self.config_path = config_path
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)

        ens_cfg = config.get("ensemble", {})
        self.xgb_weight: float = xgb_weight if xgb_weight is not None else ens_cfg.get("xgb_weight", 0.6)
        self.lstm_weight: float = lstm_weight if lstm_weight is not None else ens_cfg.get("lstm_weight", 0.4)

        self.xgb_model = XGBoostModel(config_path)
        self.lstm_model = LSTMModel(config_path)
        self._xgb_trained: bool = False
        self._lstm_trained: bool = False

    # ------------------------------------------------------------------
    # Train / predict / evaluate
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train both constituent models.

        Args:
            X_train: Training feature matrix.
            y_train: Training targets.
            X_val:   Validation feature matrix (passed to each model).
            y_val:   Validation targets.
        """
        logger.info("Training XGBoost model …")
        self.xgb_model.build()
        self.xgb_model.train(X_train, y_train, X_val, y_val)
        self._xgb_trained = True

        logger.info("Training LSTM model …")
        self.lstm_model.train(X_train, y_train, X_val, y_val)
        self._lstm_trained = True

        logger.info("Ensemble training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted ensemble predictions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Ensemble prediction array of length ``n_samples``.

        Raises:
            RuntimeError: If no constituent model has been trained.
        """
        predictions = []
        weights = []

        if self._xgb_trained:
            predictions.append(self.xgb_model.predict(X))
            weights.append(self.xgb_weight)

        if self._lstm_trained:
            lstm_preds = self.lstm_model.predict(X)
            predictions.append(lstm_preds[: len(X)])
            weights.append(self.lstm_weight)

        if not predictions:
            raise RuntimeError("No trained models available for prediction.")

        # Normalise weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]

        ensemble_preds = np.zeros(len(X))
        for pred, w in zip(predictions, weights):
            ensemble_preds += w * pred[: len(X)]

        return ensemble_preds

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble predictions against ground-truth labels.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            Dictionary with keys ``"r2_score"`` and ``"rmse"``.
        """
        preds = self.predict(X)
        min_len = min(len(preds), len(y))
        r2 = r2_score(y[:min_len], preds[:min_len])
        rmse = np.sqrt(mean_squared_error(y[:min_len], preds[:min_len]))
        metrics = {"r2_score": r2, "rmse": rmse}
        logger.info("Ensemble evaluation — R²: %.4f  RMSE: %.2f kW", r2, rmse)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, models_dir: str) -> None:
        """Save all trained models to *models_dir*.

        Args:
            models_dir: Directory in which to persist model files.
        """
        os.makedirs(models_dir, exist_ok=True)
        if self._xgb_trained:
            self.xgb_model.save(os.path.join(models_dir, "xgboost_model.json"))
        if self._lstm_trained:
            self.lstm_model.save(os.path.join(models_dir, "lstm_model.keras"))
        logger.info("Ensemble models saved to %s", models_dir)

    def load(self, models_dir: str) -> None:
        """Load models from *models_dir*.

        Args:
            models_dir: Directory that contains saved model files.
        """
        xgb_path = os.path.join(models_dir, "xgboost_model.json")
        if os.path.exists(xgb_path):
            self.xgb_model.load(xgb_path)
            self._xgb_trained = True

        lstm_path = os.path.join(models_dir, "lstm_model.keras")
        if os.path.exists(lstm_path):
            self.lstm_model.load(lstm_path)
            self._lstm_trained = True

        logger.info("Ensemble models loaded from %s", models_dir)
