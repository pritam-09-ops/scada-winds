"""XGBoost regressor for wind power prediction.

Configuration (via *config.yaml*):
    - n_estimators: 300
    - max_depth: 8
    - learning_rate: 0.05
    - subsample / colsample_bytree: 0.8
    - early_stopping_rounds: 50
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import xgboost as xgb
import yaml
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost gradient-boosted regressor for wind power prediction."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize with hyperparameters from *config.yaml*.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)

        xgb_cfg = config.get("xgboost", {})
        self.params: Dict = {
            "n_estimators": xgb_cfg.get("n_estimators", 300),
            "max_depth": xgb_cfg.get("max_depth", 8),
            "learning_rate": xgb_cfg.get("learning_rate", 0.05),
            "subsample": xgb_cfg.get("subsample", 0.8),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
            "min_child_weight": xgb_cfg.get("min_child_weight", 1),
            "gamma": xgb_cfg.get("gamma", 0.0),
        }
        self.early_stopping_rounds: int = xgb_cfg.get("early_stopping_rounds", 50)
        self.model: Optional[xgb.XGBRegressor] = None

    # ------------------------------------------------------------------
    # Build / train / predict / evaluate
    # ------------------------------------------------------------------

    def build(self, params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """Instantiate the XGBRegressor.

        Args:
            params: Optional dictionary of hyperparameters that override the
                defaults loaded from *config.yaml*.

        Returns:
            Configured (but not yet fitted) ``XGBRegressor`` instance.
        """
        model_params = {**self.params, **(params or {})}
        self.model = xgb.XGBRegressor(
            **model_params,
            objective="reg:squarederror",
            tree_method="hist",
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=42,
            verbosity=0,
        )
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> xgb.XGBRegressor:
        """Fit the model on training data with optional validation / early stopping.

        Args:
            X_train: Training feature matrix.
            y_train: Training targets.
            X_val:   Validation feature matrix (enables early stopping).
            y_val:   Validation targets.

        Returns:
            Fitted ``XGBRegressor``.
        """
        if self.model is None:
            self.build()

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        logger.info("Training XGBoost model with %d samples …", X_train.shape[0])
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        train_preds = self.model.predict(X_train)
        train_r2 = r2_score(y_train, train_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        logger.info(
            "XGBoost training — R²: %.4f  RMSE: %.2f kW", train_r2, train_rmse
        )
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for feature matrix *X*.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted power values.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions.")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute R² and RMSE on a held-out dataset.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            Dictionary with keys ``"r2_score"`` and ``"rmse"``.
        """
        predictions = self.predict(X)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        metrics = {"r2_score": r2, "rmse": rmse}
        logger.info("XGBoost evaluation — R²: %.4f  RMSE: %.2f kW", r2, rmse)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save the trained model to disk (XGBoost native JSON format).

        Args:
            filepath: Destination path (e.g. ``"models/xgboost_model.json"``).

        Raises:
            RuntimeError: If no model has been trained yet.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.model.get_booster().save_model(filepath)
        logger.info("XGBoost model saved to %s", filepath)

    def load(self, filepath: str) -> None:
        """Load a previously saved model.

        Args:
            filepath: Path to the saved model file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        logger.info("XGBoost model loaded from %s", filepath)
