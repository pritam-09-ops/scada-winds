"""LSTM deep-learning model for wind power time-series prediction.

Architecture (from *config.yaml*):
    - 2 stacked LSTM layers
    - 128 units in the first layer, 64 in the second
    - 0.3 dropout after each LSTM layer
    - Dense(1) output layer
    - Adam optimiser, MSE loss
    - EarlyStopping + ReduceLROnPlateau callbacks
"""

import logging
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM deep-learning regressor for wind power prediction."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize LSTM model with configuration from *config.yaml*.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)

        lstm_cfg = config.get("lstm", {})
        self.units: int = lstm_cfg.get("units", 128)
        self.num_layers: int = lstm_cfg.get("num_layers", 2)
        self.dropout: float = lstm_cfg.get("dropout", 0.3)
        self.batch_size: int = lstm_cfg.get("batch_size", 32)
        self.epochs: int = lstm_cfg.get("epochs", 100)
        self.sequence_length: int = lstm_cfg.get("sequence_length", 24)
        self.patience: int = lstm_cfg.get("patience", 10)

        self.model = None
        self.scaler: MinMaxScaler = MinMaxScaler()
        self._is_scaler_fitted: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_keras():
        """Lazy-import TensorFlow/Keras so that the module is importable
        even when TensorFlow is not installed."""
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from tensorflow.keras.layers import Dense, Dropout, LSTM
            from tensorflow.keras.models import Sequential, load_model

            return Sequential, LSTM, Dropout, Dense, EarlyStopping, ReduceLROnPlateau, load_model
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for the LSTM model. "
                "Install it with: pip install tensorflow"
            ) from exc

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape a flat feature matrix into overlapping LSTM sequences.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target array of length ``n_samples``.
            sequence_length: Number of time-steps per sequence.  Defaults to
                ``self.sequence_length``.

        Returns:
            Tuple ``(X_seq, y_seq)`` where ``X_seq`` has shape
            ``(n_samples - seq_len, seq_len, n_features)``.
        """
        seq_len = sequence_length or self.sequence_length
        X_seq: list = []
        y_seq: list = []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    # ------------------------------------------------------------------
    # Build / train / predict / evaluate
    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[int, int]) -> None:
        """Construct the Keras Sequential LSTM model.

        Args:
            input_shape: ``(sequence_length, n_features)`` tuple.
        """
        Sequential, LSTM, Dropout, Dense, *_ = self._import_keras()

        model = Sequential()
        # First LSTM layer
        model.add(
            LSTM(
                self.units,
                return_sequences=(self.num_layers > 1),
                input_shape=input_shape,
            )
        )
        model.add(Dropout(self.dropout))

        # Additional LSTM layers
        for i in range(1, self.num_layers):
            return_seq = i < (self.num_layers - 1)
            model.add(
                LSTM(max(1, self.units // (2 ** i)), return_sequences=return_seq)
            )
            model.add(Dropout(self.dropout))

        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model = model
        logger.info(
            "Built LSTM model: %d layers, %d units, %.1f dropout.",
            self.num_layers,
            self.units,
            self.dropout,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Scale inputs, create sequences, and fit the LSTM model.

        Args:
            X_train: Training feature matrix.
            y_train: Training targets.
            X_val:   Validation feature matrix (enables early stopping).
            y_val:   Validation targets.
        """
        *_, EarlyStopping, ReduceLROnPlateau, _ = self._import_keras()

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        self._is_scaler_fitted = True

        X_seq, y_seq = self.create_sequences(X_train_scaled, y_train)

        if self.model is None:
            self.build(input_shape=(X_seq.shape[1], X_seq.shape[2]))

        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(monitor="val_loss" if X_val is not None else "loss",
                              factor=0.5, patience=5, verbose=0),
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
            if len(X_val_seq) > 0:
                validation_data = (X_val_seq, y_val_seq)

        logger.info(
            "Training LSTM with %d sequences of length %d …",
            X_seq.shape[0],
            self.sequence_length,
        )
        self.model.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        train_preds = self.model.predict(X_seq, verbose=0).flatten()
        train_r2 = r2_score(y_seq, train_preds)
        train_rmse = np.sqrt(mean_squared_error(y_seq, train_preds))
        logger.info(
            "LSTM training — R²: %.4f  RMSE: %.2f kW", train_r2, train_rmse
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for a new feature matrix.

        The first ``sequence_length`` rows will be padded with the first
        available prediction so that the returned array has the same length
        as *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Prediction array of length ``n_samples``.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions.")
        if not self._is_scaler_fitted:
            raise RuntimeError("Scaler has not been fitted. Train the model first.")

        X_scaled = self.scaler.transform(X)
        dummy_y = np.zeros(len(X_scaled))
        X_seq, _ = self.create_sequences(X_scaled, dummy_y)

        if len(X_seq) == 0:
            return np.zeros(len(X))

        preds = self.model.predict(X_seq, verbose=0).flatten()
        # Pad first `sequence_length` values
        padded = np.concatenate([np.full(self.sequence_length, preds[0]), preds])
        return padded[: len(X)]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            Dictionary with keys ``"r2_score"`` and ``"rmse"``.
        """
        predictions = self.predict(X)
        min_len = min(len(predictions), len(y))
        r2 = r2_score(y[:min_len], predictions[:min_len])
        rmse = np.sqrt(mean_squared_error(y[:min_len], predictions[:min_len]))
        metrics = {"r2_score": r2, "rmse": rmse}
        logger.info("LSTM evaluation — R²: %.4f  RMSE: %.2f kW", r2, rmse)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save the Keras model and fitted MinMaxScaler.

        The scaler is saved alongside the model with a ``_scaler.pkl``
        suffix so that it is automatically loaded when :meth:`load` is
        called.

        Args:
            filepath: Destination path for the Keras model
                (e.g. ``"models/lstm_model.keras"``).

        Raises:
            RuntimeError: If no model has been trained yet.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.model.save(filepath)

        scaler_path = self._scaler_path(filepath)
        joblib.dump(self.scaler, scaler_path)
        logger.info("LSTM model saved to %s (scaler: %s)", filepath, scaler_path)

    def load(self, filepath: str) -> None:
        """Load a previously saved Keras model and its scaler.

        Args:
            filepath: Path to the saved Keras model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        *_, load_model = self._import_keras()
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = load_model(filepath)

        scaler_path = self._scaler_path(filepath)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self._is_scaler_fitted = True
        logger.info("LSTM model loaded from %s", filepath)

    @staticmethod
    def _scaler_path(model_path: str) -> str:
        """Derive the scaler file path from the model path."""
        base = model_path
        for ext in (".h5", ".keras"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        return base + "_scaler.pkl"
