"""Inference script for SCADA wind power prediction.

Loads trained XGBoost and LSTM models from disk, preprocesses new data,
and returns ensemble predictions with basic confidence statistics.

Usage
-----
::

    python predict.py --data new_data.csv [--models models/] [--output predictions.csv]
    python predict.py --data new_data.csv --skip-lstm   # XGBoost only (faster)
"""

import argparse
import logging
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run directly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.features.feature_engineering import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def preprocess_data(
    df: pd.DataFrame,
    config_path: str = "config.yaml",
    scaler_path: Optional[str] = None,
) -> np.ndarray:
    """Preprocess a raw DataFrame for model inference.

    Steps:
        1. Handle missing values
        2. Feature engineering
        3. Apply the saved StandardScaler (if available)

    Args:
        df:          Raw input DataFrame.
        config_path: Path to the YAML configuration file.
        scaler_path: Path to the saved ``feature_scaler.pkl``.

    Returns:
        Scaled feature matrix as a NumPy array.
    """
    import yaml

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    features_cfg = config.get("features", {})
    target_col = features_cfg.get("target_column", "LV ActivePower (kW)")

    # Handle missing values
    loader = DataLoader(config_path)
    df = loader.handle_missing_values(df)

    # Feature engineering
    engineer = FeatureEngineer(config_path)
    datetime_col = next(
        (c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None
    )
    df = engineer.engineer_features(df, datetime_col=datetime_col)

    # Select numeric features (exclude target if present)
    feature_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]
    X = df[feature_cols].values

    # Apply scaler if available
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        logger.info("Applied feature scaler from %s", scaler_path)
    else:
        logger.warning(
            "No scaler found at '%s'. Predictions may be less accurate.", scaler_path
        )

    return X


def make_predictions(
    data: pd.DataFrame,
    models_dir: str = "models/",
    config_path: str = "config.yaml",
    skip_lstm: bool = False,
) -> pd.DataFrame:
    """Generate ensemble predictions for new wind-power data.

    Args:
        data:        Raw input DataFrame.
        models_dir:  Directory containing saved model files.
        config_path: Path to the YAML configuration file.
        skip_lstm:   If ``True``, use only the XGBoost model.

    Returns:
        DataFrame with columns ``prediction_kw``, ``lower_bound``,
        ``upper_bound``, and ``confidence_interval_width``.

    Raises:
        FileNotFoundError: If the XGBoost model file is not found.
    """
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    X = preprocess_data(data, config_path=config_path, scaler_path=scaler_path)

    ensemble = EnsembleModel(config_path)

    # Always load XGBoost (required)
    xgb_path = os.path.join(models_dir, "xgboost_model.json")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(
            f"XGBoost model not found at '{xgb_path}'. "
            "Run train.py first to generate the model files."
        )
    ensemble.xgb_model.load(xgb_path)
    ensemble._xgb_trained = True

    # Optionally load LSTM
    if not skip_lstm:
        lstm_path = os.path.join(models_dir, "lstm_model.keras")
        if os.path.exists(lstm_path):
            ensemble.lstm_model.load(lstm_path)
            ensemble._lstm_trained = True
        else:
            logger.warning(
                "LSTM model not found at '%s'. Falling back to XGBoost only.", lstm_path
            )

    logger.info(
        "Running predictions (XGBoost: %s  LSTM: %s) on %d samples …",
        ensemble._xgb_trained,
        ensemble._lstm_trained,
        len(X),
    )
    predictions = ensemble.predict(X)

    # Simple confidence interval: ±5% as a rough heuristic.
    # NOTE: This is NOT a statistically derived prediction interval.
    # It does not reflect actual model uncertainty; techniques such as
    # conformal prediction, quantile regression, or bootstrap ensembles
    # would be required for statistically valid bounds.
    ci_half = np.abs(predictions) * 0.05
    result = pd.DataFrame(
        {
            "prediction_kw": predictions,
            "lower_bound": predictions - ci_half,
            "upper_bound": predictions + ci_half,
            "confidence_interval_width": 2 * ci_half,
        }
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCADA wind power ensemble inference."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the input CSV file with new wind-power data.",
    )
    parser.add_argument(
        "--models",
        default="models/",
        help="Directory containing saved model files (default: models/).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save predictions to this CSV path (optional).",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Use only the XGBoost model (faster; no LSTM).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not os.path.exists(args.data):
        logger.error("Data file not found: %s", args.data)
        sys.exit(1)

    new_data = pd.read_csv(args.data)
    logger.info("Loaded %d rows from %s", len(new_data), args.data)

    result_df = make_predictions(
        data=new_data,
        models_dir=args.models,
        config_path=args.config,
        skip_lstm=args.skip_lstm,
    )

    print(result_df.head(20).to_string(index=False))
    logger.info(
        "Prediction stats — mean: %.1f kW  min: %.1f kW  max: %.1f kW",
        result_df["prediction_kw"].mean(),
        result_df["prediction_kw"].min(),
        result_df["prediction_kw"].max(),
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result_df.to_csv(args.output, index=False)
        logger.info("Predictions saved to %s", args.output)
