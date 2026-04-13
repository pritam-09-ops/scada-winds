"""Main training pipeline for SCADA wind power prediction.

Steps
-----
1. Load and preprocess raw SCADA data
2. Feature engineering
3. Split into train / validation / test sets
4. Train XGBoost with early stopping
5. Train LSTM with early stopping
6. (Optional) Optuna hyperparameter optimisation for XGBoost
7. Ensemble evaluation
8. Save all models, scalers, and metrics

Usage
-----
::

    python train.py                        # uses config.yaml defaults
    python train.py --config my_config.yaml
    python train.py --data path/to/data.csv --skip-optuna
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run directly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.analysis.feature_importance import FeatureImportanceAnalyzer
from src.features.feature_engineering import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostModel
from src.optimization.optuna_tuner import OptunaTuner
from src.utils.data_loader import DataLoader
from src.utils.results_exporter import generate_all_results

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_file_handler(logs_dir: str) -> None:
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(logs_dir, "training.log"))
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(fh)


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train / validation / test subsets."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # val_size is relative to the original dataset → rescale
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Pipeline step helpers (keep `train` readable)
# ---------------------------------------------------------------------------


def _load_and_engineer(
    config_path: str,
    raw_data_path: str,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, list, pd.DataFrame]:
    """Load data, preprocess, and run feature engineering.

    Returns:
        Tuple of ``(X, y, feature_cols, engineered_df)``.
    """
    loader = DataLoader(config_path)
    df = loader.preprocess(raw_data_path)

    engineer = FeatureEngineer(config_path)
    datetime_col = next(
        (c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None
    )
    df = engineer.engineer_features(df, datetime_col=datetime_col)
    logger.info("Dataset shape after feature engineering: %s", df.shape)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            "Check 'features.target_column' in config.yaml."
        )

    feature_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]
    X = df[feature_cols].values
    y = df[target_col].values
    logger.info("Features: %d  |  Samples: %d", len(feature_cols), len(X))
    return X, y, feature_cols, df


def _scale_and_save(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    models_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a StandardScaler on training data and persist it."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Feature scaler saved to %s", scaler_path)
    return X_train_sc, X_val_sc, X_test_sc


def _train_xgboost(
    config_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models_dir: str,
    best_params: dict | None = None,
) -> Tuple[XGBoostModel, dict]:
    """Train, evaluate, and save the XGBoost model."""
    logger.info("--- XGBoost ---")
    model = XGBoostModel(config_path)
    model.build(params=best_params)
    model.train(X_train, y_train, X_val, y_val)
    metrics = model.evaluate(X_test, y_test)
    logger.info(
        "XGBoost test — R²: %.4f  RMSE: %.2f kW",
        metrics["r2_score"], metrics["rmse"],
    )
    model.save(os.path.join(models_dir, "xgboost_model.json"))
    return model, metrics


def _train_lstm(
    config_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models_dir: str,
) -> Tuple[LSTMModel, dict]:
    """Train, evaluate, and save the LSTM model."""
    logger.info("--- LSTM ---")
    model = LSTMModel(config_path)
    model.train(X_train, y_train, X_val, y_val)
    metrics = model.evaluate(X_test, y_test)
    logger.info(
        "LSTM test — R²: %.4f  RMSE: %.2f kW",
        metrics["r2_score"], metrics["rmse"],
    )
    model.save(os.path.join(models_dir, "lstm_model.keras"))
    return model, metrics


def _run_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list,
    target_col: str,
    logs_dir: str,
) -> None:
    """Compute and plot Random Forest feature importances."""
    logger.info("--- Feature Importance ---")
    fi_analyzer = FeatureImportanceAnalyzer()
    fi_analyzer.fit(
        pd.DataFrame(X_train, columns=feature_cols),
        pd.Series(y_train, name=target_col),
    )
    fi_analyzer.plot(
        save_path=os.path.join(logs_dir, "feature_importance.png"),
        show=False,
    )
    logger.info("Top 5 predictive features:")
    for name, score in fi_analyzer.top_features(5):
        logger.info("  %-40s %.4f", name, score)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config_path: str = "config.yaml", data_path: str = None, run_optuna: bool = True) -> None:
    """End-to-end training pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        data_path:   Override the data path from the config.
        run_optuna:  Whether to run Optuna hyperparameter optimisation.
    """
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    training_cfg = config.get("training", {})
    test_size = training_cfg.get("test_size", 0.2)
    val_size = training_cfg.get("validation_size", 0.1)
    random_state = training_cfg.get("random_state", 42)
    models_dir = training_cfg.get("models_dir", "models/")
    logs_dir = training_cfg.get("logs_dir", "logs/")

    features_cfg = config.get("features", {})
    target_col = features_cfg.get("target_column", "LV ActivePower (kW)")
    raw_data_path = data_path or config.get("data", {}).get(
        "raw_data_path", "data/raw/wind_power_data.csv"
    )

    _add_file_handler(logs_dir)
    logger.info("=" * 60)
    logger.info("SCADA Wind Power Prediction — Training Pipeline")
    logger.info("=" * 60)
    logger.info("Config: %s", config_path)
    logger.info("Data:   %s", raw_data_path)

    # ------------------------------------------------------------------
    # Load, preprocess, and engineer features
    # ------------------------------------------------------------------
    X, y, feature_cols, engineered_df = _load_and_engineer(config_path, raw_data_path, target_col)

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, test_size, val_size, random_state
    )
    logger.info(
        "Split — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # Build per-row split labels aligned with the full dataset
    n_total = len(X)
    split_labels = np.empty(n_total, dtype=object)
    # Reproduce the split indices so labels line up with engineered_df rows.
    from sklearn.model_selection import train_test_split as _tts
    idx_all = np.arange(n_total)
    idx_tmp, idx_test = _tts(idx_all, test_size=test_size, random_state=random_state)
    val_rel = val_size / (1.0 - test_size)
    idx_train, idx_val = _tts(idx_tmp, test_size=val_rel, random_state=random_state)
    split_labels[idx_train] = "train"
    split_labels[idx_val] = "val"
    split_labels[idx_test] = "test"

    # ------------------------------------------------------------------
    # Scale features
    # ------------------------------------------------------------------
    X_train_sc, X_val_sc, X_test_sc = _scale_and_save(
        X_train, X_val, X_test, models_dir
    )

    # ------------------------------------------------------------------
    # (Optional) Optuna hyperparameter optimisation
    # ------------------------------------------------------------------
    best_xgb_params = None
    if run_optuna:
        logger.info("Running Optuna hyperparameter optimisation …")
        tuner = OptunaTuner(config_path)
        best_xgb_params = tuner.optimize_xgboost(X_train_sc, y_train)
        logger.info("Best XGBoost params: %s", best_xgb_params)

    # ------------------------------------------------------------------
    # Train individual models
    # ------------------------------------------------------------------
    xgb_model, xgb_metrics = _train_xgboost(
        config_path, X_train_sc, y_train, X_val_sc, y_val,
        X_test_sc, y_test, models_dir, best_xgb_params,
    )
    lstm_model, lstm_metrics = _train_lstm(
        config_path, X_train_sc, y_train, X_val_sc, y_val,
        X_test_sc, y_test, models_dir,
    )

    # ------------------------------------------------------------------
    # Ensemble evaluation
    # ------------------------------------------------------------------
    logger.info("--- Ensemble ---")
    ensemble = EnsembleModel(config_path)
    ensemble.xgb_model = xgb_model
    ensemble._xgb_trained = True
    ensemble.lstm_model = lstm_model
    ensemble._lstm_trained = True
    ens_metrics = ensemble.evaluate(X_test_sc, y_test)
    logger.info(
        "Ensemble test — R²: %.4f  RMSE: %.2f kW",
        ens_metrics["r2_score"], ens_metrics["rmse"],
    )

    # ------------------------------------------------------------------
    # Feature importance analysis
    # ------------------------------------------------------------------
    _run_feature_importance(X_train, y_train, feature_cols, target_col, logs_dir)

    # ------------------------------------------------------------------
    # Generate consolidated CSV + results visualisations
    # ------------------------------------------------------------------
    logger.info("--- Results Export ---")

    # Collect full-dataset predictions for the main CSV
    import joblib as _joblib
    scaler = _joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
    X_all_sc = scaler.transform(X)

    xgb_all_preds = xgb_model.predict(X_all_sc)
    lstm_all_preds = lstm_model.predict(X_all_sc)
    ens_all_preds = ensemble.predict(X_all_sc)

    # Feature importance dict from the FeatureImportanceAnalyzer reuse
    fi_analyzer = FeatureImportanceAnalyzer()
    fi_analyzer.fit(
        pd.DataFrame(X_train, columns=feature_cols),
        pd.Series(y_train, name=target_col),
    )
    fi_dict = fi_analyzer.get_importances()

    results_dir = config.get("data", {}).get("results_dir", "data/results")
    main_csv_path = config.get("data", {}).get("main_csv_path", "data/main.csv")

    generate_all_results(
        raw_df=engineered_df,
        feature_cols=feature_cols,
        y_actual=y,
        xgb_preds=xgb_all_preds,
        lstm_preds=lstm_all_preds,
        ensemble_preds=ens_all_preds,
        xgb_metrics=xgb_metrics,
        lstm_metrics=lstm_metrics,
        ensemble_metrics=ens_metrics,
        feature_importance=fi_dict,
        split_labels=split_labels,
        best_xgb_params=best_xgb_params,
        results_dir=results_dir,
        main_csv_path=main_csv_path,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    target_r2 = config.get("target_metrics", {}).get("r2_score", 0.989)
    target_rmse = config.get("target_metrics", {}).get("rmse_kw", 35.7)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  XGBoost  — R²: %.4f  RMSE: %.2f kW", xgb_metrics["r2_score"], xgb_metrics["rmse"])
    logger.info("  LSTM     — R²: %.4f  RMSE: %.2f kW", lstm_metrics["r2_score"], lstm_metrics["rmse"])
    logger.info("  Ensemble — R²: %.4f  RMSE: %.2f kW", ens_metrics["r2_score"], ens_metrics["rmse"])
    logger.info("  Target   — R²: %.3f  RMSE: %.1f kW", target_r2, target_rmse)
    logger.info("  Models saved in: %s", os.path.abspath(models_dir))
    logger.info("  Main CSV:        %s", os.path.abspath(main_csv_path))
    logger.info("  Results dir:     %s", os.path.abspath(results_dir))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SCADA wind power prediction models."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override data path from config.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip Optuna hyperparameter optimisation to speed up training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        config_path=args.config,
        data_path=args.data,
        run_optuna=not args.skip_optuna,
    )
