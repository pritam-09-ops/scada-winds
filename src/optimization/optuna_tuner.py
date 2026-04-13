"""Optuna-based Bayesian hyperparameter optimisation for XGBoost.

Uses the TPE (Tree-structured Parzen Estimator) sampler and 5-fold
cross-validation to maximise R² score.

Target metrics (from *config.yaml*):
    - R² ≈ 0.989
    - RMSE ≈ 35.7 kW
"""

import logging
from typing import Dict, Optional

import numpy as np
import optuna
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    """Bayesian hyperparameter optimisation using Optuna with TPE sampler.

    Attributes:
        n_trials:   Number of optimisation trials.
        cv_folds:   Number of cross-validation folds.
        direction:  Optimisation direction (``"maximize"`` for R²).
        best_params: Best hyperparameters found (set after :meth:`optimize_xgboost`).
        study:      The ``optuna.Study`` object (set after optimisation).
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize OptunaTuner with configuration from *config.yaml*.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)

        opt_cfg = config.get("optimization", {})
        self.n_trials: int = opt_cfg.get("n_trials", 100)
        self.cv_folds: int = opt_cfg.get("cv_folds", 5)
        self.direction: str = opt_cfg.get("direction", "maximize")

        self.best_params: Optional[Dict] = None
        self.study: Optional[optuna.Study] = None

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def _xgboost_objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Optuna objective for XGBoost hyperparameter search.

        Uses *cv_folds*-fold cross-validation to estimate mean R².

        Args:
            trial: Current Optuna trial.
            X:     Feature matrix.
            y:     Target array.

        Returns:
            Mean R² across CV folds.
        """
        import xgboost as xgb  # lazy import

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        r2_scores = []

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(
                **params,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_val)
            r2_scores.append(r2_score(y_val, preds))

        return float(np.mean(r2_scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: Optional[int] = None,
    ) -> Dict:
        """Run Bayesian optimisation for XGBoost hyperparameters.

        Args:
            X:        Feature matrix.
            y:        Target array.
            n_trials: Override the number of trials from the config.

        Returns:
            Dictionary of best hyperparameters found.
        """
        trials = n_trials or self.n_trials
        logger.info(
            "Starting XGBoost optimisation: %d trials, %d-fold CV …",
            trials,
            self.cv_folds,
        )

        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
        )
        self.study.optimize(
            lambda trial: self._xgboost_objective(trial, X, y),
            n_trials=trials,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        logger.info(
            "Optimisation complete — best R²: %.4f  |  best params: %s",
            self.study.best_value,
            self.best_params,
        )
        return self.best_params

    def get_best_params(self) -> Optional[Dict]:
        """Return the best hyperparameters found by the last optimisation run.

        Returns:
            Dictionary of best hyperparameters, or ``None`` if
            :meth:`optimize_xgboost` has not been called yet.
        """
        return self.best_params
