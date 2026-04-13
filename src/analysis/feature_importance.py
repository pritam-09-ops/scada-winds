"""Random Forest feature importance analysis for SCADA wind power data.

Identifies the most predictive features (e.g. Rotor Speed, Pitch Angle)
and produces publication-ready bar charts.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Compute and visualise Random Forest feature importances."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize the analyzer.

        Args:
            n_estimators: Number of trees in the Random Forest.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.importances: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> "FeatureImportanceAnalyzer":
        """Train a Random Forest and compute feature importances.

        Args:
            X:         Feature DataFrame.
            y:         Target Series.
            test_size: Fraction of data held out for evaluation.

        Returns:
            Self (for method chaining).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)

        self.importances = self.model.feature_importances_
        self.feature_names = list(X.columns)

        score = self.model.score(X_test, y_test)
        logger.info("Random Forest R² on test set: %.4f", score)
        return self

    def get_importances(self) -> Dict[str, float]:
        """Return feature importances as an ordered dictionary.

        Returns:
            ``{feature_name: importance_score}`` sorted descending.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.importances is None or self.feature_names is None:
            raise RuntimeError("Call fit() before get_importances().")
        paired = sorted(
            zip(self.feature_names, self.importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: score for name, score in paired}

    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return the top-*n* most important features.

        Args:
            n: Number of top features to return.

        Returns:
            List of ``(feature_name, importance_score)`` tuples.
        """
        items = list(self.get_importances().items())
        return items[:n]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot a horizontal bar chart of feature importances.

        Args:
            top_n:     Number of top features to display.
            save_path: If provided, save the figure to this path.
            show:      Whether to call ``plt.show()``.
        """
        importances_dict = self.get_importances()
        names = list(importances_dict.keys())[:top_n]
        scores = list(importances_dict.values())[:top_n]

        # Reverse so the most important feature is at the top
        names = names[::-1]
        scores = scores[::-1]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
        ax.barh(names, scores, color="steelblue")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance – Random Forest (top {top_n})")
        ax.axvline(x=0, color="black", linewidth=0.8)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Feature importance plot saved to %s", save_path)

        if show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    feat_cfg = cfg.get("features", {})
    target_col = feat_cfg.get("target_column", "LV ActivePower (kW)")
    data_path = cfg.get("data", {}).get("raw_data_path", "data/raw/wind_power_data.csv")

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    analyzer = FeatureImportanceAnalyzer()
    analyzer.fit(df[feature_cols], df[target_col])
    analyzer.plot(save_path="feature_importance.png")

    print("\nTop 10 features:")
    for name, score in analyzer.top_features(10):
        print(f"  {name:<40} {score:.4f}")
