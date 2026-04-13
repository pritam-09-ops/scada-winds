"""Data loading and preprocessing utilities for SCADA wind power data."""

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses wind power SCADA data from CSV files."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize DataLoader with YAML configuration.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, "r") as fh:
            self.config = yaml.safe_load(fh)
        self.data_config = self.config.get("data", {})
        self.features_config = self.config.get("features", {})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load wind power data from a CSV file.

        Args:
            filepath: Path to the CSV data file.

        Returns:
            Raw DataFrame loaded from the CSV.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        logger.info("Loading data from %s", filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=True)
        logger.info("Loaded %d rows with %d columns", len(df), len(df.columns))
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Handle missing values using the specified strategy.

        Args:
            df: Input DataFrame.
            strategy: One of ``"drop"``, ``"forward_fill"``, or
                ``"backward_fill"``.  Falls back to the value from
                *config.yaml* when not provided.

        Returns:
            DataFrame with missing values handled.

        Raises:
            ValueError: If an unknown strategy is supplied.
        """
        strategy = strategy or self.data_config.get(
            "missing_value_strategy", "forward_fill"
        )
        missing_count = df.isnull().sum().sum()
        logger.info(
            "Handling %d missing values using strategy: %s", missing_count, strategy
        )

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "forward_fill":
            df = df.ffill().bfill()
        elif strategy == "backward_fill":
            df = df.bfill().ffill()
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Remove outliers from numeric columns using the z-score method.

        Args:
            df: Input DataFrame.
            columns: Numeric columns to check.  Defaults to all numeric
                columns in *df*.
            threshold: Z-score threshold above which a row is considered an
                outlier.  Falls back to the value from *config.yaml*.

        Returns:
            DataFrame with outlier rows removed.
        """
        threshold = threshold or self.data_config.get(
            "outlier_z_score_threshold", 3.0
        )
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(
            "Removing outliers using z-score threshold: %.1f across %d columns",
            threshold,
            len(columns),
        )
        initial_len = len(df)

        valid_columns = [c for c in columns if c in df.columns]
        if not valid_columns:
            return df

        z_scores = np.abs(stats.zscore(df[valid_columns].fillna(0)))
        mask = (z_scores < threshold).all(axis=1)
        df = df[mask]

        removed = initial_len - len(df)
        logger.info(
            "Removed %d outlier rows (%.2f%%)",
            removed,
            removed / initial_len * 100 if initial_len else 0,
        )
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity after preprocessing.

        Args:
            df: Preprocessed DataFrame to validate.

        Returns:
            ``True`` if validation passes, ``False`` otherwise.
        """
        if df.empty:
            logger.error("DataFrame is empty after preprocessing")
            return False

        target_col = self.features_config.get(
            "target_column", "LV ActivePower (kW)"
        )
        if target_col not in df.columns:
            logger.warning(
                "Target column '%s' not found; ensure the CSV uses the correct "
                "column names (configurable in config.yaml).",
                target_col,
            )

        logger.info(
            "Data validation passed: %d rows, %d columns", len(df), len(df.columns)
        )
        return True

    def preprocess(
        self,
        filepath: str,
        remove_outliers: bool = True,
    ) -> pd.DataFrame:
        """Run the full preprocessing pipeline.

        Steps:
            1. Load CSV
            2. Handle missing values
            3. (Optional) Remove outliers

        Args:
            filepath: Path to the raw CSV file.
            remove_outliers: Whether to apply z-score outlier removal.

        Returns:
            Cleaned and validated DataFrame.
        """
        df = self.load_csv(filepath)
        df = self.handle_missing_values(df)
        if remove_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = self.remove_outliers(df, numeric_cols)
        self.validate_data(df)
        return df
