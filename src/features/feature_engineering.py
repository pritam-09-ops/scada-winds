"""Feature engineering pipeline for wind power SCADA data.

Derives physical and statistical features that improve model accuracy:
- Wind Power Density  (cubic function of wind speed)
- Temporal features   (hour, day-of-week, month, is_night)
- Rolling statistics  (mean and std over configurable windows)
- Aerodynamic interactions (wind-rotor, pitch-wind, polynomial wind speed)
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for wind power prediction."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize FeatureEngineer with YAML configuration.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, "r") as fh:
            self.config = yaml.safe_load(fh)
        self.features_config = self.config.get("features", {})
        self.air_density: float = self.features_config.get("air_density", 1.225)
        self.rotor_area: float = self.features_config.get("rotor_area", 7854.0)

    # ------------------------------------------------------------------
    # Individual feature derivations
    # ------------------------------------------------------------------

    def derive_wind_power_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive Wind Power Density as a cubic function of wind speed.

        Formula: P_wind = 0.5 × ρ × A × v³

        Where:
            ρ  = air density (kg/m³)
            A  = rotor swept area (m²)
            v  = wind speed (m/s)

        Note:
            This represents the *theoretical* kinetic power available in the
            wind, not the actual turbine output.  Real turbine power is limited
            by the Betz limit (Cp_max ≈ 0.593) and mechanical/electrical
            losses.  The ``wind_power_density`` feature therefore captures the
            aerodynamic forcing term rather than the electrical yield.

        Args:
            df: Input DataFrame containing a wind-speed column.

        Returns:
            DataFrame with ``wind_power_density`` column added.
        """
        wind_col = self.features_config.get("wind_speed_column", "Wind Speed (m/s)")
        if wind_col not in df.columns:
            logger.warning(
                "Wind speed column '%s' not found – skipping wind_power_density.",
                wind_col,
            )
            return df

        df = df.copy()
        df["wind_power_density"] = (
            0.5 * self.air_density * self.rotor_area * df[wind_col] ** 3
        )
        logger.debug("Derived wind_power_density feature.")
        return df

    def create_temporal_features(
        self,
        df: pd.DataFrame,
        datetime_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create temporal features from a datetime column or index.

        Generated features:
            - ``hour``        – hour of day (0–23)
            - ``day_of_week`` – day of week (0 = Monday, 6 = Sunday)
            - ``month``       – month of year (1–12)
            - ``is_night``    – 1 if hour < 6 or hour ≥ 20, else 0

        Args:
            df: Input DataFrame.
            datetime_col: Name of a datetime column.  If ``None``, the
                function tries the DataFrame index and then the first
                datetime-typed column it finds.

        Returns:
            DataFrame with temporal feature columns appended.
        """
        df = df.copy()

        if datetime_col and datetime_col in df.columns:
            dt = pd.to_datetime(df[datetime_col])
            df["hour"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
            df["month"] = dt.dt.month
        elif isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
            df["hour"] = dt.hour
            df["day_of_week"] = dt.dayofweek
            df["month"] = dt.month
        else:
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns
            if len(datetime_cols) > 0:
                dt = pd.to_datetime(df[datetime_cols[0]])
                df["hour"] = dt.dt.hour
                df["day_of_week"] = dt.dt.dayofweek
                df["month"] = dt.dt.month
            else:
                logger.warning(
                    "No datetime column found – temporal features not created."
                )
                return df

        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)

        logger.debug("Created temporal features: hour, day_of_week, month, is_night.")
        return df

    def create_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Generate rolling mean and standard deviation features.

        Args:
            df: Input DataFrame.
            columns: Columns on which to compute rolling stats.  Defaults to
                the wind-speed and target columns from *config.yaml*.
            windows: Rolling window sizes.  Defaults to the value in
                *config.yaml* (``[6, 12, 24]``).

        Returns:
            DataFrame with ``{col}_rolling_mean_{w}`` and
            ``{col}_rolling_std_{w}`` columns appended.
        """
        df = df.copy()
        windows = windows or self.features_config.get("rolling_windows", [6, 12, 24])

        if columns is None:
            wind_col = self.features_config.get(
                "wind_speed_column", "Wind Speed (m/s)"
            )
            target_col = self.features_config.get(
                "target_column", "LV ActivePower (kW)"
            )
            columns = [c for c in [wind_col, target_col] if c in df.columns]

        for col in columns:
            if col not in df.columns:
                continue
            for window in windows:
                df[f"{col}_rolling_mean_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_rolling_std_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).std().fillna(0.0)
                )

        logger.debug("Created rolling statistics for windows: %s.", windows)
        return df

    def create_aerodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aerodynamic interaction features.

        Generated features:
            - ``wind_rotor_interaction`` – wind speed × rotor speed
            - ``pitch_wind_interaction`` – pitch angle × wind speed
            - ``wind_speed_squared``     – wind speed²
            - ``wind_speed_cubed``       – wind speed³

        The last two encode the non-linear aerodynamic power extraction
        curve described by the Betz limit model.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with aerodynamic feature columns appended.
        """
        df = df.copy()
        wind_col = self.features_config.get("wind_speed_column", "Wind Speed (m/s)")
        rotor_col = self.features_config.get("rotor_speed_column", "Rotor Speed (rpm)")
        pitch_col = self.features_config.get(
            "pitch_angle_column", "Pitch Angle (deg)"
        )

        if wind_col in df.columns and rotor_col in df.columns:
            df["wind_rotor_interaction"] = df[wind_col] * df[rotor_col]

        if pitch_col in df.columns and wind_col in df.columns:
            df["pitch_wind_interaction"] = df[pitch_col] * df[wind_col]

        if wind_col in df.columns:
            df["wind_speed_squared"] = df[wind_col] ** 2
            df["wind_speed_cubed"] = df[wind_col] ** 3

        logger.debug("Created aerodynamic interaction features.")
        return df

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def engineer_features(
        self,
        df: pd.DataFrame,
        datetime_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run the full feature engineering pipeline.

        Steps (in order):
            1. Wind Power Density derivation
            2. Temporal features
            3. Rolling statistics
            4. Aerodynamic interactions

        Args:
            df: Raw or cleaned input DataFrame.
            datetime_col: Name of the datetime column (optional).

        Returns:
            DataFrame enriched with all engineered features.
        """
        logger.info("Starting feature engineering pipeline …")
        df = self.derive_wind_power_density(df)
        df = self.create_temporal_features(df, datetime_col)
        df = self.create_rolling_statistics(df)
        df = self.create_aerodynamic_features(df)
        logger.info(
            "Feature engineering complete.  Total features: %d", len(df.columns)
        )
        return df
