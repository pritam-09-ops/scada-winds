"""Results export and visualisation utilities for SCADA wind power models.

Provides helpers to:
- Consolidate raw data, engineered features, model predictions, and
  evaluation metrics into a single ``data/main.csv`` file.
- Generate a suite of publication-ready plots saved under ``data/results/``.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette used across all plots
# ---------------------------------------------------------------------------
PALETTE = {
    "xgboost": "#2196F3",   # blue
    "lstm": "#FF9800",       # orange
    "ensemble": "#4CAF50",   # green
    "actual": "#9C27B0",     # purple
    "residual": "#F44336",   # red
}


# ---------------------------------------------------------------------------
# CSV consolidation
# ---------------------------------------------------------------------------


def build_main_csv(
    raw_df: pd.DataFrame,
    feature_cols: List[str],
    y_actual: np.ndarray,
    xgb_preds: Optional[np.ndarray],
    lstm_preds: Optional[np.ndarray],
    ensemble_preds: Optional[np.ndarray],
    xgb_metrics: Optional[Dict[str, float]],
    lstm_metrics: Optional[Dict[str, float]],
    ensemble_metrics: Optional[Dict[str, float]],
    split_labels: Optional[np.ndarray] = None,
    output_path: str = "data/main.csv",
    model_version: str = "1.0",
) -> pd.DataFrame:
    """Assemble a consolidated DataFrame and write it to *output_path*.

    The resulting CSV contains:
    - All numeric feature columns from *raw_df* (after engineering)
    - Actual target values
    - XGBoost / LSTM / Ensemble predictions (where available)
    - Confidence-interval columns (±5 % heuristic)
    - Prediction errors and residuals for each model
    - ``split`` label (``"train"`` / ``"val"`` / ``"test"``) when provided
    - Metadata columns: ``timestamp``, ``model_version``

    Args:
        raw_df:           DataFrame whose rows correspond to *y_actual* (after
                          feature engineering and outlier removal).
        feature_cols:     Names of the numeric feature columns used for
                          modelling.
        y_actual:         Ground-truth target array.
        xgb_preds:        XGBoost prediction array (or ``None``).
        lstm_preds:       LSTM prediction array (or ``None``).
        ensemble_preds:   Ensemble prediction array (or ``None``).
        xgb_metrics:      ``{"r2_score": ..., "rmse": ...}`` for XGBoost.
        lstm_metrics:     Same for LSTM.
        ensemble_metrics: Same for the ensemble.
        split_labels:     Array of ``"train"`` / ``"val"`` / ``"test"``
                          strings aligned with the rows of *raw_df*.
        output_path:      Destination path for the CSV file.
        model_version:    Version string written to the ``model_version``
                          column.

    Returns:
        The assembled DataFrame (also written to *output_path*).
    """
    n = len(y_actual)
    rows = min(n, len(raw_df))

    df = raw_df.iloc[:rows].copy().reset_index(drop=True)

    # --- actual target -------------------------------------------------------
    df["actual_kw"] = y_actual[:rows]

    # --- predictions ---------------------------------------------------------
    def _add_pred_cols(prefix: str, preds: np.ndarray) -> None:
        p = preds[:rows]
        # Confidence interval: ±5% of the absolute predicted value.
        # NOTE: This is a rough heuristic approximation, not a statistically
        # derived prediction interval.  For valid uncertainty quantification,
        # use conformal prediction, quantile regression, or bootstrap methods.
        ci_half = np.abs(p) * 0.05
        df[f"{prefix}_prediction_kw"] = p
        df[f"{prefix}_lower_bound"] = p - ci_half
        df[f"{prefix}_upper_bound"] = p + ci_half
        df[f"{prefix}_ci_width"] = 2 * ci_half
        df[f"{prefix}_error_kw"] = p - df["actual_kw"]
        df[f"{prefix}_abs_error_kw"] = np.abs(df[f"{prefix}_error_kw"])
        df[f"{prefix}_residual"] = df["actual_kw"] - p

    if xgb_preds is not None:
        _add_pred_cols("xgb", xgb_preds)
    if lstm_preds is not None:
        _add_pred_cols("lstm", lstm_preds)
    if ensemble_preds is not None:
        _add_pred_cols("ensemble", ensemble_preds)

    # --- split labels --------------------------------------------------------
    if split_labels is not None:
        df["split"] = split_labels[:rows]

    # --- metadata ------------------------------------------------------------
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    df["timestamp"] = ts
    df["model_version"] = model_version

    # --- metrics columns (repeated for every row so they are accessible) -----
    def _add_metric_cols(prefix: str, metrics: Dict[str, float]) -> None:
        df[f"{prefix}_r2"] = metrics.get("r2_score", float("nan"))
        df[f"{prefix}_rmse"] = metrics.get("rmse", float("nan"))

    if xgb_metrics:
        _add_metric_cols("xgb", xgb_metrics)
    if lstm_metrics:
        _add_metric_cols("lstm", lstm_metrics)
    if ensemble_metrics:
        _add_metric_cols("ensemble", ensemble_metrics)

    # --- write ---------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Main CSV written to %s  (%d rows × %d cols)", output_path, len(df), len(df.columns))
    return df


def append_predictions_to_main_csv(
    raw_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: str = "data/main.csv",
    model_version: str = "1.0",
) -> pd.DataFrame:
    """Append inference-time predictions to (or create) *data/main.csv*.

    Merges *raw_df* (input features) with *predictions_df* (columns
    ``prediction_kw``, ``lower_bound``, ``upper_bound``,
    ``confidence_interval_width``) and appends the result to the CSV.
    If the file does not yet exist it is created.

    Args:
        raw_df:         Raw input DataFrame as supplied to ``make_predictions``.
        predictions_df: Output of ``make_predictions``.
        output_path:    Destination CSV path.
        model_version:  Version string for metadata column.

    Returns:
        The newly appended rows as a DataFrame.
    """
    n = min(len(raw_df), len(predictions_df))
    combined = raw_df.iloc[:n].copy().reset_index(drop=True)

    # rename ensemble prediction columns to 'ensemble_' prefix
    col_map = {
        "prediction_kw": "ensemble_prediction_kw",
        "lower_bound": "ensemble_lower_bound",
        "upper_bound": "ensemble_upper_bound",
        "confidence_interval_width": "ensemble_ci_width",
    }
    pred_renamed = predictions_df.iloc[:n].rename(columns=col_map).reset_index(drop=True)
    combined = pd.concat([combined, pred_renamed], axis=1)

    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    combined["timestamp"] = ts
    combined["model_version"] = model_version
    combined["split"] = "inference"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_header = not os.path.exists(output_path)
    combined.to_csv(output_path, mode="a", header=write_header, index=False)
    logger.info(
        "%s predictions to %s  (%d rows)",
        "Wrote" if write_header else "Appended",
        output_path,
        len(combined),
    )
    return combined


# ---------------------------------------------------------------------------
# Metrics summary JSON
# ---------------------------------------------------------------------------


def save_metrics_json(
    xgb_metrics: Optional[Dict[str, float]],
    lstm_metrics: Optional[Dict[str, float]],
    ensemble_metrics: Optional[Dict[str, float]],
    best_xgb_params: Optional[Dict] = None,
    output_path: str = "data/results/metrics_summary.json",
    model_version: str = "1.0",
) -> None:
    """Write model evaluation metrics to a structured JSON file.

    Args:
        xgb_metrics:      XGBoost metrics dict.
        lstm_metrics:     LSTM metrics dict.
        ensemble_metrics: Ensemble metrics dict.
        best_xgb_params:  Best Optuna hyperparameters (optional).
        output_path:      Destination path for the JSON file.
        model_version:    Version string written to the file.
    """
    summary = {
        "model_version": model_version,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "metrics": {
            "xgboost": xgb_metrics or {},
            "lstm": lstm_metrics or {},
            "ensemble": ensemble_metrics or {},
        },
        "hyperparameters": {
            "xgboost": best_xgb_params or {},
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Metrics summary written to %s", output_path)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", path)
    plt.close(fig)


def plot_model_comparison(
    y_actual: np.ndarray,
    xgb_preds: Optional[np.ndarray],
    lstm_preds: Optional[np.ndarray],
    ensemble_preds: Optional[np.ndarray],
    xgb_metrics: Optional[Dict[str, float]],
    lstm_metrics: Optional[Dict[str, float]],
    ensemble_metrics: Optional[Dict[str, float]],
    save_dir: str = "data/results",
    max_points: int = 500,
) -> None:
    """Plot predicted vs. actual values for all available models.

    Args:
        y_actual:         Ground-truth target values.
        xgb_preds:        XGBoost predictions (or ``None``).
        lstm_preds:       LSTM predictions (or ``None``).
        ensemble_preds:   Ensemble predictions (or ``None``).
        xgb_metrics:      XGBoost evaluation metrics.
        lstm_metrics:     LSTM evaluation metrics.
        ensemble_metrics: Ensemble evaluation metrics.
        save_dir:         Directory in which to save plots.
        max_points:       Cap on the number of scatter points (for readability).
    """
    models = []
    if xgb_preds is not None:
        models.append(("XGBoost", xgb_preds, PALETTE["xgboost"], xgb_metrics))
    if lstm_preds is not None:
        models.append(("LSTM", lstm_preds, PALETTE["lstm"], lstm_metrics))
    if ensemble_preds is not None:
        models.append(("Ensemble", ensemble_preds, PALETTE["ensemble"], ensemble_metrics))

    if not models:
        logger.warning("No predictions provided – skipping model comparison plot.")
        return

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    idx = np.random.choice(len(y_actual), size=min(max_points, len(y_actual)), replace=False)
    idx.sort()

    for ax, (name, preds, colour, metrics) in zip(axes[0], models):
        y_s = y_actual[idx]
        p_s = preds[idx]
        ax.scatter(y_s, p_s, alpha=0.4, s=15, color=colour, label=name)
        lim_min = min(y_s.min(), p_s.min())
        lim_max = max(y_s.max(), p_s.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="Perfect fit")
        r2 = (metrics or {}).get("r2_score", float("nan"))
        rmse = (metrics or {}).get("rmse", float("nan"))
        ax.set_title(f"{name}\nR²={r2:.4f}  RMSE={rmse:.1f} kW")
        ax.set_xlabel("Actual (kW)")
        ax.set_ylabel("Predicted (kW)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Model Comparison – Predicted vs Actual", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, os.path.join(save_dir, "model_comparison.png"))


def plot_time_series_predictions(
    y_actual: np.ndarray,
    xgb_preds: Optional[np.ndarray],
    lstm_preds: Optional[np.ndarray],
    ensemble_preds: Optional[np.ndarray],
    save_dir: str = "data/results",
    max_points: int = 500,
) -> None:
    """Plot actual vs. predicted values over time (sample index).

    Args:
        y_actual:       Ground-truth time series.
        xgb_preds:      XGBoost predictions.
        lstm_preds:     LSTM predictions.
        ensemble_preds: Ensemble predictions.
        save_dir:       Directory in which to save plots.
        max_points:     Number of consecutive samples to display.
    """
    n = min(max_points, len(y_actual))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y_actual[:n], color=PALETTE["actual"], lw=1.5, label="Actual", zorder=5)

    if xgb_preds is not None:
        ax.plot(x, xgb_preds[:n], color=PALETTE["xgboost"], lw=1, alpha=0.8, label="XGBoost")
    if lstm_preds is not None:
        ax.plot(x, lstm_preds[:n], color=PALETTE["lstm"], lw=1, alpha=0.8, label="LSTM")
    if ensemble_preds is not None:
        ax.plot(x, ensemble_preds[:n], color=PALETTE["ensemble"], lw=1.5, ls="--", label="Ensemble")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Time-Series Predictions vs Actual")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(save_dir, "time_series_predictions.png"))


def plot_residuals(
    y_actual: np.ndarray,
    xgb_preds: Optional[np.ndarray],
    lstm_preds: Optional[np.ndarray],
    ensemble_preds: Optional[np.ndarray],
    save_dir: str = "data/results",
) -> None:
    """Residual distribution and residuals-vs-fitted plots.

    Generates two figures:
    - ``residuals_distribution.png`` – KDE histogram of residuals per model.
    - ``residuals_vs_fitted.png``    – Scatter of residuals vs. fitted values.

    Args:
        y_actual:       Ground-truth values.
        xgb_preds:      XGBoost predictions.
        lstm_preds:     LSTM predictions.
        ensemble_preds: Ensemble predictions.
        save_dir:       Directory in which to save plots.
    """
    models = []
    if xgb_preds is not None:
        models.append(("XGBoost", xgb_preds, PALETTE["xgboost"]))
    if lstm_preds is not None:
        models.append(("LSTM", lstm_preds, PALETTE["lstm"]))
    if ensemble_preds is not None:
        models.append(("Ensemble", ensemble_preds, PALETTE["ensemble"]))

    if not models:
        return

    # --- residual distributions ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, preds, colour in models:
        residuals = y_actual - preds[: len(y_actual)]
        sns.kdeplot(residuals, ax=ax, label=name, color=colour, fill=True, alpha=0.3)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Residual (kW)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution by Model")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(save_dir, "residuals_distribution.png"))

    # --- residuals vs fitted -------------------------------------------------
    n_models = len(models)
    fig2, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    for ax, (name, preds, colour) in zip(axes[0], models):
        p = preds[: len(y_actual)]
        residuals = y_actual - p
        ax.scatter(p, residuals, alpha=0.3, s=12, color=colour)
        ax.axhline(0, color="black", lw=1.2, ls="--")
        ax.set_xlabel("Fitted value (kW)")
        ax.set_ylabel("Residual (kW)")
        ax.set_title(f"{name} – Residuals vs Fitted")
        ax.grid(alpha=0.3)
    fig2.tight_layout()
    _savefig(fig2, os.path.join(save_dir, "residuals_vs_fitted.png"))


def plot_metrics_bar(
    xgb_metrics: Optional[Dict[str, float]],
    lstm_metrics: Optional[Dict[str, float]],
    ensemble_metrics: Optional[Dict[str, float]],
    save_dir: str = "data/results",
) -> None:
    """Bar chart comparing R² and RMSE across models.

    Args:
        xgb_metrics:      XGBoost metrics dict.
        lstm_metrics:     LSTM metrics dict.
        ensemble_metrics: Ensemble metrics dict.
        save_dir:         Directory in which to save plots.
    """
    entries = []
    for name, metrics in [
        ("XGBoost", xgb_metrics),
        ("LSTM", lstm_metrics),
        ("Ensemble", ensemble_metrics),
    ]:
        if metrics:
            entries.append(
                {
                    "Model": name,
                    "R²": metrics.get("r2_score", float("nan")),
                    "RMSE (kW)": metrics.get("rmse", float("nan")),
                }
            )

    if not entries:
        return

    df = pd.DataFrame(entries)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colours = [PALETTE.get(n.lower(), "#607D8B") for n in df["Model"]]

    ax1.bar(df["Model"], df["R²"], color=colours)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("R²")
    ax1.set_title("R² Score by Model")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df["R²"]):
        ax1.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

    ax2.bar(df["Model"], df["RMSE (kW)"], color=colours)
    ax2.set_ylabel("RMSE (kW)")
    ax2.set_title("RMSE by Model")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df["RMSE (kW)"]):
        ax2.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("Model Performance Summary", fontsize=13)
    fig.tight_layout()
    _savefig(fig, os.path.join(save_dir, "metrics_bar_chart.png"))


def plot_error_distribution(
    y_actual: np.ndarray,
    ensemble_preds: np.ndarray,
    save_dir: str = "data/results",
) -> None:
    """Histogram of absolute prediction errors for the ensemble model.

    Args:
        y_actual:       Ground-truth values.
        ensemble_preds: Ensemble predictions.
        save_dir:       Directory in which to save plots.
    """
    errors = np.abs(y_actual - ensemble_preds[: len(y_actual)])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=50, color=PALETTE["ensemble"], edgecolor="white", alpha=0.85)
    ax.axvline(errors.mean(), color="red", lw=1.5, ls="--", label=f"Mean={errors.mean():.1f} kW")
    ax.axvline(
        np.percentile(errors, 95),
        color="orange",
        lw=1.5,
        ls=":",
        label=f"P95={np.percentile(errors, 95):.1f} kW",
    )
    ax.set_xlabel("Absolute Error (kW)")
    ax.set_ylabel("Count")
    ax.set_title("Ensemble Absolute Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(save_dir, "error_distribution.png"))


def generate_all_results(
    raw_df: pd.DataFrame,
    feature_cols: List[str],
    y_actual: np.ndarray,
    xgb_preds: Optional[np.ndarray],
    lstm_preds: Optional[np.ndarray],
    ensemble_preds: Optional[np.ndarray],
    xgb_metrics: Optional[Dict[str, float]],
    lstm_metrics: Optional[Dict[str, float]],
    ensemble_metrics: Optional[Dict[str, float]],
    feature_importance: Optional[Dict[str, float]] = None,
    split_labels: Optional[np.ndarray] = None,
    best_xgb_params: Optional[Dict] = None,
    results_dir: str = "data/results",
    main_csv_path: str = "data/main.csv",
    model_version: str = "1.0",
) -> pd.DataFrame:
    """Run the full export and visualisation pipeline.

    Calls, in order:
    1. :func:`build_main_csv`
    2. :func:`save_metrics_json`
    3. :func:`plot_model_comparison`
    4. :func:`plot_time_series_predictions`
    5. :func:`plot_residuals`
    6. :func:`plot_metrics_bar`
    7. :func:`plot_error_distribution` (ensemble only)
    8. Feature importance bar chart (if *feature_importance* is provided)

    Args:
        raw_df:             Engineered feature DataFrame aligned with
                            *y_actual*.
        feature_cols:       Names of numeric feature columns.
        y_actual:           Ground-truth target values.
        xgb_preds:          XGBoost predictions.
        lstm_preds:         LSTM predictions.
        ensemble_preds:     Ensemble predictions.
        xgb_metrics:        XGBoost evaluation metrics.
        lstm_metrics:       LSTM evaluation metrics.
        ensemble_metrics:   Ensemble evaluation metrics.
        feature_importance: ``{feature_name: importance_score}`` dict from
                            :class:`~src.analysis.feature_importance.FeatureImportanceAnalyzer`.
        split_labels:       Per-row split labels.
        best_xgb_params:    Best Optuna hyperparameters.
        results_dir:        Directory for PNG plots and JSON summary.
        main_csv_path:      Destination path for the consolidated CSV.
        model_version:      Version string for metadata fields.

    Returns:
        The consolidated DataFrame written to *main_csv_path*.
    """
    logger.info("=" * 60)
    logger.info("Generating results: CSV + visualisations")
    logger.info("=" * 60)

    # 1. Main CSV
    main_df = build_main_csv(
        raw_df=raw_df,
        feature_cols=feature_cols,
        y_actual=y_actual,
        xgb_preds=xgb_preds,
        lstm_preds=lstm_preds,
        ensemble_preds=ensemble_preds,
        xgb_metrics=xgb_metrics,
        lstm_metrics=lstm_metrics,
        ensemble_metrics=ensemble_metrics,
        split_labels=split_labels,
        output_path=main_csv_path,
        model_version=model_version,
    )

    # 2. Metrics JSON
    save_metrics_json(
        xgb_metrics=xgb_metrics,
        lstm_metrics=lstm_metrics,
        ensemble_metrics=ensemble_metrics,
        best_xgb_params=best_xgb_params,
        output_path=os.path.join(results_dir, "metrics_summary.json"),
        model_version=model_version,
    )

    # 3–7. Plots
    plot_model_comparison(
        y_actual, xgb_preds, lstm_preds, ensemble_preds,
        xgb_metrics, lstm_metrics, ensemble_metrics,
        save_dir=results_dir,
    )
    plot_time_series_predictions(
        y_actual, xgb_preds, lstm_preds, ensemble_preds,
        save_dir=results_dir,
    )
    plot_residuals(y_actual, xgb_preds, lstm_preds, ensemble_preds, save_dir=results_dir)
    plot_metrics_bar(xgb_metrics, lstm_metrics, ensemble_metrics, save_dir=results_dir)

    if ensemble_preds is not None:
        plot_error_distribution(y_actual, ensemble_preds, save_dir=results_dir)

    # 8. Feature importance (if provided)
    if feature_importance:
        _plot_feature_importance(feature_importance, save_dir=results_dir)

    logger.info("All results written to %s and %s", main_csv_path, results_dir)
    return main_df


def _plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    save_dir: str = "data/results",
) -> None:
    """Save a feature-importance bar chart from a pre-computed dict.

    Args:
        feature_importance: ``{feature_name: importance_score}`` sorted
                            descending.
        top_n:              Number of top features to display.
        save_dir:           Directory in which to save the plot.
    """
    items = list(feature_importance.items())[:top_n]
    names = [i[0] for i in items][::-1]
    scores = [i[1] for i in items][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
    ax.barh(names, scores, color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance (top {top_n})")
    ax.axvline(x=0, color="black", lw=0.8)
    fig.tight_layout()
    _savefig(fig, os.path.join(save_dir, "feature_importance.png"))
