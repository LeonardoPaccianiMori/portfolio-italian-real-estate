"""
Model evaluation utilities for the rent prediction ML model.

This module provides functions for evaluating model performance including:
- Regression metrics (RMSE, MAE, MAPE, R2)
- Visualization (predicted vs actual, residuals)
- Permutation importance analysis

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def evaluate_model(
    Y_test: pd.Series,
    Y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.

    This function computes standard regression metrics comparing
    actual and predicted values. Metrics are computed on log-transformed
    values if the model was trained with log transformation.

    Args:
        Y_test: Actual target values (log-transformed if model uses log).
        Y_pred: Predicted target values (same scale as Y_test).

    Returns:
        Dictionary containing:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - mape: Mean Absolute Percentage Error (as decimal)
        - r2: R-squared (coefficient of determination)

    Example:
        >>> Y_pred = model.predict(X_test)
        >>> metrics = evaluate_model(Y_test, Y_pred)
        >>> print(f"R2: {metrics['r2']:.3f}")
        R2: 0.850
    """
    # Calculates all metrics.
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, Y_pred)
    mape = mean_absolute_percentage_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

    # Logs the results.
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MAPE: {mape*100:.2f}%")
    logger.info(f"R2: {r2:.4f}")

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics from evaluate_model.

    Example:
        >>> metrics = evaluate_model(Y_test, Y_pred)
        >>> print_metrics(metrics)
        RMSE: 0.25
        MAE: 0.14
        MAPE: 2.07%
    """
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']*100:.2f}%")
    print(f"R2: {metrics['r2']:.4f}")


def plot_predictions(
    Y_test: pd.Series,
    Y_pred: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot predicted vs actual values scatter plot.

    This function creates a scatter plot comparing predicted and actual
    values, with a diagonal reference line showing perfect predictions.
    The R2 score is displayed on the plot.

    Args:
        Y_test: Actual target values.
        Y_pred: Predicted target values.
        figsize: Figure size as (width, height). Default is (10, 6).
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot. Default is True.

    Example:
        >>> Y_pred = model.predict(X_test)
        >>> plot_predictions(Y_test, Y_pred, save_path='predictions.png')
    """
    # Calculates R2 for display on plot.
    r2 = r2_score(Y_test, Y_pred)

    # Creates the scatter plot.
    plt.figure(figsize=figsize)
    plt.scatter(Y_test, Y_pred, alpha=0.5, s=10)

    # Adds diagonal reference line (perfect predictions).
    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        'r--',
        label='Perfect prediction'
    )

    # Adds R2 annotation.
    # Positions text in upper left area of plot.
    x_pos = Y_test.min() + 0.1 * (Y_test.max() - Y_test.min())
    y_pos = Y_test.max() - 0.1 * (Y_test.max() - Y_test.min())
    plt.text(
        x_pos, y_pos,
        r'$\mathregular{R^2=}$' + f'{r2:.2f}',
        fontsize=14
    )

    # Labels and title.
    plt.xlabel('Actual rent (log-transformed)')
    plt.ylabel('Predicted rent (log-transformed)')
    plt.title('Predicted vs. Actual Rent Prices')

    plt.tight_layout()

    # Saves if path provided.
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved prediction plot to {save_path}")

    # Shows if requested.
    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals(
    Y_test: pd.Series,
    Y_pred: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot residuals (errors) vs predicted values.

    This function creates a scatter plot of residuals (actual - predicted)
    against predicted values. A well-fitted model should show residuals
    randomly scattered around zero.

    Patterns in the residual plot may indicate:
    - Heteroscedasticity (funnel shape)
    - Nonlinearity (curved pattern)
    - Outliers (extreme residuals)

    Args:
        Y_test: Actual target values.
        Y_pred: Predicted target values.
        figsize: Figure size as (width, height). Default is (10, 6).
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot. Default is True.

    Example:
        >>> Y_pred = model.predict(X_test)
        >>> plot_residuals(Y_test, Y_pred, save_path='residuals.png')
    """
    # Calculates residuals.
    residuals = Y_test - Y_pred

    # Creates the scatter plot.
    plt.figure(figsize=figsize)
    plt.scatter(Y_pred, residuals, alpha=0.5, s=10)

    # Adds horizontal reference line at zero.
    plt.axhline(y=0, color='r', linestyle='--', label='Zero residual')

    # Labels and title.
    plt.xlabel('Predicted rent')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')

    plt.tight_layout()

    # Saves if path provided.
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved residual plot to {save_path}")

    # Shows if requested.
    if show:
        plt.show()
    else:
        plt.close()


def compute_permutation_importance(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    n_repeats: int = 100,
    random_state: int = 2025,
    n_jobs: int = 1,
    scoring: str = 'r2'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute permutation importance for model features.

    Permutation importance measures the decrease in model performance
    when a single feature's values are randomly shuffled. This is more
    reliable than built-in feature importances for correlated features.

    Args:
        model: Trained RandomForestRegressor model.
        X_test: Test features DataFrame.
        Y_test: Test target Series.
        n_repeats: Number of times to permute each feature. Default is 100.
        random_state: Random seed for reproducibility. Default is 2025.
        n_jobs: Number of parallel jobs. Default is 1.
        scoring: Scoring metric to use. Default is 'r2'.

    Returns:
        Tuple of (importances_mean, importances_std) arrays.

    Example:
        >>> means, stds = compute_permutation_importance(model, X_test, Y_test)
        >>> print(f"Most important feature importance: {means.max():.4f}")
    """
    logger.info(f"Computing permutation importance ({n_repeats} repeats)...")

    # Computes permutation importance.
    result = permutation_importance(
        model, X_test, Y_test,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs
    )

    logger.info("Permutation importance computed")

    return result.importances_mean, result.importances_std


def plot_feature_importance(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    top_n: int = 10,
    n_repeats: int = 100,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> pd.DataFrame:
    """
    Plot permutation importance for top features.

    This function computes permutation importance and creates a horizontal
    bar chart showing the most important features.

    Args:
        model: Trained RandomForestRegressor model.
        X_test: Test features DataFrame.
        Y_test: Test target Series.
        top_n: Number of top features to show. Default is 10.
        n_repeats: Number of permutation repeats. Default is 100.
        figsize: Figure size as (width, height). Default is (8, 6).
        save_path: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot. Default is True.

    Returns:
        DataFrame with feature names and importance values.

    Example:
        >>> importance_df = plot_feature_importance(
        ...     model, X_test, Y_test,
        ...     top_n=15,
        ...     save_path='importance.png'
        ... )
    """
    # Computes permutation importance.
    importances_mean, importances_std = compute_permutation_importance(
        model, X_test, Y_test,
        n_repeats=n_repeats
    )

    # Gets sorted indices for top features.
    sorted_idx = importances_mean.argsort()[-top_n:]

    # Gets feature names and values for top features.
    top_features = [X_test.columns[i] for i in sorted_idx]
    top_importances = importances_mean[sorted_idx]
    top_stds = importances_std[sorted_idx]

    # Creates the bar plot.
    plt.figure(figsize=figsize)
    plt.barh(
        range(len(sorted_idx)),
        top_importances,
        xerr=top_stds,
        capsize=5
    )
    plt.yticks(range(len(sorted_idx)), top_features)
    plt.xlabel(r'Permutation Importance (R$\mathregular{{}^2}$)')
    plt.title(f'Top {top_n} Important Features')
    plt.tight_layout()

    # Saves if path provided.
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")

    # Shows if requested.
    if show:
        plt.show()
    else:
        plt.close()

    # Creates DataFrame with all features.
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': importances_mean,
        'importance_std': importances_std
    })
    importance_df = importance_df.sort_values(
        'importance_mean',
        ascending=False
    ).reset_index(drop=True)

    return importance_df


def full_evaluation(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    plot_dir: Optional[str] = None,
    show_plots: bool = True
) -> Dict[str, any]:
    """
    Perform full model evaluation with metrics and plots.

    This function runs a complete evaluation pipeline:
    1. Makes predictions
    2. Calculates all metrics
    3. Creates prediction and residual plots
    4. Computes and plots feature importance

    Args:
        model: Trained RandomForestRegressor model.
        X_test: Test features DataFrame.
        Y_test: Test target Series.
        plot_dir: Directory to save plots. If None, plots are not saved.
        show_plots: Whether to display plots. Default is True.

    Returns:
        Dictionary containing:
        - metrics: Dict of evaluation metrics
        - predictions: Array of predicted values
        - importance_df: DataFrame of feature importances

    Example:
        >>> results = full_evaluation(model, X_test, Y_test, plot_dir='plots/')
        >>> print(f"R2: {results['metrics']['r2']:.3f}")
    """
    # Makes predictions.
    Y_pred = model.predict(X_test)

    # Calculates metrics.
    metrics = evaluate_model(Y_test, Y_pred)

    # Determines save paths.
    pred_path = f"{plot_dir}/predicted_vs_actual.png" if plot_dir else None
    resid_path = f"{plot_dir}/residuals.png" if plot_dir else None
    import_path = f"{plot_dir}/feature_importance.png" if plot_dir else None

    # Creates plots.
    plot_predictions(Y_test, Y_pred, save_path=pred_path, show=show_plots)
    plot_residuals(Y_test, Y_pred, save_path=resid_path, show=show_plots)
    importance_df = plot_feature_importance(
        model, X_test, Y_test,
        save_path=import_path,
        show=show_plots
    )

    return {
        'metrics': metrics,
        'predictions': Y_pred,
        'importance_df': importance_df
    }
