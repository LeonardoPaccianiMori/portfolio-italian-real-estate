"""
Model training functionality for the rent prediction ML model.

This module provides functions for training a RandomForest regression model
to predict rent prices from real estate features. The model uses synthetic
rent data for training and can be saved/loaded for later use.

Key features:
1. RandomForest training with configurable parameters
2. Train-test split with reproducibility
3. Model persistence (save/load)
4. Log-transformed target handling

Author: Leonardo Pacciani-Mori
License: MIT
"""

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


# Default model parameters for RandomForest.
DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_leaf': 1,
    'n_jobs': -1,
    'random_state': 2025
}


def train_rent_model(
    X: pd.DataFrame,
    Y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 2025,
    model_params: Optional[Dict[str, Any]] = None
) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train a RandomForest model for rent prediction.

    This function trains a RandomForest regressor on the provided features
    and target. It performs a train-test split and returns both the trained
    model and the split data for evaluation.

    The target (Y) should already be log-transformed before calling this
    function to ensure positive predictions.

    Args:
        X: Feature DataFrame (after one-hot encoding).
        Y: Target Series (log-transformed prices).
        test_size: Fraction of data to use for testing. Default is 0.3.
        random_state: Random seed for reproducibility. Default is 2025.
        model_params: Dictionary of RandomForest parameters. If None,
            uses DEFAULT_MODEL_PARAMS.

    Returns:
        Tuple of (model, X_train, X_test, Y_train, Y_test):
        - model: Trained RandomForestRegressor
        - X_train: Training features
        - X_test: Test features
        - Y_train: Training target
        - Y_test: Test target

    Example:
        >>> model, X_train, X_test, Y_train, Y_test = train_rent_model(X, Y)
        >>> print(f"Model trained on {len(X_train)} samples")
    """
    # Uses default parameters if not specified.
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()
    else:
        # Merges with defaults to ensure all params are set.
        params = DEFAULT_MODEL_PARAMS.copy()
        params.update(model_params)
        model_params = params

    # Performs train-test split.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state
    )

    logger.info(
        f"Train-test split: {len(X_train)} train, {len(X_test)} test "
        f"({test_size*100:.0f}% test)"
    )

    # Creates and trains the model.
    model = RandomForestRegressor(**model_params)

    logger.info("Training RandomForest model...")
    model.fit(X_train, Y_train)

    logger.info(
        f"Model trained with {model_params['n_estimators']} estimators"
    )

    return model, X_train, X_test, Y_train, Y_test


def train_full_model(
    X: pd.DataFrame,
    Y: pd.Series,
    model_params: Optional[Dict[str, Any]] = None
) -> RandomForestRegressor:
    """
    Train a RandomForest model on the full dataset (no train-test split).

    This function is used when you want to train a final model on all
    available data, typically after hyperparameter tuning and evaluation.

    Args:
        X: Feature DataFrame (after one-hot encoding).
        Y: Target Series (log-transformed prices).
        model_params: Dictionary of RandomForest parameters. If None,
            uses DEFAULT_MODEL_PARAMS.

    Returns:
        Trained RandomForestRegressor model.

    Example:
        >>> final_model = train_full_model(X, Y)
        >>> save_model(final_model, 'rent_predictor.joblib')
    """
    # Uses default parameters if not specified.
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()
    else:
        params = DEFAULT_MODEL_PARAMS.copy()
        params.update(model_params)
        model_params = params

    # Creates and trains the model on full data.
    model = RandomForestRegressor(**model_params)

    logger.info(f"Training model on full dataset ({len(X)} samples)...")
    model.fit(X, Y)

    logger.info("Full model training complete")
    return model


def predict(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    inverse_log_transform: bool = True,
    round_to_decade: bool = True
) -> np.ndarray:
    """
    Make predictions using the trained model.

    This function generates predictions and optionally applies inverse
    log transformation and rounding.

    Args:
        model: Trained RandomForestRegressor model.
        X: Feature DataFrame for prediction.
        inverse_log_transform: Whether to apply expm1 to reverse log
            transformation. Default is True.
        round_to_decade: Whether to round predictions to nearest decade.
            Default is True.

    Returns:
        Array of predicted rent values.

    Example:
        >>> predictions = predict(model, X_test)
        >>> print(f"Mean predicted rent: {predictions.mean():.2f}")
    """
    # Makes raw predictions (log-transformed).
    Y_pred_log = model.predict(X)

    # Applies inverse transformation if requested.
    if inverse_log_transform:
        Y_pred = np.expm1(Y_pred_log)
    else:
        Y_pred = Y_pred_log

    # Rounds to nearest decade if requested.
    if round_to_decade:
        Y_pred = np.round(Y_pred / 10) * 10

    return Y_pred


def save_model(
    model: RandomForestRegressor,
    filepath: str,
    feature_names: Optional[list] = None
) -> None:
    """
    Save the trained model to disk.

    This function saves the model using joblib for efficient storage.
    Optionally, it can also save the feature names for later validation.

    Args:
        model: Trained RandomForestRegressor model.
        filepath: Path to save the model file.
        feature_names: Optional list of feature names to save alongside
            the model for validation.

    Example:
        >>> save_model(model, 'models/rent_predictor.joblib', X.columns.tolist())
    """
    # Creates directory if it doesn't exist.
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # Saves model and optionally feature names.
    save_dict = {'model': model}
    if feature_names is not None:
        save_dict['feature_names'] = feature_names

    joblib.dump(save_dict, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(
    filepath: str
) -> Tuple[RandomForestRegressor, Optional[list]]:
    """
    Load a trained model from disk.

    This function loads a model saved with save_model. If feature names
    were saved, they are also returned.

    Args:
        filepath: Path to the saved model file.

    Returns:
        Tuple of (model, feature_names). feature_names may be None if
        not saved.

    Example:
        >>> model, feature_names = load_model('models/rent_predictor.joblib')
    """
    saved_dict = joblib.load(filepath)

    # Handles both old and new save formats.
    if isinstance(saved_dict, dict):
        model = saved_dict['model']
        feature_names = saved_dict.get('feature_names')
    else:
        # Old format: just the model.
        model = saved_dict
        feature_names = None

    logger.info(f"Model loaded from {filepath}")
    return model, feature_names


def get_feature_importances(
    model: RandomForestRegressor,
    feature_names: list
) -> pd.DataFrame:
    """
    Get feature importances from the trained model.

    This function extracts the built-in feature importances from the
    RandomForest model and returns them as a sorted DataFrame.

    Note: These are Gini importances (MDI), not permutation importances.
    For more robust importance estimates, use permutation_importance from
    model_evaluator.

    Args:
        model: Trained RandomForestRegressor model.
        feature_names: List of feature names corresponding to X columns.

    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted by
        importance in descending order.

    Example:
        >>> importances = get_feature_importances(model, X.columns.tolist())
        >>> print(importances.head(10))
    """
    # Gets importances from model.
    importances = model.feature_importances_

    # Creates DataFrame and sorts.
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values(
        'importance',
        ascending=False
    ).reset_index(drop=True)

    return importance_df
