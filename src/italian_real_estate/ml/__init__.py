"""
Machine Learning module for the Italian Real Estate pipeline.

This module provides functionality for training a rent prediction model
using synthetic data and applying it to sale/auction listings.

The rent prediction workflow is:
1. Load synthetic data
2. Apply feature engineering (property mapping, heating/AC extraction)
3. Prepare data (one-hot encoding, outlier removal)
4. Train RandomForest model on rent listings
5. Evaluate model performance
6. Predict rent for sale/auction listings
7. Generate dashboard-ready data

Submodules:
    feature_engineering: Feature mapping and extraction utilities.
    data_preparation: One-hot encoding and data preparation utilities.
    model_trainer: RandomForest model training functionality.
    model_evaluator: Model evaluation metrics and visualization.
    rent_predictor: Rent prediction for sale/auction listings.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from .feature_engineering import (
    extract_heating_components,
    extract_air_conditioning_components,
    extract_window_info,
    apply_property_mapping,
    apply_heating_extraction,
    apply_air_conditioning_extraction,
    apply_window_frame_extraction,
    engineer_all_features,
    PROPERTY_MAPPING,
    PROPERTY_SECOND_MAPPING,
    HEATING_POWER_MAPPING,
    WINDOW_GLASS_MAPPING,
    WINDOW_COLUMN_MAPPINGS,
    WINDOW_COLUMNS,
)
from .data_preparation import (
    one_hot_encode_at_position,
    one_hot_encode_in_place,
    remove_outliers,
    cast_data_types,
    prepare_features_and_target,
    get_column_types,
    prepare_rent_training_data,
    DEFAULT_FLOAT_COLUMNS,
    DEFAULT_CATEGORICAL_COLUMNS,
)
from .model_trainer import (
    train_rent_model,
    train_full_model,
    predict,
    save_model,
    load_model,
    get_feature_importances,
    DEFAULT_MODEL_PARAMS,
)
from .model_evaluator import (
    evaluate_model,
    print_metrics,
    plot_predictions,
    plot_residuals,
    compute_permutation_importance,
    plot_feature_importance,
    full_evaluation,
)
from .rent_predictor import (
    predict_rent_for_listings,
    create_dashboard_data,
    get_rent_summary_by_region,
    get_rent_summary_by_listing_type,
    calculate_rent_to_price_ratio,
)

__all__ = [
    # Feature engineering
    "extract_heating_components",
    "extract_air_conditioning_components",
    "extract_window_info",
    "apply_property_mapping",
    "apply_heating_extraction",
    "apply_air_conditioning_extraction",
    "apply_window_frame_extraction",
    "engineer_all_features",
    "PROPERTY_MAPPING",
    "PROPERTY_SECOND_MAPPING",
    "HEATING_POWER_MAPPING",
    "WINDOW_GLASS_MAPPING",
    "WINDOW_COLUMN_MAPPINGS",
    "WINDOW_COLUMNS",
    # Data preparation
    "one_hot_encode_at_position",
    "one_hot_encode_in_place",
    "remove_outliers",
    "cast_data_types",
    "prepare_features_and_target",
    "get_column_types",
    "prepare_rent_training_data",
    "DEFAULT_FLOAT_COLUMNS",
    "DEFAULT_CATEGORICAL_COLUMNS",
    # Model training
    "train_rent_model",
    "train_full_model",
    "predict",
    "save_model",
    "load_model",
    "get_feature_importances",
    "DEFAULT_MODEL_PARAMS",
    # Model evaluation
    "evaluate_model",
    "print_metrics",
    "plot_predictions",
    "plot_residuals",
    "compute_permutation_importance",
    "plot_feature_importance",
    "full_evaluation",
    # Rent prediction
    "predict_rent_for_listings",
    "create_dashboard_data",
    "get_rent_summary_by_region",
    "get_rent_summary_by_listing_type",
    "calculate_rent_to_price_ratio",
]
