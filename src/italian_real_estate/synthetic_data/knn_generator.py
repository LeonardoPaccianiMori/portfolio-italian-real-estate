"""
KNN-based synthetic data generation using TensorFlow GPU acceleration.

This module provides functions for generating synthetic real estate listings
using K-Nearest Neighbors interpolation. The generation uses TensorFlow
for GPU-accelerated distance computation and feature interpolation.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import gc
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm

from ..config.logging_config import get_logger
from ..config.settings import SYNTHETIC_DATA_CHUNK_SIZE, SYNTHETIC_DATA_BATCH_SIZE
from .gpu_utils import configure_gpu_memory_growth, clear_gpu_memory

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def process_dataset(
    data: pd.DataFrame,
    name: str,
    numerical_columns: List[str],
    categorical_columns: List[str]
) -> Tuple[np.ndarray, Dict, Any, Any, Dict]:
    """
    Process a dataset for KNN synthetic data generation.

    This function creates a preprocessing pipeline that:
    1. Standardizes numerical columns using StandardScaler
    2. One-hot encodes categorical columns using OneHotEncoder

    Args:
        data: The DataFrame to process.
        name: Name identifier for the dataset (e.g., 'rent', 'auction', 'sale').
        numerical_columns: List of numerical column names.
        categorical_columns: List of categorical column names.

    Returns:
        Tuple containing:
        - transformed_data: The transformed dataset as numpy array
        - cat_column_indices: Mapping of categorical columns to their indices
        - dataset_preprocessor: The fitted ColumnTransformer
        - num_transformer: The numerical transformer component (StandardScaler)
        - cat_categories: Mapping of categorical columns to their encoder categories

    Example:
        >>> transformed, cat_idx, preprocessor, num_trans, cats = process_dataset(
        ...     rent_data, 'rent', num_cols, cat_cols
        ... )
    """
    logger.info(f"Processing {name} dataset...")

    # Creates preprocessor with numerical and categorical transformers.
    dataset_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
             categorical_columns)
        ],
        remainder='drop'
    )

    # Fits and transforms the data.
    transformed_data = dataset_preprocessor.fit_transform(data)

    # Extracts numerical transformer for inverse transforms.
    num_transformer = dataset_preprocessor.named_transformers_['num']

    # Extracts category info for decoding one-hot back to categories.
    cat_transformer = dataset_preprocessor.named_transformers_['cat']
    cat_column_indices: Dict[str, List[int]] = {}
    cat_categories: Dict[str, np.ndarray] = {}
    current_idx = len(numerical_columns)

    for i, col in enumerate(categorical_columns):
        n_categories = len(cat_transformer.categories_[i])
        cat_column_indices[col] = list(range(current_idx, current_idx + n_categories))
        cat_categories[col] = cat_transformer.categories_[i]
        current_idx += n_categories

    logger.info(
        f"Processed {name}: {transformed_data.shape[0]} samples, "
        f"{transformed_data.shape[1]} features"
    )

    return (
        transformed_data,
        cat_column_indices,
        dataset_preprocessor,
        num_transformer,
        cat_categories
    )


def generate_synthetic_data(
    data: pd.DataFrame,
    transformed_data: np.ndarray,
    distance_data: np.ndarray,
    n_synthetic_samples: int,
    cat_column_indices: Dict[str, List[int]],
    cat_categories: Dict[str, np.ndarray],
    num_transformer: Any,
    numerical_columns: List[str],
    categorical_columns: List[str],
    batch_size: int = 100,
    k: int = 5,
    disable_progress: bool = False
) -> pd.DataFrame:
    """
    Generate synthetic data using vectorized TensorFlow GPU operations.

    This function generates synthetic listings by:
    1. Randomly sampling seed points from the original data
    2. Finding K nearest neighbors based on distance features
    3. Interpolating features using inverse-distance weighting
    4. Decoding one-hot back to categorical values

    Args:
        data: Original pandas DataFrame.
        transformed_data: Transformed data as numpy array (all features).
        distance_data: Data used for distance calculations (subset of features).
        n_synthetic_samples: Total number of synthetic samples to generate.
        cat_column_indices: Mapping of categorical columns to indices.
        cat_categories: Mapping of categorical columns to their categories.
        num_transformer: Transformer for inverting numerical features.
        numerical_columns: List of numerical column names.
        categorical_columns: List of categorical column names.
        batch_size: Number of samples to process in each GPU batch.
        k: Number of nearest neighbors to use.
        disable_progress: Whether to disable the progress bar.

    Returns:
        DataFrame with synthetic samples.

    Example:
        >>> synthetic_df = generate_synthetic_data(
        ...     rent_data, transformed, distance_data, 10000,
        ...     cat_indices, cat_categories, num_transformer,
        ...     num_cols, cat_cols
        ... )
    """
    # Checks GPU availability.
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    logger.info(f"Using TensorFlow {tf.__version__} with GPU: {gpu_available}")

    # Converts data to TensorFlow tensors.
    tf_distance_data = tf.convert_to_tensor(distance_data, dtype=tf.float32)
    tf_transformed_data = tf.convert_to_tensor(transformed_data, dtype=tf.float32)

    n_samples = tf.shape(tf_distance_data)[0]
    num_batches = (n_synthetic_samples + batch_size - 1) // batch_size

    synthetic_batches = []

    # Processes in batches to manage memory.
    for batch_idx in tqdm(
        range(num_batches),
        desc="Processing batches",
        leave=False,
        disable=disable_progress
    ):
        # Determines current batch size.
        current_batch_size = tf.minimum(
            batch_size,
            n_synthetic_samples - batch_idx * batch_size
        )

        # Randomly samples seed indices from original dataset.
        seed_indices = tf.random.uniform(
            shape=[current_batch_size],
            minval=0,
            maxval=n_samples,
            dtype=tf.int32
        )
        tf_seed_points = tf.gather(tf_distance_data, seed_indices)

        # Computes pairwise squared Euclidean distances.
        expanded_seeds = tf.expand_dims(tf_seed_points, axis=1)
        expanded_data = tf.expand_dims(tf_distance_data, axis=0)
        squared_diff = tf.square(expanded_seeds - expanded_data)
        tf_dists = tf.reduce_sum(squared_diff, axis=2)

        # Gets indices and distances of K-nearest neighbors.
        _, knn_indices = tf.nn.top_k(-tf_dists, k=k)
        knn_dists = tf.gather(tf_dists, knn_indices, batch_dims=1)

        # Gathers transformed data for the K neighbors.
        neighbors_transformed = tf.gather(tf_transformed_data, knn_indices)

        # Computes inverse-distance weights.
        eps = 1e-6
        weights = 1.0 / (knn_dists + eps)
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

        # Processes numerical features (weighted average).
        num_numerical = len(numerical_columns)
        neighbors_num = neighbors_transformed[:, :, :num_numerical]
        weights_exp = tf.expand_dims(weights, axis=2)
        synthetic_num = tf.reduce_sum(neighbors_num * weights_exp, axis=1)

        # Processes categorical features (weighted vote).
        synthetic_cat_list = []
        for col in categorical_columns:
            col_idx = cat_column_indices[col]
            # Gathers one-hot vectors for this categorical feature.
            neighbors_cat = tf.gather(neighbors_transformed, col_idx, axis=2)
            # Weighted sum across K neighbors.
            weighted_cat = tf.reduce_sum(
                neighbors_cat * tf.expand_dims(weights, axis=2),
                axis=1
            )
            # Picks index with maximum weighted vote.
            pred_cat_idx = tf.argmax(weighted_cat, axis=1, output_type=tf.int32)
            # Converts to one-hot.
            one_hot_cat = tf.one_hot(
                pred_cat_idx,
                depth=len(col_idx),
                dtype=tf.float32
            )
            synthetic_cat_list.append(one_hot_cat)

        # Concatenates numerical and categorical parts.
        if synthetic_cat_list:
            synthetic_cat_concat = tf.concat(synthetic_cat_list, axis=1)
            synthetic_batch = tf.concat([synthetic_num, synthetic_cat_concat], axis=1)
        else:
            synthetic_batch = synthetic_num

        # Converts to numpy immediately to free GPU memory from intermediate tensors.
        # This prevents OOM errors during long batch processing runs.
        synthetic_batches.append(synthetic_batch.numpy())

        # Clears GPU memory periodically to prevent accumulation.
        # Every 50 batches, run garbage collection and clear TensorFlow session.
        if (batch_idx + 1) % 50 == 0:
            gc.collect()
            tf.keras.backend.clear_session()

    # Concatenates all batches (now numpy arrays).
    synthetic_np = np.concatenate(synthetic_batches, axis=0)

    # Inverse transforms numerical features.
    synthetic_num_inv = num_transformer.inverse_transform(
        synthetic_np[:, :num_numerical]
    )

    # Decodes categorical features from one-hot.
    synthetic_cat_outputs = []
    start = num_numerical

    for col in categorical_columns:
        depth = len(cat_column_indices[col])
        cat_one_hot = synthetic_np[:, start:start + depth]
        # Takes argmax for each row.
        pred_indices = np.argmax(cat_one_hot, axis=1)

        col_categories = []
        for idx in pred_indices:
            categories = cat_categories[col]
            if idx < len(categories):
                col_categories.append(categories[idx])
            else:
                # Falls back to random category if index out of bounds.
                col_categories.append(np.random.choice(categories))

        synthetic_cat_outputs.append(np.array(col_categories).reshape(-1, 1))
        start += depth

    # Combines numerical and categorical into final array.
    synthetic_data_combined = np.concatenate(
        [synthetic_num_inv] + synthetic_cat_outputs,
        axis=1
    )

    # Creates DataFrame with original column order.
    synthetic_df = pd.DataFrame(
        synthetic_data_combined,
        columns=numerical_columns + categorical_columns
    )

    return synthetic_df


def generate_synthetic_data_in_chunks(
    data: pd.DataFrame,
    transformed_data: np.ndarray,
    distance_data: np.ndarray,
    n_synthetic_samples: int,
    cat_column_indices: Dict[str, List[int]],
    cat_categories: Dict[str, np.ndarray],
    num_transformer: Any,
    numerical_columns: List[str],
    categorical_columns: List[str],
    chunk_size: int = SYNTHETIC_DATA_CHUNK_SIZE,
    batch_size: int = SYNTHETIC_DATA_BATCH_SIZE
) -> pd.DataFrame:
    """
    Generate synthetic data in smaller chunks to avoid memory issues.

    This function wraps generate_synthetic_data to process data in
    manageable chunks, clearing GPU memory between chunks. This is
    essential for generating large numbers of synthetic samples.

    Args:
        data: Original pandas DataFrame.
        transformed_data: Transformed data as numpy array.
        distance_data: Data for distance calculations.
        n_synthetic_samples: Total synthetic samples to generate.
        cat_column_indices: Mapping of categorical columns to indices.
        cat_categories: Mapping of categorical columns to categories.
        num_transformer: Transformer for inverting numerical features.
        numerical_columns: List of numerical column names.
        categorical_columns: List of categorical column names.
        chunk_size: Number of samples to generate per chunk.
        batch_size: Number of samples per GPU batch within a chunk.

    Returns:
        DataFrame with all synthetic samples concatenated.

    Example:
        >>> synthetic_df = generate_synthetic_data_in_chunks(
        ...     rent_data, transformed, distance_data, 80000,
        ...     cat_indices, cat_categories, num_transformer,
        ...     num_cols, cat_cols, chunk_size=5000, batch_size=25
        ... )
    """
    results = []

    # Configures GPU memory growth.
    configure_gpu_memory_growth()

    # Calculates number of chunks needed.
    num_chunks = (n_synthetic_samples + chunk_size - 1) // chunk_size

    # Processes each chunk.
    for i in tqdm(range(num_chunks), desc="Processing chunks", position=0):
        # Calculates size of current chunk.
        current_chunk_size = min(chunk_size, n_synthetic_samples - i * chunk_size)

        logger.info(
            f"Generating chunk {i + 1}/{num_chunks} with {current_chunk_size} samples..."
        )

        # Generates this chunk.
        chunk_data = generate_synthetic_data(
            data,
            transformed_data,
            distance_data,
            current_chunk_size,
            cat_column_indices,
            cat_categories,
            num_transformer,
            numerical_columns,
            categorical_columns,
            batch_size=batch_size,
            disable_progress=False
        )

        # Appends to results.
        results.append(chunk_data)

        # Clears GPU memory.
        clear_gpu_memory()

        logger.info(f"Chunk {i + 1}/{num_chunks} completed.")

    # Concatenates all chunks.
    logger.info("Combining chunks...")
    return pd.concat(results, ignore_index=True)
