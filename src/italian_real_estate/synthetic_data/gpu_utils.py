"""
GPU memory management utilities for TensorFlow.

This module provides functions for managing GPU memory when using
TensorFlow for synthetic data generation. It helps prevent out-of-memory
errors by configuring memory growth and clearing memory between batches.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import gc

import tensorflow as tf

from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def configure_gpu_memory_growth() -> None:
    """
    Configure GPU memory growth to avoid pre-allocating all memory.

    By default, TensorFlow allocates all available GPU memory at startup.
    This function enables memory growth mode, which allocates memory
    incrementally as needed. This is important when processing data
    in chunks to avoid running out of memory.

    Side Effects:
        Modifies TensorFlow's GPU memory allocation settings.

    Example:
        >>> configure_gpu_memory_growth()
        Memory growth enabled for 1 GPU(s)
    """
    # Gets list of available GPUs.
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enables memory growth for each GPU.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"Memory growth enabled for {len(gpus)} GPU(s)")

        except RuntimeError as e:
            # Memory growth must be set before GPU initialization.
            logger.error(f"Error configuring GPU memory growth: {e}")
    else:
        logger.warning("No GPUs available. Running on CPU only.")


def clear_gpu_memory() -> None:
    """
    Aggressively clear GPU memory.

    This function clears the Keras backend session, forces garbage
    collection, and resets GPU memory statistics. It should be called
    between chunks during synthetic data generation to free up memory.

    Side Effects:
        - Clears Keras backend session
        - Forces Python garbage collection
        - Resets TensorFlow memory statistics

    Example:
        >>> # After processing a batch of data
        >>> clear_gpu_memory()
    """
    # Clears Keras backend session, which releases GPU memory.
    tf.keras.backend.clear_session()

    # Forces Python garbage collection to free up memory.
    gc.collect()

    # Tries to reset memory statistics if available.
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.reset_memory_stats(gpu)
        except Exception:
            # This might not be available in all TF versions.
            pass


def get_gpu_info() -> dict:
    """
    Get information about available GPUs.

    Returns:
        Dictionary containing:
        - gpu_available: Whether GPU is available
        - gpu_count: Number of GPUs
        - gpu_names: List of GPU device names
        - tensorflow_version: TensorFlow version string

    Example:
        >>> info = get_gpu_info()
        >>> print(f"GPUs available: {info['gpu_available']}")
    """
    gpus = tf.config.list_physical_devices('GPU')

    info = {
        'gpu_available': len(gpus) > 0,
        'gpu_count': len(gpus),
        'gpu_names': [gpu.name for gpu in gpus],
        'tensorflow_version': tf.__version__
    }

    return info
