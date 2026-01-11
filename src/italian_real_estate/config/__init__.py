"""
Configuration module for the Italian Real Estate pipeline.

This module provides centralized configuration settings and logging setup
used throughout the pipeline.

Submodules:
    settings: All configuration constants, database parameters, and defaults.
    logging_config: Centralized logging configuration.
"""

from .settings import *
from .logging_config import setup_logging
