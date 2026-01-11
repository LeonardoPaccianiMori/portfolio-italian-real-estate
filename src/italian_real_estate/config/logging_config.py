"""
Logging configuration for the Italian Real Estate pipeline.

This module provides centralized logging setup used across all pipeline
components. It configures consistent log formatting, log levels, and
handlers for the entire application.

The logging configuration follows best practices:
    - Consistent timestamp format across all modules
    - Module name included in log messages for traceability
    - Log level indicator for quick scanning
    - Suppression of verbose third-party library logs

Usage:
    from italian_real_estate.config.logging_config import setup_logging, get_logger

    # At application startup
    setup_logging()

    # In each module
    logger = get_logger(__name__)
    logger.info("Processing started")

Author: Leonardo Pacciani-Mori
License: MIT
"""

import logging
import sys
from typing import Optional


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Default log format string.
# Includes: timestamp, logger name, log level, and the actual message.
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default date format for timestamps in log messages.
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default logging level for the application.
DEFAULT_LOG_LEVEL = logging.INFO


# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    suppress_third_party: bool = True
) -> None:
    """
    Configure logging for the entire application.

    This function sets up the root logger with consistent formatting and
    optionally suppresses verbose logging from third-party libraries that
    can clutter the log output.

    This function should be called once at application startup, before any
    logging calls are made. Subsequent calls will update the configuration.

    Args:
        level: The logging level threshold. Messages below this level will
            not be logged. Common values are logging.DEBUG, logging.INFO,
            logging.WARNING, logging.ERROR. Defaults to INFO.
        log_format: The format string for log messages. Uses Python's
            logging format syntax with placeholders like %(asctime)s.
        date_format: The strftime format string for timestamps in log
            messages. Defaults to ISO 8601 format without milliseconds.
        suppress_third_party: If True, sets third-party library loggers to
            WARNING level to reduce noise. This affects Airflow, PostgreSQL,
            and other commonly verbose libraries.

    Returns:
        None

    Example:
        >>> setup_logging(level=logging.DEBUG)
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message will now be shown")
    """
    # Configure the root logger with the specified settings.
    # This affects all loggers that don't have explicit handlers.
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # StreamHandler writes to stderr by default, which is appropriate
            # for container environments and Airflow task logs.
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Suppress verbose logging from third-party libraries if requested.
    # These libraries can generate excessive log output that obscures
    # our application's important messages.
    if suppress_third_party:
        _suppress_third_party_logging()


def _suppress_third_party_logging() -> None:
    """
    Suppress verbose logging from third-party libraries.

    This internal function sets specific third-party library loggers to
    WARNING level, reducing noise in the logs while still capturing
    important warnings and errors.

    The suppressed libraries include:
        - Airflow PostgreSQL and SQL hooks (very verbose during queries)
        - urllib3 (connection pool messages)
        - asyncio (event loop debugging)
        - aiohttp (HTTP client debugging)
        - selenium (WebDriver status messages)

    This function is called automatically by setup_logging() when
    suppress_third_party=True (the default).

    Returns:
        None
    """
    # List of third-party logger names to suppress.
    # These are set to WARNING to hide INFO and DEBUG messages.
    loggers_to_suppress = [
        # Airflow database hooks generate verbose SQL logging
        "airflow.providers.postgres.hooks.postgres",
        "airflow.providers.common.sql.hooks.sql",
        "airflow.providers.mongo.hooks.mongo",

        # HTTP libraries can be very chatty
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "aiohttp",

        # Async event loop debugging
        "asyncio",

        # Selenium WebDriver status messages
        "selenium",
        "selenium.webdriver",

        # MongoDB driver
        "pymongo",

        # TensorFlow GPU messages
        "tensorflow",
        "absl",
    ]

    # Set each logger to WARNING level
    for logger_name in loggers_to_suppress:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    This function provides a convenient way to get a properly configured
    logger for any module in the application. It should be called at the
    module level with __name__ as the argument.

    The returned logger inherits settings from the root logger configured
    by setup_logging(). Module-specific settings can be applied to the
    returned logger if needed.

    Args:
        name: The name of the logger, typically the module's __name__.
            If None, returns the root logger. Using __name__ creates a
            hierarchical logger structure that mirrors the package structure.

    Returns:
        logging.Logger: A configured logger instance for the specified name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing listing %d", listing_id)
        2024-01-15 10:30:45 - italian_real_estate.etl.extractors - INFO - Processing listing 12345
    """
    return logging.getLogger(name)


def set_log_level(level: int, logger_name: Optional[str] = None) -> None:
    """
    Dynamically change the log level for a specific logger or the root logger.

    This function allows runtime adjustment of log levels, which is useful
    for debugging specific components without restarting the application.

    Args:
        level: The new logging level to set. Common values are:
            - logging.DEBUG (10): Detailed diagnostic information
            - logging.INFO (20): General operational information
            - logging.WARNING (30): Warning messages for potential issues
            - logging.ERROR (40): Error messages for failures
        logger_name: The name of the logger to modify. If None, modifies
            the root logger which affects all loggers.

    Returns:
        None

    Example:
        >>> # Enable debug logging for the ETL module only
        >>> set_log_level(logging.DEBUG, "italian_real_estate.etl")
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)


# =============================================================================
# MODULE-LEVEL INITIALIZATION
# =============================================================================

# Create a module-level logger for this configuration module.
# This is useful for logging configuration-related messages.
_logger = get_logger(__name__)
