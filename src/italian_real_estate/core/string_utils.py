"""
String manipulation utilities for the Italian Real Estate pipeline.

This module provides functions for cleaning, formatting, and transforming
strings used throughout the data pipeline. These utilities handle common
data quality issues like extracting numbers from formatted currency strings
and preprocessing text for translation.

Functions in this module are designed to be robust against malformed input
and return None or the original value when processing fails.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import re
from typing import Optional


def remove_non_numbers(string: str) -> Optional[str]:
    """
    Extract numeric characters from a string for conversion to int/float.

    This function processes strings that contain formatted numbers (such as
    currency values) and extracts only the numeric portion. It handles common
    European number formatting conventions where commas are used as decimal
    separators.

    The function is commonly used to parse price values like "€300.000,00"
    into a string "300000.00" that can be converted to a numeric type.

    Processing steps:
        1. Replace commas with dots (European decimal notation)
        2. Extract only digits and decimal points
        3. Return None if no numeric content found

    Args:
        string: The input string potentially containing formatted numbers.
            Examples: "€300.000,00", "120 mq", "15,5%"

    Returns:
        Optional[str]: A string containing only numeric characters and decimal
            points, suitable for conversion to int or float. Returns None if
            the input is empty or contains no numeric characters.

    Example:
        >>> remove_non_numbers("€300.000,00")
        '300000.00'
        >>> remove_non_numbers("120 mq")
        '120'
        >>> remove_non_numbers("no numbers here")
        None
    """
    # Handle None or empty input
    if not string:
        return None

    # Replace commas with dots to normalize decimal notation.
    # In European formatting, "300.000,50" means 300,000.50 in US format.
    string = string.replace(",", ".")

    # Build a new string containing only numeric characters and decimal points.
    # We iterate character by character to filter out currency symbols, spaces,
    # and other non-numeric characters.
    temp_string = ""
    for char in string:
        if char.isnumeric() or char == ".":
            temp_string += char

    # Return None if no numeric content was found.
    # This handles cases like "N/A" or purely text strings.
    if temp_string == "":
        return None

    return temp_string


def preprocess_text(text: str) -> str:
    """
    Preprocess text for improved translation quality.

    This function cleans and normalizes Italian text before sending it to
    the translation API. Preprocessing improves translation accuracy by:
        - Ensuring consistent spacing around punctuation
        - Handling common abbreviations and special characters
        - Normalizing whitespace

    The preprocessing is specifically designed for Italian real estate
    descriptions and handles domain-specific patterns.

    Processing steps:
        1. Add spaces after punctuation marks if missing
        2. Normalize spaces around slashes and hyphens
        3. Collapse multiple spaces into single spaces
        4. Strip leading and trailing whitespace

    Args:
        text: The Italian text to preprocess. Can contain property
            descriptions, addresses, or feature lists.

    Returns:
        str: The preprocessed text with normalized spacing and punctuation.
            Returns the original text if input is None or not a string.

    Example:
        >>> preprocess_text("Appartamento,3 locali/2 bagni")
        'Appartamento, 3 locali / 2 bagni'
        >>> preprocess_text("Via Roma,15-Milano")
        'Via Roma, 15 - Milano'
    """
    # Handle None or non-string input gracefully
    if not text or not isinstance(text, str):
        return text if text else ""

    # Add a space after punctuation marks if followed directly by a letter.
    # This fixes cases like "word,word" to "word, word"
    # Matches: period, comma, colon, semicolon, exclamation, question mark
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)

    # Add spaces around slashes for readability.
    # Changes "3 locali/2 bagni" to "3 locali / 2 bagni"
    text = re.sub(r'(\S)/(\S)', r'\1 / \2', text)

    # Add spaces around hyphens when connecting words (not in compound words).
    # This helps with Italian addresses like "Via Roma-Centro"
    # But preserves compound words if they have spaces
    text = re.sub(r'(\S)-(\S)', r'\1 - \2', text)

    # Normalize multiple spaces into single spaces.
    # This handles cases where previous substitutions created double spaces.
    text = re.sub(r' +', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def clean_province_name(name: str) -> str:
    """
    Normalize province names for consistent storage and comparison.

    Italian province names can appear in different formats across data sources.
    This function normalizes them to a consistent lowercase, hyphen-separated
    format suitable for use as keys in dictionaries and database lookups.

    Processing steps:
        1. Convert to lowercase
        2. Replace spaces with hyphens
        3. Handle special characters in province names

    Args:
        name: The province name to normalize. Can be in mixed case with
            spaces or other separators.

    Returns:
        str: The normalized province name in lowercase with hyphens.

    Example:
        >>> clean_province_name("L'Aquila")
        "l'aquila"
        >>> clean_province_name("Reggio Emilia")
        'reggio-emilia'
        >>> clean_province_name("MILANO")
        'milano'
    """
    if not name:
        return ""

    # Convert to lowercase for case-insensitive matching
    normalized = name.lower()

    # Replace spaces with hyphens for URL-friendly format
    normalized = normalized.replace(" ", "-")

    return normalized


def clean_region_name(name: str) -> str:
    """
    Normalize region names for consistent storage and comparison.

    Similar to clean_province_name(), this function normalizes Italian region
    names to a consistent format. Region names are used for grouping and
    filtering data across the pipeline.

    Args:
        name: The region name to normalize.

    Returns:
        str: The normalized region name in lowercase with hyphens.

    Example:
        >>> clean_region_name("Emilia Romagna")
        'emilia-romagna'
        >>> clean_region_name("FRIULI VENEZIA GIULIA")
        'friuli-venezia-giulia'
    """
    # Region names follow the same normalization rules as provinces
    return clean_province_name(name)


def extract_first_value_before_separator(text: str, separator: str = "|") -> str:
    """
    Extract the first value from a separator-delimited string.

    Some fields in the source data contain multiple values separated by a
    delimiter (commonly "|"). This function extracts just the first value,
    which is typically the most relevant.

    Args:
        text: The delimited string to process.
        separator: The delimiter character. Defaults to "|".

    Returns:
        str: The first value before the separator, stripped of whitespace.
            Returns the original text if no separator is found.

    Example:
        >>> extract_first_value_before_separator("apartment | villa")
        'apartment'
        >>> extract_first_value_before_separator("single value")
        'single value'
    """
    if not text:
        return ""

    # Split on the separator and take the first part
    parts = text.split(separator, maxsplit=1)

    # Return the first part, stripped of whitespace
    return parts[0].strip()


def safe_lower(text: str) -> Optional[str]:
    """
    Safely convert text to lowercase, handling None and non-string values.

    This utility function provides null-safe lowercase conversion, which is
    commonly needed when processing data that may contain missing values.

    Args:
        text: The text to convert to lowercase. Can be None.

    Returns:
        Optional[str]: The lowercase string, or None if input was None.

    Example:
        >>> safe_lower("APARTMENT")
        'apartment'
        >>> safe_lower(None)
        None
    """
    if text is None:
        return None

    if not isinstance(text, str):
        return str(text).lower()

    return text.lower()


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length with an optional suffix.

    Useful for logging and display purposes where long strings need to be
    shortened while indicating truncation occurred.

    Args:
        text: The string to potentially truncate.
        max_length: The maximum length of the result including suffix.
        suffix: String to append when truncation occurs. Defaults to "...".

    Returns:
        str: The original string if shorter than max_length, otherwise
            a truncated version with the suffix appended.

    Example:
        >>> truncate_string("This is a long description", 15)
        'This is a lo...'
    """
    if not text or len(text) <= max_length:
        return text or ""

    # Calculate how many characters from the original we can keep
    truncate_at = max_length - len(suffix)

    if truncate_at <= 0:
        return suffix[:max_length]

    return text[:truncate_at] + suffix
