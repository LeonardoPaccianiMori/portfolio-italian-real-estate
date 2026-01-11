"""
Date conversion utilities for the Italian Real Estate pipeline.

This module provides functions for parsing, converting, and manipulating
date values used throughout the data pipeline. It handles the various date
formats found in MongoDB documents and web scraped data, converting them
to Python datetime objects suitable for PostgreSQL storage.

The primary challenge is that dates in the source data appear in multiple
formats (ISO format, European format) and need to be normalized for
consistent processing and storage.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import datetime
from typing import Optional, Union
from dateutil import parser as dateutil_parser


def convert_mongo_date(date_str: str) -> Optional[datetime.date]:
    """
    Convert date strings from MongoDB into Python date objects.

    This function handles the various date formats found in the MongoDB
    warehouse and converts them to Python date objects suitable for
    PostgreSQL storage. It attempts multiple parsing strategies in order
    of likelihood.

    Supported formats:
        - ISO format: "2024-01-15" (YYYY-MM-DD)
        - European format: "15/01/2024" (DD/MM/YYYY)
        - Abbreviated year: "15/01/24" (DD/MM/YY)

    The function is designed to be robust against malformed input and
    returns None for unparseable dates rather than raising exceptions.

    Args:
        date_str: A string representation of a date. Can be in ISO format
            (YYYY-MM-DD) or European format (DD/MM/YYYY).

    Returns:
        Optional[datetime.date]: A Python date object representing the
            parsed date, or None if the string couldn't be parsed or was
            empty/None.

    Example:
        >>> convert_mongo_date("2024-01-15")
        datetime.date(2024, 1, 15)
        >>> convert_mongo_date("15/01/2024")
        datetime.date(2024, 1, 15)
        >>> convert_mongo_date(None)
        None
        >>> convert_mongo_date("invalid")
        None
    """
    # Handle None or empty input
    if not date_str:
        return None

    # Attempt to parse as ISO format first (most common in our data)
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        pass  # Try the next format

    # Attempt to parse as European format DD/MM/YYYY
    try:
        return datetime.datetime.strptime(date_str, '%d/%m/%Y').date()
    except (ValueError, TypeError):
        pass  # Try the next format

    # Attempt to parse as abbreviated European format DD/MM/YY
    try:
        return datetime.datetime.strptime(date_str, '%d/%m/%y').date()
    except (ValueError, TypeError):
        pass  # Unable to parse

    # Return None if all parsing attempts failed
    return None


def calculate_date_difference_days(
    date1: Union[str, datetime.date],
    date2: Union[str, datetime.date]
) -> Optional[int]:
    """
    Calculate the difference in days between two dates.

    This function computes the number of days between two dates, where
    date1 is subtracted from date2. The result is positive if date2 is
    after date1, negative if before.

    This is commonly used to calculate:
        - Listing age (scraping_date - creation_date)
        - Time since last update (scraping_date - last_update_date)

    Args:
        date1: The first (earlier) date, either as a string or date object.
        date2: The second (later) date, either as a string or date object.

    Returns:
        Optional[int]: The number of days between the dates, or None if
            either date couldn't be parsed.

    Example:
        >>> calculate_date_difference_days("2024-01-01", "2024-01-15")
        14
        >>> calculate_date_difference_days("2024-01-15", "2024-01-01")
        -14
    """
    # Convert string dates to date objects if needed
    if isinstance(date1, str):
        date1 = convert_mongo_date(date1)
    if isinstance(date2, str):
        date2 = convert_mongo_date(date2)

    # Return None if either conversion failed
    if date1 is None or date2 is None:
        return None

    # Calculate and return the difference in days
    delta = date2 - date1
    return delta.days


def parse_work_dates(work_dates_str: str) -> tuple:
    """
    Parse construction work dates from the workDates field.

    The workDates field in property listings contains construction start
    and optionally end dates in the format "DD/MM/YYYY - DD/MM/YYYY" or
    just "DD/MM/YYYY" for ongoing projects.

    Args:
        work_dates_str: A string containing one or two dates separated by
            " - ". For example: "01/03/2023 - 15/09/2024" or "01/03/2023".

    Returns:
        tuple: A tuple (start_date, end_date) where each element is either
            a date string in ISO format (YYYY-MM-DD) or None.

    Example:
        >>> parse_work_dates("01/03/2023 - 15/09/2024")
        ('2023-03-01', '2024-09-15')
        >>> parse_work_dates("01/03/2023")
        ('2023-03-01', None)
    """
    if not work_dates_str:
        return None, None

    # Split on the separator and clean up whitespace
    dates = work_dates_str.replace(" ", "").split("-")

    work_start_date = None
    work_end_date = None

    if len(dates) >= 1 and dates[0]:
        # Parse the start date
        date_parts = dates[0].split("/")
        if len(date_parts) == 3:
            # Convert DD/MM/YYYY to YYYY-MM-DD format
            work_start_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"

    if len(dates) >= 2 and dates[1]:
        # Parse the end date
        date_parts = dates[1].split("/")
        if len(date_parts) == 3:
            # Convert DD/MM/YYYY to YYYY-MM-DD format
            work_end_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"

    return work_start_date, work_end_date


def calculate_work_completion(
    scraping_date: str,
    work_start_date: str,
    work_end_date: str
) -> Optional[float]:
    """
    Calculate construction work completion percentage.

    This function calculates how far along construction work is, expressed
    as a ratio where:
        - 0.0 = work just started
        - 1.0 = work completed
        - negative values = work hasn't started yet
        - values > 1.0 = past the expected completion date

    Args:
        scraping_date: The date the data was scraped (YYYY-MM-DD format).
        work_start_date: The construction start date (YYYY-MM-DD format).
        work_end_date: The expected completion date (YYYY-MM-DD format).

    Returns:
        Optional[float]: The completion ratio, or None if dates are invalid
            or if the total work duration is zero.

    Example:
        >>> calculate_work_completion("2024-06-15", "2024-01-01", "2024-12-31")
        0.46  # About halfway through
    """
    # Return None if any date is missing
    if not all([scraping_date, work_start_date, work_end_date]):
        return None

    # Parse dates using dateutil for flexibility
    try:
        scrape_date = dateutil_parser.parse(scraping_date, dayfirst=True)
        start_date = dateutil_parser.parse(work_start_date, dayfirst=True)
        end_date = dateutil_parser.parse(work_end_date, dayfirst=True)
    except (ValueError, TypeError):
        return None

    # Calculate the durations
    days_from_start = scrape_date - start_date
    total_work_duration = end_date - start_date

    # Avoid division by zero
    if total_work_duration.days == 0:
        return None

    # Calculate and return the completion ratio
    completion = days_from_start / total_work_duration
    return completion


def get_todays_date_string() -> str:
    """
    Get today's date as a string in ISO format.

    This utility function returns the current date formatted as YYYY-MM-DD,
    which is used as the scraping date for new data collection.

    Returns:
        str: Today's date in ISO format (YYYY-MM-DD).

    Example:
        >>> get_todays_date_string()  # Run on January 15, 2024
        '2024-01-15'
    """
    return datetime.datetime.today().strftime('%Y-%m-%d')


def parse_timestamp_to_date(timestamp: int) -> Optional[str]:
    """
    Convert a Unix timestamp to a date string.

    The listing creation date is sometimes stored as a Unix timestamp
    (seconds since epoch). This function converts it to an ISO format
    date string.

    Args:
        timestamp: A Unix timestamp (integer seconds since 1970-01-01).

    Returns:
        Optional[str]: The date in ISO format (YYYY-MM-DD), or None if
            the timestamp is invalid.

    Example:
        >>> parse_timestamp_to_date(1705363200)  # Jan 16, 2024
        '2024-01-16'
    """
    if timestamp is None:
        return None

    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError, OSError):
        # OSError can occur for timestamps outside the valid range
        return None


def fix_abbreviated_year(date_str: str) -> str:
    """
    Fix dates with abbreviated two-digit years.

    Some date strings use two-digit years (e.g., "15/01/24"). This function
    expands them to four-digit years, assuming years 00-99 refer to
    2000-2099.

    Args:
        date_str: A date string potentially containing an abbreviated year.

    Returns:
        str: The date string with a four-digit year, or the original string
            if no fix was needed or possible.

    Example:
        >>> fix_abbreviated_year("24-01-15")  # YYYY-MM-DD format, already OK
        '24-01-15'
        >>> fix_abbreviated_year("24-1-15")  # YY-M-DD format needs fix
        '2024-1-15'
    """
    if not date_str or len(date_str) < 8:
        # Too short to be a valid date string, return as-is
        return date_str

    # Check if it looks like a date with abbreviated year (YY-MM-DD)
    # by checking if the first part is only 2 characters
    parts = date_str.split("-")
    if len(parts) == 3 and len(parts[0]) == 2:
        try:
            year = int(parts[0])
            if year < 100:  # Two-digit year
                parts[0] = "20" + parts[0]
                return "-".join(parts)
        except ValueError:
            pass

    return date_str
