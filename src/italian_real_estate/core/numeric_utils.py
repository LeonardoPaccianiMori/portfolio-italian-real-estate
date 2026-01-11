"""
Numeric calculation utilities for the Italian Real Estate pipeline.

This module provides functions for financial calculations (mortgage payments)
and numeric processing (rounding, random rounding for synthetic data).
These utilities support both the ETL pipeline and the synthetic data
generation process.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import random
from typing import Optional, Union
import math


def mortgage_monthly_payment(
    principal: float,
    interest: float,
    term: int = 30
) -> Optional[float]:
    """
    Calculate the monthly mortgage payment given principal, interest rate, and term.

    This function implements the standard amortization formula to calculate
    fixed monthly mortgage payments. It is used in the ETL pipeline to
    compute estimated monthly payments for sale listings.

    The formula used is:
        P = L * [c(1 + c)^n] / [(1 + c)^n - 1]

    Where:
        P = monthly payment
        L = loan principal (loan amount)
        c = monthly interest rate (annual rate / 12)
        n = total number of payments (years * 12)

    Args:
        principal: The mortgage principal (loan amount) in euros.
        interest: The annual interest rate as a decimal (e.g., 0.04 for 4%).
            Note: NOT a percentage - use 0.04, not 4.
        term: The mortgage term in years. Defaults to 30 years.

    Returns:
        Optional[float]: The monthly payment amount rounded to 2 decimal
            places, or None if either principal or interest is None/invalid.

    Example:
        >>> mortgage_monthly_payment(200000, 0.04, 30)
        954.83
        >>> mortgage_monthly_payment(None, 0.04)
        None
    """
    # Validate inputs - both principal and interest must be valid
    if principal is None or interest is None:
        return None

    if principal <= 0 or interest <= 0:
        return None

    # Convert annual interest rate to monthly rate
    monthly_rate = interest / 12

    # Calculate total number of monthly payments
    num_payments = term * 12

    # Calculate monthly payment using the amortization formula
    # P = L * [c(1 + c)^n] / [(1 + c)^n - 1]
    payment = principal * (
        monthly_rate * (1 + monthly_rate) ** num_payments
    ) / (
        (1 + monthly_rate) ** num_payments - 1
    )

    return round(payment, 2)


def calculate_monthly_payment(
    price: float,
    mortgage_rate: float,
    loan_term_years: int = 30,
    down_payment_percent: float = 20
) -> Optional[float]:
    """
    Calculate monthly mortgage payment for a property purchase.

    This is an extended version of mortgage_monthly_payment that takes into
    account the down payment. It calculates the loan amount after down payment
    and then computes the monthly payment.

    This function is used during the PostgreSQL migration to compute monthly
    payments for properties where only the price and mortgage rate are known.

    Args:
        price: The full purchase price of the property in euros.
        mortgage_rate: The annual interest rate as a decimal (e.g., 0.035 for 3.5%).
        loan_term_years: The mortgage term in years. Defaults to 30.
        down_payment_percent: The down payment as a percentage of price.
            Defaults to 20 (i.e., 20%).

    Returns:
        Optional[float]: The monthly payment amount, or None if inputs are
            invalid (None, zero, or negative).

    Example:
        >>> calculate_monthly_payment(300000, 0.035, 30, 20)
        1078.59  # For 80% of 300k at 3.5% over 30 years
    """
    # Validate inputs
    if price is None or mortgage_rate is None:
        return None

    if price <= 0 or mortgage_rate <= 0:
        return None

    # Calculate the loan amount (price minus down payment)
    loan_amount = price * (1 - down_payment_percent / 100)

    # Convert annual rate to monthly rate
    monthly_interest_rate = mortgage_rate / 12

    # Calculate total number of payments
    num_payments = loan_term_years * 12

    # Calculate monthly payment
    if monthly_interest_rate == 0:
        # Special case: 0% interest rate
        monthly_payment = loan_amount / num_payments
    else:
        # Standard amortization formula
        monthly_payment = loan_amount * (
            monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments
        ) / (
            (1 + monthly_interest_rate) ** num_payments - 1
        )

    return monthly_payment


def random_round(x: float) -> int:
    """
    Randomly round a number to handle exact midpoints probabilistically.

    When a number is exactly at the midpoint between two integers (e.g., 2.5),
    standard rounding always goes the same direction. This function introduces
    randomness at midpoints to avoid systematic bias in synthetic data.

    For non-midpoint values, standard rounding rules apply.

    This function is primarily used in synthetic data generation to create
    more natural-looking distributions of integer values like room counts.

    Args:
        x: The floating-point number to round.

    Returns:
        int: The rounded integer value.

    Example:
        >>> random_round(2.5)  # Will randomly return 2 or 3
        2  # or 3
        >>> random_round(2.3)
        2
        >>> random_round(2.7)
        3
    """
    # Get the fractional part
    fractional_part = x - int(x)

    # Check if it's exactly at the midpoint (0.5)
    if fractional_part == 0.5:
        # Randomly choose to round up or down
        if random.random() < 0.5:
            return int(x)
        else:
            return int(x) + 1
    else:
        # Use standard rounding
        return round(x)


def round_to_nearest(value: float, nearest: int) -> float:
    """
    Round a value to the nearest specified increment.

    This is useful for rounding prices to natural increments (e.g., nearest
    €1,000) or surfaces to whole numbers. The function handles both positive
    and negative values correctly.

    Args:
        value: The value to round.
        nearest: The increment to round to (e.g., 10, 100, 1000).

    Returns:
        float: The value rounded to the nearest increment.

    Example:
        >>> round_to_nearest(147500, 1000)
        148000.0
        >>> round_to_nearest(147499, 1000)
        147000.0
    """
    if nearest <= 0:
        return value

    return round(value / nearest) * nearest


def constrain_to_range(
    value: float,
    min_value: float,
    max_value: float
) -> float:
    """
    Constrain a value to be within a specified range.

    This function clips values to ensure they fall within valid bounds.
    It is useful for post-processing synthetic data where generated values
    may occasionally fall outside realistic ranges.

    Args:
        value: The value to constrain.
        min_value: The minimum allowed value (inclusive).
        max_value: The maximum allowed value (inclusive).

    Returns:
        float: The constrained value, guaranteed to be within [min_value, max_value].

    Example:
        >>> constrain_to_range(150, 1, 100)
        100
        >>> constrain_to_range(-5, 1, 100)
        1
        >>> constrain_to_range(50, 1, 100)
        50
    """
    return max(min_value, min(max_value, value))


def safe_float_conversion(value: Union[str, int, float, None]) -> Optional[float]:
    """
    Safely convert a value to float, handling various input types.

    This utility handles common data quality issues where numeric values
    may be stored as strings or may contain non-numeric characters.

    Args:
        value: The value to convert. Can be string, int, float, or None.

    Returns:
        Optional[float]: The float value, or None if conversion fails.

    Example:
        >>> safe_float_conversion("123.45")
        123.45
        >>> safe_float_conversion(100)
        100.0
        >>> safe_float_conversion("invalid")
        None
    """
    if value is None:
        return None

    if isinstance(value, float):
        return value

    if isinstance(value, int):
        return float(value)

    if isinstance(value, str):
        try:
            # Handle European decimal notation (comma as decimal separator)
            clean_value = value.replace(",", ".")
            return float(clean_value)
        except ValueError:
            return None

    return None


def safe_int_conversion(value: Union[str, int, float, None]) -> Optional[int]:
    """
    Safely convert a value to integer, handling various input types.

    Similar to safe_float_conversion but for integer values. Handles the
    special case where the source data contains "3+" style notation for
    "3 or more" which should be converted to 4.

    Args:
        value: The value to convert. Can be string, int, float, or None.

    Returns:
        Optional[int]: The integer value, or None if conversion fails.

    Example:
        >>> safe_int_conversion("3+")
        4
        >>> safe_int_conversion(5.7)
        5
        >>> safe_int_conversion("invalid")
        None
    """
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        # Handle "X+" notation (meaning X or more)
        if value.endswith("+"):
            try:
                return int(value[:-1]) + 1
            except ValueError:
                pass

        # Handle range notation like "3-5" (take the higher value)
        if "-" in value:
            parts = value.split("-")
            try:
                return int(parts[-1].replace("+", "")) + (1 if "+" in parts[-1] else 0)
            except ValueError:
                pass

        # Standard conversion
        try:
            return int(value)
        except ValueError:
            return None

    return None


def calculate_price_per_sqm(
    price: Optional[float],
    surface: Optional[float]
) -> Optional[float]:
    """
    Calculate the price per square meter.

    This metric is commonly used to compare property values across different
    sizes and is a key feature in the machine learning model.

    Args:
        price: The property price in euros.
        surface: The property surface area in square meters.

    Returns:
        Optional[float]: The price per square meter rounded to 2 decimal
            places, or None if either input is None or surface is zero.

    Example:
        >>> calculate_price_per_sqm(200000, 85)
        2352.94
    """
    if price is None or surface is None:
        return None

    if surface == 0:
        return None

    return round(price / surface, 2)
