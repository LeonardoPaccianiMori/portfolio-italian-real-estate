"""
Web scraping module for the Italian Real Estate pipeline.

This module provides functionality for scraping real estate listings from
listing.website and populating the MongoDB datalake with raw HTML data.

Submodules:
    http_client: Asynchronous HTTP client for fetching web pages.
    selenium_handler: Selenium WebDriver utilities for JavaScript-rendered content.
    listing_parser: BeautifulSoup parsing utilities for listing data.
    datalake_populator: Main orchestration for datalake population.
"""

from .http_client import get_single_url, get_multiple_urls
from .selenium_handler import extract_html_with_selenium, initialize_webdriver
from .listing_parser import has_sublistings, is_auction_listing
from .datalake_populator import (
    datalake_populate_links_range,
    datalake_populate_links_province_async,
    log_datalake_statistics,
)
