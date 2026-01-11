"""
Datalake population orchestration for the Italian Real Estate pipeline.

This module provides the main orchestration logic for scraping real estate
listings from listing.website and storing them in the MongoDB datalake. It
coordinates the HTTP client, Selenium handler, and listing parser to
efficiently scrape listings by province and price range.

The scraping strategy handles the website's pagination limit (80 pages / 2000
listings) by using price range windows that can be recursively narrowed when
too many results are returned.

Key components:
    - Price range iteration for comprehensive coverage
    - Pagination handling with automatic range splitting
    - Parent/child listing relationship tracking
    - Duplicate detection via scraping date

Author: Leonardo Pacciani-Mori
License: MIT
"""

import asyncio
import random
import time
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from italian_real_estate.config.settings import (
    PROVINCES_CSV_PATH,
    SALE_PRICE_MIN,
    SALE_PRICE_MAX,
    SALE_PRICE_STEP,
    RENT_PRICE_MIN,
    RENT_PRICE_MAX,
    RENT_PRICE_STEP,
    VALID_LISTING_TYPES,
    LISTING_TYPE_URL_SEGMENTS,
    SCRAPING_HEADERS,
    SCRAPING_USE_SELENIUM,
    SCRAPING_SELENIUM_HEADLESS,
    SCRAPING_SELENIUM_PAGE_DELAY,
    SCRAPING_SELENIUM_PAGE_LOAD_TIMEOUT,
    SCRAPING_HUMANIZE,
    SCRAPING_HUMANIZE_MIN_WAIT,
    SCRAPING_HUMANIZE_MAX_WAIT,
    SCRAPING_POLITE_MODE,
    SCRAPING_POLITE_MIN_WAIT,
    SCRAPING_POLITE_MAX_WAIT,
)
from italian_real_estate.config.logging_config import get_logger
from italian_real_estate.core.connections import (
    get_mongodb_client,
    get_datalake_collections,
)
from italian_real_estate.scraping.http_client import get_multiple_urls
from italian_real_estate.scraping.listing_parser import (
    parse_page_count,
    extract_listing_links_from_search,
)
from italian_real_estate.scraping.selenium_handler import (
    webdriver_session,
    extract_html_with_selenium,
    load_url_with_retries,
)

# Initialize module logger
logger = get_logger(__name__)


def _humanize_http_wait() -> None:
    if not (SCRAPING_POLITE_MODE or SCRAPING_HUMANIZE):
        return
    if SCRAPING_POLITE_MODE:
        min_wait = max(0.0, SCRAPING_POLITE_MIN_WAIT)
        max_wait = max(min_wait, SCRAPING_POLITE_MAX_WAIT)
    else:
        min_wait = max(0.0, SCRAPING_HUMANIZE_MIN_WAIT)
        max_wait = max(min_wait, SCRAPING_HUMANIZE_MAX_WAIT)
    time.sleep(random.uniform(min_wait, max_wait))

# Load province data at module level for efficiency.
# This CSV contains province names, codes, and region mappings.
_province_df = None
_prov_to_code = None
_prov_to_region = None


def _load_province_data():
    """
    Load province data from CSV file into module-level variables.

    This internal function lazily loads the province data on first access
    and caches it for subsequent calls. The data includes province names,
    two-letter codes, and region mappings.

    The province data is used to:
        - Generate scraping tasks for each province
        - Store province metadata with each listing
        - Map provinces to their regions

    Returns:
        None (updates module-level variables)
    """
    global _province_df, _prov_to_code, _prov_to_region

    if _province_df is None:
        logger.info(f"Loading province data from {PROVINCES_CSV_PATH}")
        _province_df = pd.read_csv(PROVINCES_CSV_PATH, sep=",", na_filter=False)
        _prov_to_code = dict(zip(_province_df["Province"], _province_df["Code"]))
        _prov_to_region = dict(zip(_province_df["Province"], _province_df["Region"]))


def get_province_code(province_name: str) -> str:
    """
    Get the two-letter code for a province name.

    Province codes are used in the MongoDB documents to provide a compact
    identifier for each province.

    Args:
        province_name: The full name of the province (e.g., "Milano").

    Returns:
        str: The two-letter province code (e.g., "MI").

    Example:
        >>> get_province_code("Milano")
        'MI'
    """
    _load_province_data()
    return _prov_to_code.get(province_name, "")


def get_province_region(province_name: str) -> str:
    """
    Get the region name for a province.

    Args:
        province_name: The full name of the province.

    Returns:
        str: The name of the region containing the province.

    Example:
        >>> get_province_region("Milano")
        'Lombardia'
    """
    _load_province_data()
    return _prov_to_region.get(province_name, "")


def get_all_provinces() -> List[str]:
    """
    Get a list of all Italian province names.

    This is used to generate the list of scraping tasks when populating
    the datalake for all provinces.

    Returns:
        list: A list of all province names.

    Example:
        >>> provinces = get_all_provinces()
        >>> print(len(provinces))  # 107 provinces in Italy
    """
    _load_province_data()
    return _province_df["Province"].tolist()


def _fetch_urls_with_selenium(urls: List[str]) -> List[Optional[str]]:
    if not urls:
        return []

    pages = []
    with webdriver_session(headless=SCRAPING_SELENIUM_HEADLESS) as driver:
        driver.set_page_load_timeout(SCRAPING_SELENIUM_PAGE_LOAD_TIMEOUT)
        for url in urls:
            try:
                logger.info(f"Loading page with Selenium: {url}")
                page_source = load_url_with_retries(
                    driver,
                    url,
                    render_wait=SCRAPING_SELENIUM_PAGE_DELAY,
                )
                pages.append(page_source)
            except Exception as e:
                logger.error(f"Error loading page with Selenium: {url}: {str(e)}")
                pages.append(None)

    return pages


def get_todays_date() -> str:
    """
    Get today's date as a string in YYYY-MM-DD format.

    This date is used as the scraping_date when storing listings.

    Returns:
        str: Today's date in ISO format.
    """
    return datetime.today().strftime('%Y-%m-%d')


async def extract_html_source_code(
    url: str,
    html_source: str,
    province_name: str,
    listing_type: str,
    date: str,
    parent_listing: Optional[int] = None,
    use_selenium: bool = False,
) -> None:
    """
    Extract and store listing data in the MongoDB datalake.

    This function processes a single listing's HTML and stores it in the
    appropriate MongoDB collection. It handles:
        - Parent/child listing relationships
        - Sub-listing expansion and recursive processing
        - Duplicate detection by scraping date
        - Auction listing filtering for sale scrapes

    The function inserts new listings or appends scraping data to existing
    listings, tracking when each version was scraped.

    Args:
        url: The URL of the listing page.
        html_source: The HTML content of the listing page.
        province_name: The name of the province being scraped.
        listing_type: The type of listing ("sale", "rent", or "auction").
        date: The scraping date in YYYY-MM-DD format.
        parent_listing: The ID of the parent listing if this is a sub-listing,
            or None for top-level listings.
        use_selenium: If True, use Selenium for sub-listing fetches.

    Returns:
        None

    Example:
        >>> await extract_html_source_code(
        ...     "https://example.com/listing/123",
        ...     html_content,
        ...     "Milano",
        ...     "sale",
        ...     "2024-01-15"
        ... )
    """
    if listing_type not in VALID_LISTING_TYPES:
        raise ValueError(f"listing_type must be one of {VALID_LISTING_TYPES}")

    if html_source is None:
        logger.warning(f"Received None HTML source for URL: {url}, skipping...")
        return

    raise RuntimeError(
        "Scraping redacted in the public portfolio version. "
        "Re-implement extract_html_source_code() to enable scraping."
    )

    # Intended flow (the actual function body is redacted to avoid out-of-the-box scraping):
    # 1) Connect to MongoDB and select the datalake collection.
    # 2) Extract listing_id from the URL and filter auctions.
    # 3) Parse HTML and expand sub-listings with Selenium if needed.
    # 4) Validate listing metadata presence.
    # 5) Upsert parent/child listings and HTML snapshots into MongoDB.


async def datalake_populate_links_range(
    province: str,
    listing_type: str,
    mode: str,
    price1: str,
    price2: Optional[str] = None,
    scraping_date: Optional[str] = None,
) -> None:
    """
    Populate the datalake with listings for a specific province and price range.

    This function handles the core scraping logic, fetching all listings that
    match the given criteria and storing them in the datalake. It handles
    pagination and recursively narrows the price range when too many results
    are returned (>80 pages / 2000 listings).

    Args:
        province: The name of the province to scrape.
        listing_type: The type of listing ("sale", "rent", or "auction").
        mode: The price range mode:
            - "up": Prices above price1
            - "down": Prices below price1
            - "between": Prices between price1 and price2
        price1: The first price boundary (as a string).
        price2: The second price boundary for "between" mode (as a string).
        scraping_date: The scraping date in YYYY-MM-DD format.

    Returns:
        None

    Example:
        >>> await datalake_populate_links_range(
        ...     "Milano", "sale", "between", "100000", "150000"
        ... )
    """
    # Validate mode argument
    if mode not in ["up", "down", "between"]:
        raise ValueError("mode must be 'up', 'down', or 'between'")

    date = scraping_date or get_todays_date()

    # Get the URL segment for the listing type
    search_string = LISTING_TYPE_URL_SEGMENTS[listing_type]
    province_slug = f"{province}-provincia"
    base_url = f"https://www.listing.website/{search_string}/{province_slug}/?criterio=rilevanza"

    # Build the search URL based on mode
    if mode == "up":
        log_msg = f"province {province}, {listing_type}, price above {price1}"
        url = f"{base_url}&prezzoMinimo={price1}"
    elif mode == "down":
        log_msg = f"province {province}, {listing_type}, price below {price1}"
        url = f"{base_url}&prezzoMassimo={price1}"
    else:  # between
        log_msg = f"province {province}, {listing_type}, price {price1}-{price2}"
        url = f"{base_url}&prezzoMinimo={price1}&prezzoMassimo={price2}"

    logger.info(f"Starting to fetch data for {log_msg}")

    # Create a session with retry adapter
    session = requests.Session()
    session.headers.update(SCRAPING_HEADERS)
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    html_source = None
    fetched_with_selenium = SCRAPING_USE_SELENIUM or SCRAPING_POLITE_MODE
    status_code = None

    if fetched_with_selenium:
        html_source = extract_html_with_selenium(
            url,
            page_load_timeout=SCRAPING_SELENIUM_PAGE_LOAD_TIMEOUT,
            render_wait=SCRAPING_SELENIUM_PAGE_DELAY,
            headless=SCRAPING_SELENIUM_HEADLESS,
        )
        if html_source is None:
            logger.error("Selenium failed to fetch main listings page")
            return
    else:
        try:
            # Fetch the first page of results
            _humanize_http_wait()
            req = session.get(url, timeout=60)
            status_code = req.status_code
            if status_code == 403:
                raise requests.exceptions.HTTPError("403 Client Error: Forbidden")
            req.raise_for_status()
            html_source = req.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching main listings page: {str(e)}")
            logger.info("Retrying...")
            try:
                _humanize_http_wait()
                req = session.get(url, timeout=120)
                status_code = req.status_code
                if status_code == 403:
                    raise requests.exceptions.HTTPError("403 Client Error: Forbidden")
                req.raise_for_status()
                html_source = req.text
            except requests.exceptions.RequestException as e2:
                logger.error(f"Retry failed: {str(e2)}")
                logger.info("Falling back to Selenium for main listings page...")
                html_source = extract_html_with_selenium(
                    url,
                    page_load_timeout=SCRAPING_SELENIUM_PAGE_LOAD_TIMEOUT,
                    render_wait=SCRAPING_SELENIUM_PAGE_DELAY,
                    headless=SCRAPING_SELENIUM_HEADLESS,
                )
                if html_source is None:
                    logger.error("Selenium fallback failed for main listings page")
                    return
                fetched_with_selenium = True

    # Extract page count from pagination elements
    page_number = parse_page_count(html_source)

    if fetched_with_selenium:
        logger.info("HTML source obtained with Selenium")
    else:
        logger.info(f"HTML source obtained with status code {status_code}")
    logger.info(f"{page_number} pages returned")

    # Check if we're under the 80 page limit
    if page_number < 80:
        # We can process all results without splitting
        logger.info(f"Processing {page_number} page(s)")

        # Generate URLs for all pages
        pages_urls = [f"{url}&pag={i}" for i in range(1, page_number + 1)]

        # Fetch all pages
        if fetched_with_selenium:
            all_pages_data = _fetch_urls_with_selenium(pages_urls)
        else:
            all_pages_data = await get_multiple_urls(pages_urls)

        # Parse all pages and extract listing links
        soups_pages = [
            BeautifulSoup(x, 'html.parser')
            for x in all_pages_data
            if x is not None
        ]

        # Extract listing links from all pages
        listing_links = []
        for soup in soups_pages:
            links = extract_listing_links_from_search(soup)
            listing_links.extend(links)

        listing_number = len(listing_links)
        logger.info(f"Found {listing_number} listings")

        if listing_number == 0:
            logger.info("No listings found for this price range")
            return

        # Fetch all listing pages
        if fetched_with_selenium:
            all_listings_data = _fetch_urls_with_selenium(listing_links)
        else:
            all_listings_data = await get_multiple_urls(listing_links)

        # Process each listing
        for i, listing_html in enumerate(all_listings_data):
            await extract_html_source_code(
                listing_links[i],
                listing_html,
                province,
                listing_type,
                date,
                use_selenium=fetched_with_selenium,
            )
            logger.info(f"Processed listing {i + 1}/{listing_number}")

        logger.info(f"Finished fetching data for {log_msg}")

    else:
        # Too many results - need to split the price range
        logger.info("Too many results (>=80 pages). Splitting price range...")

        if mode == "down":
            mid_point = str(int(int(price1) / 2))
            await datalake_populate_links_range(
                province,
                listing_type,
                "down",
                mid_point,
                scraping_date=date,
            )
            await datalake_populate_links_range(
                province,
                listing_type,
                "between",
                mid_point,
                price1,
                scraping_date=date,
            )

        elif mode == "between":
            mid_point = str(int((int(price1) + int(price2)) / 2))
            await datalake_populate_links_range(
                province,
                listing_type,
                mode,
                price1,
                mid_point,
                scraping_date=date,
            )
            await datalake_populate_links_range(
                province,
                listing_type,
                mode,
                mid_point,
                price2,
                scraping_date=date,
            )

        elif mode == "up":
            mid_point = str(int(int(price1) * 2))
            await datalake_populate_links_range(
                province,
                listing_type,
                "between",
                price1,
                mid_point,
                scraping_date=date,
            )
            await datalake_populate_links_range(
                province,
                listing_type,
                "up",
                mid_point,
                scraping_date=date,
            )


async def datalake_populate_links_province_async(
    province: str,
    listing_type: str,
    scraping_date: Optional[str] = None,
) -> None:
    """
    Populate the datalake with all listings for a province and listing type.

    This function orchestrates the scraping of an entire province by iterating
    through price ranges. It ensures comprehensive coverage by scanning from
    the minimum to maximum expected prices.

    The price ranges are different for sale/auction (€50k-€1M in €50k steps)
    versus rent (€200-€5k in €400 steps).

    Args:
        province: The name of the province to scrape.
        listing_type: The type of listing ("sale", "rent", or "auction").
        scraping_date: The scraping date in YYYY-MM-DD format.

    Returns:
        None

    Example:
        >>> await datalake_populate_links_province_async("Milano", "rent")
    """
    # Define price ranges based on listing type
    if listing_type == "rent":
        prices = np.arange(
            RENT_PRICE_MIN,
            RENT_PRICE_MAX + RENT_PRICE_STEP,
            RENT_PRICE_STEP
        ).astype(int).astype(str)
    else:
        prices = np.arange(
            SALE_PRICE_MIN,
            SALE_PRICE_MAX + SALE_PRICE_STEP,
            SALE_PRICE_STEP
        ).astype(int).astype(str)

    logger.info(f"Starting extraction for {province}, {listing_type}")
    date = scraping_date or get_todays_date()

    # Fetch listings below the minimum price
    await datalake_populate_links_range(
        province,
        listing_type,
        "down",
        prices[0],
        scraping_date=date,
    )

    # Fetch listings in each price range
    for i in range(1, len(prices)):
        await datalake_populate_links_range(
            province,
            listing_type,
            "between",
            prices[i - 1],
            prices[i],
            scraping_date=date,
        )

    # Fetch listings above the maximum price
    await datalake_populate_links_range(
        province,
        listing_type,
        "up",
        prices[-1],
        scraping_date=date,
    )

    logger.info(f"Finished extraction for {province}, {listing_type}")


def datalake_populate_links_province(
    province: str,
    listing_type: str,
    scraping_date: Optional[str] = None,
) -> None:
    """
    Synchronous wrapper for datalake_populate_links_province_async.

    This wrapper function allows the async scraping function to be called
    from synchronous code, such as Airflow's PythonOperator.

    Args:
        province: The name of the province to scrape.
        listing_type: The type of listing ("sale", "rent", or "auction").
        scraping_date: The scraping date in YYYY-MM-DD format.

    Returns:
        None
    """
    asyncio.run(
        datalake_populate_links_province_async(
            province,
            listing_type,
            scraping_date=scraping_date,
        )
    )


def log_datalake_statistics() -> None:
    """
    Log summary statistics for the MongoDB datalake.

    This function connects to the datalake and reports the count of listings
    in each collection. It is typically called at the end of a scraping run
    to provide a summary of the data collected.

    Returns:
        None
    """
    client = get_mongodb_client()
    sale_coll, rent_coll, auction_coll = get_datalake_collections(client)

    # Count documents in each collection
    sale_count = sale_coll.count_documents({})
    rent_count = rent_coll.count_documents({})
    auction_count = auction_coll.count_documents({})
    total_count = sale_count + rent_count + auction_count

    # Log the results
    logger.info("=" * 30)
    logger.info("DATALAKE LISTING COUNTS")
    logger.info("=" * 30)
    logger.info(f"Sale listings:     {sale_count}")
    logger.info(f"Rent listings:     {rent_count}")
    logger.info(f"Auction listings:  {auction_count}")
    logger.info("-" * 30)
    logger.info(f"TOTAL LISTINGS:    {total_count}")
    logger.info("=" * 30)

    client.close()

    logger.info("Datalake population completed successfully.")
