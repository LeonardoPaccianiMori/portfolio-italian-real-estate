"""
Selenium WebDriver utilities for the Italian Real Estate pipeline.

This module provides functionality for handling JavaScript-rendered content
that cannot be fetched with simple HTTP requests. It uses Selenium with
Chrome/Chromium or Firefox in headless mode to interact with dynamic web pages.

The primary use case is expanding sub-listings on property pages, where
clicking a button reveals additional property links that are not present
in the initial HTML.

Key features:
    - Headless Firefox browser automation
    - Automatic driver cleanup
    - Click handling for dynamic content
    - Error handling with graceful fallbacks

Author: Leonardo Pacciani-Mori
License: MIT
"""

import os
import pathlib
import random
import tempfile
import shutil
import time
from typing import Optional, List, Tuple
from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
)
from bs4 import BeautifulSoup

from italian_real_estate.config.logging_config import get_logger
from italian_real_estate.config.settings import (
    SCRAPING_USER_AGENT,
    SCRAPING_HUMANIZE,
    SCRAPING_HUMANIZE_MIN_WAIT,
    SCRAPING_HUMANIZE_MAX_WAIT,
    SCRAPING_HUMANIZE_SCROLL_MIN,
    SCRAPING_HUMANIZE_SCROLL_MAX,
    SCRAPING_HUMANIZE_SCROLL_WAIT_MIN,
    SCRAPING_HUMANIZE_SCROLL_WAIT_MAX,
    SCRAPING_POLITE_MODE,
    SCRAPING_POLITE_MIN_WAIT,
    SCRAPING_POLITE_MAX_WAIT,
    SCRAPING_POLITE_MAX_RETRIES,
    SCRAPING_POLITE_BACKOFF_BASE,
)
from italian_real_estate.scraping.listing_parser import is_blocked_page

# Initialize module logger
logger = get_logger(__name__)


def humanize_before_fetch() -> None:
    if not (SCRAPING_POLITE_MODE or SCRAPING_HUMANIZE):
        return
    if SCRAPING_POLITE_MODE:
        min_wait = max(0.0, SCRAPING_POLITE_MIN_WAIT)
        max_wait = max(min_wait, SCRAPING_POLITE_MAX_WAIT)
    else:
        min_wait = max(0.0, SCRAPING_HUMANIZE_MIN_WAIT)
        max_wait = max(min_wait, SCRAPING_HUMANIZE_MAX_WAIT)
    time.sleep(random.uniform(min_wait, max_wait))


def humanize_after_load(driver: webdriver.Remote) -> None:
    if not (SCRAPING_POLITE_MODE or SCRAPING_HUMANIZE):
        return
    min_scrolls = max(1, SCRAPING_HUMANIZE_SCROLL_MIN)
    max_scrolls = max(min_scrolls, SCRAPING_HUMANIZE_SCROLL_MAX)
    scrolls = random.randint(min_scrolls, max_scrolls)
    wait_min = max(0.0, SCRAPING_HUMANIZE_SCROLL_WAIT_MIN)
    wait_max = max(wait_min, SCRAPING_HUMANIZE_SCROLL_WAIT_MAX)

    try:
        scroll_height = driver.execute_script(
            "return document.body.scrollHeight || document.documentElement.scrollHeight || 0;"
        )
    except Exception:
        scroll_height = 0

    if not scroll_height or scroll_height <= 0:
        for _ in range(scrolls):
            try:
                driver.execute_script(
                    "window.scrollBy(0, Math.max(200, (window.innerHeight || 800) * 0.6));"
                )
            except Exception:
                break
            time.sleep(random.uniform(wait_min, wait_max))
        return

    for i in range(scrolls):
        position = int(((i + 1) / scrolls) * scroll_height)
        driver.execute_script("window.scrollTo(0, arguments[0]);", position)
        time.sleep(random.uniform(wait_min, wait_max))


def load_url_with_retries(
    driver: webdriver.Remote,
    url: str,
    render_wait: float = 0.0,
    scroll_after_load: bool = True,
) -> Optional[str]:
    max_retries = SCRAPING_POLITE_MAX_RETRIES if SCRAPING_POLITE_MODE else 1
    max_retries = max(1, max_retries)
    backoff_base = max(0.0, SCRAPING_POLITE_BACKOFF_BASE)
    last_html: Optional[str] = None

    for attempt in range(max_retries):
        try:
            humanize_before_fetch()
            driver.get(url)
            if render_wait:
                time.sleep(render_wait)
            if scroll_after_load:
                humanize_after_load(driver)
            last_html = driver.page_source
        except Exception as exc:
            logger.error(f"Error loading page with Selenium: {url}: {exc}")
            last_html = None

        if not SCRAPING_POLITE_MODE:
            return last_html

        if last_html and not is_blocked_page(last_html):
            return last_html

        if attempt < max_retries - 1:
            backoff = backoff_base * (2 ** attempt)
            logger.warning(
                f"Blocked or empty page for {url}. Retrying in {backoff:.1f}s..."
            )
            time.sleep(backoff)

    return last_html

def _select_browser() -> str:
    choice = os.getenv("SCRAPING_SELENIUM_BROWSER", "").strip().lower()
    if choice in ("chrome", "chromium"):
        return "chrome"
    if choice == "firefox":
        return "firefox"

    if os.getenv("CHROMEDRIVER_PATH") or os.getenv("CHROME_BIN"):
        return "chrome"
    if shutil.which("chromedriver"):
        return "chrome"

    return "firefox"


def _prepare_profile_dir() -> Tuple[Optional[str], bool]:
    profile_dir = os.getenv("SCRAPING_SELENIUM_PROFILE_DIR")
    if profile_dir:
        path = pathlib.Path(profile_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_path = path / ".write_test"
            test_path.write_text("ok")
            test_path.unlink(missing_ok=True)
            return str(path), False
        except Exception as exc:
            logger.warning(
                "SCRAPING_SELENIUM_PROFILE_DIR '%s' is not writable; using a temp profile (%s).",
                profile_dir,
                exc,
            )

    temp_dir = tempfile.mkdtemp(prefix="selenium-profile-")
    return temp_dir, True


def initialize_webdriver(
    headless: bool = True,
    profile_dir: Optional[str] = None,
) -> webdriver.Remote:
    """
    Initialize and return a configured Selenium WebDriver instance.

    This function creates a Selenium WebDriver with recommended settings for
    web scraping. By default, it runs in headless mode (no
    visible browser window) which is faster and works in server environments.

    The driver should be closed when no longer needed to free resources.
    Consider using the webdriver_session context manager for automatic cleanup.

    Args:
        headless: If True, runs the browser without a visible window.
            Defaults to True for server/automation use.

    Returns:
        webdriver.Remote: A configured WebDriver instance.

    Example:
        >>> driver = initialize_webdriver()
        >>> try:
        ...     driver.get("https://example.com")
        ...     html = driver.page_source
        ... finally:
        ...     driver.quit()
    """
    browser = _select_browser()
    logger.info(f"Initializing Selenium WebDriver ({browser})")

    if browser == "chrome":
        options = ChromeOptions()

        if headless:
            options.add_argument("--headless=new")
            options.add_argument("--remote-debugging-pipe")

        # Additional options for stability and performance
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        if SCRAPING_USER_AGENT:
            options.add_argument(f"--user-agent={SCRAPING_USER_AGENT}")

        if profile_dir:
            options.add_argument(f"--user-data-dir={profile_dir}")

        chrome_bin = os.getenv("CHROME_BIN")
        if chrome_bin:
            options.binary_location = chrome_bin

        chromedriver_path = os.getenv("CHROMEDRIVER_PATH")
        if chromedriver_path:
            service = ChromeService(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        return driver

    # Firefox fallback
    options = FirefoxOptions()

    if headless:
        options.add_argument("--headless")

    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    if SCRAPING_USER_AGENT:
        options.set_preference("general.useragent.override", SCRAPING_USER_AGENT)

    driver = webdriver.Firefox(options=options)

    return driver


@contextmanager
def webdriver_session(headless: bool = True):
    """
    Context manager for Selenium WebDriver with automatic cleanup.

    This context manager ensures that the WebDriver is properly closed
    even if an exception occurs during page interaction. It is the
    recommended way to use WebDriver for short-lived operations.

    Args:
        headless: If True, runs the browser without a visible window.

    Yields:
        webdriver.Remote: A configured WebDriver instance.

    Example:
        >>> with webdriver_session() as driver:
        ...     driver.get("https://example.com")
        ...     html = driver.page_source
        # Driver is automatically closed here
    """
    driver = None
    profile_dir = None
    cleanup_profile = False

    try:
        # Initialize the driver
        profile_dir, cleanup_profile = _prepare_profile_dir()
        driver = initialize_webdriver(headless, profile_dir=profile_dir)
        yield driver

    finally:
        # Always close the driver, even if an exception occurred
        if driver:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {str(e)}")
        if cleanup_profile and profile_dir:
            shutil.rmtree(profile_dir, ignore_errors=True)


def click_element_safely(
    driver: webdriver.Remote,
    xpath: str,
    timeout: int = 10
) -> bool:
    """
    Safely click an element identified by XPath with wait and error handling.

    This function attempts to click an element on the page, waiting for it
    to be clickable first. If the click fails (e.g., element not found or
    not clickable), it returns False instead of raising an exception.

    Args:
        driver: An active Selenium WebDriver instance.
        xpath: The XPath expression to locate the element to click.
        timeout: Maximum seconds to wait for the element to be clickable.

    Returns:
        bool: True if the click was successful, False otherwise.

    Example:
        >>> with webdriver_session() as driver:
        ...     driver.get(url)
        ...     if click_element_safely(driver, "//button[@id='show-more']"):
        ...         # Process the expanded page
        ...         pass
    """
    try:
        # Wait for the element to be clickable
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )

        # Execute click via JavaScript for reliability.
        # Direct .click() can fail if the element is obscured.
        driver.execute_script("arguments[0].click();", element)

        return True

    except TimeoutException:
        # Element didn't become clickable within the timeout
        logger.debug(f"Element not found or not clickable: {xpath}")
        return False

    except NoSuchElementException:
        # Element doesn't exist on the page
        logger.debug(f"Element not found: {xpath}")
        return False

    except ElementClickInterceptedException:
        # Another element is blocking the click
        logger.debug(f"Element click intercepted: {xpath}")
        return False

    except Exception as e:
        # Catch any other exceptions to prevent crashes
        logger.warning(f"Error clicking element {xpath}: {str(e)}")
        return False


def extract_html_with_selenium(
    url: str,
    click_xpaths: Optional[List[str]] = None,
    wait_after_click: float = 1.0,
    page_load_timeout: int = 30,
    render_wait: float = 1.0,
    headless: bool = True,
) -> Optional[str]:
    """
    Load a page with Selenium and optionally click elements before extracting HTML.

    This function is used when a page requires JavaScript execution or button
    clicks to reveal all content. It loads the page, optionally clicks specified
    elements, and returns the final HTML source.

    The primary use case is expanding sub-listings on property pages where
    a "show all" button needs to be clicked to reveal additional property links.

    Args:
        url: The URL to load.
        click_xpaths: Optional list of XPath expressions for elements to click.
            Elements are clicked in order. If an element isn't found, it's skipped.
        wait_after_click: Seconds to wait after each click for dynamic content
            to load. Defaults to 1.0 second.
        page_load_timeout: Maximum seconds to wait for the initial page load.
            Defaults to 30 seconds.
        render_wait: Seconds to wait after initial page load before interacting.
            Defaults to 1.0 second.
        headless: If True, runs the browser without a visible window.

    Returns:
        str or None: The HTML source of the page after all interactions,
            or None if the page couldn't be loaded.

    Example:
        >>> # Expand sub-listings and get the full page HTML
        >>> html = extract_html_with_selenium(
        ...     "https://example.com/listing/123",
        ...     click_xpaths=["//button[contains(@class, 'show-all')]"]
        ... )
        >>> if html:
        ...     soup = BeautifulSoup(html, 'html.parser')
    """
    with webdriver_session(headless=headless) as driver:
        try:
            # Set page load timeout
            driver.set_page_load_timeout(page_load_timeout)

            # Load the page
            logger.info(f"Loading page with Selenium: {url}")
            page_source = load_url_with_retries(
                driver,
                url,
                render_wait=render_wait,
                scroll_after_load=not bool(click_xpaths),
            )
            if page_source is None:
                return None

            # Click specified elements if any
            if click_xpaths:
                for xpath in click_xpaths:
                    if click_element_safely(driver, xpath):
                        # Wait for dynamic content to load after the click
                        time.sleep(wait_after_click)

            if click_xpaths:
                humanize_after_load(driver)

            # Return the final page source
            return driver.page_source

        except TimeoutException:
            logger.error(f"Timeout loading page: {url}")
            return None

        except Exception as e:
            logger.error(f"Error during Selenium session for {url}: {str(e)}")
            return None


def expand_sublistings_and_get_links(
    url: str,
    single_sublisting_xpath: str,
    multiple_sublistings_xpath: str,
    link_extraction_selector: str
) -> tuple:
    """
    Expand sub-listings on a page and extract all listing links.

    This function handles the specific pattern used on listing.website where
    property listings can have multiple sub-listings (e.g., multiple apartments
    in the same building). It clicks the appropriate button to reveal all
    sub-listing links and extracts them.

    There are two types of buttons:
        - One for when there's exactly one additional sub-listing
        - Another for when there are multiple additional sub-listings

    Args:
        url: The URL of the listing page to process.
        single_sublisting_xpath: XPath for the button shown when there's
            exactly one additional sub-listing.
        multiple_sublistings_xpath: XPath for the button shown when there
            are multiple additional sub-listings.
        link_extraction_selector: CSS selector or pattern for extracting
            the sub-listing links after expansion.

    Returns:
        tuple: (parent_html, sublisting_links, sublisting_ids) where:
            - parent_html: The original page HTML before expansion
            - sublisting_links: List of URLs to sub-listings
            - sublisting_ids: List of listing IDs extracted from URLs

    Example:
        >>> parent_html, links, ids = expand_sublistings_and_get_links(
        ...     "https://example.com/listing/123",
        ...     "//button[@class='single']",
        ...     "//button[@class='multiple']",
        ...     "a.sublisting-link"
        ... )
    """
    sublisting_links = []
    sublisting_ids = []
    parent_html = None

    with webdriver_session() as driver:
        try:
            # Load the page
            driver.get(url)

            # Store the parent's original HTML
            parent_html = driver.page_source

            # Try clicking the single sub-listing button first
            clicked = click_element_safely(driver, single_sublisting_xpath)

            if not clicked:
                # Try the multiple sub-listings button
                clicked = click_element_safely(driver, multiple_sublistings_xpath)

            if clicked:
                # Wait for dynamic content to load
                import time
                time.sleep(1.5)

                # Parse the expanded page to extract links
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Extract links using the provided selector
                link_elements = soup.select(link_extraction_selector)

                for element in link_elements:
                    if 'href' in element.attrs:
                        link = element['href']
                        sublisting_links.append(link)

                        # Extract listing ID from URL (last numeric segment)
                        try:
                            listing_id = int(link.split('/')[-2])
                            sublisting_ids.append(listing_id)
                        except (ValueError, IndexError):
                            pass

        except Exception as e:
            logger.error(f"Error expanding sub-listings for {url}: {str(e)}")

    return parent_html, sublisting_links, sublisting_ids


def get_page_source_after_interaction(
    driver: webdriver.Remote,
    interaction_xpaths: List[str],
    wait_between: float = 1.0
) -> str:
    """
    Execute a series of interactions and return the resulting page source.

    This utility function clicks multiple elements in sequence and returns
    the final page state. It's useful for pages that require multiple
    interactions to reveal all content.

    Args:
        driver: An active Selenium WebDriver instance with a page loaded.
        interaction_xpaths: List of XPath expressions for elements to click,
            in the order they should be clicked.
        wait_between: Seconds to wait between each interaction.

    Returns:
        str: The HTML source after all interactions are complete.

    Example:
        >>> with webdriver_session() as driver:
        ...     driver.get(url)
        ...     html = get_page_source_after_interaction(
        ...         driver,
        ...         ["//button[@id='expand']", "//button[@id='show-details']"]
        ...     )
    """
    import time

    for xpath in interaction_xpaths:
        if click_element_safely(driver, xpath):
            time.sleep(wait_between)

    return driver.page_source
