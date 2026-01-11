"""
Filename: warehouse_migration_mongoDB_postgreSQL.py
Author: Leonardo Pacciani-Mori
Date: YYYY-MM-DD

Description:
    This script gets migrates data from the MongoDB warehouse to the PostgreSQL
    warehouse, both hosted locally on the same machine where the script is executed.
    This script uses Apache Airflow to divide the data migration into smaller,
    independent tasks that can be executed in parallel.

License:
    This code is released under the GNU General Public License v3.
    See https://www.gnu.org/licenses/gpl-3.0.en.html for details.
"""

# -------------------------------------------------------------------------------
# --------------LIBRARY IMPORTS AND GLOBAL VARIABLES DEFINITION------------------
# -------------------------------------------------------------------------------

import datetime
from datetime import timedelta
import json
import re
import logging
import requests
from bson import json_util
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from pymongo import MongoClient
import hashlib
import subprocess
import time
import atexit
from lingua import Language, LanguageDetectorBuilder
import sqlite3
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

try:
    from airflow.utils.dates import days_ago
except ImportError:
    try:
        from airflow.sdk import timezone

        def days_ago(n: int):
            return timezone.utcnow() - timedelta(days=n)
    except ImportError:
        from datetime import datetime, timezone as dt_timezone

        def days_ago(n: int):
            return datetime.now(dt_timezone.utc) - timedelta(days=n)

# These variables are needed to configure the batch processing
BATCH_SIZE = 10000  # Number of records to process in each batch
MAX_RECORDS_PER_COLLECTION = None  # Set to None to process all records, or a number for testing

# Sets the logging level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppresses excessive SQL logging from Airflow's PostgresHook
logging.getLogger("airflow.providers.postgres.hooks.postgres").setLevel(logging.WARNING)
logging.getLogger("airflow.providers.common.sql.hooks.sql").setLevel(logging.WARNING)

# URL of the LibreTranslate instance (using the local one)
LIBRETRANSLATE_URL = "http://localhost:5000/translate"

# Initializes the language detector
LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.ITALIAN
).build()
    
# Custom translations dictionary with regex patterns, needed when translating to english
# Format: 'regex_pattern': 'english_translation'
CUSTOM_TRANSLATIONS = {
    r'\boccupat[oiae]?\b': "inhabited",   # This translates "occupato", "occupati", "occupata" and "occupate" to "inhabited"
    r'\bocupad[oiae]\b': "inhabited",
    r'\bliber[oiae]?\b': "vacant",        # Similarly, this translates "libero", "liberi", "libera" and "libere" to "vacant"
    r'\blibr[oiae]\b': "vacant",
    r'\blibrr[oa]\b': "vacant",
    r'\bvaci[oa]\b': "vacant",          # There is at least one listing with this word, which is actually spanish
    r'\bautonom[oa]\b': "independent",
    r'\bv.\b': "via",
    r'\bn.\b': "number",
    r'\bp.zza\b': "square",
    r'\btelematic[oia]\b': "on line",
    r'\btelematiche\b': "on line",
    r'\bmq\b': "squared meters",
    r'\bpl.\b': "floor",
    r'\bimm.\b': "real estate",
    r'\bv.le\b': "avenue",
    r'\bviale\b': "avenue",
    r'\b(?<!in )corso\b': "boulevard",
    r'\bc.so\b': "boulevard",
    r'\bcucina abitabile\b': "eat-in kitchen",
    r'\bcucina cucinotto\b': "kitchenette",
    r'\bcucina a vista\b': "open kitchen",
    r'\bcucina semi abitabile\b': "semi eat-in kitchen",
    r'\bcucina angolo cottura\b': "cooking corner",
    r'\bportiere\b': "doorman",
    r'\bpasso carrabile\b': "driveway",
    r'\bbox privato\b': "private parking spot",
    r'\bterratetto\b': "terraced house",
    r'\bvilletta a schiera\b': "terraced house",
    r'\bvilla a schiera\b': "terraced house",
    r'\barea accoglienza\b': "front desk",
    r'\binfissi\b': "window frames",
    r'\bdb_fullname_external_fixtures. ♪\b': "window frames outdoor glass / wood"
    
}

# -------------------------------------------------------------------------------
# ----------------------------CLASS DEFINITIONS----------------------------------
# -------------------------------------------------------------------------------

class DateEncoder(json.JSONEncoder):
    """
    This definition overrides the default method of the custom JSON encoder so that
    it can handle datetime objects, which it normally would not be possible.
    This script uses Airflow's XCom mechanism, which needs data to be serializable.
    """

    def default(self, obj):
        """Encode datetime.date objects into a serializable mapping for JSON."""
        if isinstance(obj, datetime.date):
            return {"__date__": True, "year": obj.year, "month": obj.month, "day": obj.day}
        return super().default(obj)

# -------------------------------------------------------------------------------
# ---------------------------FUNCTION DEFINITIONS--------------------------------
# -------------------------------------------------------------------------------

def initialize_translation_cache():
    """
    This function creates a SQLite database to use as a cache for translations. This
    way we avoid re-translating the same text multiple times
    """

    # Defines the cache location and name
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(cache_dir, 'mongoDB_PostgreSQL_migration_translation_cache.db')
    
    # Creates the cache table if it doesn't exist
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS translation_cache (
        source_text TEXT PRIMARY KEY,
        translated_text TEXT,
        source_lang TEXT,
        target_lang TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()
    
    logger.info(f"Translation cache initialized at {db_path}")
    return db_path

# -------------------------------------------------------------------------------

def get_from_cache(text, source_lang="it", target_lang="en", cache_path=None):
    """
    This function gets a translation from the cache, if it exists
    """

    if not cache_path:
        return None
        
    try:
        # Opens SQLite connection and defines cursor object
        conn = sqlite3.connect(cache_path)
        cursor = conn.cursor()

        # Executes query to get the translation
        cursor.execute(
            "SELECT translated_text FROM translation_cache WHERE source_text=? AND source_lang=? AND target_lang=?", 
            (text, source_lang, target_lang)
        )
        result = cursor.fetchone()

        # Closes the SQLite connection
        conn.close()
        
        if result:
            return result[0]
        return None
    
    except Exception as e:
        logger.error(f"Error retrieving from translation cache: {str(e)}")
        return None

# -------------------------------------------------------------------------------

def save_to_cache(text, translated_text, source_lang="it", target_lang="en", cache_path=None):
    """
    This function saves a translation into to the SQLite cache
    """

    if not cache_path:
        return
        
    try:
        # Opens SQLite connection and defines cursor object
        conn = sqlite3.connect(cache_path)
        cursor = conn.cursor()

        # Executes query to save the translation
        cursor.execute(
            "INSERT OR REPLACE INTO translation_cache (source_text, translated_text, source_lang, target_lang) VALUES (?, ?, ?, ?)",
            (text, translated_text, source_lang, target_lang)
        )
        conn.commit()

        # Closes the SQLite connection
        conn.close()

    except Exception as e:
        logger.error(f"Error saving to translation cache: {str(e)}")

# -------------------------------------------------------------------------------

def batch_translate(texts, source_lang="it", target_lang="en", batch_size=100, max_retries=10, cache_path=None):
    """
    This function translates a batch of texts at once using LibreTranslate with custom translation placeholders
    
    Args:
        texts: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code
        batch_size: Maximum number of texts per batch
        max_retries: Maximum number of retry attempts
        cache_path: Path to translation cache
        
    Returns:
        dict: Dictionary mapping original texts to their translations
    """
    if not texts:
        return {}
        
    results = {}
    
    # First check cache for any existing translations
    cached_results = {}
    texts_to_translate = []
    
    for text in texts:
        cached = get_from_cache(text, source_lang, target_lang, cache_path)
        if cached:
            cached_results[text] = cached
        else:
            texts_to_translate.append(text)
    
    # Return early if all translations were in cache
    if not texts_to_translate:
        return cached_results
    
    # For texts that need translation, prepare placeholders
    text_placeholders = {}
    placeholder_texts = []
    
    for text in texts_to_translate:
        # Preprocess the text
        preprocessed = preprocess_text(text)
        
        # Special case: if the entire text exactly matches a pattern, use translation directly
        direct_match = False
        for pattern, translation in CUSTOM_TRANSLATIONS.items():
            if re.match(f'^{pattern}$', preprocessed.lower(), re.IGNORECASE):
                cached_results[text] = translation
                save_to_cache(text, translation, source_lang, target_lang, cache_path)
                direct_match = True
                break
        
        if direct_match:
            continue
        
        # Create placeholders for custom translations
        placeholders = {}
        placeholder_text = preprocessed
        
        # Replace custom translation patterns with placeholders
        for pattern, translation in CUSTOM_TRANSLATIONS.items():
            # Prepare the search pattern with word boundaries if needed
            if not (pattern.startswith(r'\b') or pattern.endswith(r'\b')):
                search_pattern = r'\b' + pattern + r'\b'
            else:
                search_pattern = pattern
            
            # Replace all occurrences at once
            matches = list(re.finditer(search_pattern, placeholder_text, flags=re.IGNORECASE))
            
            # Process matches in reverse order (to avoid index issues)
            for i, match in enumerate(reversed(matches)):
                placeholder = f"[PLACEHOLDER_{len(placeholders)}]"
                placeholders[placeholder] = translation
                start, end = match.span()
                placeholder_text = placeholder_text[:start] + placeholder + placeholder_text[end:]
        
        # Store the placeholder mapping for this text
        text_placeholders[text] = placeholders
        placeholder_texts.append((text, placeholder_text))
    
    # Processes translations in batches to avoid overwhelming the API
    for i in range(0, len(placeholder_texts), batch_size):
        batch = placeholder_texts[i:i+batch_size]
        
        for attempt in range(max_retries):
            try:
                # Prepares the request payload
                payload = {
                    "q": [item[1] for item in batch],  # Send the placeholder texts
                    "source": source_lang,
                    "target": target_lang
                }
                
                # Sends the request to LibreTranslate
                response = requests.post(LIBRETRANSLATE_URL, json=payload, timeout=30)
                
                if response.status_code == 200:
                    # If the request was successful, parses the response
                    translated_batch = response.json().get("translatedText", [])
                    
                    # Process translations and replace placeholders
                    for j, (original_text, placeholder_text) in enumerate(batch):
                        if j < len(translated_batch):
                            translated = translated_batch[j]
                            
                            # Replace placeholders with their translations
                            placeholders = text_placeholders.get(original_text, {})
                            for placeholder, custom_translation in placeholders.items():
                                translated = translated.replace(placeholder, custom_translation)
                            
                            # Remove excessive underscores (common issue in translations)
                            # translated = re.sub(r'_+', ' ', translated).strip()
                            
                            # Store the result and save to cache
                            cached_results[original_text] = translated
                            save_to_cache(original_text, translated, source_lang, target_lang, cache_path)
                    
                    # Break out of retry loop (success)
                    break
                else:
                    # Logs error message and waits 2s before retrying
                    logger.warning(f"Batch translation attempt {attempt+1} failed with status code {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2)

            except Exception as e:
                # Logs error message and waits 2s before retrying
                logger.error(f"Error during batch translation attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2) 
    
    # For any texts that failed batch translation, try individual translation
    for text in texts_to_translate:
        if text not in cached_results:
            translated = translate_text(text, source_lang, target_lang, max_retries, cache_path)
            cached_results[text] = translated
    
    return cached_results

# -------------------------------------------------------------------------------

def calculate_monthly_payment(price, mortgage_rate, loan_term_years=30, down_payment_percent=20):
    """
    This function calculates the monthly mortgage payment for a given price and mortgage rate.
    It assumes a 30-year mortgage with 20% down payment.
    Returns the value of the monthly payment,
    """

    if price is None or mortgage_rate is None or price <= 0 or mortgage_rate <= 0:
        return None
    
    # Calculates the loan amount (price minus down payment)
    loan_amount = price * ( 1 - down_payment_percent / 100)
    
    # Converts annual rate to monthly decimal rate
    monthly_interest_rate = mortgage_rate / 12
    
    # Number of monthly payments
    num_payments = loan_term_years * 12
    
    # Calculates the monthly payment using the formula: P = L[c(1 + c)^n]/[(1 + c)^n - 1]
    # Where P = payment, L = loan amount, c = monthly interest rate, n = number of payments
    if monthly_interest_rate == 0:
        # If the interest rate is 0, divides the loan amount by the number of payments
        monthly_payment = loan_amount / num_payments
    else:
        monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) / ((1 + monthly_interest_rate)**num_payments - 1)
    
    return monthly_payment

# -------------------------------------------------------------------------------

def compute_adjusted_mortgage_rates():
    """
    This function computes the mortgage rates to apply to each scraping date. It looks at
    'sale' and 'auction' listings only. If only one rate is present for a given scraping_date,
    that is the rate associated to that date. If more than one rate is present for a given
    scraping_date, the average of those rates is applied to that date. 
    Returns the dictionary {scraping_date: adjusted_rate}
    """

    logger.info("Computing adjusted mortgage rates...")
    
    # Connects to MongoDB
    client = get_mongo_client()
    try:
        db = client['listing_website_warehouse']

        # List containing the names of the collections we will process ("sale" and "auction")
        collections = ['sale', 'auction']
    
        # Dictionary to store date -> [mortgage_rates] mapping
        date_mortgage_rates = {}

        # Processes each collection
        for collection_name in collections:
            logger.info("Processing data to compute adjusted mortgage rates for "+collection_name+" collection")
            logger.info("Gathering data from mongoDB warehouse...")
            # Gets all the documents in the collection
            cursor = db[collection_name].find({})
            logger.info("Processing data...")
            for document in cursor:
                # Processes each data entry in the document
                for data_entry in document.get('data', []):
                    # Gets the scraping date
                    scraping_date = data_entry.get('scraping_date')
                    if not scraping_date:
                        continue

                    # Initializes a list for this date if needed
                    if scraping_date not in date_mortgage_rates:
                        date_mortgage_rates[scraping_date] = []

                    # Checks if there's a mortgage rate
                    listing_features = data_entry.get('listing_features', {})
                    additional_costs = listing_features.get('additional_costs', {})
                    mortgage_rate = additional_costs.get('mortgage_rate')

                    # Adds non-null rates only
                    if mortgage_rate is not None:
                        date_mortgage_rates[scraping_date].append(mortgage_rate)
    finally:
        client.close()
    
    # Calculates adjusted rates
    adjusted_rates = {}
    for date, rates in date_mortgage_rates.items():
        logger.info("Computing adjusted mortgage rates for data scraped on "+str(date))
        if len(rates) == 1:
            adjusted_rates[date] = rates[0]
        elif len(rates) > 1:
            adjusted_rates[date] = sum(rates) / len(rates)
        else:
            logger.warning(f"No non-null mortgage rates found for date {date}")
    
    logger.info(f"Computed adjusted mortgage rates for {len(adjusted_rates)} dates")
    return adjusted_rates

# -------------------------------------------------------------------------------

def convert_dict_tuple_keys_to_str(d):
    """
    This function converts all tuple keys in a dictionary to string keys. This is
    needed for JSON serialization. The PostgreSQL database dimension mappings below
    use tuple as dictionary keys, which are not natively supported by JSON.
    """

    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        if isinstance(k, tuple):
            # Use custom encoder for JSON serialization to handle date objects
            result[json.dumps(k, cls=DateEncoder)] = v
        else:
            result[k] = v
    return result

# -------------------------------------------------------------------------------

def convert_nested_dict_tuple_keys_to_str(d):
    """
    This function converts all tuple keys in a nested dictionary to string keys. Like
    the function above, this is needed for JSON serialization.
    """

    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = convert_nested_dict_tuple_keys_to_str(v)
        else:
            result[k] = v
    return convert_dict_tuple_keys_to_str(result)

# -------------------------------------------------------------------------------

def convert_dict_str_keys_to_tuple(d):
    """
    This function convert string keys that look like JSON arrays back to tuple keys.
    It is the reverse operation of the function 'convert_dict_tuple_keys_to_str', and
    it's needed to reverse the tuple-to-string to deserialize JSON data.
    """

    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        try:
            if isinstance(k, str) and k.startswith('[') and k.endswith(']'):
                # This looks like a JSON array, try to parse it and convert to tuple
                parsed = json.loads(k)
                if isinstance(parsed, list):
                    # Process the list to convert any date dictionaries back to date objects
                    processed_list = []
                    for item in parsed:
                        if isinstance(item, dict) and item.get('__date__') is True:
                            # Convert back to date object
                            processed_list.append(
                                datetime.date(item['year'], item['month'], item['day'])
                            )
                        else:
                            processed_list.append(item)
                    result[tuple(processed_list)] = v
                else:
                    result[k] = v
            else:
                result[k] = v
        except (json.JSONDecodeError, TypeError):
            # If there's an error parsing, just use the original key
            result[k] = v
    return result

# -------------------------------------------------------------------------------

def convert_nested_dict_str_keys_to_tuple(d):
    """
    Similarly to above.
    """

    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = convert_nested_dict_str_keys_to_tuple(v)
        else:
            result[k] = v
    return convert_dict_str_keys_to_tuple(result)

# -------------------------------------------------------------------------------

def convert_mongo_date(date_str):
    """
    This function converts date strings into PostgreSQL date format. It is needed when
    loading dates from the mongoDB warehouse (where they are stored as strings) into the
    PostgreSQL warehouse (where they are stored as DATE objects).
    """

    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        try:
            return datetime.datetime.strptime(date_str, '%d/%m/%Y').date()
        except (ValueError, TypeError):
            return None
        
# -------------------------------------------------------------------------------

def execute_query_silent(postgres_hook, query, params=None):
    """
    This function sxecutes an SQL query without logging the full query. It is
    needed to avoid unnecessarily long and verbose task logs in Airflow.
    """

    # Connects to PostgreSQL
    conn = postgres_hook.get_conn()

    # Defines the cursor
    cursor = conn.cursor()

    try:
        # Executes the query
        cursor.execute(query, params)

        # Commits the query
        conn.commit()

        if cursor.description:
            # If the query actually retrieved data, returns it
            return cursor.fetchall()
        return None
    
    finally:
        # Closes PostgreSQL connection
        cursor.close()

# -------------------------------------------------------------------------------

def check_all_nulls(values):
    """
    This function simply checks if all values in the list passed as argunent are None.
    """

    return all(v is None for v in values)

# -------------------------------------------------------------------------------

def get_existing_null_record_id(postgres_hook, table_name, id_column, condition_columns=None):
    """
    This function gets the ID of an existing record with all NULL values for a given table and column.
    This function is needed to avoid the creation of duplicate records with all NULL values in dimension tables.
    """

    if condition_columns is None:
        # If this argument is not set, uses all columns except the ID column for the SQL query
        query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        AND column_name != '{id_column}'
        """
        # Executes the query and returns the results
        column_results = execute_query_silent(postgres_hook, query)
        condition_columns = [row[0] for row in column_results]
    
    # WHERE clause for all NULL columns
    where_clause = " AND ".join([f"{col} IS NULL" for col in condition_columns])
    
    # Query for ID of a record with all NULL values
    query = f"""
    SELECT {id_column} 
    FROM {table_name} 
    WHERE {where_clause}
    LIMIT 1
    """
    
    # Executes the query and returns the results
    result = execute_query_silent(postgres_hook, query)
    return result[0][0] if result else None

# -------------------------------------------------------------------------------

def process_total_room_number(value):
    """
    This function processes the total_room_number field to handle special string values like '2 - 5+'.
    In fact, this field is sometimes stored as an integer and sometimes as a string in the mongoDB warehouse.
    """

    # If the 
    if value is None:
        return None
    
    # If it's already an integer or float, return it directly
    if isinstance(value, (int, float)):
        return int(value)
    
    # Convert to string and remove whitespaces
    value_str = str(value).replace(" ", "")
    
    try:
        # If it's a simple number, return it
        return int(value_str)
    except ValueError:
        # Handle ranges like "2-5" or "2-5+"
        if "-" in value_str:
            parts = value_str.split("-")
            last_part = parts[-1]
            
            # Handle the case where the last part has a + sign
            if "+" in last_part:
                # Remove the + and add 1 to the number
                num = int(last_part.replace("+", ""))
                return num + 1
            else:
                # Return the largest number in the range
                return int(last_part)
        
        # Handle cases with just a + sign like "5+"
        elif "+" in value_str:
            num = int(value_str.replace("+", ""))
            return num + 1
        
        # If we can't parse it, return None
        logger.warning(f"Could not parse total_room_number value: {value}")
        return None
    
# -------------------------------------------------------------------------------

def translate_text(text, source_lang="it", target_lang="en", max_retries=10, cache_path=None):
    """
    This function translates text using LibreTranslate, and uses the regex-based custom translations
    defined at the beginning for specific Italian words or expressions.
    
    Uses a placeholder approach:
    1. Replace custom translations with unique placeholders
    2. Send text with placeholders to LibreTranslate
    3. After translation, replace placeholders with English translations
    
    Args:
        text: Text to translate
        source_lang: Source language code (default: "it" for Italian)
        target_lang: Target language code (default: "en" for English)
        max_retries: Maximum number of retry attempts
        cache_path: Path to the translation cache database
        
    Returns:
        str: Translated text or original text if translation fails
    """
    if not text or not isinstance(text, str):
        return text
    
    # Checks cache first
    cached_translation = get_from_cache(text, source_lang, target_lang, cache_path)
    if cached_translation:
        return cached_translation
    
    # Preprocesses text to improve readability and translation quality
    preprocessed_text = preprocess_text(text)
    
    # Detects the language
    detected_lang = LANGUAGE_DETECTOR.detect_language_of(preprocessed_text)
    
    # If already in English, returns the preprocessed text
    if detected_lang == Language.ENGLISH:
        return preprocessed_text
    
    # If not Italian or English, logs a warning but proceed with translation attempt
    if detected_lang and detected_lang != Language.ITALIAN:
        logger.warning(f"Text detected as language other than Italian or English: {detected_lang}")
    
    # Special case: if the entire text exactly matches a pattern, return the translation directly
    for pattern, translation in CUSTOM_TRANSLATIONS.items():
        if re.match(f'^{pattern}$', preprocessed_text.lower(), re.IGNORECASE):
            # Saves to cache for future use
            save_to_cache(text, translation, source_lang, target_lang, cache_path)
            return translation
    
    # Create a dictionary to track placeholders and their translations
    placeholders = {}
    placeholder_text = preprocessed_text
    
    # Replace custom translation patterns with placeholders
    for pattern, translation in CUSTOM_TRANSLATIONS.items():
        # Prepare the search pattern with word boundaries if needed
        if not (pattern.startswith(r'\b') or pattern.endswith(r'\b')):
            search_pattern = r'\b' + pattern + r'\b'
        else:
            search_pattern = pattern
        
        # Find all matches and replace with placeholders
        matches = list(re.finditer(search_pattern, placeholder_text, flags=re.IGNORECASE))
        
        # Process matches in reverse order (to avoid index issues)
        for i, match in enumerate(reversed(matches)):
            placeholder = f"__CUSTOM_TRANS_{len(placeholders)}__"
            placeholders[placeholder] = translation
            start, end = match.span()
            placeholder_text = placeholder_text[:start] + placeholder + placeholder_text[end:]
    
    # Translate the text with placeholders
    for attempt in range(max_retries):
        try:
            # Prepares the request payload
            payload = {
                "q": placeholder_text,
                "source": source_lang,
                "target": target_lang
            }
            
            # Sends the request to LibreTranslate
            response = requests.post(LIBRETRANSLATE_URL, json=payload, timeout=10)
            
            # Checks if the request was successful
            if response.status_code == 200:
                # Parses the response
                result = response.json()
                # Gets the translated text
                translated_text = result.get("translatedText", placeholder_text)
                
                # Replace placeholders with their translations
                final_text = translated_text
                for placeholder, custom_translation in placeholders.items():
                    final_text = final_text.replace(placeholder, custom_translation)
                
                # Remove excessive underscores (common issue in translations)
                final_text = re.sub(r'_+', ' ', final_text).strip()
                
                # Saves to cache for future use
                save_to_cache(text, final_text, source_lang, target_lang, cache_path)
                
                return final_text
            else:
                # Logs an error message and waits 1 second before retrying
                logger.warning(f"Translation attempt {attempt+1} failed with status code {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        except Exception as e:
            # Logs an error message and waits 1 second before retrying
            logger.error(f"Error during translation attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1) 
    
    # If translation failed, returns preprocessed text
    return preprocessed_text

# -------------------------------------------------------------------------------

def translate_values_parallel(values, source_lang="it", target_lang="en", max_workers=10, cache_path=None):
    """
    This function translates a list of values in parallel using ThreadPoolExecutor
    and now properly handles custom translations with placeholders
    
    Args:
        values: List of strings to translate
        source_lang: Source language code
        target_lang: Target language code
        max_workers: Maximum number of worker threads
        cache_path: Path to the translation cache database
        
    Returns:
        dict: Dictionary mapping original values to their translations
    """
    if not values:
        return {}
    
    # Deduplicates values to avoid unnecessary translations
    unique_values = list(set(values))
    logger.info(f"Translating {len(unique_values)} unique values in parallel (max workers: {max_workers})")
    
    results = {}
    
    # Try batch translation first with our updated batch_translate function that supports placeholders
    batch_results = batch_translate(unique_values, source_lang, target_lang, cache_path=cache_path)
    
    # For any values that weren't translated in the batch, uses individual translation
    values_to_translate = [v for v in unique_values if v not in batch_results]

    if values_to_translate:
        logger.info(f"Batch translation completed. {len(values_to_translate)} values need individual translation.")

        # Defines a worker function for translation
        def translate_worker(text):
            """Translate a single string with caching and return the original/pair."""
            return text, translate_text(text, source_lang, target_lang, cache_path=cache_path)

        # Uses ThreadPoolExecutor to translate in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for original, translated in executor.map(translate_worker, values_to_translate):
                results[original] = translated
    
    # Combines batch results with individual results
    results.update(batch_results)
    
    return results

# -------------------------------------------------------------------------------

def count_batches():
    '''Count the total number of batches needed for processing'''
    logger.info("Counting total batches needed...")

    # Connect directly to MongoDB to get collection counts
    client = get_mongo_client()
    try:
        db = client['listing_website_warehouse']

        collections = ['rent', 'auction', 'sale']
        total_batches = 0

        for collection_name in collections:
            # Count documents where child_listings is null
            query = {"child_listings": None}
            total_count = db[collection_name].count_documents(query)

            if MAX_RECORDS_PER_COLLECTION:
                process_count = min(total_count, MAX_RECORDS_PER_COLLECTION)
            else:
                process_count = total_count

            # Calculate number of batches for this collection
            collection_batches = (process_count + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
            total_batches += collection_batches

            logger.info(f"Collection '{collection_name}': Total={total_count}, To Process={process_count}, Batches={collection_batches}")

        logger.info(f"Total batches to process: {total_batches}")
        return total_batches
    finally:
        client.close()

# -------------------------------------------------------------------------------

def setup_batch_processing():
    '''This function sets up the batch processing parameters and stores them for subsequent tasks'''
    logger.info("Setting up batch processing parameters...")

    # Compute adjusted mortgage rates and store in Airflow Variables
    adjusted_rates = compute_adjusted_mortgage_rates()
    Variable.set("adjusted_mortgage_rates", json.dumps(adjusted_rates))
    logger.info(f"Stored adjusted mortgage rates for {len(adjusted_rates)} dates")
    
    # Connect directly to MongoDB to get collection counts
    client = get_mongo_client()
    try:
        db = client['listing_website_warehouse']

        collections = ['rent', 'auction', 'sale']
        batch_info = {}

        for collection_name in collections:
            # Count documents where child_listings is null
            query = {"child_listings": None}
            total_count = db[collection_name].count_documents(query)

            if MAX_RECORDS_PER_COLLECTION:
                process_count = min(total_count, MAX_RECORDS_PER_COLLECTION)
            else:
                process_count = total_count

            num_batches = (process_count + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division

            batch_info[collection_name] = {
                'total_count': total_count,
                'process_count': process_count,
                'num_batches': num_batches
            }

            logger.info(f"Collection '{collection_name}': Total={total_count}, To Process={process_count}, Batches={num_batches}")
    finally:
        client.close()

    # Store batch info in Airflow Variables for downstream tasks
    Variable.set("batch_info", json.dumps(batch_info))

    # Prepare a list of all batches to process
    all_batches = []
    batch_num = 0

    for collection_name in collections:
        for i in range(batch_info[collection_name]['num_batches']):
            all_batches.append({
                'collection': collection_name,
                'batch_num': batch_num,
                'skip': i * BATCH_SIZE,
                'limit': BATCH_SIZE
            })
            batch_num += 1

    Variable.set("all_batches", json.dumps(all_batches))
    logger.info(f"Setup complete. Total batches to process: {len(all_batches)}")

    return len(all_batches)

# -------------------------------------------------------------------------------

def process_batch(**kwargs):
    '''Process a specific batch of data'''
    ti = kwargs['ti']
    batch_num = kwargs.get('batch_num', 0)

    # Get all batches info
    all_batches = json.loads(Variable.get("all_batches"))
    if batch_num >= len(all_batches):
        logger.error(f"Invalid batch number: {batch_num}. Total batches: {len(all_batches)}")
        return None

    batch = all_batches[batch_num]
    collection_name = batch['collection']
    skip = batch['skip']
    limit = batch['limit']

    logger.info(f"Processing batch {batch_num}: Collection={collection_name}, Skip={skip}, Limit={limit}")

    # Connect to MongoDB
    client = get_mongo_client()
    try:
        # Only get listings where child_listings is null
        query = {"child_listings": None}
        collection = client["listing_website_warehouse"][collection_name]
        cursor = collection.find(query).skip(skip).limit(limit)

        # Convert cursor to list
        batch_listings = list(cursor)
    finally:
        client.close()
    logger.info(f"Retrieved {len(batch_listings)} listings for this batch")

    for listing in batch_listings:
        listing['collection_name'] = collection_name.lower()  # Convert collection name to lowercase

    # Convert to JSON to handle MongoDB specific types
    json_str = json_util.dumps(batch_listings)
    batch_data = json.loads(json_str)

    # Store the processed dimension mappings for this batch
    dimension_mappings = process_dimensions_for_batch(batch_data)

    # Convert tuple keys to string keys for XCom compatibility
    serializable_dimension_mappings = convert_nested_dict_tuple_keys_to_str(dimension_mappings)

    # Store mappings for this batch in XCom
    ti.xcom_push(key=f'batch_{batch_num}_dimension_mappings', value=serializable_dimension_mappings)
    ti.xcom_push(key=f'batch_{batch_num}_data', value=batch_data)

    logger.info(f"Batch {batch_num} processing complete. Processed {len(batch_data)} records.")
    return f"Batch {batch_num} complete"

# -------------------------------------------------------------------------------

def get_mongo_client():
    host = os.getenv("MONGODB_HOST", "mongodb")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    user = os.getenv("MONGODB_USER")
    password = os.getenv("MONGODB_PASSWORD")
    auth_source = os.getenv("MONGODB_AUTH_SOURCE", "admin")

    if user and password:
        return MongoClient(
            host=host,
            port=port,
            username=user,
            password=password,
            authSource=auth_source,
        )
    return MongoClient(host=host, port=port)

# -------------------------------------------------------------------------------

def process_dimensions_for_batch(batch_data):
    '''Process all dimension tables for a batch of data'''
    logger.info("Processing dimensions for batch data...")
    start_time = time.time()
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    execute_query_silent(postgres_hook, "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;")

    # Load adjusted mortgage rates
    adjusted_rates = json.loads(Variable.get("adjusted_mortgage_rates", "{}"))
    logger.info(f"Loaded adjusted mortgage rates for {len(adjusted_rates)} dates")

    dimension_mappings = {}
    total_listings = len(batch_data)
    dimension_counts = {}  # To track counts for summary

    # Process date dimension
    logger.info("Processing date dimension...")
    date_mappings = {}
    date_rows = set()

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            scraping_date = data_entry.get('scraping_date')
            if scraping_date:
                date_rows.add(scraping_date)

    initial_count = len(date_rows)
    processed_count = 0

    for date_str in date_rows:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == initial_count:
            logger.info(f"Processing date dimension: {processed_count}/{initial_count}")

        date_obj = convert_mongo_date(date_str)
        if date_obj:
            # Convert month name to lowercase
            month_name = date_obj.strftime('%B').lower()

            # Using INSERT ... ON CONFLICT DO NOTHING to handle duplicates
            query = """
            WITH inserted AS (
                INSERT INTO dim_date (date_value, year, month_number, month_name, day_of_month)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (date_value) DO NOTHING
                RETURNING date_id
            )
            SELECT date_id FROM inserted
            UNION ALL
            SELECT date_id FROM dim_date WHERE date_value = %s
            LIMIT 1
            """
            params = (date_obj, date_obj.year, date_obj.month, month_name, date_obj.day, date_obj)
            result = execute_query_silent(postgres_hook, query, params)

            date_id = result[0][0] if result else None
            if date_id:
                date_mappings[date_str] = date_id
            else:
                logger.warning(f"Failed to get date_id for {date_obj}")

    dimension_mappings['date'] = date_mappings
    dimension_counts['date'] = len(date_mappings)

    # Process listing_type dimension
    logger.info("Processing listing_type dimension...")
    listing_type_mappings = {}
    listing_types = {'rent', 'auction', 'sale'}

    # Log progress for listing_type
    processed_count = 0
    total_count = len(listing_types)
    
    for lt in listing_types:
        processed_count += 1
        if processed_count % 1 == 0 or processed_count == total_count:  # Log for each one since there are few
            logger.info(f"Processing listing_type: {processed_count}/{total_count}")
            
        
        lt_lower = lt.lower()

        # Using INSERT ... ON CONFLICT DO NOTHING to handle duplicates
        query = """
        WITH inserted AS (
            INSERT INTO dim_listing_type (listing_type)
            VALUES (%s)
            ON CONFLICT (listing_type) DO NOTHING
            RETURNING listing_type_id
        )
        SELECT listing_type_id FROM inserted
        UNION ALL
        SELECT listing_type_id FROM dim_listing_type WHERE listing_type = %s
        LIMIT 1
        """
        params = (lt_lower, lt_lower)
        result = execute_query_silent(postgres_hook, query, params)

        listing_type_id = result[0][0] if result else None
        if listing_type_id:
            listing_type_mappings[lt_lower] = listing_type_id
        else:
            logger.warning(f"Failed to get listing_type_id for {lt_lower}")

    dimension_mappings['listing_type'] = listing_type_mappings
    dimension_counts['listing_type'] = len(listing_type_mappings)

    # Process seller_type dimension
    logger.info("Processing seller_type dimension...")
    seller_type_mappings = {}
    seller_types = set()

    # Check for existing NULL seller_type
    null_seller_type_id = get_existing_null_record_id(postgres_hook, 'dim_seller_type', 'seller_type_id', ['seller_type'])
    if null_seller_type_id:
        seller_type_mappings[None] = null_seller_type_id
    
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing seller_type: listing #{idx+1}/{len(batch_data)}")
        
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                listing_info = listing_features.get('listing_info')
                if listing_info:
                    seller_type = listing_info.get('seller_type')
                    if seller_type:
                        seller_types.add(seller_type.lower())  

    processed_count = 0
    total_count = len(seller_types)
    
    for st in seller_types:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(f"Processing seller_type values: {processed_count}/{total_count}")
            
        # Using parameterized query
        query = """
        WITH inserted AS (
            INSERT INTO dim_seller_type (seller_type)
            VALUES (%s)
            ON CONFLICT (seller_type) DO NOTHING
            RETURNING seller_type_id
        )
        SELECT seller_type_id FROM inserted
        UNION ALL
        SELECT seller_type_id FROM dim_seller_type WHERE seller_type = %s
        LIMIT 1
        """
        params = (st, st)
        result = execute_query_silent(postgres_hook, query, params)

        seller_type_id = result[0][0] if result else None
        if seller_type_id:
            seller_type_mappings[st] = seller_type_id
        else:
            logger.warning(f"Failed to get seller_type_id for {st}")

    dimension_mappings['seller_type'] = seller_type_mappings
    dimension_counts['seller_type'] = len(seller_type_mappings)

    # Process listing_info dimension
    logger.info("Processing listing_info dimension...")
    listing_info_mappings = {}
    processed_count = 0

    # Define all_null_key upfront
    all_null_key = (None, None, None, None)
    
    # Check for existing all-NULL record in listing_info
    listing_info_columns = ['listing_age', 'listing_last_update', 'number_of_pictures', 'seller_type_id']
    null_listing_info_id = get_existing_null_record_id(postgres_hook, 'dim_listing_info', 'listing_info_id', listing_info_columns)
    
    # If we have an all-NULL record, use it
    if null_listing_info_id:
        listing_info_mappings[all_null_key] = null_listing_info_id
        logger.info(f"Found existing all-NULL listing_info record with ID {null_listing_info_id}")

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing listing_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            listing_info = listing_features.get('listing_info')
            if not listing_info:
                continue

            seller_type = listing_info.get('seller_type')
            if seller_type and isinstance(seller_type, str):
                seller_type = seller_type.lower()  
            seller_type_id = seller_type_mappings.get(seller_type)

            # If no seller_type_id, try to use the NULL mapping
            if not seller_type_id and None in seller_type_mappings:
                seller_type_id = seller_type_mappings[None]

            if seller_type_id:
                listing_age = listing_info.get('listing_age')
                listing_last_update = listing_info.get('listing_last_update')
                number_of_photos = listing_info.get('number_of_photos')

                # Create a unique key for this listing_info
                listing_info_key = (
                    listing_age,
                    listing_last_update,
                    number_of_photos,
                    seller_type_id
                )

                # Check if this is an all-NULL key (except seller_type_id which can't be NULL due to FK constraint)
                if listing_age is None and listing_last_update is None and number_of_photos is None:
                    # Skip if we already have an all-NULL record
                    if all_null_key in listing_info_mappings:
                        listing_info_mappings[listing_info_key] = listing_info_mappings[all_null_key]
                        continue

                # Skip if we've already processed this combination
                if listing_info_key in listing_info_mappings:
                    continue

                processed_count += 1

                # Prepare query with parameters
                query = """
                WITH inserted AS (
                    INSERT INTO dim_listing_info (
                        listing_age, listing_last_update, number_of_pictures, seller_type_id
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (listing_age, listing_last_update, number_of_pictures, seller_type_id)
                    DO NOTHING
                    RETURNING listing_info_id
                )
                SELECT listing_info_id FROM inserted
                UNION ALL
                SELECT listing_info_id FROM dim_listing_info
                WHERE (listing_age IS NULL AND %s IS NULL OR listing_age = %s)
                AND (listing_last_update IS NULL AND %s IS NULL OR listing_last_update = %s)
                AND (number_of_pictures IS NULL AND %s IS NULL OR number_of_pictures = %s)
                AND seller_type_id = %s
                LIMIT 1
                """
                params = (
                    listing_age, listing_last_update, number_of_photos, seller_type_id,
                    listing_age, listing_age, listing_last_update, listing_last_update,
                    number_of_photos, number_of_photos, seller_type_id
                )
                result = execute_query_silent(postgres_hook, query, params)

                listing_info_id = result[0][0] if result else None
                if listing_info_id:
                    listing_info_mappings[listing_info_key] = listing_info_id
                else:
                    logger.warning(f"Failed to get listing_info_id for key (reduced detail in logs)")

    dimension_mappings['listing_info'] = listing_info_mappings
    dimension_counts['listing_info'] = len(listing_info_mappings)

    # Process availability dimension
    logger.info("Processing availability dimension...")
    availability_mappings = {}
    availability_values = set()

    # Check for existing NULL availability
    null_availability_id = get_existing_null_record_id(postgres_hook, 'dim_availability', 'availability_id', ['availability'])
    if null_availability_id:
        availability_mappings[None] = null_availability_id
    
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing availability: listing #{idx+1}/{len(batch_data)}")
            
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                availability = listing_features.get('availability')
                if availability:
                    availability_values.add(availability.lower())  

    processed_count = 0
    total_count = len(availability_values)
    
    for availability in availability_values:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(f"Processing availability values: {processed_count}/{total_count}")
            
        # Using parameterized query
        query = """
        WITH inserted AS (
            INSERT INTO dim_availability (availability)
            VALUES (%s)
            ON CONFLICT (availability) DO NOTHING
            RETURNING availability_id
        )
        SELECT availability_id FROM inserted
        UNION ALL
        SELECT availability_id FROM dim_availability WHERE availability = %s
        LIMIT 1
        """
        params = (availability, availability)
        result = execute_query_silent(postgres_hook, query, params)

        availability_id = result[0][0] if result else None
        if availability_id:
            availability_mappings[availability] = availability_id
        else:
            logger.warning(f"Failed to get availability_id for {availability}")

    dimension_mappings['availability'] = availability_mappings
    dimension_counts['availability'] = len(availability_mappings)

    # Process type_of_property dimension
    logger.info("Processing type_of_property dimension...")
    type_of_property_mappings = {}
    type_of_property_values = set()

    # Check for existing NULL type_of_property
    null_type_id = get_existing_null_record_id(postgres_hook, 'dim_type_of_property', 'type_of_property_id', ['type_of_property'])
    if null_type_id:
        type_of_property_mappings[None] = null_type_id
    
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing type_of_property: listing #{idx+1}/{len(batch_data)}")
            
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            type_of_property = listing_features.get('type_of_property')
            if type_of_property and type_of_property.get('class') and isinstance(type_of_property.get('class'), str):
                type_of_property_values.add(type_of_property.get('class').lower())  

    processed_count = 0
    total_count = len(type_of_property_values)
    
    for property_type in type_of_property_values:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(f"Processing type_of_property values: {processed_count}/{total_count}")
            
        # Using parameterized query
        query = """
        WITH inserted AS (
            INSERT INTO dim_type_of_property (type_of_property)
            VALUES (%s)
            ON CONFLICT (type_of_property) DO NOTHING
            RETURNING type_of_property_id
        )
        SELECT type_of_property_id FROM inserted
        UNION ALL
        SELECT type_of_property_id FROM dim_type_of_property WHERE type_of_property = %s
        LIMIT 1
        """
        params = (property_type, property_type)
        result = execute_query_silent(postgres_hook, query, params)

        type_id = result[0][0] if result else None
        if type_id:
            type_of_property_mappings[property_type] = type_id
        else:
            logger.warning(f"Failed to get type_of_property_id for {property_type}")

    dimension_mappings['type_of_property'] = type_of_property_mappings
    dimension_counts['type_of_property'] = len(type_of_property_mappings)

    # Process condition dimension
    logger.info("Processing condition dimension...")
    condition_mappings = {}
    condition_values = set()

    # Check for existing NULL condition
    null_condition_id = get_existing_null_record_id(postgres_hook, 'dim_condition', 'condition_id', ['condition'])
    if null_condition_id:
        condition_mappings[None] = null_condition_id
    
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing condition: listing #{idx+1}/{len(batch_data)}")
            
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                condition = listing_features.get('condition')
                if condition:
                    condition_values.add(condition.lower())  

    processed_count = 0
    total_count = len(condition_values)
    
    for condition in condition_values:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(f"Processing condition values: {processed_count}/{total_count}")
            
        # Using parameterized query
        query = """
        WITH inserted AS (
            INSERT INTO dim_condition (condition)
            VALUES (%s)
            ON CONFLICT (condition) DO NOTHING
            RETURNING condition_id
        )
        SELECT condition_id FROM inserted
        UNION ALL
        SELECT condition_id FROM dim_condition WHERE condition = %s
        LIMIT 1
        """
        params = (condition, condition)
        result = execute_query_silent(postgres_hook, query, params)

        condition_id = result[0][0] if result else None
        if condition_id:
            condition_mappings[condition] = condition_id
        else:
            logger.warning(f"Failed to get condition_id for {condition}")

    dimension_mappings['condition'] = condition_mappings
    dimension_counts['condition'] = len(condition_mappings)

    # Process features dimension
    logger.info("Processing features dimension...")
    features_mappings = {}
    feature_values = set()

    # Check for existing NULL feature
    null_feature_id = get_existing_null_record_id(postgres_hook, 'dim_features', 'feature_id', ['feature_name'])
    if null_feature_id:
        features_mappings[None] = null_feature_id
    
    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            features = listing_features.get('additional_features')
            if features:
                for feature in features:
                    if feature:
                        feature_values.add(feature.lower())  

    feature_count = 0
    for feature in feature_values:
        feature_count += 1
        if feature_count % 10 == 0 or feature_count == len(feature_values):
            logger.info(f"Processing features: {feature_count}/{len(feature_values)}")

        # Using parameterized query
        query = """
        WITH inserted AS (
            INSERT INTO dim_features (feature_name)
            VALUES (%s)
            ON CONFLICT (feature_name) DO NOTHING
            RETURNING feature_id
        )
        SELECT feature_id FROM inserted
        UNION ALL
        SELECT feature_id FROM dim_features WHERE feature_name = %s
        LIMIT 1
        """
        params = (feature, feature)
        result = execute_query_silent(postgres_hook, query, params)

        feature_id = result[0][0] if result else None
        if feature_id:
            features_mappings[feature] = feature_id
        else:
            logger.warning(f"Failed to get feature_id for {feature}")

    dimension_mappings['features'] = features_mappings
    dimension_counts['features'] = len(features_mappings)

    # Process additional_costs dimension
    logger.info("Processing additional_costs dimension...")
    additional_costs_mappings = {}
    
    # Define all_null_key upfront
    all_null_key = (None, None, None, None)
    
    # Check for existing all-NULL record in additional_costs
    cost_columns = ['condominium_monthly_expenses', 'heating_yearly_expenses', 'mortgage_rate', 'monthly_payment']
    null_costs_id = get_existing_null_record_id(postgres_hook, 'dim_additional_costs', 'additional_costs_id', cost_columns)
    
    # If we have an all-NULL record, use it
    if null_costs_id:
        additional_costs_mappings[all_null_key] = null_costs_id
        logger.info(f"Found existing all-NULL additional_costs record with ID {null_costs_id}")

    processed_count = 0
    unique_costs = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing additional_costs: listing #{idx+1}/{len(batch_data)}")

        # Get the listing type for applying mortgage rate adjustments
        listing_type = listing['collection_name']
        
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            scraping_date = data_entry.get('scraping_date')
            price = listing_features.get('price')
            
            additional_costs = listing_features.get('additional_costs')
            if not additional_costs:
                continue

            # Initialize other_costs as empty dict if None
            other_costs = additional_costs.get('other_costs')
            if other_costs is None:
                other_costs = {}

            # Get original mortgage rate
            mortgage_rate = additional_costs.get('mortgage_rate')
            
            # For sale and auction listings, use adjusted rate if mortgage_rate is None
            if listing_type in ['sale', 'auction'] and mortgage_rate is None:
                mortgage_rate = adjusted_rates.get(scraping_date)
                logger.debug(f"Using adjusted mortgage rate {mortgage_rate} for date {scraping_date}")
            
            # Extract condominium expenses and heating expenses
            condominium_monthly_expenses = other_costs.get('condominium_expenses')  # Match actual field name
            heating_yearly_expenses = other_costs.get('heating_expenses')  # Match actual field name

            # Extract numeric values from text if present
            if condominium_monthly_expenses and isinstance(condominium_monthly_expenses, str):
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', condominium_monthly_expenses)
                condominium_monthly_expenses = float(numeric_match.group(1)) if numeric_match else None

            if heating_yearly_expenses and isinstance(heating_yearly_expenses, str):
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', heating_yearly_expenses)
                heating_yearly_expenses = float(numeric_match.group(1)) if numeric_match else None

            # Calculate monthly payment for sale and auction listings
            monthly_payment = None
            if listing_type in ['sale', 'auction'] and mortgage_rate is not None and price is not None:
                monthly_payment = calculate_monthly_payment(price, mortgage_rate)
                logger.debug(f"Calculated monthly payment {monthly_payment} for price {price} and rate {mortgage_rate}")

            # Create a unique key including the monthly payment
            cost_key = (
                condominium_monthly_expenses,
                heating_yearly_expenses,
                mortgage_rate,
                monthly_payment
            )

            # Check if this is an all-NULL key
            if check_all_nulls([condominium_monthly_expenses, heating_yearly_expenses, mortgage_rate, monthly_payment]):
                # Skip if we already have an all-NULL record
                if all_null_key in additional_costs_mappings:
                    additional_costs_mappings[cost_key] = additional_costs_mappings[all_null_key]
                    continue

            if cost_key in unique_costs or cost_key in additional_costs_mappings:
                continue

            unique_costs.add(cost_key)
            processed_count += 1

            # Check if this combination already exists using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_additional_costs (
                    condominium_monthly_expenses, heating_yearly_expenses, mortgage_rate, monthly_payment
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (condominium_monthly_expenses, heating_yearly_expenses, mortgage_rate, monthly_payment)
                DO NOTHING
                RETURNING additional_costs_id
            )
            SELECT additional_costs_id FROM inserted
            UNION ALL
            SELECT additional_costs_id FROM dim_additional_costs
            WHERE (condominium_monthly_expenses IS NULL AND %s IS NULL OR condominium_monthly_expenses = %s)
            AND (heating_yearly_expenses IS NULL AND %s IS NULL OR heating_yearly_expenses = %s)
            AND (mortgage_rate IS NULL AND %s IS NULL OR mortgage_rate = %s)
            AND (monthly_payment IS NULL AND %s IS NULL OR monthly_payment = %s)
            LIMIT 1
            """
            params = (
                condominium_monthly_expenses, heating_yearly_expenses, mortgage_rate, monthly_payment,
                condominium_monthly_expenses, condominium_monthly_expenses,
                heating_yearly_expenses, heating_yearly_expenses,
                mortgage_rate, mortgage_rate,
                monthly_payment, monthly_payment
            )
            result = execute_query_silent(postgres_hook, query, params)

            additional_costs_id = result[0][0] if result else None
            if additional_costs_id:
                additional_costs_mappings[cost_key] = additional_costs_id

    dimension_mappings['additional_costs'] = additional_costs_mappings
    dimension_counts['additional_costs'] = len(additional_costs_mappings)

    # Process category dimension
    logger.info("Processing category dimension...")
    category_mappings = {}
    category_values = set()

    # Check for existing NULL category
    null_category_id = get_existing_null_record_id(postgres_hook, 'dim_category', 'category_id', ['category_name'])
    if null_category_id:
        category_mappings[None] = null_category_id
    
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing category: listing #{idx+1}/{len(batch_data)}")
            
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            type_of_property = listing_features.get('type_of_property')
            if type_of_property:
                category = type_of_property.get('category')
                if category and isinstance(category, str):
                    category_values.add(category.lower())  

            # Also check for auction's lot category
            auction_info = listing_features.get('auction_info')
            if auction_info:
                lot_category = auction_info.get('lot_category')
                if lot_category and lot_category.get('name') and isinstance(lot_category.get('name'), str):
                    category_values.add(lot_category.get('name').lower())  

    processed_count = 0
    total_count = len(category_values)
    
    for category in category_values:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(f"Processing category values: {processed_count}/{total_count}")
            
        # Using parameterized query
        query = """
        WITH inserted AS (
            INSERT INTO dim_category (category_name)
            VALUES (%s)
            ON CONFLICT (category_name) DO NOTHING
            RETURNING category_id
        )
        SELECT category_id FROM inserted
        UNION ALL
        SELECT category_id FROM dim_category WHERE category_name = %s
        LIMIT 1
        """
        params = (category, category)
        result = execute_query_silent(postgres_hook, query, params)

        category_id = result[0][0] if result else None
        if category_id:
            category_mappings[category] = category_id
        else:
            logger.warning(f"Failed to get category_id for {category}")

    dimension_mappings['category'] = category_mappings
    dimension_counts['category'] = len(category_mappings)

    # Process rooms_info dimension
    logger.info("Processing rooms_info dimension...")
    rooms_info_mappings = {}
    
    # Define all_null_key upfront
    all_null_key = (None, None, None, None, None, None)
    
    # Check for existing all-NULL record in rooms_info
    rooms_columns = ['bathrooms_number', 'bedrooms_number', 'total_room_number', 'kitchen_status', 'garage', 'floor']
    null_rooms_id = get_existing_null_record_id(postgres_hook, 'dim_rooms_info', 'rooms_info_id', rooms_columns)
    
    # If we have an all-NULL record, use it
    if null_rooms_id:
        rooms_info_mappings[all_null_key] = null_rooms_id
        logger.info(f"Found existing all-NULL rooms_info record with ID {null_rooms_id}")

    processed_count = 0
    unique_rooms = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing rooms_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            rooms_info = listing_features.get('rooms_info')
            if not rooms_info:
                continue

            bathrooms_number = rooms_info.get('bathrooms_number')
            bedrooms_number = rooms_info.get('bedrooms_number')
            
            # Process total_room_number with the new helper function
            raw_total_room_number = rooms_info.get('total_room_number')
            total_room_number = process_total_room_number(raw_total_room_number)
            
            kitchen_status = rooms_info.get('kitchen_status')
            if kitchen_status and isinstance(kitchen_status, str):
                kitchen_status = kitchen_status.lower()  
            garage = rooms_info.get('garage')
            if garage and isinstance(garage, str):
                garage = garage.lower()  

            # Process floor
            floor = None
            property_floor = rooms_info.get('property_floor')
            if property_floor:
                if isinstance(property_floor, str):
                    property_floor = property_floor.lower()  
                    if "rialzato" in property_floor:
                        floor = 0
                    elif "terra" in property_floor:
                        floor = 0
                    elif "seminterrato" in property_floor or "interrato" in property_floor:
                        floor = -1
                    else:
                        # Extract any numbers from the floor string
                        floor_match = re.search(r'(\d+)', property_floor)
                        if floor_match:
                            floor = int(floor_match.group(1))
                elif isinstance(property_floor, int) or isinstance(property_floor, float):
                    floor = int(property_floor)

            # Create a unique key
            rooms_key = (
                bathrooms_number,
                bedrooms_number,
                total_room_number,
                kitchen_status,
                garage,
                floor
            )

            # Check if this is an all-NULL key
            if check_all_nulls(rooms_key):
                # Skip if we already have an all-NULL record
                if all_null_key in rooms_info_mappings:
                    rooms_info_mappings[rooms_key] = rooms_info_mappings[all_null_key]
                    continue

            if rooms_key in unique_rooms or rooms_key in rooms_info_mappings:
                continue

            unique_rooms.add(rooms_key)
            processed_count += 1

            # Check if this combination already exists using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_rooms_info (
                    bathrooms_number, bedrooms_number, total_room_number, kitchen_status, garage, floor
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (bathrooms_number, bedrooms_number, total_room_number, kitchen_status, garage, floor)
                DO NOTHING
                RETURNING rooms_info_id
            )
            SELECT rooms_info_id FROM inserted
            UNION ALL
            SELECT rooms_info_id FROM dim_rooms_info
            WHERE (bathrooms_number IS NULL AND %s IS NULL OR bathrooms_number = %s)
            AND (bedrooms_number IS NULL AND %s IS NULL OR bedrooms_number = %s)
            AND (total_room_number IS NULL AND %s IS NULL OR total_room_number = %s)
            AND (kitchen_status IS NULL AND %s IS NULL OR kitchen_status = %s)
            AND (garage IS NULL AND %s IS NULL OR garage = %s)
            AND (floor IS NULL AND %s IS NULL OR floor = %s)
            LIMIT 1
            """
            params = (
                bathrooms_number, bedrooms_number, total_room_number, kitchen_status, garage, floor,
                bathrooms_number, bathrooms_number,
                bedrooms_number, bedrooms_number,
                total_room_number, total_room_number,
                kitchen_status, kitchen_status,
                garage, garage,
                floor, floor
            )
            result = execute_query_silent(postgres_hook, query, params)

            rooms_info_id = result[0][0] if result else None
            if rooms_info_id:
                rooms_info_mappings[rooms_key] = rooms_info_id

    dimension_mappings['rooms_info'] = rooms_info_mappings
    dimension_counts['rooms_info'] = len(rooms_info_mappings)

    # Process building_info dimension
    logger.info("Processing building_info dimension...")
    building_info_mappings = {}
    
    # Define all_null_key upfront
    all_null_key = (None, None, None, None, None, None, None, None)
    
    # Check for existing all-NULL record in building_info
    building_columns = ['has_elevator', 'building_usage', 'building_year', 'total_building_floors',
                       'total_number_of_residential_units', 'work_start_date', 'work_end_date', 'work_completion']
    null_building_id = get_existing_null_record_id(postgres_hook, 'dim_building_info', 'building_info_id', building_columns)
    
    # If we have an all-NULL record, use it
    if null_building_id:
        building_info_mappings[all_null_key] = null_building_id
        logger.info(f"Found existing all-NULL building_info record with ID {null_building_id}")

    processed_count = 0
    unique_buildings = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing building_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            building_info = listing_features.get('building_info')
            if not building_info:
                continue

            has_elevator = building_info.get('has_elevator')
            building_usage = building_info.get('building_usage')
            if building_usage and isinstance(building_usage, str):
                building_usage = building_usage.lower()  
            
            # Convert building_year to integer if it's a string
            building_year = building_info.get('building_year')
            if isinstance(building_year, str):
                try:
                    building_year = int(building_year)
                except (ValueError, TypeError):
                    building_year = None
            
            # Convert total_building_floors to integer if it's a string
            total_building_floors = building_info.get('total_building_floors')
            if isinstance(total_building_floors, str):
                try:
                    total_building_floors = int(total_building_floors)
                except (ValueError, TypeError):
                    total_building_floors = None
            
            # Convert total_number_of_residential_units to integer if it's a string
            total_number_of_residential_units = building_info.get('total_number_of_residential_units')
            if isinstance(total_number_of_residential_units, str):
                try:
                    total_number_of_residential_units = int(total_number_of_residential_units)
                except (ValueError, TypeError):
                    total_number_of_residential_units = None
            
            # Convert work_start_date and work_end_date to date objects
            work_start_date_str = building_info.get('work_start_date')
            work_end_date_str = building_info.get('work_end_date')
            
            # Convert to date objects using the existing convert_mongo_date function
            work_start_date = convert_mongo_date(work_start_date_str)
            work_end_date = convert_mongo_date(work_end_date_str)
            
            # Convert work_completion to float if it's a string
            work_completion = building_info.get('work_completion')
            if isinstance(work_completion, str):
                try:
                    work_completion = float(work_completion)
                except (ValueError, TypeError):
                    work_completion = None

            # Create a unique key
            building_key = (
                has_elevator,
                building_usage,
                building_year,
                total_building_floors,
                total_number_of_residential_units,
                work_start_date,
                work_end_date,
                work_completion
            )

            # Check if this is an all-NULL key
            if check_all_nulls(building_key):
                # Skip if we already have an all-NULL record
                if all_null_key in building_info_mappings:
                    building_info_mappings[building_key] = building_info_mappings[all_null_key]
                    continue

            if building_key in unique_buildings or building_key in building_info_mappings:
                continue

            unique_buildings.add(building_key)
            processed_count += 1

            # Check if this combination already exists using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_building_info (
                    has_elevator, building_usage, building_year, total_building_floors,
                    total_number_of_residential_units, work_start_date, work_end_date, work_completion
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (has_elevator, building_usage, building_year, total_building_floors,
                            total_number_of_residential_units, work_start_date, work_end_date, work_completion)
                DO NOTHING
                RETURNING building_info_id
            )
            SELECT building_info_id FROM inserted
            UNION ALL
            SELECT building_info_id FROM dim_building_info
            WHERE (has_elevator IS NULL AND %s IS NULL OR has_elevator = %s)
            AND (building_usage IS NULL AND %s IS NULL OR building_usage = %s)
            AND (building_year IS NULL AND %s IS NULL OR building_year = %s)
            AND (total_building_floors IS NULL AND %s IS NULL OR total_building_floors = %s)
            AND (total_number_of_residential_units IS NULL AND %s IS NULL OR total_number_of_residential_units = %s)
            AND (work_start_date IS NULL AND %s IS NULL OR work_start_date = %s)
            AND (work_end_date IS NULL AND %s IS NULL OR work_end_date = %s)
            AND (work_completion IS NULL AND %s IS NULL OR work_completion = %s)
            LIMIT 1
            """
            params = (
                has_elevator, building_usage, building_year, total_building_floors,
                total_number_of_residential_units, work_start_date, work_end_date, work_completion,
                has_elevator, has_elevator,
                building_usage, building_usage,
                building_year, building_year,
                total_building_floors, total_building_floors,
                total_number_of_residential_units, total_number_of_residential_units,
                work_start_date, work_start_date,
                work_end_date, work_end_date,
                work_completion, work_completion
            )
            result = execute_query_silent(postgres_hook, query, params)

            building_info_id = result[0][0] if result else None
            if building_info_id:
                building_info_mappings[building_key] = building_info_id

    dimension_mappings['building_info'] = building_info_mappings
    dimension_counts['building_info'] = len(building_info_mappings)

    # Process energy_info dimension
    logger.info("Processing energy_info dimension...")
    energy_info_mappings = {}
    
    # Define all_null_key upfront
    all_null_key = (None, None, None, None)
    
    # Check for existing all-NULL record in energy_info
    energy_columns = ['is_zero_energy_building', 'heating_type', 'energy_class', 'air_conditioning']
    null_energy_id = get_existing_null_record_id(postgres_hook, 'dim_energy_info', 'energy_info_id', energy_columns)
    
    # If we have an all-NULL record, use it
    if null_energy_id:
        energy_info_mappings[all_null_key] = null_energy_id
        logger.info(f"Found existing all-NULL energy_info record with ID {null_energy_id}")

    processed_count = 0
    unique_energy = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing energy_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            energy_info = listing_features.get('energy_info')
            if not energy_info:
                continue

            is_zero_energy_building = energy_info.get('is_zero_energy_building')
            heating_type = energy_info.get('heating_type')
            if heating_type and isinstance(heating_type, str):
                heating_type = heating_type.lower()  
            energy_class = energy_info.get('energy_class')
            # Handle energy_class being a dictionary with a name field
            if energy_class and isinstance(energy_class, dict) and 'name' in energy_class:
                energy_class = energy_class['name']

            if energy_class and isinstance(energy_class, str):
                energy_class = energy_class.lower()  
                if energy_class == "a":
                    energy_class = "a1"     #'a' and 'a1' are the same, so we make the database consistent
            air_conditioning = energy_info.get('air_conditioning')
            if air_conditioning and isinstance(air_conditioning, str):
                air_conditioning = air_conditioning.lower()  

            # Create a unique key
            energy_key = (
                is_zero_energy_building,
                heating_type,
                energy_class,
                air_conditioning
            )

            # Check if this is an all-NULL key
            if check_all_nulls(energy_key):
                # Skip if we already have an all-NULL record
                if all_null_key in energy_info_mappings:
                    energy_info_mappings[energy_key] = energy_info_mappings[all_null_key]
                    continue

            if energy_key in unique_energy or energy_key in energy_info_mappings:
                continue

            unique_energy.add(energy_key)
            processed_count += 1

            # Check if this combination already exists using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_energy_info (
                    is_zero_energy_building, heating_type, energy_class, air_conditioning
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (is_zero_energy_building, heating_type, energy_class, air_conditioning)
                DO NOTHING
                RETURNING energy_info_id
            )
            SELECT energy_info_id FROM inserted
            UNION ALL
            SELECT energy_info_id FROM dim_energy_info
            WHERE (is_zero_energy_building IS NULL AND %s IS NULL OR is_zero_energy_building = %s)
            AND (heating_type IS NULL AND %s IS NULL OR heating_type = %s)
            AND (energy_class IS NULL AND %s IS NULL OR energy_class = %s)
            AND (air_conditioning IS NULL AND %s IS NULL OR air_conditioning = %s)
            LIMIT 1
            """
            params = (
                is_zero_energy_building, heating_type, energy_class, air_conditioning,
                is_zero_energy_building, is_zero_energy_building,
                heating_type, heating_type,
                energy_class, energy_class,
                air_conditioning, air_conditioning
            )
            result = execute_query_silent(postgres_hook, query, params)

            energy_info_id = result[0][0] if result else None
            if energy_info_id:
                energy_info_mappings[energy_key] = energy_info_id

    dimension_mappings['energy_info'] = energy_info_mappings
    dimension_counts['energy_info'] = len(energy_info_mappings)

    # Process location_info dimension
    logger.info("Processing location_info dimension...")
    location_info_mappings = {}
    
    # Define all_null_key upfront
    all_null_key = (None, None, None, None, None, None, None, None)
    
    # Check for existing all-NULL record in location_info
    location_columns = ['latitude', 'longitude', 'region', 'province', 'province_code', 'city', 'macrozone', 'microzone']
    null_location_id = get_existing_null_record_id(postgres_hook, 'dim_location_info', 'location_info_id', location_columns)
    
    # If we have an all-NULL record, use it
    if null_location_id:
        location_info_mappings[all_null_key] = null_location_id
        logger.info(f"Found existing all-NULL location_info record with ID {null_location_id}")

    processed_count = 0
    unique_locations = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing location_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            location_info = listing_features.get('location_info')
            if not location_info:
                continue

            latitude = location_info.get('latitude')
            longitude = location_info.get('longitude')
            region = location_info.get('region')
            if region and isinstance(region, str):
                region = region.lower()  
            province = location_info.get('province')
            if province and isinstance(province, str):
                province = province.lower()  
            province_code = location_info.get('province_code')
            if province_code and isinstance(province_code, str):
                province_code = province_code.lower()  
            city = location_info.get('city')
            if city and isinstance(city, str):
                city = city.lower()  
            macrozone = location_info.get('macrozone')
            if macrozone and isinstance(macrozone, str):
                macrozone = macrozone.lower()  
            microzone = location_info.get('microzone')
            if microzone and isinstance(microzone, str):
                microzone = microzone.lower()  

            # Create a unique key
            location_key = (
                latitude,
                longitude,
                region,
                province,
                province_code,
                city,
                macrozone,
                microzone
            )

            # Check if this is an all-NULL key
            if check_all_nulls(location_key):
                # Skip if we already have an all-NULL record
                if all_null_key in location_info_mappings:
                    location_info_mappings[location_key] = location_info_mappings[all_null_key]
                    continue

            if location_key in unique_locations or location_key in location_info_mappings:
                continue

            unique_locations.add(location_key)
            processed_count += 1

            # Modify the query to handle cases with no result more gracefully
            # Using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_location_info (
                    latitude, longitude, region, province, province_code, city, macrozone, microzone
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (latitude, longitude, region, province, province_code, city, macrozone, microzone)
                DO UPDATE SET
                    latitude = COALESCE(EXCLUDED.latitude, dim_location_info.latitude),
                    longitude = COALESCE(EXCLUDED.longitude, dim_location_info.longitude)
                RETURNING location_info_id
            )
            SELECT location_info_id FROM inserted
            UNION ALL
            SELECT location_info_id FROM dim_location_info
            WHERE (latitude IS NULL AND %s IS NULL OR latitude = %s)
            AND (longitude IS NULL AND %s IS NULL OR longitude = %s)
            AND (region IS NULL AND %s IS NULL OR region = %s)
            AND (province IS NULL AND %s IS NULL OR province = %s)
            AND (province_code IS NULL AND %s IS NULL OR province_code = %s)
            AND (city IS NULL AND %s IS NULL OR city = %s)
            AND (macrozone IS NULL AND %s IS NULL OR macrozone = %s)
            AND (microzone IS NULL AND %s IS NULL OR microzone = %s)
            LIMIT 1
            """
            params = (
                latitude, longitude, region, province, province_code, city, macrozone, microzone,
                latitude, latitude,
                longitude, longitude,
                region, region,
                province, province,
                province_code, province_code,
                city, city,
                macrozone, macrozone,
                microzone, microzone
            )

            try:
                result = execute_query_silent(postgres_hook, query, params)
                if result:
                    location_info_id = result[0][0]
                    location_info_mappings[location_key] = location_info_id
            except Exception as e:
                logger.error(f"Error processing location: {str(e)}")

    dimension_mappings['location_info'] = location_info_mappings
    dimension_counts['location_info'] = len(location_info_mappings)

    # Process surface_composition dimension
    logger.info("Processing surface_composition dimension...")
    surface_composition_mappings = {}
    
    # Define all_null_key upfront
    all_null_key = (None, None, None, None, None)
    
    # Check for existing all-NULL record in surface_composition
    surface_columns = ['element_name', 'floor', 'surface', 'percentage', 'commercial_surface']
    null_surface_id = get_existing_null_record_id(postgres_hook, 'dim_surface_composition', 'surface_composition_id', surface_columns)
    
    # If we have an all-NULL record, use it
    if null_surface_id:
        surface_composition_mappings[all_null_key] = null_surface_id
        logger.info(f"Found existing all-NULL surface_composition record with ID {null_surface_id}")

    processed_count = 0
    unique_surfaces = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing surface_composition: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            surface_composition = listing_features.get('surface_composition')
            if not surface_composition:
                continue

            element_name = surface_composition.get('element')
            if element_name and isinstance(element_name, str):
                element_name = element_name.lower()  
            floor_str = surface_composition.get('floor')
            if floor_str and isinstance(floor_str, str):
                floor_str = floor_str.lower()  
            surface = surface_composition.get('surface')
            percentage = surface_composition.get('percentage')
            commercial_surface = surface_composition.get('commercial_surface')

            # Try to extract floor number from string
            floor = None
            if floor_str:
                if isinstance(floor_str, str):
                    if "rialzato" in floor_str:
                        floor = 0
                    elif "terra" in floor_str:
                        floor = 0
                    elif "seminterrato" in floor_str or "interrato" in floor_str:
                        floor = -1
                    else:
                        floor_match = re.search(r'(\d+)', floor_str)
                        if floor_match:
                            floor = int(floor_match.group(1))
                elif isinstance(floor_str, int) or isinstance(floor_str, float):
                    floor = int(floor_str)

            # Create a unique key
            surface_key = (
                element_name,
                floor,
                surface,
                percentage,
                commercial_surface
            )

            # Check if this is an all-NULL key
            if check_all_nulls(surface_key):
                # Skip if we already have an all-NULL record
                if all_null_key in surface_composition_mappings:
                    surface_composition_mappings[surface_key] = surface_composition_mappings[all_null_key]
                    continue

            if surface_key in unique_surfaces or surface_key in surface_composition_mappings:
                continue

            unique_surfaces.add(surface_key)
            processed_count += 1

            # Check if this combination already exists using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_surface_composition (
                    element_name, floor, surface, percentage, commercial_surface
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (element_name, floor, surface, percentage, commercial_surface) DO NOTHING
                RETURNING surface_composition_id
            )
            SELECT surface_composition_id FROM inserted
            UNION ALL
            SELECT surface_composition_id FROM dim_surface_composition
            WHERE (element_name IS NULL AND %s IS NULL OR element_name = %s)
            AND (floor IS NULL AND %s IS NULL OR floor = %s)
            AND (surface IS NULL AND %s IS NULL OR surface = %s)
            AND (percentage IS NULL AND %s IS NULL OR percentage = %s)
            AND (commercial_surface IS NULL AND %s IS NULL OR commercial_surface = %s)
            LIMIT 1
            """
            params = (
                element_name, floor, surface, percentage, commercial_surface,
                element_name, element_name,
                floor, floor,
                surface, surface,
                percentage, percentage,
                commercial_surface, commercial_surface
            )
            result = execute_query_silent(postgres_hook, query, params)

            surface_composition_id = result[0][0] if result else None
            if surface_composition_id:
                surface_composition_mappings[surface_key] = surface_composition_id

    dimension_mappings['surface_composition'] = surface_composition_mappings
    dimension_counts['surface_composition'] = len(surface_composition_mappings)

    # Process auction_info dimension
    logger.info("Processing auction_info dimension...")
    auction_info_mappings = {}

    # Define all_null_key upfront
    all_null_key = (None, None, None, None, None, None, None, None, None, None)  # Added one more None for deposit_modality_hash

    # Check for existing all-NULL record in auction_info
    auction_columns = ['auction_end_date', 'deposit_modality', 'deposit_modality_hash', 'auction_type', 'is_open', 'minimum_offer',
                    'procedure_number', 'auction_court', 'lot_category_id', 'lot_category_name']
    null_auction_id = get_existing_null_record_id(postgres_hook, 'dim_auction_info', 'auction_info_id', auction_columns)

    # If we have an all-NULL record, use it
    if null_auction_id:
        auction_info_mappings[all_null_key] = null_auction_id
        logger.info(f"Found existing all-NULL auction_info record with ID {null_auction_id}")

    processed_count = 0
    unique_auctions = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing auction_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            auction_info = listing_features.get('auction_info')
            if not auction_info:
                continue

            auction_end_date = convert_mongo_date(auction_info.get('auction_end_date'))
            deposit_modality = auction_info.get('deposit_modality')
            if deposit_modality and isinstance(deposit_modality, str):
                deposit_modality = deposit_modality.lower()  
            
            # Generate hash for deposit_modality - NEW CODE
            deposit_modality_hash = None
            if deposit_modality is not None:
                deposit_modality_hash = hashlib.sha256(deposit_modality.encode('utf-8')).hexdigest()
                
            auction_type = auction_info.get('auction_type')
            if auction_type and isinstance(auction_type, str):
                auction_type = auction_type.lower()  
            is_open = auction_info.get('is_open')
            minimum_offer = auction_info.get('minimum_offer')
            procedure_number = auction_info.get('procedure_number')
            if procedure_number and isinstance(procedure_number, str):
                procedure_number = procedure_number.lower()  
            auction_court = auction_info.get('auction_court')
            if auction_court and isinstance(auction_court, str):
                auction_court = auction_court.lower()  

            lot_category = auction_info.get('lot_category')
            lot_category_id = None
            lot_category_name = None
            if lot_category:
                lot_category_id = lot_category.get('id')
                lot_category_name = lot_category.get('name')
                if lot_category_name and isinstance(lot_category_name, str):
                    lot_category_name = lot_category_name.lower()  

            # Create a unique key
            auction_key = (
                auction_end_date,
                deposit_modality,         
                deposit_modality_hash,
                auction_type,
                is_open,
                minimum_offer,
                procedure_number,
                auction_court,
                lot_category_id,
                lot_category_name
            )

            # Check if this is an all-NULL key
            if check_all_nulls(auction_key):
                # Skip if we already have an all-NULL record
                if all_null_key in auction_info_mappings:
                    auction_info_mappings[auction_key] = auction_info_mappings[all_null_key]
                    continue

            if auction_key in unique_auctions or auction_key in auction_info_mappings:
                continue

            unique_auctions.add(auction_key)
            processed_count += 1

            # Check if this combination already exists - Using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_auction_info (
                    auction_end_date, deposit_modality, deposit_modality_hash, auction_type, is_open, minimum_offer,
                    procedure_number, auction_court, lot_category_id, lot_category_name
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING auction_info_id
            )
            SELECT auction_info_id FROM inserted
            UNION ALL
            SELECT auction_info_id FROM dim_auction_info
            WHERE (auction_end_date IS NULL AND %s IS NULL OR auction_end_date = %s)
            AND (deposit_modality_hash IS NULL AND %s IS NULL OR deposit_modality_hash = %s)
            AND (auction_type IS NULL AND %s IS NULL OR auction_type = %s)
            AND (is_open IS NULL AND %s IS NULL OR is_open = %s)
            AND (minimum_offer IS NULL AND %s IS NULL OR minimum_offer = %s)
            AND (procedure_number IS NULL AND %s IS NULL OR procedure_number = %s)
            AND (auction_court IS NULL AND %s IS NULL OR auction_court = %s)
            AND (lot_category_id IS NULL AND %s IS NULL OR lot_category_id = %s)
            AND (lot_category_name IS NULL AND %s IS NULL OR lot_category_name = %s)
            LIMIT 1
            """
            params = (
                auction_end_date, deposit_modality, deposit_modality_hash, auction_type, is_open, minimum_offer,
                procedure_number, auction_court, lot_category_id, lot_category_name,
                auction_end_date, auction_end_date,
                deposit_modality_hash, deposit_modality_hash,
                auction_type, auction_type,
                is_open, is_open,
                minimum_offer, minimum_offer,
                procedure_number, procedure_number,
                auction_court, auction_court,
                lot_category_id, lot_category_id,
                lot_category_name, lot_category_name
            )
            
            try:
                result = execute_query_silent(postgres_hook, query, params)
                auction_info_id = result[0][0] if result else None
                if auction_info_id:
                    auction_info_mappings[auction_key] = auction_info_id
                else:
                    # If we couldn't find a row, try a more targeted approach
                    # This handles potential edge cases with very long text fields
                    logger.warning(f"Could not get auction_info_id for key using standard query, trying fallback approach")
                    
                    # Targeted query using just a few key fields
                    fallback_query = """
                    SELECT auction_info_id FROM dim_auction_info
                    WHERE (auction_end_date IS NULL AND %s IS NULL OR auction_end_date = %s)
                    AND (is_open IS NULL AND %s IS NULL OR is_open = %s)
                    LIMIT 1
                    """
                    fallback_params = (auction_end_date, auction_end_date, is_open, is_open)
                    fallback_result = execute_query_silent(postgres_hook, fallback_query, fallback_params)
                    auction_info_id = fallback_result[0][0] if fallback_result else None
                    
                    if auction_info_id:
                        auction_info_mappings[auction_key] = auction_info_id
                        logger.info(f"Found auction_info_id {auction_info_id} using fallback query")
                    else:
                        logger.warning(f"Could not get auction_info_id for key using fallback query")
            except Exception as e:
                logger.error(f"Error processing auction info: {str(e)}")

    dimension_mappings['auction_info'] = auction_info_mappings
    dimension_counts['auction_info'] = len(auction_info_mappings)

    # Process cadastral_info dimension
    logger.info("Processing cadastral_info dimension...")
    cadastral_info_mappings = {}

    # Define all_null_key upfront
    all_null_key = (None, None, None)

    # Check for existing all-NULL record in cadastral_info
    cadastral_columns = ['cadastral', 'cadastral_additional_info', 'sub_cadastral_info']
    null_cadastral_id = get_existing_null_record_id(postgres_hook, 'dim_cadastral_info', 'cadastral_info_id', cadastral_columns)

    # If we have an all-NULL record, use it
    if null_cadastral_id:
        cadastral_info_mappings[all_null_key] = null_cadastral_id
        logger.info(f"Found existing all-NULL cadastral_info record with ID {null_cadastral_id}")

    processed_count = 0
    unique_cadastrals = set()

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing cadastral_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            cadastral_info = listing_features.get('cadastral_info')
            if not cadastral_info:
                continue

            cadastral = cadastral_info.get('cadastral')
            if cadastral and isinstance(cadastral, str):
                cadastral = cadastral.lower()  
            cadastral_additional_info = cadastral_info.get('cadastral_additional_info')
            if cadastral_additional_info and isinstance(cadastral_additional_info, str):
                cadastral_additional_info = cadastral_additional_info.lower()  
            sub_cadastral_info = cadastral_info.get('sub_cadastral_info')
            if sub_cadastral_info and isinstance(sub_cadastral_info, str):
                sub_cadastral_info = sub_cadastral_info.lower()  

            # Create a unique key
            cadastral_key = (
                cadastral,
                cadastral_additional_info,
                sub_cadastral_info
            )

            # Check if this is an all-NULL key
            if check_all_nulls(cadastral_key):
                # Skip if we already have an all-NULL record
                if all_null_key in cadastral_info_mappings:
                    cadastral_info_mappings[cadastral_key] = cadastral_info_mappings[all_null_key]
                    continue

            if cadastral_key in unique_cadastrals or cadastral_key in cadastral_info_mappings:
                continue

            unique_cadastrals.add(cadastral_key)
            processed_count += 1

            # Check if this combination already exists using parameterized query
            query = """
            WITH inserted AS (
                INSERT INTO dim_cadastral_info (
                    cadastral, cadastral_additional_info, sub_cadastral_info
                ) VALUES (%s, %s, %s)
                ON CONFLICT (cadastral, cadastral_additional_info, sub_cadastral_info)
                DO NOTHING
                RETURNING cadastral_info_id
            )
            SELECT cadastral_info_id FROM inserted
            UNION ALL
            SELECT cadastral_info_id FROM dim_cadastral_info
            WHERE (cadastral IS NULL AND %s IS NULL OR cadastral = %s)
            AND (cadastral_additional_info IS NULL AND %s IS NULL OR cadastral_additional_info = %s)
            AND (sub_cadastral_info IS NULL AND %s IS NULL OR sub_cadastral_info = %s)
            LIMIT 1
            """
            params = (
                cadastral, cadastral_additional_info, sub_cadastral_info,
                cadastral, cadastral,
                cadastral_additional_info, cadastral_additional_info,
                sub_cadastral_info, sub_cadastral_info
            )
            result = execute_query_silent(postgres_hook, query, params)

            cadastral_info_id = result[0][0] if result else None
            if cadastral_info_id:
                cadastral_info_mappings[cadastral_key] = cadastral_info_id

    dimension_mappings['cadastral_info'] = cadastral_info_mappings
    dimension_counts['cadastral_info'] = len(cadastral_info_mappings)

    # Log summary of processed dimensions
    elapsed_time = time.time() - start_time
    logger.info(f"All dimensions processed for this batch in {elapsed_time:.2f} seconds.")
    logger.info("Dimension processing summary:")
    for dim_name, count in dimension_counts.items():
        logger.info(f"  - {dim_name}: {count} entries")

    return dimension_mappings

# -------------------------------------------------------------------------------

def load_fact_table_for_batch(**kwargs):
    '''Load data into the fact_listing table for a specific batch'''
    ti = kwargs['ti']
    batch_num = kwargs.get('batch_num', 0)
    logger.info(f"Loading fact table for batch {batch_num}...")
    start_time = time.time()

    # Retrieve batch data and dimension mappings
    batch_data = ti.xcom_pull(key=f'batch_{batch_num}_data')
    serializable_dimension_mappings = ti.xcom_pull(key=f'batch_{batch_num}_dimension_mappings')

    # Convert string keys back to tuple keys
    dimension_mappings = convert_nested_dict_str_keys_to_tuple(serializable_dimension_mappings)

    if not batch_data or not dimension_mappings:
        logger.error(f"Error: Could not retrieve data for batch {batch_num}")
        return f"Failed to load fact table for batch {batch_num}"

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    # Track existing records to avoid duplicates
    existing_records = set()
    query = "SELECT listing_id, date_id FROM fact_listing"
    for record in execute_query_silent(postgres_hook, query):
        existing_records.add((record[0], record[1]))

    # Process each listing in this batch
    insert_count = 0
    skip_count = 0
    feature_count = 0
    surface_count = 0
    total_listings = len(batch_data)

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 10 == 0 or idx + 1 == total_listings:
            logger.info(f"Processing listing {idx+1}/{total_listings} for fact table")

        listing_id = listing['_id']
        listing_type = listing['collection_name'].lower()  # Collection name is already lowercase
        listing_type_id = dimension_mappings['listing_type'][listing_type]

        for data_entry in listing.get('data', []):
            scraping_date = data_entry.get('scraping_date')
            date_id = dimension_mappings['date'].get(scraping_date)

            if not date_id:
                logger.warning(f"No date_id found for scraping_date {scraping_date}, skipping.")
                continue

            # Skip if already in database
            if (listing_id, date_id) in existing_records:
                skip_count += 1
                continue

            listing_features = data_entry.get('listing_features', {})

            # Extract basic data
            price = listing_features.get('price')
            total_surface = listing_features.get('total_surface')
            price_per_sq_mt = listing_features.get('price_per_sq_mt')

            # Get dimension IDs
            # Listing Info
            listing_info_id = None
            listing_info = listing_features.get('listing_info')
            if listing_info:
                seller_type = listing_info.get('seller_type')
                if seller_type and isinstance(seller_type, str):
                    seller_type = seller_type.lower()  
                seller_type_id = dimension_mappings['seller_type'].get(seller_type)

                if seller_type_id:
                    listing_info_key = (
                        listing_info.get('listing_age'),
                        listing_info.get('listing_last_update'),
                        listing_info.get('number_of_photos'),
                        seller_type_id
                    )
                    listing_info_id = dimension_mappings['listing_info'].get(listing_info_key)

            # Availability
            availability_id = None
            availability = listing_features.get('availability')
            if availability and isinstance(availability, str):
                availability = availability.lower()  
                availability_id = dimension_mappings['availability'].get(availability)

            # Type of Property
            type_of_property_id = None
            type_of_property = listing_features.get('type_of_property')
            if type_of_property and type_of_property.get('class'):
                property_class = type_of_property.get('class')
                if property_class and isinstance(property_class, str):
                    property_class = property_class.lower()  
                type_of_property_id = dimension_mappings['type_of_property'].get(property_class)

            # Category
            category_id = None
            if type_of_property and type_of_property.get('category'):
                category = type_of_property.get('category')
                if category and isinstance(category, str):
                    category = category.lower()  
                category_id = dimension_mappings['category'].get(category)

            # Condition
            condition_id = None
            condition = listing_features.get('condition')
            if condition and isinstance(condition, str):
                condition = condition.lower()  
                condition_id = dimension_mappings['condition'].get(condition)

            # Additional Costs
            additional_costs_id = None
            additional_costs = listing_features.get('additional_costs')
            if additional_costs:
                # Initialize other_costs as empty dict if None
                other_costs = additional_costs.get('other_costs')
                if other_costs is None:
                    other_costs = {}

                mortgage_rate = additional_costs.get('mortgage_rate')
                condominium_monthly_expenses = None
                heating_yearly_expenses = None
                
                # Check if other_costs is not None before trying to access its properties
                if other_costs:
                    condominium_monthly_expenses = other_costs.get('condominium_expenses')
                    heating_yearly_expenses = other_costs.get('heating_expenses')

                    # Extract numeric values from text if present
                    if condominium_monthly_expenses and isinstance(condominium_monthly_expenses, str):
                        numeric_match = re.search(r'(\d+(?:\.\d+)?)', condominium_monthly_expenses)
                        condominium_monthly_expenses = float(numeric_match.group(1)) if numeric_match else None

                    if heating_yearly_expenses and isinstance(heating_yearly_expenses, str):
                        numeric_match = re.search(r'(\d+(?:\.\d+)?)', heating_yearly_expenses)
                        heating_yearly_expenses = float(numeric_match.group(1)) if numeric_match else None

                cost_key = (
                    condominium_monthly_expenses,
                    heating_yearly_expenses,
                    mortgage_rate,
                    None
                )

                additional_costs_id = dimension_mappings['additional_costs'].get(cost_key)

            # Rooms Info
            rooms_info_id = None
            rooms_info = listing_features.get('rooms_info')
            if rooms_info:
                bathrooms_number = rooms_info.get('bathrooms_number')
                bedrooms_number = rooms_info.get('bedrooms_number')
                
                # Process total_room_number with the new helper function
                raw_total_room_number = rooms_info.get('total_room_number')
                total_room_number = process_total_room_number(raw_total_room_number)
                
                kitchen_status = rooms_info.get('kitchen_status')
                if kitchen_status and isinstance(kitchen_status, str):
                    kitchen_status = kitchen_status.lower()  
                garage = rooms_info.get('garage')
                if garage and isinstance(garage, str):
                    garage = garage.lower()  

                # Process floor
                floor = None
                property_floor = rooms_info.get('property_floor')
                if property_floor:
                    if isinstance(property_floor, str):
                        property_floor = property_floor.lower()  
                        if "rialzato" in property_floor:
                            floor = 0
                        elif "terra" in property_floor:
                            floor = 0
                        elif "seminterrato" in property_floor or "interrato" in property_floor:
                            floor = -1
                        else:
                            floor_match = re.search(r'(\d+)', property_floor)
                            if floor_match:
                                floor = int(floor_match.group(1))
                    elif isinstance(property_floor, int) or isinstance(property_floor, float):
                        floor = int(property_floor)

                rooms_key = (
                    bathrooms_number,
                    bedrooms_number,
                    total_room_number,
                    kitchen_status,
                    garage,
                    floor
                )

                rooms_info_id = dimension_mappings['rooms_info'].get(rooms_key)

            # Building Info
            building_info_id = None
            building_info = listing_features.get('building_info')
            if building_info:
                has_elevator = building_info.get('has_elevator')
                building_usage = building_info.get('building_usage')
                if building_usage and isinstance(building_usage, str):
                    building_usage = building_usage.lower()  
                
                # Convert building_year to integer if it's a string
                building_year = building_info.get('building_year')
                if isinstance(building_year, str):
                    try:
                        building_year = int(building_year)
                    except (ValueError, TypeError):
                        building_year = None
                
                # Convert total_building_floors to integer if it's a string
                total_building_floors = building_info.get('total_building_floors')
                if isinstance(total_building_floors, str):
                    try:
                        total_building_floors = int(total_building_floors)
                    except (ValueError, TypeError):
                        total_building_floors = None
                
                # Convert total_number_of_residential_units to integer if it's a string
                total_number_of_residential_units = building_info.get('total_number_of_residential_units')
                if isinstance(total_number_of_residential_units, str):
                    try:
                        total_number_of_residential_units = int(total_number_of_residential_units)
                    except (ValueError, TypeError):
                        total_number_of_residential_units = None
                
                # Convert dates using convert_mongo_date function
                work_start_date_str = building_info.get('work_start_date')
                work_end_date_str = building_info.get('work_end_date')
                
                work_start_date = convert_mongo_date(work_start_date_str)
                work_end_date = convert_mongo_date(work_end_date_str)
                
                # Convert work_completion to float if it's a string
                work_completion = building_info.get('work_completion')
                if isinstance(work_completion, str):
                    try:
                        # Handle percentage strings (e.g., "75%")
                        if '%' in work_completion:
                            work_completion = float(work_completion.rstrip('%')) / 100
                        else:
                            work_completion = float(work_completion)
                    except (ValueError, TypeError):
                        work_completion = None

                building_key = (
                    has_elevator,
                    building_usage,
                    building_year,
                    total_building_floors,
                    total_number_of_residential_units,
                    work_start_date,
                    work_end_date,
                    work_completion
                )

                building_info_id = dimension_mappings['building_info'].get(building_key)

            # Energy Info
            energy_info_id = None
            energy_info = listing_features.get('energy_info')
            if energy_info:
                is_zero_energy_building = energy_info.get('is_zero_energy_building')
                heating_type = energy_info.get('heating_type')
                if heating_type and isinstance(heating_type, str):
                    heating_type = heating_type.lower()  
                energy_class = energy_info.get('energy_class')
                # Handle energy_class being a dictionary with a name field
                if energy_class and isinstance(energy_class, dict) and 'name' in energy_class:
                    energy_class = energy_class['name']

                if energy_class and isinstance(energy_class, str):
                    energy_class = energy_class.lower()  
                air_conditioning = energy_info.get('air_conditioning')
                if air_conditioning and isinstance(air_conditioning, str):
                    air_conditioning = air_conditioning.lower()  

                energy_key = (
                    is_zero_energy_building,
                    heating_type,
                    energy_class,
                    air_conditioning
                )

                energy_info_id = dimension_mappings['energy_info'].get(energy_key)

            # Location Info
            location_info_id = None
            location_info = listing_features.get('location_info')
            if location_info:
                latitude = location_info.get('latitude')
                longitude = location_info.get('longitude')
                region = location_info.get('region')
                if region and isinstance(region, str):
                    region = region.lower()  
                province = location_info.get('province')
                if province and isinstance(province, str):
                    province = province.lower()  
                province_code = location_info.get('province_code')
                if province_code and isinstance(province_code, str):
                    province_code = province_code.lower()  
                city = location_info.get('city')
                if city and isinstance(city, str):
                    city = city.lower()  
                macrozone = location_info.get('macrozone')
                if macrozone and isinstance(macrozone, str):
                    macrozone = macrozone.lower()  
                microzone = location_info.get('microzone')
                if microzone and isinstance(microzone, str):
                    microzone = microzone.lower()  

                location_key = (
                    latitude,
                    longitude,
                    region,
                    province,
                    province_code,
                    city,
                    macrozone,
                    microzone
                )

                location_info_id = dimension_mappings['location_info'].get(location_key)

            # Auction Info
            auction_info_id = None
            auction_info = listing_features.get('auction_info')
            if auction_info:
                auction_end_date = convert_mongo_date(auction_info.get('auction_end_date'))
                deposit_modality = auction_info.get('deposit_modality')
                if deposit_modality and isinstance(deposit_modality, str):
                    deposit_modality = deposit_modality.lower()  
                
                # Generate hash for deposit_modality
                deposit_modality_hash = None
                if deposit_modality is not None:
                    deposit_modality_hash = hashlib.sha256(deposit_modality.encode('utf-8')).hexdigest()
                
                auction_type = auction_info.get('auction_type')
                if auction_type and isinstance(auction_type, str):
                    auction_type = auction_type.lower()  
                is_open = auction_info.get('is_open')
                minimum_offer = auction_info.get('minimum_offer')
                procedure_number = auction_info.get('procedure_number')
                if procedure_number and isinstance(procedure_number, str):
                    procedure_number = procedure_number.lower()  
                auction_court = auction_info.get('auction_court')
                if auction_court and isinstance(auction_court, str):
                    auction_court = auction_court.lower()  

                lot_category = auction_info.get('lot_category')
                lot_category_id = None
                lot_category_name = None
                if lot_category:
                    lot_category_id = lot_category.get('id')
                    lot_category_name = lot_category.get('name')
                    if lot_category_name and isinstance(lot_category_name, str):
                        lot_category_name = lot_category_name.lower()  

                # Update the key to include deposit_modality_hash
                auction_key = (
                    auction_end_date,
                    deposit_modality,
                    deposit_modality_hash,
                    auction_type,
                    is_open,
                    minimum_offer,
                    procedure_number,
                    auction_court,
                    lot_category_id,
                    lot_category_name
                )

                auction_info_id = dimension_mappings['auction_info'].get(auction_key)

            # Cadastral Info
            cadastral_info_id = None
            cadastral_info = listing_features.get('cadastral_info')
            if cadastral_info:
                cadastral = cadastral_info.get('cadastral')
                if cadastral and isinstance(cadastral, str):
                    cadastral = cadastral.lower()  
                cadastral_additional_info = cadastral_info.get('cadastral_additional_info')
                if cadastral_additional_info and isinstance(cadastral_additional_info, str):
                    cadastral_additional_info = cadastral_additional_info.lower()  
                sub_cadastral_info = cadastral_info.get('sub_cadastral_info')
                if sub_cadastral_info and isinstance(sub_cadastral_info, str):
                    sub_cadastral_info = sub_cadastral_info.lower()  

                cadastral_key = (
                    cadastral,
                    cadastral_additional_info,
                    sub_cadastral_info
                )

                cadastral_info_id = dimension_mappings['cadastral_info'].get(cadastral_key)

            # Insert into fact_listing using parameterized query
            query = """
            INSERT INTO fact_listing (
                listing_id, date_id, listing_type_id, price, surface, price_per_sq_mt,
                listing_info_id, additional_costs_id, availability_id, type_of_property_id,
                category_id, condition_id, rooms_info_id, building_info_id, energy_info_id,
                location_info_id, auction_info_id, cadastral_info_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                listing_id, date_id, listing_type_id, price, total_surface, price_per_sq_mt,
                listing_info_id, additional_costs_id, availability_id, type_of_property_id,
                category_id, condition_id, rooms_info_id, building_info_id, energy_info_id,
                location_info_id, auction_info_id, cadastral_info_id
            )
            execute_query_silent(postgres_hook, query, params)
            insert_count += 1

            # Process features bridge table if features exist
            features = listing_features.get('additional_features')
            if features:
                for feature in features:
                    if feature:
                        feature_lower = feature.lower()  
                        feature_id = dimension_mappings['features'].get(feature_lower)
                        if feature_id:
                            # Check if this feature relation already exists
                            bridge_query = """
                            SELECT 1 FROM listing_features_bridge
                            WHERE listing_id = %s
                            AND date_id = %s
                            AND feature_id = %s
                            """
                            bridge_params = (listing_id, date_id, feature_id)
                            bridge_result = execute_query_silent(postgres_hook, bridge_query, bridge_params)

                            if not bridge_result:
                                # Insert new feature relation using parameterized query
                                bridge_insert_query = """
                                INSERT INTO listing_features_bridge (listing_id, date_id, feature_id)
                                VALUES (%s, %s, %s)
                                """
                                bridge_insert_params = (listing_id, date_id, feature_id)
                                execute_query_silent(postgres_hook, bridge_insert_query, bridge_insert_params)
                                feature_count += 1

            # Process surface_composition bridge table if surface_composition exists
            surface_composition = listing_features.get('surface_composition')
            if surface_composition:
                element_name = surface_composition.get('element')
                if element_name and isinstance(element_name, str):
                    element_name = element_name.lower()  
                floor_str = surface_composition.get('floor')
                if floor_str and isinstance(floor_str, str):
                    floor_str = floor_str.lower()  
                surface = surface_composition.get('surface')
                percentage = surface_composition.get('percentage')
                commercial_surface = surface_composition.get('commercial_surface')

                # Process floor
                floor = None
                if floor_str:
                    if isinstance(floor_str, str):
                        if "rialzato" in floor_str:
                            floor = 0
                        elif "terra" in floor_str:
                            floor = 0
                        elif "seminterrato" in floor_str or "interrato" in floor_str:
                            floor = -1
                        else:
                            floor_match = re.search(r'(\d+)', floor_str)
                            if floor_match:
                                floor = int(floor_match.group(1))
                    elif isinstance(floor_str, int) or isinstance(floor_str, float):
                        floor = int(floor_str)

                surface_key = (
                    element_name,
                    floor,
                    surface,
                    percentage,
                    commercial_surface
                )

                surface_composition_id = dimension_mappings['surface_composition'].get(surface_key)

                if surface_composition_id:
                    # Check if this surface_composition relation already exists
                    bridge_query = """
                    SELECT 1 FROM surface_composition_bridge
                    WHERE listing_id = %s
                    AND date_id = %s
                    AND surface_composition_id = %s
                    """
                    bridge_params = (listing_id, date_id, surface_composition_id)
                    bridge_result = execute_query_silent(postgres_hook, bridge_query, bridge_params)

                    if not bridge_result:
                        # Insert new surface_composition relation using parameterized query
                        bridge_insert_query = """
                        INSERT INTO surface_composition_bridge (listing_id, date_id, surface_composition_id)
                        VALUES (%s, %s, %s)
                        """
                        bridge_insert_params = (listing_id, date_id, surface_composition_id)
                        execute_query_silent(postgres_hook, bridge_insert_query, bridge_insert_params)
                        surface_count += 1

    elapsed_time = time.time() - start_time
    logger.info(f"Fact table loading complete for batch {batch_num} in {elapsed_time:.2f} seconds.")
    logger.info(f"Inserted {insert_count} new records, skipped {skip_count} existing records.")
    logger.info(f"Bridge tables: Added {feature_count} feature relationships and {surface_count} surface composition relationships.")

    return f"Batch {batch_num} loaded into fact table successfully"

# -------------------------------------------------------------------------------

def start_libretranslate_service():
    """Start a local LibreTranslate instance and return the process"""
    logger.info("Starting local LibreTranslate service...")
    
    # Start LibreTranslate as a subprocess
    # Adjust the command based on how LibreTranslate is installed on your system
    process = subprocess.Popen(
        ["libretranslate", "--host", "localhost", "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Register cleanup function to ensure the process is terminated when the script exits
    def cleanup():
        """Ensure the LibreTranslate subprocess is terminated on exit."""
        if process.poll() is None:  # If the process is still running
            logger.info("Terminating LibreTranslate process...")
            process.terminate()
            try:
                process.wait(timeout=10)  # Wait up to 10 seconds for graceful termination
            except subprocess.TimeoutExpired:
                logger.warning("LibreTranslate process did not terminate gracefully, forcing...")
                process.kill()
    
    atexit.register(cleanup)
    
    # Wait for the service to become available
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:5000/languages")
            if response.status_code == 200:
                logger.info("LibreTranslate service is up and running!")
                return process
        except requests.exceptions.ConnectionError:
            pass
        
        logger.info(f"Waiting for LibreTranslate to start (attempt {attempt+1}/{max_attempts})...")
        time.sleep(2)
    
    # If we've exhausted all attempts, raise an error
    logger.error("Failed to start LibreTranslate service")
    cleanup()
    raise RuntimeError("Failed to start LibreTranslate service after multiple attempts")

# -------------------------------------------------------------------------------

def stop_libretranslate_service(process):
    """Stop the LibreTranslate process"""
    if process and process.poll() is None:  # If the process is still running
        logger.info("Stopping LibreTranslate service...")
        process.terminate()
        try:
            process.wait(timeout=10)  # Wait up to 10 seconds for graceful termination
        except subprocess.TimeoutExpired:
            logger.warning("LibreTranslate process did not terminate gracefully, forcing...")
            process.kill()

# -------------------------------------------------------------------------------

def get_primary_key_column(postgres_hook, table_name):
    """Get the primary key column name for a table"""
    query = """
    SELECT a.attname
    FROM   pg_index i
    JOIN   pg_attribute a ON a.attrelid = i.indrelid
                         AND a.attnum = ANY(i.indkey)
    WHERE  i.indrelid = %s::regclass
    AND    i.indisprimary
    """
    params = (table_name,)
    result = execute_query_silent(postgres_hook, query, params)
    if result:
        return result[0][0]
    return None

# -------------------------------------------------------------------------------

def preprocess_text(text):
    """
    Preprocess text before translation:
    1. Add spaces after punctuation marks (., ,, ;, :) when followed by a word
    2. Add spaces before and after slashes (/) and hyphens (-) when they separate words
    3. Remove excessive underscores
    
    Args:
        text: Original text string
        
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return text
    
    # Remove excessive underscores (convert them to a single space)
    text = re.sub(r'_+', ' ', text)
    
    # Add space after punctuation marks (., ,, ;, :) when followed by a word
    # Pattern: punctuation followed by a letter (not a number or space)
    text = re.sub(r'([.,;:])([a-zA-Z])', r'\1 \2', text)
    
    # Add spaces around slashes between words (but not in dates)
    # Pattern: word/word (not number/number)
    text = re.sub(r'([a-zA-Z])\/([a-zA-Z])', r'\1 / \2', text)
    
    # Add spaces around hyphens between words (but not in dates or numbers)
    # Pattern: word-word (not number-number)
    text = re.sub(r'([a-zA-Z])\-([a-zA-Z])', r'\1 - \2', text)
    
    # Clean up any double spaces that might have been created
    text = re.sub(r' +', ' ', text).strip()
    
    return text

# -------------------------------------------------------------------------------

def resolve_conflicts_in_batch(postgres_hook, table, primary_key, conflicts, fk_relationships):
    """
    Resolve translation conflicts in batch mode for better performance
    
    Args:
        postgres_hook: PostgreSQL connection hook
        table: Table name
        primary_key: Primary key column name
        conflicts: Dictionary of conflicts {translated_value: [id1, id2, ...]}
        fk_relationships: List of foreign key relationships for this table
    """
    start_time = time.time()
    logger.info(f"Starting batch conflict resolution for {len(conflicts)} conflicts in {table}")
    
    # Get a connection from the hook
    conn = postgres_hook.get_conn()
    
    try:
        # Group operations by referencing table to minimize transaction overhead
        updates_by_table = {}
        for translated_value, conflicting_ids in conflicts.items():
            primary_id = conflicting_ids[0]
            secondary_ids = conflicting_ids[1:]
            
            if not secondary_ids:
                continue
                
            # Format list of secondary IDs for SQL
            secondary_ids_str = ','.join(str(id) for id in secondary_ids)
            
            # For each table that references this dimension table
            for fk_rel in fk_relationships:
                referencing_table = fk_rel['referencing_table']
                referencing_column = fk_rel['referencing_column']
                
                if referencing_table not in updates_by_table:
                    updates_by_table[referencing_table] = []
                
                updates_by_table[referencing_table].append({
                    'primary_id': primary_id,
                    'secondary_ids': secondary_ids_str,
                    'referencing_column': referencing_column
                })
        
        # Apply updates efficiently, one referencing table at a time
        with conn.cursor() as cursor:
            for referencing_table, updates in updates_by_table.items():
                if not updates:
                    continue
                    
                logger.info(f"Processing {len(updates)} conflict resolutions for table {referencing_table}")
                
                # Create temporary table for bulk update mapping
                cursor.execute(f"""
                    CREATE TEMP TABLE temp_id_mapping (
                        old_id INTEGER NOT NULL,
                        new_id INTEGER NOT NULL
                    ) ON COMMIT DROP
                """)
                
                # Build and execute batch insert of all mappings
                mapping_values = []
                mapping_params = []
                for i, update_info in enumerate(updates):
                    for secondary_id in update_info['secondary_ids'].split(','):
                        if secondary_id:  # Skip empty strings
                            mapping_values.append(f"(%s, %s)")
                            mapping_params.extend([int(secondary_id), update_info['primary_id']])
                
                if mapping_values:
                    # Insert all mappings in one statement
                    cursor.execute(
                        f"INSERT INTO temp_id_mapping (old_id, new_id) VALUES {', '.join(mapping_values)}",
                        mapping_params
                    )
                    
                    # Get the referencing column from the first update (they're all the same per table)
                    referencing_column = updates[0]['referencing_column']
                    
                    # Update references using a join with the temporary table
                    cursor.execute(f"""
                        UPDATE {referencing_table} AS t
                        SET {referencing_column} = m.new_id
                        FROM temp_id_mapping AS m
                        WHERE t.{referencing_column} = m.old_id
                    """)
                    
                    updated_rows = cursor.rowcount
                    logger.info(f"Updated {updated_rows} rows in {referencing_table}")
                
                # Drop the temporary table (happens automatically with ON COMMIT DROP)
            
            # Delete the secondary records that are no longer referenced
            all_secondary_ids = []
            for translated_value, conflicting_ids in conflicts.items():
                all_secondary_ids.extend(conflicting_ids[1:])
            
            if all_secondary_ids:
                # Delete in batches to avoid huge IN clauses
                BATCH_SIZE = 1000
                for i in range(0, len(all_secondary_ids), BATCH_SIZE):
                    batch_ids = all_secondary_ids[i:i+BATCH_SIZE]
                    ids_str = ','.join(str(id) for id in batch_ids)
                    
                    try:
                        cursor.execute(f"DELETE FROM {table} WHERE {primary_key} IN ({ids_str})")
                        deleted_rows = cursor.rowcount
                        logger.info(f"Deleted {deleted_rows} secondary records from {table} (batch {i//BATCH_SIZE + 1})")
                    except Exception as e:
                        logger.warning(f"Could not delete some secondary records: {str(e)}")
            
            # Commit all changes
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error during batch conflict resolution: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Batch conflict resolution completed in {elapsed_time:.2f} seconds")


# -------------------------------------------------------------------------------

def bulk_update_translations(postgres_hook, table, column, primary_key, id_to_translated, conflicts):
    """
    Update translations in bulk for better performance
    
    Args:
        postgres_hook: PostgreSQL connection hook
        table: Table name
        column: Column name to update
        primary_key: Primary key column name
        id_to_translated: Dictionary mapping ID to translated value
        conflicts: Dictionary of conflicts (to skip secondary IDs)
    """
    start_time = time.time()
    logger.info(f"Starting bulk update of translations for {table}.{column}")
    
    # Create a list of all secondary IDs that should be skipped
    skip_ids = []
    for translated_value, conflicting_ids in conflicts.items():
        skip_ids.extend(conflicting_ids[1:])
    skip_ids_set = set(skip_ids)
    
    # Create a parameter list for the update, skipping secondary IDs
    update_params = []
    for record_id, translated_value in id_to_translated.items():
        if record_id not in skip_ids_set:
            update_params.append((translated_value, record_id))
    
    if not update_params:
        logger.info(f"No updates needed for {table}.{column}")
        return
    
    # Execute updates in batches
    BATCH_SIZE = 1000
    conn = postgres_hook.get_conn()
    update_query = f"UPDATE {table} SET {column} = %s WHERE {primary_key} = %s"
    
    try:
        with conn.cursor() as cursor:
            total_updated = 0
            
            for i in range(0, len(update_params), BATCH_SIZE):
                batch = update_params[i:i+BATCH_SIZE]
                
                # Use executemany for efficiency (if supported by the driver)
                cursor.executemany(update_query, batch)
                total_updated += cursor.rowcount
                
                # Log progress for large updates
                if (i + BATCH_SIZE) % 10000 == 0 or i + BATCH_SIZE >= len(update_params):
                    logger.info(f"Updated {i + len(batch)}/{len(update_params)} translations in {table}.{column}")
            
            conn.commit()
            logger.info(f"Total of {total_updated} translations updated in {table}.{column}")
            
    except Exception as e:
        logger.error(f"Error during bulk translation update: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Bulk update completed in {elapsed_time:.2f} seconds")


# -------------------------------------------------------------------------------

def translate_database_fields(**kwargs):
    '''Translate Italian field values to English for the specified columns'''
    logger.info("Starting translation of database fields from Italian to English...")
    start_time = time.time()
    
    # Initialize translation cache
    cache_path = initialize_translation_cache()
    
    # Start LibreTranslate service
    libretranslate_process = None
    try:
        libretranslate_process = start_libretranslate_service()
        
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Define columns to translate and their corresponding tables
        columns_to_translate = {
            'dim_seller_type': ['seller_type'],
            'dim_category': ['category_name'],
            'dim_energy_info': ['heating_type', 'air_conditioning'],
            'dim_type_of_property': ['type_of_property'],
            'dim_condition': ['condition'],
            'dim_rooms_info': ['kitchen_status', 'garage'],
            'dim_building_info': ['building_usage'],
            'dim_features': ['feature_name']
        }
        
        total_translations = 0
        
        # Cache for foreign key relationships (to avoid repeated lookups)
        fk_relationship_cache = {}
    
        for table, columns in columns_to_translate.items():
            # Pre-fetch all foreign key relationships for this table
            primary_key = get_primary_key_column(postgres_hook, table)
            if not primary_key:
                logger.warning(f"Could not determine primary key for table {table}, skipping")
                continue
                
            # Cache the foreign key relationships for this table
            fk_query = f"""
                SELECT
                    tc.constraint_name,
                    tc.table_name as referencing_table,
                    kcu.column_name as referencing_column
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE
                    tc.constraint_type = 'FOREIGN KEY'
                    AND ccu.table_name = '{table}'
                    AND ccu.column_name = '{primary_key}'
            """
            fk_results = execute_query_silent(postgres_hook, fk_query)
            
            # Store results in our cache
            fk_relationship_cache[table] = []
            for fk_row in fk_results:
                constraint_name, referencing_table, referencing_column = fk_row
                fk_relationship_cache[table].append({
                    'constraint_name': constraint_name,
                    'referencing_table': referencing_table,
                    'referencing_column': referencing_column
                })
            
            logger.info(f"Found {len(fk_relationship_cache[table])} foreign key relationships for table {table}")
            
            for column in columns:
                logger.info(f"Processing translations for {table}.{column}")
                
                # Get all non-null distinct values for the column
                query = f"SELECT DISTINCT {primary_key}, {column} FROM {table} WHERE {column} IS NOT NULL"
                results = execute_query_silent(postgres_hook, query)
                
                if not results:
                    logger.info(f"No non-null values found for {table}.{column}")
                    continue
                
                logger.info(f"Found {len(results)} distinct values for {table}.{column}")
                
                # Extract all values that need translation
                original_values = [row[1] for row in results]
                
                # Translate all values in parallel
                translations = translate_values_parallel(
                    original_values, 
                    max_workers=20,
                    cache_path=cache_path
                )
                
                # Update ID-to-translation mapping
                id_to_translated = {row[0]: translations.get(row[1], row[1]) for row in results}
                
                # Count unique translations
                total_translations += len(set(translations.values()))
                
                # Check for conflicts (multiple Italian terms translating to the same English term)
                translated_to_ids = {}
                for record_id, translated_value in id_to_translated.items():
                    if translated_value not in translated_to_ids:
                        translated_to_ids[translated_value] = []
                    translated_to_ids[translated_value].append(record_id)
                
                # Find conflicts (translations with multiple record IDs)
                conflicts = {t: ids for t, ids in translated_to_ids.items() if len(ids) > 1}
                
                if conflicts:
                    logger.info(f"Found {len(conflicts)} translation conflicts for {table}.{column}")
                    
                    # Batch conflict resolution for better performance
                    resolve_conflicts_in_batch(
                        postgres_hook, 
                        table, 
                        primary_key, 
                        conflicts, 
                        fk_relationship_cache[table]
                    )
                
                # Bulk update approach: update in batches to reduce database calls
                bulk_update_translations(
                    postgres_hook,
                    table,
                    column,
                    primary_key,
                    id_to_translated,
                    conflicts
                )
                
                logger.info(f"Completed processing for {table}.{column}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Translation completed in {elapsed_time:.2f} seconds.")
        logger.info(f"Translated {total_translations} unique values across the database.")
        
        return f"Translation task completed successfully"

    finally:
        # Always stop the LibreTranslate service, even if there's an error
        if libretranslate_process:
            stop_libretranslate_service(libretranslate_process)

# -------------------------------------------------------------------------------

def finalize_migration(**kwargs):
    '''Finalize the migration by reporting overall statistics'''
    ti = kwargs['ti']
    
    # Get the batch info to know how many batches were processed
    all_batches = json.loads(Variable.get("all_batches"))
    num_batches = len(all_batches)

    logger.info(f"Migration completed. Total batches processed: {num_batches}")

    # Count total records in fact table
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    result = execute_query_silent(postgres_hook, "SELECT COUNT(*) FROM fact_listing")
    total_records = result[0][0] if result else 0

    logger.info(f"Total records in fact table: {total_records}")

    # Count records in bridge tables
    features_result = execute_query_silent(postgres_hook, "SELECT COUNT(*) FROM listing_features_bridge")
    features_count = features_result[0][0] if features_result else 0

    surface_result = execute_query_silent(postgres_hook, "SELECT COUNT(*) FROM surface_composition_bridge")
    surface_count = surface_result[0][0] if surface_result else 0

    logger.info(f"Total feature relationships: {features_count}")
    logger.info(f"Total surface composition relationships: {surface_count}")

    # Clean up Airflow variables
    try:
        Variable.delete("batch_info")
        Variable.delete("all_batches")
        logger.info("Cleaned up temporary variables")
    except:
        logger.warning("Could not clean up temporary variables")

    return {
        "batches_processed": num_batches,
        "fact_table_records": total_records,
        "feature_relationships": features_count,
        "surface_relationships": surface_count
    }

# -------------------------------------------------------------------------------

def create_dynamic_tasks(dag):
    """Function to dynamically create process and load tasks for all batches"""
    setup_task = PythonOperator(
        task_id='setup_batch_processing',
        python_callable=setup_batch_processing,
        dag=dag,
    )

    # Create a task for translation
    translate_task = PythonOperator(
        task_id='translate_database_fields',
        python_callable=translate_database_fields,
        provide_context=True,
        dag=dag,
    )

    finalize_task = PythonOperator(
        task_id='finalize_migration',
        python_callable=finalize_migration,
        provide_context=True,
        dag=dag,
    )

    # Get total number of batches
    total_batches = count_batches()
    logger.info(f"Creating tasks for {total_batches} batches")

    # Create tasks for each batch
    batch_tasks = []
    for batch_num in range(total_batches):
        process_batch_task = PythonOperator(
            task_id=f'process_batch_{batch_num}',
            python_callable=process_batch,
            op_kwargs={'batch_num': batch_num},
            provide_context=True,
            dag=dag,
        )

        load_fact_task = PythonOperator(
            task_id=f'load_fact_table_for_batch_{batch_num}',
            python_callable=load_fact_table_for_batch,
            op_kwargs={'batch_num': batch_num},
            provide_context=True,
            dag=dag,
        )

        # Define dependencies
        setup_task >> process_batch_task >> load_fact_task
        batch_tasks.append(load_fact_task)

    # All batch tasks must complete before translation
    for task in batch_tasks:
        task >> translate_task
    
    # Translation must complete before finalization
    translate_task >> finalize_task

    return setup_task, translate_task, finalize_task, batch_tasks

# -------------------------------------------------------------------------------
#-----------------------------DAG DEFINITION------------------------------------
# -------------------------------------------------------------------------------

#Defines the DAG's default arguments
default_args = {
"owner": "Leonardo Pacciani-Mori",
"start_date": days_ago(0),
"max_active_tasks": 32,
"retries": 3,
"retry_delay": timedelta(minutes=1)
}

#Defines the DAG
dag = DAG(
'listing.website_MongoDB_to_PostgreSQL_migration',
default_args=default_args,
description='DAG to migrate data from the data warehouse from MongoDB to PostgreSQL',
schedule=None,
)

#Create dynamic tasks
setup_task, translate_task, finalize_task, batch_tasks = create_dynamic_tasks(dag)
