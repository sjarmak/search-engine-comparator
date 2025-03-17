"""
Academic Search Results Comparator - Backend API
Main FastAPI application entry point
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Set, Optional, Any
import httpx
import asyncio
import re
import os
from scholarly import scholarly, ProxyGenerator
import math
from dotenv import load_dotenv
import logging
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from functools import lru_cache
import random
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import urllib.parse
import platform
import sys
import ssl
import certifi
import signal
from contextlib import contextmanager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_platform_specific_fixes():
    """Apply fixes specific to the operating system platform."""
    system = platform.system()
    logger.info(f"Detected platform: {system}")
    
    if system == "Darwin":  # macOS
        # Fix for macOS SSL certificate verification issues
        try:
            # Override SSL default context with certifi's certificate bundle
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
            
            # Set environment variables for requests/urllib3
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            
            logger.info(f"SSL certificate path set to: {certifi.where()}")
            logger.info("Successfully applied macOS-specific SSL fixes")
        except Exception as e:
            logger.error(f"Failed to apply macOS-specific SSL fixes: {str(e)}")

# Apply platform-specific fixes BEFORE any other operations
apply_platform_specific_fixes()

# Load environment variables
load_dotenv()

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

# Service configuration with fallback settings
SERVICE_CONFIG = {
    "ads": {
        "enabled": True,
        "priority": 1,  # Lower number = higher priority
        "timeout": 15,  # seconds
        "min_results": 5,  # Minimum acceptable results
    },
    "scholar": {
        "enabled": True,
        "priority": 2,
        "timeout": 20,
        "min_results": 3,
    },
    "semanticScholar": {
        "enabled": True,
        "priority": 3,
        "timeout": 15,
        "min_results": 5,
    },
    "webOfScience": {
        "enabled": True,
        "priority": 4,
        "timeout": 20,
        "min_results": 3,
    }
}

# Initialize FastAPI
app = FastAPI(
    title="Academic Search Results Comparator API",
    description="API for comparing search results from multiple academic search engines",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Constants
ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
WOS_API_URL = "https://api.clarivate.com/apis/wos-starter/v1/"
NUM_RESULTS = 20
TIMEOUT_SECONDS = 60  # Increased from 30 to 60

# Get API keys from environment variables
# ADS_API_KEY = os.getenv("ADS_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
WOS_API_KEY = os.getenv("WOS_API_KEY")

# Add these variables at the module level
last_proxy_refresh_time = 0
PROXY_REFRESH_INTERVAL = 3600  # Refresh proxy every hour

def refresh_scholarly_proxy_if_needed():
    """Check if scholarly proxy needs to be refreshed based on time elapsed."""
    global last_proxy_refresh_time
    current_time = time.time()
    
    # If more than PROXY_REFRESH_INTERVAL seconds have passed since last refresh
    if current_time - last_proxy_refresh_time > PROXY_REFRESH_INTERVAL:
        logger.info("Refreshing Google Scholar proxy due to timeout")
        success = setup_scholarly_proxy()
        if success:
            last_proxy_refresh_time = current_time
            logger.info("Successfully refreshed Google Scholar proxy")
        else:
            logger.error("Failed to refresh Google Scholar proxy")

# Add at module level
last_scholar_request_time = 0
MIN_REQUEST_INTERVAL = 10  # seconds

# Add this check before making requests
async def respect_rate_limit():
    global last_scholar_request_time
    current_time = time.time()
    time_since_last = current_time - last_scholar_request_time
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        wait_time = MIN_REQUEST_INTERVAL - time_since_last
        logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
        await asyncio.sleep(wait_time)
        
    last_scholar_request_time = time.time()

def setup_scholarly_proxy() -> bool:
    """
    Set up proxy for Google Scholar with better error handling and fallbacks.
    Returns True if any setup method was successful, False otherwise.
    """
    try:
        pg = ProxyGenerator()
        
        # Try free proxies first
        logger.info("Attempting to set up free proxies for Google Scholar")
        try:
            success = pg.FreeProxies(timeout=15)
            if success:
                logger.info("Successfully set up free proxies for Google Scholar")
                scholarly.use_proxy(pg)
                scholarly.set_timeout(20)  # Slightly higher timeout
                
                # Additional scholarly settings to reduce CAPTCHA risk
                if hasattr(scholarly, '_SESSION_HEADER'):
                    scholarly._SESSION_HEADER.update({
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9'
                    })
                
                return True
        except Exception as e:
            logger.warning(f"Free proxies setup failed: {str(e)}")
        
        # Try with a specific country's proxies
        try:
            success = pg.FreeProxies(country="us", timeout=15)
            if success:
                logger.info("Successfully set up US free proxies for Google Scholar")
                scholarly.use_proxy(pg)
                scholarly.set_timeout(20)
                return True
        except Exception as e:
            logger.warning(f"Country-specific proxies setup failed: {str(e)}")
        
        # Last resort: direct connection
        logger.info("Using direct connection for Google Scholar (no proxy)")
        scholarly.use_proxy(None)
        scholarly.set_timeout(25)  # Longer timeout for direct connection
        return True
        
    except Exception as e:
        logger.error(f"All proxy setup methods failed: {str(e)}")
        try:
            # Final attempt with no proxy
            scholarly.use_proxy(None)
            return True
        except:
            return False

# Initialize scholarly proxy
proxy_success = setup_scholarly_proxy()
if not proxy_success:
    logger.warning("Failed to set up any proxy for Google Scholar. This may affect results.")

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    sources: List[str]
    metrics: List[str]
    fields: List[str]

class SearchResult(BaseModel):
    title: str
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    source: str
    rank: int

# Create a timeout context manager
class timeout:
    def __init__(self, seconds, *, timeout_message='Operation timed out'):
        self.seconds = seconds
        self.timeout_message = timeout_message
        self.original_timeout_handler = None

    def __enter__(self):
        if hasattr(signal, 'SIGALRM'):  # Only on Unix systems
            self.original_timeout_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, 'SIGALRM'):  # Only on Unix systems
            signal.alarm(0)  # Disable the alarm
            signal.signal(signal.SIGALRM, self.original_timeout_handler)
        return exc_type is TimeoutError

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

async def safe_api_request(client, method, url, **kwargs):
    """
    Makes an API request with better error handling and retry logic.
    
    Args:
        client: httpx.AsyncClient instance
        method: HTTP method (get, post, etc.)
        url: Target URL
        **kwargs: Additional arguments for the request
        
    Returns:
        Response object or None if all retries failed
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            request_func = getattr(client, method.lower())
            
            # Add timeout if not specified
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 30.0  # 30 seconds timeout
                
            response = await request_func(url, **kwargs)
            
            # Log detailed info for non-200 responses
            if response.status_code != 200:
                logger.error(
                    f"API request failed: {method} {url}, "
                    f"Status: {response.status_code}, "
                    f"Response: {response.text[:500]}..."  # Limit response text size in logs
                )
                
                # Specific handling based on status code
                if response.status_code == 429:
                    # Rate limiting - use exponential backoff
                    retry_seconds = retry_delay * (2 ** attempt)
                    logger.info(f"Rate limited. Waiting {retry_seconds}s before retry {attempt+1}/{max_retries}")
                    await asyncio.sleep(retry_seconds)
                    continue
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        retry_seconds = retry_delay * (attempt + 1)
                        logger.info(f"Server error. Waiting {retry_seconds}s before retry {attempt+1}/{max_retries}")
                        await asyncio.sleep(retry_seconds)
                        continue
            
            return response
            
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(f"Connection error on attempt {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                retry_seconds = retry_delay * (attempt + 1)
                logger.info(f"Waiting {retry_seconds}s before retry")
                await asyncio.sleep(retry_seconds)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error during API request: {str(e)}")
            return None
    
    return None

# Helper functions for text normalization and cleaning
def normalize_text(text: str) -> str:
    """Normalize text for better comparison."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def stem_text(text: str) -> str:
    """Apply stemming to text."""
    if not text:
        return ""
    
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def preprocess_text(text: str, apply_stemming: bool = True) -> str:
    """Preprocess text by normalizing and optionally stemming."""
    text = normalize_text(text)
    if apply_stemming:
        text = stem_text(text)
    return text

# Functions for similarity metrics
def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets: J(A,B) = |A ∩ B| / |A ∪ B|"""
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_rank_based_overlap(list1: List[Any], list2: List[Any], k: int = None) -> float:
    """
    Calculate rank-based overlap (RBO) between two ranked lists.
    Implementation based on "A Similarity Measure for Indefinite Rankings" by Webber et al.
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        k: Depth of evaluation (if None, use the length of the shorter list)
        
    Returns:
        Float value between 0 and 1, where 1 is perfect agreement
    """
    if not list1 or not list2:
        return 0.0
    
    # Use the smaller list's length if k is not specified
    if k is None:
        k = min(len(list1), len(list2))
    else:
        k = min(k, min(len(list1), len(list2)))
    
    # Weight parameter (typically 0.9 or 0.98)
    p = 0.9
    
    # Calculate the weighted overlap at each depth
    sum_weight = 0
    curr_weight = (1 - p)
    total = 0
    
    for depth in range(1, k + 1):
        set1 = set(list1[:depth])
        set2 = set(list2[:depth])
        overlap = len(set1.intersection(set2))
        agreement = overlap / depth
        total += curr_weight * agreement
        sum_weight += curr_weight
        curr_weight *= p
    
    # Normalize by the sum of weights
    return total / sum_weight if sum_weight > 0 else 0.0

def calculate_cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    """Calculate cosine similarity between two term frequency vectors."""
    if not vec1 or not vec2:
        return 0.0
    
    # Find common keys
    common_keys = set(vec1.keys()).intersection(set(vec2.keys()))
    
    # Calculate dot product
    dot_product = sum(vec1[key] * vec2[key] for key in common_keys)
    
    # Calculate magnitudes
    mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    
    # Calculate cosine similarity
    if mag1 * mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)

# Functions to get search results from different sources
async def get_ads_results(query: str, fields: List[str], num_results: int = NUM_RESULTS) -> List[SearchResult]:
    """
    Get results from NASA ADS (SciX) with caching.
    
    Args:
        query: Search query string
        fields: List of fields to retrieve
        num_results: Number of results to return
        
    Returns:
        List of SearchResult objects
    """
    # Get API key from environment
    api_key = os.getenv("ADS_API_KEY")
    if not api_key:
        logger.error("ADS API key not found in environment variables")
        return []
        
    logger.info(f"Starting ADS search with query: {query}")
    
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "q": query,
        "rows": num_results,
        "fl": "title,author,abstract,doi,year,bibcode",
        "sort": "score desc",  # Changed from "date desc" to "score desc"
    }
    
    logger.info(f"Making ADS API request with params: {params}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await safe_api_request(
                client,
                "GET",
                ADS_API_URL,
                headers=headers,
                params=params
            )
            
        if response and response.status_code != 200:
            logger.error(f"ADS API error: Status {response.status_code}")
            logger.error(f"ADS API response: {response.text[:500]}")  # Log first 500 chars of error
            return []
            
        if not response:
            logger.error("No response from ADS API")
            return []
            
        data = response.json()
        docs = data.get('response', {}).get('docs', [])
        logger.info(f"Got {len(docs)} results from ADS")
        
        results = []
        for i, doc in enumerate(docs, 1):
            try:
                result = SearchResult(
                    title=doc.get('title', [''])[0],
                    authors=doc.get('author'),
                    abstract=doc.get('abstract'),
                    doi=doc.get('doi', [''])[0] if doc.get('doi') else None,
                    year=doc.get('year'),
                    url=f"https://ui.adsabs.harvard.edu/abs/{doc['bibcode']}/abstract",
                    source="ads",
                    rank=i
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing result {i}: {str(e)}")
                continue
                
        return results
        
    except Exception as e:
        logger.error(f"Error in get_ads_results: {str(e)}")
        return []

async def get_scholar_direct_html(query: str, num_results: int = 20) -> List[SearchResult]:
    """
    Get Google Scholar results by directly parsing the HTML response.
    This approach is more reliable but may trigger CAPTCHAs if used too frequently.
    """
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&as_sdt=0,5&num={num_results}"
    
    # Rotate between different user agents to reduce CAPTCHA risk
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://scholar.google.com/",
        "DNT": "1",
    }
    
    logger.info("Using direct HTML parsing approach for Google Scholar")
    
    try:
        # Use SSL verification explicitly from certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        async with httpx.AsyncClient(verify=certifi.where()) as client:
            response = await client.get(
                url, 
                headers=headers, 
                follow_redirects=True, 
                timeout=15.0
            )
            
        if response.status_code != 200:
            logger.error(f"Google Scholar direct HTML request failed with status {response.status_code}")
            return []
            
        # Check for CAPTCHA
        if "sorry" in response.text.lower() and "captcha" in response.text.lower():
            logger.error("Google Scholar returned a CAPTCHA challenge")
            return []
        
        # Parse results using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('div', class_='gs_ri')
        
        results = []
        for i, article in enumerate(articles[:num_results]):
            try:
                # Extract title (remove [HTML], [PDF], etc. markers)
                title_element = article.find('h3', class_='gs_rt')
                if not title_element:
                    continue
                    
                title = title_element.text
                for marker in ['[HTML]', '[PDF]', '[BOOK]', '[CITATION]']:
                    title = title.replace(marker, '')
                title = title.strip()
                
                # Extract authors and publication venue
                authors_venue = article.find('div', class_='gs_a')
                authors_text = authors_venue.text if authors_venue else ""
                
                # Try to extract authors from the text
                authors = []
                if authors_text:
                    # Authors are typically before the first dash
                    author_part = authors_text.split('-')[0] if '-' in authors_text else authors_text
                    # Split by commas, but be careful with "and"
                    author_list = author_part.replace(' and ', ', ').split(',')
                    authors = [a.strip() for a in author_list if a.strip()]
                
                # Try to extract year
                year = None
                if authors_text:
                    # Year is typically in the middle part
                    year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
                    if year_match:
                        try:
                            year = int(year_match.group(0))
                        except:
                            pass
                
                # Extract snippet/abstract
                snippet = article.find('div', class_='gs_rs')
                abstract = snippet.text if snippet else ""
                
                # Extract URL if available
                url = ""
                if title_element and title_element.find('a'):
                    url = title_element.find('a').get('href', '')
                
                # Create result object
                result = SearchResult(
                    title=title,
                    authors=authors[:3],  # Limit to first 3 authors
                    abstract=abstract,
                    year=year,
                    url=url,
                    source="scholar",
                    rank=i + 1
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error extracting Google Scholar result: {str(e)}")
                continue
                
        logger.info(f"Direct HTML parsing returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in Google Scholar direct HTML method: {str(e)}")
        return []

async def get_scholar_results(query: str, fields: List[str], num_results: int = NUM_RESULTS) -> List[SearchResult]:
    """
    Get results from Google Scholar using a multi-approach strategy with fallbacks.
    
    Args:
        query: Search query string
        fields: List of fields to retrieve
        num_results: Number of results to return
        
    Returns:
        List of SearchResult objects
    """
    logger.info(f"Starting Google Scholar search with query: {query}")
    
    # Check cache first
    cache_key = get_cache_key("scholar", query, fields)
    cached_results = load_from_cache(cache_key)
    
    if cached_results is not None:
        logger.info(f"Retrieved {len(cached_results)} Google Scholar results from cache")
        return cached_results
    
    # Wait to respect rate limits
    await respect_rate_limit()
    
    # APPROACH 1: Try direct HTML parsing first (most reliable)
    try:
        logger.info("Attempting direct HTML parsing approach for Google Scholar")
        results = await get_scholar_direct_html(query, num_results)
        
        if results and len(results) >= 3:  # If we got at least 3 results
            logger.info(f"Direct HTML approach successful, got {len(results)} results")
            save_to_cache(cache_key, results)
            return results
        else:
            logger.warning("Direct HTML approach didn't return enough results, trying scholarly library")
    except Exception as e:
        logger.error(f"Direct HTML approach failed: {str(e)}")
    
    # APPROACH 2: Try the scholarly library
    refresh_scholarly_proxy_if_needed()
    
    try:
        logger.info("Attempting scholarly library approach for Google Scholar")
        
        # Get results while allowing for asyncio to yield control
        search_query = scholarly.search_pubs(query)
        
        results = []
        for i in range(num_results):
            try:
                # Allow other async tasks to run by yielding control periodically
                if i % 3 == 0:  # Yield every 3 items to prevent blocking too long
                    await asyncio.sleep(0.1)
                
                # Get the next publication with a timeout
                pub = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: next(search_query, None)),
                    timeout=8.0  # Timeout for individual item fetch
                )
                
                if pub is None:
                    # No more results
                    break
                
                if 'bib' in pub:
                    # Extract authors (handle both list and string formats)
                    authors = pub['bib'].get('author', [])
                    if isinstance(authors, str):
                        authors = [authors]
                    
                    # Get year with fallback
                    year = None
                    if 'pub_year' in pub['bib'] and pub['bib']['pub_year']:
                        try:
                            year = int(pub['bib']['pub_year'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Create search result
                    result = SearchResult(
                        title=pub['bib'].get('title', ''),
                        authors=authors[:3],  # Limit to first 3 authors like other sources
                        abstract=pub['bib'].get('abstract', ''),
                        year=year,
                        url=pub.get('pub_url', ''),
                        source="scholar",
                        rank=i + 1
                    )
                    results.append(result)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching Google Scholar result #{i+1}")
                break  # Break on timeout to avoid hanging
            except Exception as e:
                logger.warning(f"Error processing Google Scholar result #{i+1}: {str(e)}")
                if "CAPTCHA" in str(e):
                    break  # Break on CAPTCHA to try fallback
                continue
                
        if results and len(results) >= 3:  # If we got at least 3 results
            logger.info(f"Scholarly approach successful, got {len(results)} results")
            save_to_cache(cache_key, results)
            return results
                
    except Exception as e:
        logger.error(f"Error in scholarly approach: {str(e)}")
    
    # APPROACH 3: Use the fallback method as last resort
    try:
        logger.info("Attempting fallback method for Google Scholar")
        results = await get_scholar_results_fallback(query, num_results)
        
        if results:
            logger.info(f"Fallback method successful, got {len(results)} results")
            save_to_cache(cache_key, results)
            return results
    except Exception as e:
        logger.error(f"Fallback method failed: {str(e)}")
    
    # If all approaches failed, return empty list
    logger.error("All Google Scholar approaches failed, returning empty list")
    return []

async def get_scholar_results_fallback(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    A simplified fallback method for getting Google Scholar results that doesn't use the scholarly library.
    This serves as a last resort if the scholarly library fails.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of simplified result dictionaries
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&as_sdt=0,5&num={num_results}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        async with httpx.AsyncClient(verify=certifi.where()) as client:
            response = await client.get(url, headers=headers, follow_redirects=True, timeout=15.0)
            
        if response.status_code != 200:
            logger.error(f"Google Scholar fallback failed with status {response.status_code}")
            return []
            
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('div', class_='gs_ri')
        
        results = []
        for i, article in enumerate(articles[:num_results]):
            try:
                # Extract title
                title_element = article.find('h3', class_='gs_rt')
                title = title_element.text if title_element else "Unknown Title"
                
                # Clean title
                for marker in ['[HTML]', '[PDF]', '[BOOK]', '[CITATION]']:
                    title = title.replace(marker, '')
                title = title.strip()
                
                # Extract authors and publication venue
                authors_venue = article.find('div', class_='gs_a')
                authors_text = authors_venue.text if authors_venue else ""
                
                # Extract snippet/abstract
                snippet = article.find('div', class_='gs_rs')
                abstract = snippet.text if snippet else ""
                
                # Extract URL if available
                url = ""
                if title_element and title_element.find('a'):
                    url = title_element.find('a').get('href', '')
                
                # Try to extract authors
                authors = []
                if authors_text:
                    # Authors are typically before the first dash
                    author_part = authors_text.split('-')[0] if '-' in authors_text else authors_text
                    # Split by commas, but be careful with "and"
                    author_list = author_part.replace(' and ', ', ').split(',')
                    authors = [a.strip() for a in author_list if a.strip()]
                
                # Try to extract year
                year = None
                if authors_text:
                    # Year is typically in the middle part
                    year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
                    if year_match:
                        try:
                            year = int(year_match.group(0))
                        except:
                            pass
                
                # Create a simplified result
                result = SearchResult(
                    title=title,
                    authors=authors[:3] if authors else [],
                    abstract=abstract,
                    url=url,
                    year=year,
                    source="scholar",
                    rank=i + 1
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error extracting Google Scholar result: {str(e)}")
                continue
                
        return results
        
    except Exception as e:
        logger.error(f"Error in Google Scholar fallback method: {str(e)}")
        return []

async def get_semantic_scholar_results(query: str, fields: List[str], num_results: int = NUM_RESULTS) -> List[SearchResult]:
    """
    Get results from Semantic Scholar API with caching and rate limit handling.
    
    Args:
        query: Search query string
        fields: List of fields to retrieve
        num_results: Number of results to return
        
    Returns:
        List of SearchResult objects
    """
    # Check cache first
    cache_key = get_cache_key("semanticScholar", query, fields)
    cached_results = load_from_cache(cache_key)
    
    if cached_results is not None:
        logger.info(f"Retrieved {len(cached_results)} Semantic Scholar results from cache")
        return cached_results
    
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Try with API key if available
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}
    
    params = {
        "query": query,
        "limit": num_results,
        "fields": "title,authors,abstract,year,externalIds,url"
    }
    
    # Implement progressive backoff for retries
    max_retries = 5
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Add delay on retries - exponential backoff
            if attempt > 0:
                retry_delay = base_delay * (2 ** (attempt - 1))  # 2, 4, 8, 16...
                logger.info(f"Semantic Scholar retry {attempt}/{max_retries}, waiting {retry_delay}s")
                await asyncio.sleep(retry_delay)
            
            logger.info(f"Making Semantic Scholar API request (attempt {attempt+1}/{max_retries})")
            
            async with httpx.AsyncClient() as client:
                response = await safe_api_request(
                    client,
                    "GET",
                    base_url,
                    headers=headers,
                    params=params
                )
                
            if not response:
                logger.error("No response from Semantic Scholar API")
                continue  # Try again
                
            if response.status_code == 429:
                logger.warning("Semantic Scholar rate limit exceeded, will retry with backoff")
                # Extract retry-after header if available
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after) + 1  # Add 1 second buffer
                    logger.info(f"Semantic Scholar API requested wait time: {wait_time}s")
                    await asyncio.sleep(wait_time)
                continue  # Try again after waiting
                
            if response.status_code != 200:
                logger.error(f"Error fetching from Semantic Scholar. Status: {response.status_code}")
                if response.status_code >= 500:  # Server error, retry
                    continue
                else:  # Client error, don't retry
                    break
                
            data = response.json()
            papers = data.get("data", [])
            
            if not papers:
                logger.warning("Semantic Scholar returned no results")
                break
                
            logger.info(f"Received {len(papers)} results from Semantic Scholar")
            
            results = []
            for i, paper in enumerate(papers, 1):
                try:
                    # Extract authors (maximum 3 to match other engines)
                    authors = []
                    if paper.get("authors"):
                        for author in paper["authors"][:3]:
                            author_name = author.get("name", "")
                            if author_name:
                                authors.append(author_name)
                    
                    # Get DOI if available
                    doi = None
                    if paper.get("externalIds") and paper["externalIds"].get("DOI"):
                        doi = paper["externalIds"]["DOI"]
                    
                    # Get URL (paper URL or DOI URL)
                    url = paper.get("url", "")
                    if not url and doi:
                        url = f"https://doi.org/{doi}"
                    
                    result = SearchResult(
                        title=paper.get("title", ""),
                        authors=authors,
                        abstract=paper.get("abstract", ""),
                        doi=doi,
                        year=paper.get("year"),
                        url=url,
                        source="semanticScholar",
                        rank=i
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing Semantic Scholar result {i}: {str(e)}")
            
            # Save to cache and return if we have results
            if results:
                save_to_cache(cache_key, results)
                return results
            
            # No results found after successful request
            break
                
        except Exception as e:
            logger.error(f"Error in Semantic Scholar request (attempt {attempt+1}): {str(e)}")
            # Continue to next retry
    
    # If we reach here, all attempts failed or returned no results
    logger.warning(f"No results found in Semantic Scholar for '{query}' after {max_retries} attempts")
    
    # Create a placeholder result
    no_results_msg = f"The term '{query}' did not match any documents in Semantic Scholar, or the API rate limit was exceeded."
    placeholder = SearchResult(
        title="[No results found in Semantic Scholar]",
        authors=[],
        abstract=no_results_msg,
        year=None,
        url="https://www.semanticscholar.org/",
        source="semanticScholar",
        rank=1
    )
    results = [placeholder]
    
    # Save empty results to cache
    save_to_cache(cache_key, results)
    return results

async def get_web_of_science_results(query: str, fields: List[str], num_results: int = NUM_RESULTS) -> List[SearchResult]:
    """
    Get results from Web of Science Starter API with caching.
    
    Args:
        query: Search query string
        fields: List of fields to retrieve
        num_results: Number of results to return
        
    Returns:
        List of SearchResult objects
    """
    # Check cache first
    cache_key = get_cache_key("webOfScience", query, fields)
    cached_results = load_from_cache(cache_key)
    
    if cached_results is not None:
        logger.info(f"Retrieved {len(cached_results)} Web of Science results from cache")
        return cached_results
        
    if not WOS_API_KEY:
        logger.error("Web of Science API key not found in environment variables.")
        return []
    
    # Format query with proper WoS syntax
    wos_query = f'AU=({query})'
    
    headers = {
        "X-ApiKey": WOS_API_KEY,
        "Accept": "application/json"
    }
    
    # The correct WoS Starter API endpoint
    base_url = f"{WOS_API_URL}documents"
    
    params = {
        "db": "WOS",
        "q": wos_query,  # FIXED: Don't add ALL=() twice
        "limit": min(num_results, 50),
        "page": 1
    }
    
    logger.info(f"Making Web of Science API request with query: {wos_query}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                base_url,
                headers=headers,
                params=params,
                timeout=30.0
            )
            
        if response.status_code != 200:
            logger.warning(f"WoS API error: Status {response.status_code}")
            logger.debug(f"Response: {response.text[:500]}")
            # Return placeholder for error case
            no_results_msg = f"Error accessing Web of Science API: Status {response.status_code}"
            placeholder = SearchResult(
                title="[Web of Science API Error]",
                authors=[],
                abstract=no_results_msg,
                year=None,
                url="https://webofknowledge.com",
                source="webOfScience",
                rank=1
            )
            results = [placeholder]
            save_to_cache(cache_key, results)
            return results
            
        data = response.json()
        
        # Check if there are results
        documents = data.get('hits', [])
        total = data.get('metadata', {}).get('total', 0)
        
        logger.info(f"WoS query returned {total} total results, {len(documents)} in this page")
        
        if not documents:
            # No results found with the all fields search
            logger.warning(f"No results found in Web of Science for '{query}'")
            no_results_msg = f"The term '{query}' did not match any documents in the Web of Science Core Collection."
            placeholder = SearchResult(
                title="[No results found in Web of Science database]",
                authors=[],
                abstract=no_results_msg,
                year=None,
                url="https://webofknowledge.com",
                source="webOfScience",
                rank=1
            )
            results = [placeholder]
            save_to_cache(cache_key, results)
            return results
            
        # Process the results
        results = []
        for i, doc in enumerate(documents[:num_results], 1):
            try:
                # Get URL - use DOI or the Web of Science URL
                doi = doc.get("identifiers", {}).get("doi", None)
                url = None
                if doi is not None:
                    url = f"https://doi.org/{doi}"
                
                result = SearchResult(
                    title=doc.get("title"),
                    authors=[a.get("displayName", "") for a in doc.get("names", {}).get("authors", [])[:3]],
                    abstract='',
                    doi=doi,
                    year=doc.get("source", {}).get("publishYear"),
                    url=url,
                    source="webOfScience",
                    rank=i
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing WoS result {i}: {str(e)}")
                continue
        
        # Save results to cache and return
        save_to_cache(cache_key, results)
        return results
            
    except Exception as e:
        logger.error(f"Error in WoS API request: {str(e)}")
        
        # Create a placeholder result for exception case
        no_results_msg = f"Error accessing Web of Science API: {str(e)}"
        placeholder = SearchResult(
            title="[Web of Science API Error]",
            authors=[],
            abstract=no_results_msg,
            year=None,
            url="https://webofknowledge.com",
            source="webOfScience",
            rank=1
        )
        results = [placeholder]
        save_to_cache(cache_key, results)
        return results

# Add this function to implement service fallback
async def get_results_with_fallback(query: str, sources: List[str], fields: List[str], attempts: int = 2) -> Dict[str, List[SearchResult]]:
    """
    Get results from requested sources with fallback mechanism.
    
    Args:
        query: Search query
        sources: List of source names to query
        fields: List of fields to retrieve
        attempts: Number of attempts to try for each service
        
    Returns:
        Dictionary mapping source names to their results
    """
    sources_results = {}
    
    # Sort sources by priority
    prioritized_sources = sorted(
        [s for s in sources if s in SERVICE_CONFIG and SERVICE_CONFIG[s]["enabled"]], 
        key=lambda s: SERVICE_CONFIG[s]["priority"]
    )
    
    for source in prioritized_sources:
        config = SERVICE_CONFIG[source]
        results = []
        
        for attempt in range(attempts):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt+1}/{attempts} for {source}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                if source == "ads":
                    results = await get_ads_results(query, fields)
                elif source == "scholar":
                    # For Google Scholar, implement a multi-level fallback approach
                    try:
                        # First try the standard scholarly approach
                        logger.info("Trying primary Google Scholar method")
                        results = await get_scholar_results(query, fields)
                        
                        # If we didn't get enough results, try the fallback approach
                        if len(results) < config["min_results"]:
                            logger.info(f"Primary Google Scholar method returned only {len(results)} results, trying fallback")
                            fallback_results = await get_scholar_results_fallback(query, NUM_RESULTS)
                            
                            # If fallback gave us more results, use those instead
                            if len(fallback_results) > len(results):
                                results = fallback_results
                                logger.info(f"Using fallback Google Scholar results ({len(results)} items)")
                            
                    except Exception as scholar_error:
                        logger.error(f"Primary Google Scholar method failed: {str(scholar_error)}")
                        logger.info("Falling back to secondary Google Scholar method")
                        results = await get_scholar_results_fallback(query, NUM_RESULTS)
                        
                elif source == "semanticScholar":
                    results = await get_semantic_scholar_results(query, fields)
                elif source == "webOfScience":
                    results = await get_web_of_science_results(query, fields)
                else:
                    logger.warning(f"Unknown source: {source}")
                    results = []
                
                # Check if we got sufficient results
                if len(results) >= config["min_results"]:
                    logger.info(f"Successfully retrieved {len(results)} results from {source}")
                    sources_results[source] = results
                    break  # Success, no need for more attempts
                else:
                    logger.warning(f"{source} returned only {len(results)} results, minimum is {config['min_results']}")
                    # Store what we have but continue to next attempt
                    if results:
                        sources_results[source] = results
                
            except Exception as e:
                logger.error(f"Error fetching from {source} (attempt {attempt+1}/{attempts}): {str(e)}")
                # Continue to next attempt
        
        # If we still don't have any results after all attempts, store an empty list
        if source not in sources_results:
            sources_results[source] = []
    
    # Ensure all requested sources have an entry
    for source in sources:
        if source not in sources_results:
            sources_results[source] = []
    
    return sources_results

# Function to compare results and calculate similarities
def compare_results(sources_results: Dict[str, List[SearchResult]], metrics: List[str], fields: List[str]) -> Dict[str, Any]:
    """
    Compare results from different sources and calculate similarity metrics.
    
    Args:
        sources_results: Dictionary mapping source names to their search results
        metrics: List of metrics to calculate
        fields: List of fields to consider in comparisons
        
    Returns:
        Dictionary containing similarity metrics and other comparison data
    """
    # Create sets of documents from each source based on DOI or title
    source_sets = {}
    source_lists = {}
    source_texts = {}
    
    # Store full results by DOI and by normalized title for matching
    source_by_doi = {}
    source_by_title = {}
    
    for source, results in sources_results.items():
        # Initialize containers
        source_by_doi[source] = {}
        source_by_title[source] = {}
        
        # Create a set of identifiers (both DOI and title)
        identifiers_set = set()
        
        for result in results:
            # Store by DOI if available
            if result.doi:
                doi = result.doi.lower().strip()
                source_by_doi[source][doi] = result
                identifiers_set.add(f"doi:{doi}")
                
            # Always store by normalized title
            norm_title = result.title.lower().strip()
            source_by_title[source][norm_title] = result
            identifiers_set.add(f"title:{norm_title}")
        
        # Store the set for similarity calculations
        source_sets[source] = identifiers_set
        
        # Create ordered lists for rank-based comparison
        source_lists[source] = [
            f"doi:{result.doi}" if result.doi else f"title:{result.title.lower().strip()}" 
            for result in results
        ]
        
        # Create concatenated text for text-based comparisons
        if "abstract" in fields:
            # Include abstracts if requested
            source_texts[source] = " ".join([
                preprocess_text(result.title + " " + (result.abstract or ""))
                for result in results
            ])
        else:
            # Use only titles otherwise
            source_texts[source] = " ".join([preprocess_text(result.title) for result in results])
    
    # Calculate similarity metrics between all pairs of sources
    similarity_metrics = {}
    pairwise_overlap = {}
    detailed_overlap = {}
    
    source_names = list(sources_results.keys())
    for i, source1 in enumerate(source_names):
        for source2 in source_names[i+1:]:
            key = f"{source1}_vs_{source2}"
            pairwise_metrics = {}
            
            # Match papers using the same logic as the frontend
            overlapping_papers = []
            
            # First, match by DOI
            matched_ids_source1 = set()
            for doi, paper1 in source_by_doi[source1].items():
                if doi in source_by_doi[source2]:
                    paper2 = source_by_doi[source2][doi]
                    overlapping_papers.append({
                        "identifier": doi,
                        "match_type": "doi",
                        "title": paper1.title,
                        "source1_rank": paper1.rank,
                        "source2_rank": paper2.rank
                    })
                    matched_ids_source1.add(paper1.title.lower().strip())
            
            # Then match remaining papers by title
            for title, paper1 in source_by_title[source1].items():
                # Skip if already matched by DOI
                if title in matched_ids_source1:
                    continue
                    
                if title in source_by_title[source2]:
                    paper2 = source_by_title[source2][title]
                    overlapping_papers.append({
                        "identifier": title,
                        "match_type": "title",
                        "title": paper1.title,
                        "source1_rank": paper1.rank,
                        "source2_rank": paper2.rank
                    })
            
            overlap_count = len(overlapping_papers)
            
            # Log the detailed overlap for debugging
            logger.info(f"Found {overlap_count} overlapping papers between {source1} and {source2}")
            if overlap_count > 0:
                logger.info(f"Overlapping papers: {', '.join([p['title'][:50] + '...' for p in overlapping_papers])}")
            
            # Store the detailed overlap information
            detailed_overlap[key] = overlapping_papers
            
            # Calculate Jaccard similarity
            if "jaccard" in metrics:
                pairwise_metrics["jaccard"] = calculate_jaccard_similarity(
                    source_sets[source1], source_sets[source2]
                )
            
            # Calculate rank-based overlap
            if "rankBased" in metrics:
                pairwise_metrics["rankBased"] = calculate_rank_based_overlap(
                    source_lists[source1], source_lists[source2]
                )
            
            # Calculate cosine similarity
            if "cosine" in metrics:
                # Create term frequency vectors
                vec1 = {}
                for word in source_texts[source1].split():
                    vec1[word] = vec1.get(word, 0) + 1
                    
                vec2 = {}
                for word in source_texts[source2].split():
                    vec2[word] = vec2.get(word, 0) + 1
                
                pairwise_metrics["cosine"] = calculate_cosine_similarity(vec1, vec2)
            
            # Store metrics for this pair
            similarity_metrics[key] = pairwise_metrics
            
            # Calculate overlap data for visualization
            pairwise_overlap[key] = {
                "overlap": overlap_count,
                "source1_only": len(sources_results[source1]) - overlap_count,
                "source2_only": len(sources_results[source2]) - overlap_count,
                "source1_name": source1,
                "source2_name": source2,
                "overlapping_papers": overlapping_papers  # Add detailed paper info
            }
    
    # Prepare result
    return {
        "sourceResults": sources_results,
        "metrics": similarity_metrics,
        "overlap": pairwise_overlap,
        "detailedOverlap": detailed_overlap,
        "allResults": [result for results in sources_results.values() for result in results]
    }

# Add this cache configuration and helper functions
CACHE_ENABLED = True  # Set to False to disable caching
CACHE_DIR = Path("./cache")  # Cache directory
CACHE_EXPIRY = 60 * 60 * 24  # 24 hours in seconds

# Create cache directory if it doesn't exist
if CACHE_ENABLED:
    CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(source: str, query: str, fields: List[str]) -> str:
    """Generate a unique cache key for a query."""
    # Create a string representation of the query parameters
    query_str = f"{source}:{query}:{','.join(sorted(fields))}"
    # Generate a hash
    return hashlib.md5(query_str.encode()).hexdigest()

def save_to_cache(key: str, data: List[SearchResult], expiry: int = CACHE_EXPIRY) -> bool:
    """Save search results to cache."""
    if not CACHE_ENABLED:
        return False
        
    try:
        cache_path = CACHE_DIR / f"{key}.json"
        
        # Convert to JSON-serializable format
        serializable_data = []
        for result in data:
            serializable_data.append(result.dict())
        
        cache_data = {
            "timestamp": time.time(),
            "expiry": expiry,
            "data": serializable_data
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving to cache: {str(e)}")
        return False

def load_from_cache(key: str) -> Optional[List[SearchResult]]:
    """Load search results from cache if available and not expired."""
    if not CACHE_ENABLED:
        return None
        
    try:
        cache_path = CACHE_DIR / f"{key}.json"
        
        if not cache_path.exists():
            return None
            
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            
        # Check if cache is expired
        timestamp = cache_data.get("timestamp", 0)
        expiry = cache_data.get("expiry", CACHE_EXPIRY)
        
        if time.time() - timestamp > expiry:
            logger.info(f"Cache expired for key {key}")
            return None
            
        # Convert back to SearchResult objects
        results = []
        for item in cache_data.get("data", []):
            results.append(SearchResult(**item))
            
        logger.info(f"Retrieved {len(results)} results from cache for key {key}")
        return results
        
    except Exception as e:
        logger.error(f"Error loading from cache: {str(e)}")
        return None

# Main API endpoint
@app.post("/api/compare")
async def compare_search_results(request: SearchRequest):
    """
    Compare search results from selected sources based on specified metrics.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not request.sources:
        raise HTTPException(status_code=400, detail="At least one source must be selected")
    
    if not request.metrics:
        raise HTTPException(status_code=400, detail="At least one metric must be selected")
    
    if not request.fields:
        raise HTTPException(status_code=400, detail="At least one field must be selected")
    
    # Get results from selected sources
    sources_results = {}
    tasks = []
    
    # Map tasks to sources for easier tracking
    source_task_map = {}
    
    if "ads" in request.sources:
        task = asyncio.create_task(get_ads_results(request.query, request.fields))
        tasks.append(task)
        source_task_map[task] = "ads"
    
    if "scholar" in request.sources:
        task = asyncio.create_task(get_scholar_results(request.query, request.fields))
        tasks.append(task)
        source_task_map[task] = "scholar"
    
    if "semanticScholar" in request.sources:
        task = asyncio.create_task(get_semantic_scholar_results(request.query, request.fields))
        tasks.append(task)
        source_task_map[task] = "semanticScholar"
    
    if "webOfScience" in request.sources:
        logger.info(f'WoS {request.query}')
        task = asyncio.create_task(get_web_of_science_results(request.query, request.fields))
        tasks.append(task)
        source_task_map[task] = "webOfScience"
    
    try:
        # Execute all tasks with longer timeout
        completed_tasks, pending_tasks = await asyncio.wait(
            tasks,
            timeout=TIMEOUT_SECONDS,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending_tasks:
            task.cancel()
            source = source_task_map[task]
            logger.warning(f"Task for source {source} timed out after {TIMEOUT_SECONDS}s")
            sources_results[source] = []
        
        # Process completed tasks
        for task in completed_tasks:
            source = source_task_map[task]
            try:
                result = await task
                sources_results[source] = result
            except Exception as e:
                logger.error(f"Error processing results from {source}: {str(e)}")
                sources_results[source] = []
        
    except asyncio.CancelledError:
        # Handle explicit cancellation (e.g., client disconnected)
        logger.warning("Request was cancelled (client may have disconnected)")
        # Cancel any ongoing tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        raise HTTPException(status_code=499, detail="Request cancelled")
        
    except Exception as e:
        logger.error(f"Unexpected error during search task execution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing search: {str(e)}")
    
    # Ensure all requested sources have an entry, even if empty
    for source in request.sources:
        if source not in sources_results:
            sources_results[source] = []
    
    # Compare results if we have any
    if any(len(results) > 0 for results in sources_results.values()):
        comparison_result = compare_results(sources_results, request.metrics, request.fields)
        return comparison_result
    else:
        # Return a structured empty response if no results found
        return {
            "sourceResults": sources_results,
            "metrics": {},
            "overlap": {},
            "allResults": []
        }

# Endpoint to modify SciX ranking
@app.post("/api/modify-scix")
async def modify_scix_ranking(data: dict):
    """
    Apply custom modifications to SciX ranking and return re-ranked results.
    """
    query = data.get("query")
    original_results = data.get("results", [])
    modifications = data.get("modifications", {})
    
    if not query or not original_results:
        raise HTTPException(status_code=400, detail="Query and original results are required")
    
    # Apply modifications (example implementation)
    modified_results = original_results.copy()
    
    # Example: Boost papers with specific keywords in title
    if "titleKeywords" in modifications and modifications["titleKeywords"]:
        keywords = modifications["titleKeywords"].lower().split(",")
        boost_factor = float(modifications.get("keywordBoostFactor", 1.5))
        
        for result in modified_results:
            title = result.get("title", "").lower()
            for keyword in keywords:
                if keyword.strip() in title:
                    result["boosted"] = True
                    result["originalRank"] = result["rank"]
                    result["rank"] = result["rank"] / boost_factor
    
    # Example: Boost recent papers
    if "boostRecent" in modifications and modifications["boostRecent"]:
        recency_factor = float(modifications.get("recencyFactor", 1.2))
        current_year = 2023  # This should be updated or obtained dynamically
        
        for result in modified_results:
            if "year" in result and result["year"]:
                age = current_year - result["year"]
                if age <= 5:  # Papers from the last 5 years
                    result["boosted"] = True
                    result["originalRank"] = result.get("originalRank", result["rank"])
                    result["rank"] = result["rank"] / (recency_factor * (1 - age/10))
    
    # Sort by new rank
    modified_results.sort(key=lambda x: x["rank"])
    
    # Update ranks after sorting
    for i, result in enumerate(modified_results):
        result["newRank"] = i + 1
    
    return {
        "originalResults": original_results,
        "modifiedResults": modified_results,
        "modifications": modifications
    }

# Root endpoint for API status check
@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"status": "ok", "message": "Academic Search Results Comparator API is running"}

# If running directly, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
