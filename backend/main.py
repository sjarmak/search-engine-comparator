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
import rbo
from datetime import datetime
import traceback
from utils.ads_utils import get_citation_count
import requests

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
    allow_origins=[
        "https://search-engine-comparator.onrender.com", 
        "http://localhost:3000",
        "https://search-engine-comparator-1.onrender.com",
        "https://search-engine-comparator-web.onrender.com",
        "https://search-engine-comparator-api.onrender.com",
        "https://search.sjarmak.ai"  # Add your custom domain
    ],
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
ADS_API_KEY = os.getenv("ADS_API_KEY")
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
            # Updated to match new FreeProxy API
            success = pg.FreeProxies()
            if success:
                logger.info("Successfully set up free proxies for Google Scholar")
                scholarly.use_proxy(pg)
                scholarly.set_timeout(20)  # Slightly higher timeout
                return True
        except Exception as e:
            logger.warning(f"Free proxies setup failed: {str(e)}")
        
        # Try built-in Scholarly methods
        try:
            success = pg.Tor_External(tor_sock_port=9050, tor_control_port=9051)
            if success:
                logger.info("Successfully set up Tor proxy for Google Scholar")
                scholarly.use_proxy(pg)
                return True
        except Exception as e:
            logger.warning(f"Tor proxy setup failed: {str(e)}")
            
        try:
            success = pg.Luminati(username='username', password='password', proxy_port=22225)
            if success:
                logger.info("Successfully set up Luminati proxy for Google Scholar")
                scholarly.use_proxy(pg)
                return True
        except Exception as e:
            logger.warning(f"Luminati proxy setup failed: {str(e)}")
        
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

async def get_bibcode_from_doi(doi):
    """Convert a DOI to an ADS bibcode"""
    if not doi:
        return None
        
    try:
        # Get API token
        ads_token = os.environ.get("ADS_API_KEY")
        if not ads_token:
            logger.warning("ADS_API_KEY not found, cannot fetch bibcode")
            return None
            
        # Query ADS API for the bibcode
        url = "https://api.adsabs.harvard.edu/v1/search/query"
        params = {
            "q": f"doi:{doi}",
            "fl": "bibcode",
            "rows": 1
        }
        headers = {
            "Authorization": f"Bearer {ads_token}"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            
            if docs and "bibcode" in docs[0]:
                return docs[0]["bibcode"]
                
    except Exception as e:
        logger.exception(f"Error fetching bibcode for DOI {doi}: {str(e)}")
        
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

def calculate_rank_based_overlap(list1, list2, p=0.98):
    """
    Calculate the rank-biased overlap between two lists.
    
    Args:
        list1: First list of results (SearchResult objects or dicts with 'title' keys)
        list2: Second list of results (SearchResult objects or dicts with 'title' keys)
        p: The persistence parameter (default: 0.98, matching rbo library default)
    
    Returns:
        float: A score between 0 and 1, where 1 indicates identical rankings
    """
    try:
        # Extract titles, handling both dictionary and object formats
        titles1 = []
        for item in list1:
            if isinstance(item, dict):
                titles1.append(item.get('title', '').lower().strip())
            else:
                # Get the title attribute value first, then process it
                title_value = getattr(item, 'title', '')
                # Check if it's a callable (method)
                if callable(title_value):
                    title_value = title_value()
                titles1.append(str(title_value).lower().strip())
                
        titles2 = []
        for item in list2:
            if isinstance(item, dict):
                titles2.append(item.get('title', '').lower().strip())
            else:
                # Get the title attribute value first, then process it
                title_value = getattr(item, 'title', '')
                # Check if it's a callable (method)
                if callable(title_value):
                    title_value = title_value()
                titles2.append(str(title_value).lower().strip())
        
        # If either list is empty, return 0
        if not titles1 or not titles2:
            logger.warning("Empty list passed to calculate_rank_based_overlap")
            return 0.0
        
        # Log the first few titles from each list for debugging
        logger.info(f"List 1 titles (first 3): {titles1[:3]}")
        logger.info(f"List 2 titles (first 3): {titles2[:3]}")
        
        # Calculate RBO using the imported library
        similarity = rbo.RankingSimilarity(titles1, titles2)
        rbo_value = similarity.rbo()  # Using default p=0.98
        
        # Log the calculated RBO value
        logger.info(f"RBO value: {rbo_value} (p={p})")
        
        return rbo_value
        
    except Exception as e:
        logger.error(f"Error calculating rank-biased overlap: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 0.0

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
            
        if response.status_code != 200:
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

    #Allowed tags are AI, AU, CS, DO, DT, FPY, IS, OG, PG, PMID, PY, SO, SUR, TI, TS, UT, VL.
    # use AU for author, FPY for final publication year, OG for organization-enhanced, PY for year published
    # use PMID for PubMed ID, SO for publication name,
    # TS for topic terms in Title, Abstract, Author Keywords, Keywords Plus fields.
    # UT for accession number (a UUID)
    # Likely best query is AU=(q) OR TS=(q)
    wos_query = f'AU=({query}) OR TS=({query})'
    
    headers = {
        "X-ApiKey": WOS_API_KEY,
        "Accept": "application/json"
    }
    
    # The correct WoS Starter API endpoint
    base_url = f"{WOS_API_URL}documents"
    
    params = {
        "db": "WOS",
        "q": f'{wos_query}',  # FIXED: Don't add ALL=() twice
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
            if "rankBiased" in metrics:
                pairwise_metrics["rankBiased"] = calculate_rank_based_overlap(
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

# Test endpoint 
@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint that returns valid JSON"""
    logger.info("Test endpoint called")
    return {"status": "ok", "message": "API is working"}

# Root endpoint for API status check
@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"status": "ok", "message": "Academic Search Results Comparator API is running"}

@app.post("/api/boost-experiment")
async def boost_experiment(data: dict):
    """
    Apply configurable boost factors to search results using ADS metadata.
    
    This implementation:
    1. Preserves original scores and rankings
    2. Fetches metadata for each result from ADS API
    3. Calculates boosts based on this metadata
    4. Adds boosts to original scores
    5. Returns re-ranked results with detailed boost information
    """
    try:
        logger.info("Starting boost experiment with ADS metadata enrichment")
        
        # Get input data
        query = data.get("query", "")
        original_results = data.get("results", [])
        boost_config = data.get("boostConfig", {})
        
        # Default boost configuration (matching test script)
        default_boost_config = {
            "enableCiteBoost": True,
            "citeBoostWeight": 1.0,
            "enableRecencyBoost": True,
            "recencyBoostWeight": 1.0,
            "recencyFunction": "exponential",
            "recencyMultiplier": 0.01,
            "enableDoctypeBoost": True,
            "doctypeBoostWeight": 1.0,
            "enableRefereedBoost": True,
            "refereedBoostWeight": 1.0,
        }
        
        # Merge provided config with defaults
        for key, value in default_boost_config.items():
            if key not in boost_config:
                boost_config[key] = value
        
        if not original_results:
            return {
                "status": "error",
                "message": "No results to process"
            }
            
        logger.info(f"Processing {len(original_results)} results with ADS metadata")
        
        # Step 1: Extract bibcodes and preserve original info
        bibcodes = []
        original_info = {}  # Store original rank, score, etc.
        
        for i, result in enumerate(original_results):
            # Store original ranking
            original_rank = i + 1
            original_score = result.get("score", 1.0)
            
            # Get identifier
            bibcode = result.get("bibcode", "")
            if not bibcode or bibcode.startswith("10."):  # Looks like a DOI
                doi = result.get("doi", "") or bibcode
                if doi:
                    # Try to get bibcode from DOI
                    real_bibcode = await get_bibcode_from_doi(doi)
                    if real_bibcode:
                        bibcode = real_bibcode
                        bibcodes.append(bibcode)
                        logger.info(f"Converted DOI to bibcode: {doi} -> {bibcode}")
            else:
                # For results without bibcodes, create placeholder
                identifier = result.get("doi", "") or result.get("id", "") or f"result-{i}"
                original_info[identifier] = {
                    "result": result, 
                    "rank": original_rank,
                    "score": original_score
                }
                logger.info(f"No bibcode, using identifier: {identifier}")
        
        # Step 2: Fetch metadata from ADS API
        ads_metadata = {}
        
        if bibcodes or True:
            # Get API token
            ads_token = os.environ.get("ADS_API_KEY")
            if not ads_token:
                logger.warning("ADS_API_KEY not found, using limited metadata")
            else:
                try:
                    # Use ADS batch API to get metadata
                    logger.info(f"Fetching metadata for {len(bibcodes)} records from ADS API")
                    
                    # Process in batches to avoid API limits
                    batch_size = 50
                    for i in range(0, len(bibcodes), batch_size):
                        batch = bibcodes[i:i+batch_size]
                        
                        # Format query for batch lookup
                        bibcode_query = "bibcode:(" + " OR ".join(batch) + ")"
                        
                        # Make API call
                        url = "https://api.adsabs.harvard.edu/v1/search/query"
                        params = {
                            "q": bibcode_query,
                            "fl": "bibcode,title,author,year,pubdate,citation_count,doctype,property",
                            "rows": batch_size
                        }
                        headers = {
                            "Authorization": f"Bearer {ads_token}"
                        }
                        
                        response = requests.get(url, params=params, headers=headers, timeout=30)
                        
                        if response.status_code == 200:
                            data = response.json()
                            docs = data.get("response", {}).get("docs", [])
                            
                            # Store metadata
                            for doc in docs:
                                if doc.get("bibcode"):
                                    ads_metadata[doc["bibcode"]] = doc
                                    # Log citation data for debugging
                                    citation_count = doc.get("citation_count", 0)
                                    # In your boost_experiment function, add this fallback
                                    # After trying to extract citation_count from metadata

                                    # Fallback: Check if we have citation info in the original result
                                    if citation_count == 0:
                                        for original_result in original_results:
                                            original_citations = original_result.get("citations", 0)
                                            if original_citations and original_citations > 0:
                                                citation_count = original_citations
                                                logger.info(f"Using original citation count for {bibcode}: {citation_count}")
                                    logger.info(f"Retrieved metadata for {doc['bibcode']} with citation_count: {citation_count}")
                        else:
                            logger.error(f"ADS API error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    logger.exception(f"Error fetching ADS metadata: {str(e)}")
        
        # Step 3: Process each result, apply boosts based on metadata
        processed_results = []
        
        for identifier, info in original_info.items():
            original_result = info["result"]
            bibcode = original_result.get("bibcode", "")
            
            # Create working copy with original data
            result_copy = {
                "originalRank": info["rank"],
                "originalScore": info["score"]
            }
            
            # Merge original result and ADS metadata
            if bibcode and bibcode in ads_metadata:
                # We have ADS metadata, use it
                metadata = ads_metadata[bibcode]
                
                # Extract key fields from metadata
                title = metadata.get("title", [""])[0] if isinstance(metadata.get("title"), list) else metadata.get("title", "")
                authors = metadata.get("author", [])
                
                # Extract year and pubdate
                year = metadata.get("year", 0)
                try:
                    year = int(year) if year and str(year).isdigit() else 0
                except (ValueError, TypeError):
                    year = 0
                
                pubdate = metadata.get("pubdate", "")
                
                # Extract citation count - with robust extraction
                citation_count = 0
                if metadata.get("citation_count") is not None:
                    try:
                        citation_count = int(metadata.get("citation_count"))
                        logger.info(f"Extracted citation_count for {bibcode}: {citation_count}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid citation_count for {bibcode}: {metadata.get('citation_count')}")
                
                # Get doctype
                doctype = metadata.get("doctype", "").lower()
                
                # Get property array - normalize as list
                property_array = metadata.get("property", [])
                if isinstance(property_array, str):
                    property_array = [property_array]
                
                # Normalize property array to all uppercase for consistent matching
                property_array = [p.upper() if isinstance(p, str) else str(p).upper() for p in property_array]
                
                # Check for refereed status
                is_refereed = "REFEREED" in property_array
                
                # Create clean result with metadata - with all fields needed by frontend
                result_copy.update({
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "pubdate": pubdate,
                    "citations": citation_count,
                    "citation_count": citation_count,
                    "source": original_result.get("source", ""),
                    "collection": original_result.get("collection", "general"),
                    "doctype": doctype,
                    "bibcode": bibcode,
                    "identifier": bibcode,
                    "property": property_array,
                    "refereed": is_refereed,
                    "score": info["score"],  # Original score
                    "boostFactors": {},
                    # Preserve any original fields that might be used by frontend
                    "display": original_result.get("display", {})
                })
                
                logger.info(f"Using ADS metadata for {bibcode}: {citation_count} citations, {doctype} doctype")
            else:
                # No ADS metadata, use original data
                result_copy.update(original_result)
                
                # Ensure we have basic fields with proper defaults
                if "boostFactors" not in result_copy:
                    result_copy["boostFactors"] = {}
                    
                if "identifier" not in result_copy:
                    result_copy["identifier"] = identifier
                
                if "bibcode" not in result_copy and "identifier" in result_copy:
                    result_copy["bibcode"] = result_copy["identifier"]
                    
                if "property" not in result_copy:
                    result_copy["property"] = []
                else:
                    # Normalize property array if it exists
                    if isinstance(result_copy["property"], str):
                        result_copy["property"] = [result_copy["property"]]
                    result_copy["property"] = [p.upper() if isinstance(p, str) else str(p).upper() 
                                               for p in result_copy["property"]]
                
                # Ensure citation fields exist
                if "citations" not in result_copy:
                    result_copy["citations"] = 0
                if "citation_count" not in result_copy:
                    result_copy["citation_count"] = result_copy.get("citations", 0)
                
                if "doctype" not in result_copy:
                    result_copy["doctype"] = "article"  # Default
                    
                if "year" not in result_copy:
                    result_copy["year"] = 0
                
                if "refereed" not in result_copy:
                    result_copy["refereed"] = False
                
                logger.info(f"Using original data for {identifier} (no ADS metadata) with citations: {result_copy.get('citations', 0)}")
            
            # Store for boost processing
            processed_results.append(result_copy)
        
        # Step 4: Apply boosts based on metadata
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        for result in processed_results:
            result_id = result.get("identifier", "unknown")
            
            # 1. Citation boost
            if boost_config.get("enableCiteBoost", True):
                cite_weight = float(boost_config.get("citeBoostWeight", 1.0))
                citations = result.get("citations", 0)
                
                # Simple citation boost with log scale
                if citations > 0:
                    cite_boost = math.log10(1 + citations) * cite_weight / 2
                    result["boostFactors"]["citeBoost"] = cite_boost
                    # Add direct access property for easier frontend access
                    result["citeBoost"] = cite_boost
                    logger.info(f"Citation boost for {result_id}: {cite_boost:.4f} ({citations} citations)")
                else:
                    result["boostFactors"]["citeBoost"] = 0.0
                    result["citeBoost"] = 0.0
                    logger.info(f"Citation boost for {result_id}: 0.0 (no citations)")
            
            # 2. Recency boost
            if boost_config.get("enableRecencyBoost", True):
                recency_weight = float(boost_config.get("recencyBoostWeight", 1.0))
                recency_function = boost_config.get("recencyFunction", "exponential")
                pubyear = result.get("year", 0)
                
                if pubyear > 0:
                    # Calculate age in months (approximate)
                    age_years = current_year - pubyear
                    age_months = age_years * 12 + current_month
                    
                    # Apply exponential decay function
                    multiplier = float(boost_config.get("recencyMultiplier", 0.01))
                    recency_boost = math.exp(-multiplier * age_months)
                    
                    # Apply weight and normalize
                    recency_boost = min(recency_boost * recency_weight, 2.0)
                    result["boostFactors"]["recencyBoost"] = recency_boost
                    # Add direct access property
                    result["recencyBoost"] = recency_boost
                    logger.info(f"Recency boost for {result_id}: {recency_boost:.4f} (age: {age_years} years)")
                else:
                    result["boostFactors"]["recencyBoost"] = 0.0
                    result["recencyBoost"] = 0.0
                    logger.info(f"Recency boost for {result_id}: 0.0 (no year available)")
            
            # 3. Doctype boost
            if boost_config.get("enableDoctypeBoost", True):
                doctype_weight = float(boost_config.get("doctypeBoostWeight", 1.0))
                
                # Get document type
                doctype = result.get("doctype", "").lower()
                
                # Map to canonical doctype rankings - use detailed ranking system
                doctype_ranks = {
                    "article": 1,       # Highest academic value
                    "book": 2,
                    "inbook": 2,
                    "inproceedings": 3,
                    "phdthesis": 4,
                    "eprint": 5,
                    "proceedings": 6,
                    "mastersthesis": 7,
                    "techreport": 8,
                    "catalog": 9,
                    "software": 10,
                    "abstract": 11,
                    "talk": 12,
                    "bookreview": 13,
                    "proposal": 14,
                    "circular": 15,
                    "newsletter": 16,
                    "erratum": 17,
                    "obituary": 18, 
                    "pressrelease": 19,
                    "misc": 20,
                    "": 21              # Unknown type gets lowest rank
                }
                
                rank = doctype_ranks.get(doctype, 21)
                # Transform rank to a 0-1 boost factor
                unique_ranks = len(doctype_ranks)
                doctype_boost = 1.0 - ((rank - 1) / (unique_ranks - 1))
                doctype_boost *= doctype_weight
                
                result["boostFactors"]["doctypeBoost"] = doctype_boost
                # Add direct access property
                result["doctypeBoost"] = doctype_boost
                logger.info(f"Doctype boost for {result_id}: {doctype_boost:.4f} (doctype: {doctype}, rank: {rank})")
            
            # 4. Refereed boost
            if boost_config.get("enableRefereedBoost", True):
                refereed_weight = float(boost_config.get("refereedBoostWeight", 1.0))
                
                # Get refereed status - either from metadata or inferred
                is_refereed = result.get("refereed", False)
                
                # Determine refereed boost (binary: either full boost or no boost)
                refereed_boost = refereed_weight if is_refereed else 0.0
                
                result["boostFactors"]["refereedBoost"] = refereed_boost
                # Add direct access property
                result["refereedBoost"] = refereed_boost
                logger.info(f"Refereed boost for {result_id}: {refereed_boost:.4f}")
            
            # 5. Open Access boost
            if boost_config.get("enableOpenAccessBoost", False):
                oa_weight = float(boost_config.get("openAccessBoostWeight", 0.2))
                property_array = result.get("property", [])
                
                is_oa = False
                oa_properties = ["OPENACCESS", "PUB_OPENACCESS", "EPRINT_OPENACCESS", 
                                 "ADS_OPENACCESS", "AUTHOR_OPENACCESS"]
                for oa_prop in oa_properties:
                    if oa_prop in property_array:
                        is_oa = True
                        break
                
                oa_boost = oa_weight if is_oa else 0.0
                
                if is_oa:
                    result["boostFactors"]["openAccessBoost"] = oa_boost
                    # Add direct access property
                    result["openAccessBoost"] = oa_boost
                    logger.info(f"Open Access boost for {result_id}: {oa_boost:.4f}")
            
            # 6. Combine boosts
            combination_method = boost_config.get("combinationMethod", "sum")
            boosts = list(result["boostFactors"].values())
            
            if combination_method == "sum":
                total_boost = sum(boosts)
            elif combination_method == "product":
                if boosts:
                    total_boost = 1.0
                    for boost in boosts:
                        total_boost *= (1.0 + boost)
                    total_boost -= 1.0
                else:
                    total_boost = 0.0
            elif combination_method == "max":
                total_boost = max(boosts) if boosts else 0.0
            else:
                total_boost = sum(boosts)
            
            # Apply overall weight
            total_boost *= float(boost_config.get("overallBoostWeight", 1.0))
            
            # Cap maximum boost
            max_boost = float(boost_config.get("maxBoost", 5.0))
            total_boost = min(total_boost, max_boost)
            
            # Add boost to original score
            original_score = result.get("originalScore", 1.0)
            result["score"] = original_score + total_boost
            result["totalBoost"] = total_boost
            
            logger.info(f"Total boost for {result_id}: {total_boost:.4f} (original score: {original_score:.4f}, new score: {result['score']:.4f})")
        
        # Step 5: Re-rank results based on new scores
        ranked_results = sorted(processed_results, key=lambda x: x["score"], reverse=True)
        
        # Add new rank and rank change
        for i, result in enumerate(ranked_results):
            result["rank"] = i + 1
            result["rankChange"] = result["originalRank"] - result["rank"]
        
        # Step 6: Return results with validation
        return {
            "status": "success",
            "query": query,
            "results": ranked_results,
            "boostConfig": boost_config,
            "totalResults": len(ranked_results),
            "metadata": {
                "avgCitations": sum(r.get("citations", 0) for r in ranked_results) / len(ranked_results) if ranked_results else 0,
                "maxBoost": max(r.get("totalBoost", 0) for r in ranked_results) if ranked_results else 0,
                "minBoost": min(r.get("totalBoost", 0) for r in ranked_results) if ranked_results else 0,
                "avgBoost": sum(r.get("totalBoost", 0) for r in ranked_results) / len(ranked_results) if ranked_results else 0
            }
        }
    
    except Exception as e:
        logger.exception(f"Error in boost experiment: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing boost experiment: {str(e)}"
        }

@app.get("/api/debug/paper/{doi}")
async def debug_paper(doi: str):
    """Debug endpoint to get raw paper data from ADS by DOI"""
    try:
        # First convert DOI to bibcode
        bibcode = await get_bibcode_from_doi(doi)
        
        if not bibcode:
            return {"status": "error", "message": f"Could not find bibcode for DOI: {doi}"}
            
        # Now fetch all metadata for this bibcode
        ads_token = os.environ.get("ADS_API_KEY")
        if not ads_token:
            return {"status": "error", "message": "ADS_API_KEY not configured"}
            
        url = "https://api.adsabs.harvard.edu/v1/search/query"
        params = {
            "q": f"bibcode:{bibcode}",
            "fl": "*",  # Get all fields
            "rows": 1
        }
        headers = {
            "Authorization": f"Bearer {ads_token}"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return {
                "status": "error", 
                "message": f"ADS API error: {response.status_code}",
                "details": response.text
            }
            
        data = response.json()
        
        return {
            "status": "success",
            "bibcode": bibcode,
            "doi": doi,
            "raw_data": data.get("response", {}).get("docs", [{}])[0]
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/api/debug/citation/{bibcode}")
async def debug_citation(bibcode: str):
    """
    Debug endpoint to test citation data retrieval for a specific bibcode.
    This helps diagnose issues with the ADS API connection and citation extraction.
    """
    try:
        logger.info(f"Debug citation request for bibcode: {bibcode}")
        
        # Get API token
        ads_token = os.environ.get("ADS_API_KEY")
        if not ads_token:
            return {
                "status": "error", 
                "message": "ADS_API_KEY not found in environment variables"
            }
        
        # Make API call to ADS
        url = "https://api.adsabs.harvard.edu/v1/search/query"
        params = {
            "q": f"bibcode:{bibcode}",
            "fl": "bibcode,title,author,year,citation_count,doctype,property",
            "rows": 1
        }
        headers = {
            "Authorization": f"Bearer {ads_token}"
        }
        
        # Log the request
        logger.info(f"Making ADS API request for bibcode: {bibcode}")
        logger.info(f"Request URL: {url}")
        logger.info(f"Request params: {params}")
        
        # Execute request with timeout
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        # Log the response
        logger.info(f"ADS API response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            
            if docs:
                doc = docs[0]
                
                # Extract citation count
                citation_count = None
                if "citation_count" in doc:
                    try:
                        citation_count = int(doc["citation_count"])
                    except (ValueError, TypeError):
                        citation_count = f"Error converting: {doc['citation_count']}"
                
                # Extract and normalize properties
                property_array = doc.get("property", [])
                if isinstance(property_array, str):
                    property_array = [property_array]
                
                # Check for refereed status
                is_refereed = "REFEREED" in property_array
                
                # Return formatted result
                return {
                    "status": "success",
                    "bibcode": bibcode,
                    "found": True,
                    "record": {
                        "title": doc.get("title", [""])[0] if isinstance(doc.get("title"), list) else doc.get("title", ""),
                        "authors": doc.get("author", []),
                        "year": doc.get("year", ""),
                        "citation_count": citation_count,
                        "doctype": doc.get("doctype", ""),
                        "refereed": is_refereed,
                        "property": property_array
                    },
                    "raw_response": doc  # Include the raw response for debugging
                }
            else:
                return {
                    "status": "not_found",
                    "bibcode": bibcode,
                    "found": False,
                    "message": "No document found with the provided bibcode",
                    "raw_response": data.get("response", {})
                }
        else:
            # Handle API error
            error_detail = None
            try:
                error_detail = response.json()
            except:
                error_detail = response.text[:500]  # Limit text size
                
            return {
                "status": "error",
                "message": f"ADS API returned an error: {response.status_code}",
                "error_detail": error_detail
            }
            
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "ADS API request timed out"
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": "Connection error when accessing ADS API"
        }
    except Exception as e:
        logger.exception(f"Error in debug citation endpoint: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


@app.post("/api/debug/boost-data")
async def debug_boost_data(data: dict):
    """
    Debug endpoint to analyze boost data structure and field presence.
    This helps identify where citation count and boost fields might be missing or malformed.
    """
    try:
        # Get the results to analyze
        results = data.get("results", [])
        
        if not results:
            return {"status": "error", "message": "No results provided"}
        
        # Analyze field presence
        field_analysis = {
            "total_records": len(results),
            "citation_fields": {},
            "boost_fields": {}
        }
        
        # Key fields to check
        citation_fields = ["citation_count", "citations", "citationCount"]
        boost_fields = ["boostFactors", "citeBoost", "recencyBoost", "doctypeBoost", 
                       "refereedBoost", "totalBoost"]
        
        # Check field presence
        for field in citation_fields:
            field_analysis["citation_fields"][field] = {
                "present_count": sum(1 for r in results if field in r),
                "percentage": round(sum(1 for r in results if field in r) / len(results) * 100, 1),
                "sample_values": [r.get(field) for r in results[:3] if field in r]
            }
        
        for field in boost_fields:
            field_analysis["boost_fields"][field] = {
                "present_count": sum(1 for r in results if field in r),
                "percentage": round(sum(1 for r in results if field in r) / len(results) * 100, 1)
            }
            
            # For boostFactors, check inner structure
            if field == "boostFactors":
                inner_factors = {}
                for r in results:
                    if field in r and isinstance(r[field], dict):
                        for inner_field in r[field].keys():
                            if inner_field not in inner_factors:
                                inner_factors[inner_field] = 0
                            inner_factors[inner_field] += 1
                
                field_analysis["boost_fields"][field]["inner_structure"] = {
                    k: round(v / len(results) * 100, 1) 
                    for k, v in inner_factors.items()
                }
        
        # Sample data for inspection
        samples = []
        for i, result in enumerate(results[:3]):
            sample = {
                "index": i,
                "citation_data": {},
                "boost_data": {}
            }
            
            for field in citation_fields:
                if field in result:
                    sample["citation_data"][field] = result[field]
            
            for field in boost_fields:
                if field in result:
                    if field == "boostFactors" and isinstance(result[field], dict):
                        sample["boost_data"][field] = dict(result[field])
                    else:
                        sample["boost_data"][field] = result[field]
            
            samples.append(sample)
        
        # Diagnosis and recommendations
        diagnosis = []
        
        # Check citation fields
        if all(field_analysis["citation_fields"][field]["present_count"] == 0 
               for field in citation_fields):
            diagnosis.append("No citation fields found in any result")
        elif any(field_analysis["citation_fields"][field]["present_count"] > 0 
                and field_analysis["citation_fields"][field]["present_count"] < len(results)
                for field in citation_fields):
            diagnosis.append("Citation fields are inconsistently present across results")
        
        # Check boost fields
        if field_analysis["boost_fields"]["boostFactors"]["present_count"] == 0:
            diagnosis.append("No boostFactors object found in results")
        elif field_analysis["boost_fields"]["boostFactors"]["present_count"] < len(results):
            diagnosis.append("BoostFactors object inconsistently present across results")
        
        if all(field_analysis["boost_fields"][field]["present_count"] == 0 
               for field in boost_fields if field != "boostFactors"):
            diagnosis.append("No direct boost properties found (citeBoost, recencyBoost, etc.)")
        
        # Return the analysis
        return {
            "status": "success",
            "field_analysis": field_analysis,
            "samples": samples,
            "diagnosis": diagnosis,
            "recommendations": [
                "Ensure all results have citation_count and citations fields",
                "Make sure the boostFactors object is present in all results",
                "Add direct boost properties (citeBoost, recencyBoost, etc.) for easier frontend access",
                "Check that the ADS API token is correctly set in environment variables",
                "Verify that the ADS API is returning citation data for your bibcodes"
            ]
        }
        
    except Exception as e:
        logger.exception(f"Error in debug boost data endpoint: {str(e)}")
        return {
            "status": "error",
            "message": f"Error analyzing boost data: {str(e)}"
        }
      
@app.post("/api/debug-boost-fields")
async def debug_boost_fields(data: dict):
    """
    Debug endpoint to specifically diagnose issues with citation counts and boost factors.
    Helps identify field name mismatches and data parsing issues.
    """
    try:
        # Get results from request
        results = data.get("results", [])
        
        if not results:
            return {"status": "error", "message": "No results provided"}
        
        # Initial field analysis
        field_analysis = {
            "total_records": len(results),
            "citation_fields": {},
            "boost_fields": {},
            "record_samples": []
        }
        
        # Citation field candidates
        citation_fields = ["citation_count", "citations", "cited_by_count", "citationCount"]
        
        # Boost field candidates
        boost_fields = ["boost", "totalBoost", "boostFactors", "citeBoost", "recencyBoost", 
                       "doctypeBoost", "refereedBoost", "openAccessBoost"]
        
        # Analyze each result
        for idx, result in enumerate(results[:10]):  # Analyze first 10 results
            # Check for citation fields
            for field in citation_fields:
                if field in result:
                    if field not in field_analysis["citation_fields"]:
                        field_analysis["citation_fields"][field] = {
                            "count": 0,
                            "values": []
                        }
                    field_analysis["citation_fields"][field]["count"] += 1
                    field_analysis["citation_fields"][field]["values"].append(result[field])
            
            # Check for boost fields
            for field in boost_fields:
                if field in result:
                    if field not in field_analysis["boost_fields"]:
                        field_analysis["boost_fields"][field] = {
                            "count": 0,
                            "values": []
                        }
                    field_analysis["boost_fields"][field]["count"] += 1
                    if field == "boostFactors" and isinstance(result[field], dict):
                        field_analysis["boost_fields"][field]["values"].append(
                            {k: v for k, v in result[field].items()})
                    else:
                        field_analysis["boost_fields"][field]["values"].append(result[field])
            
            # Add sample record with relevant fields
            sample = {
                "index": idx,
                "bibcode": result.get("bibcode", "N/A"),
                "identified_fields": {}
            }
            
            # Add citation fields if present
            for field in citation_fields:
                if field in result:
                    sample["identified_fields"][field] = result[field]
            
            # Add boost fields if present
            for field in boost_fields:
                if field in result:
                    if field == "boostFactors" and isinstance(result[field], dict):
                        sample["identified_fields"][field] = {k: v for k, v in result[field].items()}
                    else:
                        sample["identified_fields"][field] = result[field]
            
            field_analysis["record_samples"].append(sample)
        
        # Get ADS metadata for comparison if bibcodes are present
        ads_data = {}
        bibcodes = [r.get("bibcode") for r in results[:5] if r.get("bibcode")]
        
        if bibcodes:
            ads_token = os.environ.get("ADS_API_KEY")
            if ads_token:
                # Query ADS API directly for comparison
                try:
                    bibcode_query = "bibcode:(" + " OR ".join(bibcodes) + ")"
                    url = "https://api.adsabs.harvard.edu/v1/search/query"
                    params = {
                        "q": bibcode_query,
                        "fl": "bibcode,citation_count,citations,cited_by_count",
                        "rows": len(bibcodes)
                    }
                    headers = {"Authorization": f"Bearer {ads_token}"}
                    
                    response = requests.get(url, params=params, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        docs = data.get("response", {}).get("docs", [])
                        
                        for doc in docs:
                            if doc.get("bibcode"):
                                ads_data[doc["bibcode"]] = {
                                    "citation_count": doc.get("citation_count"),
                                    "citations": doc.get("citations"),
                                    "cited_by_count": doc.get("cited_by_count")
                                }
                except Exception as e:
                    ads_data["error"] = str(e)
        
        # Prepare response with diagnostic information
        return {
            "status": "success",
            "field_analysis": field_analysis,
            "ads_comparison": ads_data,
            "recommendations": {
                "citation_fields": "Ensure citation data is stored consistently in one of these fields: " + 
                                  ", ".join(citation_fields),
                "boost_fields": "Ensure boost data is stored in boostFactors object and/or individual boost fields"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error diagnosing boost fields: {str(e)}"
        }
    
@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API and environment variables"""
    ads_token_configured = bool(os.environ.get("ADS_API_KEY"))
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ads_token_configured": ads_token_configured
        }
    }

# If running directly, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
