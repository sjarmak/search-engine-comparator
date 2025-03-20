"""
Utilities for interacting with the ADS API
"""

import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_citation_count(bibcode: str) -> Optional[int]:
    """
    Get citation count for a specific bibcode from the ADS API.
    
    Args:
        bibcode: The ADS bibcode to query
        
    Returns:
        The citation count as an integer, or None if unavailable
    """
    token = os.environ.get("ADS_API_TOKEN")
    if not token:
        logger.warning("ADS_API_TOKEN environment variable not set")
        return None
        
    base_url = "https://api.adsabs.harvard.edu/v1/search/query"
    
    params = {
        "q": f"bibcode:{bibcode}",
        "fl": "citation_count",
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            if docs:
                citation_count = docs[0].get("citation_count", 0)
                return int(citation_count) if citation_count is not None else 0
                
        logger.warning(f"Failed to get citation count for {bibcode}: {response.status_code}")
        return None
        
    except Exception as e:
        logger.exception(f"Error fetching citation count for {bibcode}: {str(e)}")
        return None 