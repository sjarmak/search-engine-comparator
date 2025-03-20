#!/usr/bin/env python
"""
Debug script for citation processing in boost_experiment function
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("boost-citation-debug")

def mock_boost_experiment(sample_file: str = None):
    """
    Mock the boost_experiment function with sample data or a provided JSON file
    to diagnose citation processing issues.
    """
    if sample_file and os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            data = json.load(f)
    else:
        # Use a minimal sample result
        data = {
            "query": "black holes",
            "boostConfig": {
                "enableCiteBoost": True,
                "citeBoostWeight": 1.0,
                "enableRecencyBoost": True,
                "recencyBoostWeight": 1.0,
                "enableDoctypeBoost": True,
                "doctypeBoostWeight": 1.0,
                "enableRefereedBoost": True, 
                "refereedBoostWeight": 1.0
            },
            "results": [
                {
                    "bibcode": "2018PhDT........18A",
                    "title": "Sample PhD Thesis",
                    "author": ["Author, A."],
                    "doctype": "phdthesis",
                    "property": ["ESOURCE", "NONARTICLE", "REFEREED"],
                    "year": "2018",
                    "pubdate": "2018-00-00",
                    "citation_count": 5  # Test with a non-zero value
                }
            ]
        }
    
    logger.info("Starting citation debugging with sample data")
    logger.info(f"Input data structure: {json.dumps(data, indent=2)[:500]}...")
    
    # Extract key components
    original_results = data.get("results", [])
    boost_config = data.get("boostConfig", {})
    
    # Process first result in detail to debug citation handling
    if original_results:
        result = original_results[0]
        logger.info("\n" + "="*80)
        logger.info("PROCESSING FIRST RESULT IN DETAIL")
        logger.info("="*80)
        
        # Log the raw structure
        logger.info(f"Raw result keys: {sorted(result.keys())}")
        
        # Check citation_count specifically
        if "citation_count" in result:
            logger.info(f"citation_count exists: {result['citation_count']} (type: {type(result['citation_count'])})")
            
            # Test our extraction logic
            if str(result.get("citation_count", "")).isdigit():
                citations = int(result.get("citation_count"))
                logger.info(f"✅ Citation extraction works! Value: {citations}")
            else:
                logger.info(f"❌ Citation extraction would fail: '{result.get('citation_count')}' is not recognized as a digit string")
                logger.info(f"Debug info - type: {type(result.get('citation_count'))}, isdigit test: {str(result.get('citation_count', '')).isdigit()}")
        else:
            logger.info("❌ citation_count field is missing from the input")
            logger.info(f"Available fields that might contain citation info: {[k for k in result.keys() if 'cit' in k.lower()]}")
        
        # Log debug info about doctype and properties
        logger.info(f"doctype: {result.get('doctype', 'missing')}")
        logger.info(f"property: {result.get('property', 'missing')}")
        
        # Extract fields as done in main.py
        clean_result = {
            "title": str(result.get("title", "")),
            "authors": result.get("author", []),
            "year": int(result.get("year", 0)) if result.get("year") and str(result.get("year", "")).isdigit() else 0,
            "citations": int(result.get("citation_count", 0)) if result.get("citation_count") and str(result.get("citation_count", "")).isdigit() else 0,
            "doctype": result.get("doctype", ""),
            "property": result.get("property", []),
            "boostFactors": {}
        }
        
        logger.info("\nCleaned result:")
        logger.info(f"citations: {clean_result['citations']}")
        logger.info(f"doctype: {clean_result['doctype']}")
        
        # Mock the citation boost calculation
        if boost_config.get("enableCiteBoost", True):
            import math
            citations = clean_result["citations"]
            if citations > 0:
                cite_boost = math.log10(1 + citations) * float(boost_config.get("citeBoostWeight", 1.0)) / 2
                clean_result["boostFactors"]["citeBoost"] = cite_boost
                logger.info(f"Citation boost calculated: {cite_boost:.4f}")
            else:
                clean_result["boostFactors"]["citeBoost"] = 0.0
                logger.info("No citation boost (citations = 0)")
    else:
        logger.error("No results provided for analysis")

def main():
    """Main function entry point."""
    import sys
    
    # Check if a sample file was provided
    sample_file = None
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
        logger.info(f"Using sample file: {sample_file}")
    
    mock_boost_experiment(sample_file)
    
    logger.info("\n\nDEBUGGING SUGGESTIONS:")
    logger.info("1. Check if 'citation_count' is present in the ADS API response")
    logger.info("2. Make sure the field has the exact name 'citation_count' (case-sensitive)")
    logger.info("3. Verify the citation_count value is a string or number that can be converted to an integer")
    logger.info("4. Add a console.log(results) in the frontend to see if citations are being passed from backend")

if __name__ == "__main__":
    main() 