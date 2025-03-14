import asyncio
import os
from dotenv import load_dotenv
from backend.main import get_ads_results

async def test_live_ads_search():
    """Test live ADS search with actual API."""
    # Load environment variables
    load_dotenv()
    
    # Print API key status (first/last 4 chars only for security)
    api_key = os.getenv("ADS_API_KEY")
    if api_key:
        print(f"ADS API key found: {api_key[:4]}...{api_key[-4:]}")
    else:
        print("No ADS API key found!")
        return
        
    # Test parameters
    query = "exoplanet detection"
    fields = ["authors", "abstract", "doi", "year"]
    
    print(f"\nTesting ADS search with query: '{query}'")
    print(f"Requesting fields: {fields}")
    
    try:
        results = await get_ads_results(query, fields)
        
        print(f"\nFound {len(results)} results")
        
        # Print first 3 results
        for i, result in enumerate(results[:3], 1):
            print(f"\nResult {i}:")
            print(f"Title: {result.title}")
            print(f"Authors: {result.authors}")
            print(f"Year: {result.year}")
            print(f"DOI: {result.doi}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error during search: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_live_ads_search()) 