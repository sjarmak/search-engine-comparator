import os
import httpx
import asyncio
from dotenv import load_dotenv

async def test_ads_direct():
    """Test ADS API directly."""
    load_dotenv()
    api_key = os.getenv("ADS_API_KEY")
    
    # Print first/last 4 chars of API key to verify it's loaded
    print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
    
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Test different queries
    queries = [
        "machine learning",
        "exoplanet",
        "\"machine learning\""  # Try with quotes
    ]
    
    async with httpx.AsyncClient() as client:
        for query in queries:
            print(f"\nTesting query: {query}")
            
            params = {
                "q": query,
                "rows": 5,
                "fl": "title,author,abstract,doi,year",
                "sort": "date desc"
            }
            
            try:
                response = await client.get(url, headers=headers, params=params)
                print(f"Status code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    num_results = len(data.get('response', {}).get('docs', []))
                    print(f"Number of results: {num_results}")
                    
                    # Print first result if any
                    if num_results > 0:
                        first_result = data['response']['docs'][0]
                        print("\nFirst result:")
                        print(f"Title: {first_result.get('title', ['No title'])[0]}")
                else:
                    print(f"Error response: {response.text}")
                    
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_ads_direct()) 