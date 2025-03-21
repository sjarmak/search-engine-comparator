# Academic Search Results Comparison Tool

A web-based tool for comparing search result similarities across multiple academic search indexers, including SciX, Google Scholar, Web of Science, and Semantic Scholar. The tool also includes advanced boosting algorithms for re-ranking search results based on citation counts, publication date, document type, and other scholarly metadata.

NOTE: Web of Science is not yet configured in the current code. 

## Features

- **Multi-Source Search**: Query multiple academic search engines simultaneously
- **Advanced Similarity Metrics**: Compare results using Jaccard similarity, rank-biased overlap, and more
- **Visualization**: Interactive charts and Venn diagrams to visualize overlap and similarities
- **SciX Ranking Modifier**: Experiment with modifications to SciX's ranking algorithm
- **Citation-Based Boosting**: Promote highly-cited papers using configurable weights
- **Recency Boosting**: Prioritize recent publications with adjustable decay functions
- **Document Type Boosting**: Customize relevance by publication type (articles, reviews, etc.)
- **Refereed Boosting**: Promote peer-reviewed content
- **Open Access Boosting**: Highlight freely accessible research
- **Enhanced Metadata**: Rich metadata for each result including citation counts and publication properties
- **Flexible Metadata Comparison**: Configure which metadata fields to consider in comparisons
- **Detailed Analysis**: View comprehensive tables and charts of comparison results

## Quick Setup

For a streamlined setup experience, you can use the included startup script. Run the following commands, 

```bash
git clone https://github.com/sjarmak/search-engine-comparator.git
cd search-engine-comparator
chmod +x startup.sh
./startup.sh
```
If you do not already have a .env file setup in the backend folder then prior to carrying out search engine comparisons requiring API keys (ADS/SciX, WoS) you will need to:

```
cd backend
nano .env
```
And then add your API keys to the environment file and ctrl+X to save and exit. Then cd back to the root directory and run ./startup.sh

This script will automatically:
- Check prerequisites
- Create and activate a Python virtual environment
- Install backend and frontend dependencies
- Create a template .env file if one doesn't exist
- Apply SSL certificate fixes for macOS users
- Start both the backend and frontend servers

After running the script, the application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

Press Ctrl+C in the terminal running the script to stop both services.

## Project Structure

```
academic-search-comparator/
├── startup.sh               # Automated setup and startup script
├── frontend/                # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ComparisonResults.js
│   │   │   ├── MetricsTable.js
│   │   │   ├── ResultsTable.js
│   │   │   ├── SciXModifier.js
│   │   │   ├── VennDiagram.js
│   │   │   └── BoostExperiment.js  # Ranking modification component
│   │   ├── App.js          # Main application component
│   │   └── index.js        # Entry point
│   └── package.json        # Frontend dependencies
│
└── backend/                # FastAPI backend
    ├── main.py             # Main API implementation
    ├── utils/              # Utility functions
    │   ├── __init__.py
    │   └── ads_utils.py    # ADS API utilities
    ├── fix_macos_certs.py  # macOS SSL certificate fix script
    └── requirements.txt    # Backend dependencies
```

### Manual Setup

If you prefer to set up each component manually, or if you encounter issues with the startup script, follow these instructions:

### Prerequisites

- Python 3.8+ for the backend
- Node.js and npm for the frontend
- API keys for the academic search services:
  - NASA SciX (ADS) API key
  - Semantic Scholar API key (optional)
  - Web of Science API key (if using WoS)

### Backend Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Make sure your requirements.txt includes:
```
fastapi
uvicorn
httpx
python-dotenv
nltk
scholarly
beautifulsoup4
certifi
python-certifi-win32  # for Windows users
requests
```

3. Create a `.env` file in the backend directory with your API keys:

```
ADS_API_KEY=your_ads_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
WOS_API_KEY=your_web_of_science_api_key
```

4. **For macOS users**: Fix SSL certificate issues by running:

```bash
python fix_macos_certs.py
```

Then add the recommended environment variables to your shell profile (`.zshrc` or `.bash_profile`).

5. Run the backend server:

**For macOS users:**
```bash
PYTHONHTTPSVERIFY=0 python -m uvicorn main:app --reload
```

**For Windows/Linux users:**
```bash
uvicorn main:app --reload
```

### Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Start the development server:

```bash
npm start
```

The application should now be running at `http://localhost:3000`.

## Troubleshooting

### Startup Script Issues
If you encounter issues with the startup script:

1. Check that you have the necessary permissions to execute the script
2. Try running the script with bash explicitly: `bash startup.sh`
3. Verify that both frontend and backend directories exist in the same folder as the script
4. Follow the manual setup instructions as an alternative

### SSL Certificate Issues on macOS
If you encounter SSL certificate verification errors on macOS:

1. Run the `fix_macos_certs.py` script to configure your certificates
2. Add the environment variables to your shell profile as instructed by the script
3. Run the backend server with the `PYTHONHTTPSVERIFY=0` flag as shown above

### Google Scholar Connection Issues
Google Scholar sometimes blocks automated access. If you encounter issues:

1. Ensure you're using the latest version of the code with the direct HTML parsing approach
2. Try different search terms to test functionality
3. The application has fallback mechanisms to use alternative Google Scholar access methods

### Citation Count Issues
If citation counts aren't displaying properly:

1. Verify your ADS API key is correctly set in the .env file
2. Check the browser console for API errors
3. Make sure the frontend is properly displaying the metadata section returned by the API

### Proxy Configuration
If you need to use proxies for enhanced reliability:

1. The application uses free proxies by default
2. For production use, consider configuring a dedicated proxy service
3. Update the `setup_scholarly_proxy()` function in `main.py` with your proxy details

## Usage Guide

1. **Enter a Search Query**: Type your academic search query in the main search box.

2. **Select Data Sources**: Choose which academic search engines to query.

3. **Choose Similarity Metrics**: Select which metrics to use for comparing results.

4. **Select Metadata Fields**: Choose which fields to consider in comparisons (Title, Abstract, Authors, DOI, Year).

5. **Run Comparison**: Click "Compare Search Results" to execute the search and view results.

6. **Analyze Results**: Explore the Overview, Detailed Results, and SciX Modifier tabs to analyze the comparison data.

7. **Experiment with SciX Ranking**: In the SciX Modifier tab, adjust parameters to see how they affect the ranking of search results.

8. **Apply Boost Algorithms**: Use the Boost Experiment tab to apply configurable boosts:
   - Citation boost: Promote highly-cited papers
   - Recency boost: Prioritize recent publications
   - Document type boost: Customize relevance by publication type
   - Refereed boost: Promote peer-reviewed content

## Advanced Usage

### Modifying SciX Ranking

The SciX Modifier tab allows you to experiment with different ranking parameters:

1. **Title Keywords**: Boost articles with specific keywords in the title.
2. **Recency Boost**: Give higher ranking to more recent publications.
3. **Weight Adjustments**: Modify the weights for authors and citations in the ranking algorithm.

Apply modifications to see how they affect the ranked results compared to the original SciX ordering.

### Configuring Boost Algorithms

The Boost Experiment tab offers advanced configuration options:

1. **Citation Boost**: Promote papers with high citation counts using logarithmic scaling.
2. **Recency Boost**: Choose from different decay functions (exponential, inverse, linear, sigmoid).
3. **Document Type Boost**: Assign different weights to articles, reviews, proceedings, etc.
4. **Refereed Boost**: Promote peer-reviewed publications.
5. **Open Access Boost**: Highlight freely accessible research.

Adjust these parameters to find the optimal ranking algorithm for your specific needs.

## API Documentation

The backend API is documented using FastAPI's automatic documentation. Once the backend server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

Key endpoints include:

### `/api/compare`
Compare search results from multiple sources with configurable metrics.

### `/api/boost-experiment`
Apply configurable boost factors to search results:
- Citation boost based on citation counts
- Recency boost with configurable decay functions
- Document type boosting with custom weights
- Refereed and open access boosting

### `/api/debug-metadata`
Debug endpoint to analyze the structure of result metadata.

## Adding New Data Sources

To add a new academic search source, you'll need to:

1. Implement a new function in the backend to retrieve results from the source
2. Update the frontend to include the new source in the selection options
3. Modify the comparison functions to handle the new source

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA ADS/SciX for providing the ADS API
- Google Scholar for their academic search service
- Semantic Scholar for their open research API
- Web of Science for their comprehensive academic database
