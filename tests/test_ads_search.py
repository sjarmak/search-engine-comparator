import pytest
from typing import List, Dict, Any, Optional
import httpx
import asyncio
from unittest.mock import Mock, patch
from typing import TYPE_CHECKING
import os
import logging

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from pytest_mock import MockerFixture

# Import the function under test
from backend.main import get_ads_results, SearchResult

# Test data
SAMPLE_ADS_RESPONSE = {
    "response": {
        "docs": [
            {
                "title": ["Test Paper 1"],
                "author": ["Author 1", "Author 2"],
                "abstract": "Test abstract 1",
                "doi": ["10.1234/test1"],
                "year": 2023,
                "bibcode": "2023test1"
            },
            {
                "title": ["Test Paper 2"],
                "author": ["Author 3", "Author 4"],
                "abstract": "Test abstract 2",
                "doi": ["10.1234/test2"],
                "year": 2022,
                "bibcode": "2022test2"
            }
        ]
    }
}

# Create a fixture for each test case with its own response data
@pytest.fixture
def success_fixture(monkeypatch):
    """Fixture for successful API response"""
    monkeypatch.setenv("ADS_API_KEY", "dummy_api_key")
    
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self._json_data = SAMPLE_ADS_RESPONSE
            self.text = "OK"
            
        def json(self):
            return self._json_data
    
    class MockClient:
        async def get(self, *args, **kwargs):
            return MockResponse()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", lambda: MockClient())
    
@pytest.fixture
def empty_response_fixture(monkeypatch):
    """Fixture for empty response"""
    monkeypatch.setenv("ADS_API_KEY", "dummy_api_key")
    
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self._json_data = {"response": {"docs": []}}
            self.text = "OK"
            
        def json(self):
            return self._json_data
    
    class MockClient:
        async def get(self, *args, **kwargs):
            return MockResponse()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", lambda: MockClient())
    
@pytest.fixture
def error_response_fixture(monkeypatch):
    """Fixture for error response"""
    monkeypatch.setenv("ADS_API_KEY", "dummy_api_key")
    
    class MockResponse:
        def __init__(self):
            self.status_code = 401
            self._json_data = {"error": "Unauthorized"}
            self.text = "Unauthorized"
            
        def json(self):
            return self._json_data
    
    class MockClient:
        async def get(self, *args, **kwargs):
            return MockResponse()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", lambda: MockClient())
    
@pytest.fixture
def missing_fields_fixture(monkeypatch):
    """Fixture for response with missing fields"""
    monkeypatch.setenv("ADS_API_KEY", "dummy_api_key")
    
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self._json_data = {
                "response": {
                    "docs": [
                        {
                            "title": ["Test Paper"],
                            "bibcode": "2023test"
                        }
                    ]
                }
            }
            self.text = "OK"
            
        def json(self):
            return self._json_data
    
    class MockClient:
        async def get(self, *args, **kwargs):
            return MockResponse()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", lambda: MockClient())
    
@pytest.fixture
def malformed_response_fixture(monkeypatch):
    """Fixture for malformed response"""
    monkeypatch.setenv("ADS_API_KEY", "dummy_api_key")
    
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self._json_data = {"invalid": "response"}
            self.text = "OK"
            
        def json(self):
            return self._json_data
    
    class MockClient:
        async def get(self, *args, **kwargs):
            return MockResponse()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", lambda: MockClient())
    
@pytest.fixture
def network_error_fixture(monkeypatch):
    """Fixture for network error"""
    monkeypatch.setenv("ADS_API_KEY", "dummy_api_key")
    
    class ErrorClient:
        async def __aenter__(self):
            raise httpx.NetworkError("Connection failed")
            
        async def __aexit__(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", lambda: ErrorClient())

@pytest.mark.asyncio
async def test_get_ads_results_success(success_fixture):
    """Test successful ADS search with all fields."""
    # Execute search
    results = await get_ads_results("test query", ["authors", "abstract", "doi", "year"])
    
    # Verify results
    assert len(results) == 2
    
    # Check first result
    first_result = results[0]
    assert isinstance(first_result, SearchResult)
    assert first_result.title == "Test Paper 1"
    assert first_result.authors == ["Author 1", "Author 2"]
    assert first_result.abstract == "Test abstract 1"
    assert first_result.doi == "10.1234/test1"
    assert first_result.year == 2023
    assert first_result.source == "SciX"
    assert first_result.rank == 1
    assert "ui.adsabs.harvard.edu" in first_result.url

@pytest.mark.asyncio
async def test_get_ads_results_empty_response(empty_response_fixture, caplog):
    """Test ADS search with empty response."""
    # Force a higher log level for the test
    logging.getLogger().setLevel(logging.INFO)
    
    # Clear log before test
    caplog.clear()
    
    # Execute search
    results = await get_ads_results("test query", ["authors"])
    
    # Verify results
    assert len(results) == 0
    
    # Print the actual log for debugging
    print(f"Captured log: {caplog.text}")
    
    # Check if message is there with more flexible matching
    assert any("No results found" in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_get_ads_results_error_handling(error_response_fixture, caplog):
    """Test ADS search error handling."""
    # Clear log before test
    caplog.clear()
    
    # Execute search
    results = await get_ads_results("test query", ["authors"])
    
    # Verify results
    assert len(results) == 0
    assert "Error fetching from ADS" in caplog.text
    assert "Status: 401" in caplog.text

@pytest.mark.asyncio
async def test_get_ads_results_missing_fields(missing_fields_fixture):
    """Test ADS search with missing optional fields."""
    # Execute search
    results = await get_ads_results("test query", ["authors", "abstract", "doi", "year"])
    
    # Verify results
    assert len(results) == 1
    result = results[0]
    assert result.title == "Test Paper"
    assert result.authors is None
    assert result.abstract is None
    assert result.doi is None
    assert result.year is None
    assert result.source == "SciX"
    assert result.rank == 1

@pytest.mark.asyncio
async def test_get_ads_results_malformed_response(malformed_response_fixture, caplog):
    """Test ADS search with malformed response."""
    # Clear log before test
    caplog.clear()
    
    # Execute search
    results = await get_ads_results("test query", ["authors"])
    
    # Verify results
    assert len(results) == 0
    assert "Error processing ADS response" in caplog.text

@pytest.mark.asyncio
async def test_get_ads_results_network_error(network_error_fixture, caplog):
    """Test ADS search with network error."""
    # Clear log before test
    caplog.clear()
    
    # Execute search
    results = await get_ads_results("test query", ["authors"])
    
    # Verify results
    assert len(results) == 0
    assert "Error fetching from ADS" in caplog.text
    assert "Connection failed" in caplog.text