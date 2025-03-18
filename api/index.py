import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend'))

# Import the FastAPI app from main.py
from backend.main import app

# Export the app for Vercel
handler = app
