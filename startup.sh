#!/usr/bin/env bash

# Academic Search Results Comparator - Startup Script
# This script sets up and starts the application

echo "=== Academic Search Results Comparator ==="
echo "Setting up environment and starting services..."

# Check prerequisites
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required but not installed."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: npm is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install backend dependencies."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    cat > .env << EOL
# Add your API keys below
ADS_API_KEY=your_ads_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
WOS_API_KEY=your_wos_api_key
EOL
    echo "Created .env file. Please edit it to add your API keys."
fi

# Platform-specific fixes
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS-specific fixes
    echo "Applying macOS-specific fixes..."
    export PYTHONHTTPSVERIFY=0
    export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
    export REQUESTS_CA_BUNDLE=$(python -c "import certifi; print(certifi.where())")
    
    echo "SSL certificate path set to: $SSL_CERT_FILE"
    echo "To make these fixes permanent, add the following to your .zshrc or .bash_profile:"
    echo "export PYTHONHTTPSVERIFY=0"
    echo "export SSL_CERT_FILE=$SSL_CERT_FILE"
    echo "export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE"
fi

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd ../frontend
npm install
if [ $? -ne 0 ]; then
    echo "Error: Failed to install frontend dependencies."
    exit 1
fi

# Starting services
echo "Starting services..."

# Start backend in background
cd ../backend
if [[ "$OSTYPE" == "darwin"* ]]; then
    # For macOS
    echo "Starting backend server with macOS SSL fixes..."
    PYTHONHTTPSVERIFY=0 uvicorn main:app --reload &
else
    # For other platforms
    echo "Starting backend server..."
    uvicorn main:app --reload &
fi
backend_pid=$!

# Start frontend
cd ../frontend
echo "Starting frontend server..."
npm start &
frontend_pid=$!

echo "Both services have been started:"
echo "- Backend: http://localhost:8000"
echo "- Frontend: http://localhost:3000"
echo "- API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services."

# Setup trap to kill both processes on exit
trap "kill $backend_pid $frontend_pid; exit" INT TERM EXIT

# Wait for any process to exit
wait
