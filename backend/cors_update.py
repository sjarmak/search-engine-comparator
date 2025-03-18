#!/usr/bin/env python
import sys

# Path to your main.py file
file_path = 'main.py'

try:
    # Read the main.py file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the CORS middleware section
    cors_section = "app.add_middleware(\n    CORSMiddleware,\n    allow_origins=[\"*\"],  # In production, specify your domain"
    
    # Replace with a more specific configuration
    new_cors = "app.add_middleware(\n    CORSMiddleware,\n    allow_origins=[\"https://search-engine-comparator.onrender.com\", \"http://localhost:3000\"],  # Frontend URLs"
    
    # Perform the replacement
    new_content = content.replace(cors_section, new_cors)
    
    # Write the changes back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print("✅ CORS settings updated successfully!")
    
except Exception as e:
    print(f"❌ Error updating CORS settings: {e}")
    sys.exit(1)
