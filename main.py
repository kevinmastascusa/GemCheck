"""
Main entry point for GemCheck PSA Card Pre-Grader application.
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Change working directory to app for relative imports
os.chdir(app_dir)

# Now import and run the main application
from app.main import main

if __name__ == "__main__":
    main()