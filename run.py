#!/usr/bin/env python3
"""
GemCheck - PSA Card Pre-Grader Application
Main entry point that sets up the Python path correctly.
"""

import sys
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Add the project root to Python path so we can import from app/
sys.path.insert(0, str(PROJECT_ROOT))

# Set working directory to project root
os.chdir(PROJECT_ROOT)

def main():
    """Main entry point for the application."""
    try:
        # Import streamlit and run the app
        import subprocess
        
        # Run streamlit from the project root
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(PROJECT_ROOT / "app" / "main.py"),
            "--server.address", "localhost",
            "--server.port", "8501"
        ]
        
        print("🎯 Starting GemCheck PSA Card Pre-Grader...")
        print("🌐 Open your browser to: http://localhost:8501")
        print("📋 Press Ctrl+C to stop the application")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 GemCheck application stopped.")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Make sure you've installed the requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()