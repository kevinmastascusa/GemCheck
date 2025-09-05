"""
Run script for GemCheck PSA Card Pre-Grader application.
This script properly handles the imports and runs the Streamlit app.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the main app
if __name__ == "__main__":
    # Change to app directory for proper imports
    os.chdir(project_root / "app")
    
    # Import streamlit and run the app
    import subprocess
    import sys
    
    # Run streamlit with the main.py file
    cmd = [sys.executable, "-m", "streamlit", "run", "main.py"]
    subprocess.run(cmd)