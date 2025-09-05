#!/usr/bin/env python3
"""
Debug script to identify specific test issues in GemCheck.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test all imports individually."""
    print("Testing imports...")
    
    imports_to_test = [
        # Basic imports
        ("numpy", "import numpy as np"),
        ("cv2", "import cv2"),
        ("pytest", "import pytest"),
        
        # App modules
        ("app.schema", "from app.schema import CenteringFindings"),
        ("app.metrics.centering", "from app.metrics.centering import analyze_centering"),
        ("app.metrics.edges_corners", "from app.metrics.edges_corners import analyze_edges"),
        ("app.metrics.surface", "from app.metrics.surface import analyze_surface"),
        ("app.metrics.glare", "from app.metrics.glare import analyze_glare"),
        ("app.scoring", "from app.scoring import calculate_sub_scores"),
    ]
    
    failed_imports = []
    
    for name, import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            print(f"OK {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed_imports.append((name, str(e)))
    
    return failed_imports

def test_basic_functionality():
    """Test basic functionality of each module."""
    print("\nTesting basic functionality...")
    
    try:
        # Test centering
        print("Testing centering module...")
        from app.metrics.centering import calculate_centering_score
        score = calculate_centering_score(0.1)
        print(f"OK Centering score test: {score}")
        
        # Test schema
        print("Testing schema...")
        from app.schema import CenteringFindings
        finding = CenteringFindings(
            left_margin_px=30.0, right_margin_px=30.0,
            top_margin_px=40.0, bottom_margin_px=40.0,
            horizontal_error=0.1, vertical_error=0.1, combined_error=0.14,
            max_error_threshold=0.25, inner_frame_detected=True,
            detection_method="edge_based", centering_score=85.0
        )
        print(f"OK Schema test: {finding.centering_score}")
        
        # Test with actual image
        print("Testing with actual image...")
        import numpy as np
        test_image = np.ones((400, 300, 3), dtype=np.uint8) * 200
        
        from app.metrics.centering import analyze_centering
        result = analyze_centering(test_image)
        print(f"OK Image analysis test: score={result.centering_score}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def run_single_test():
    """Run a single test to identify the exact issue."""
    print("\nRunning single test...")
    
    try:
        # Import test class
        sys.path.insert(0, str(PROJECT_ROOT / "tests"))
        from test_centering import TestCenteringAnalysis
        
        # Create test instance
        test_instance = TestCenteringAnalysis()
        
        # Run specific test
        print("Running test_calculate_centering_score_perfect...")
        test_instance.test_calculate_centering_score_perfect()
        print("OK Single test passed!")
        
        return True
        
    except Exception as e:
        print(f"FAIL Single test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("GemCheck Test Debug Tool")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    if failed_imports:
        print(f"\nFAIL {len(failed_imports)} import failures found:")
        for name, error in failed_imports:
            print(f"  - {name}: {error}")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\nFAIL Basic functionality test failed!")
        return False
    
    # Run single test
    if not run_single_test():
        print("\nFAIL Single test failed!")
        return False
    
    print("\nSUCCESS All debug tests passed!")
    print("\nNow try running the full test suite:")
    print("python run_tests.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)