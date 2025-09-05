#!/usr/bin/env python3
"""
Test runner for GemCheck with comprehensive logging and error reporting.
"""

import sys
import os
import logging
import subprocess
import time
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def setup_logging():
    """Set up comprehensive logging for test runs."""
    log_dir = PROJECT_ROOT / "tests" / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting GemCheck test suite")
    logger.info(f"📝 Log file: {log_file}")
    logger.info(f"🏠 Project root: {PROJECT_ROOT}")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info("-" * 80)
    
    return logger

def run_pytest_with_logging(test_files=None):
    """Run pytest with comprehensive logging."""
    logger = logging.getLogger(__name__)
    
    # Base pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "-s",  # Don't capture output
        "--tb=long",  # Long traceback format
        "--capture=no",  # Don't capture stdout/stderr
        "--log-cli-level=DEBUG",  # Show debug logs in CLI
        "--log-cli-format=%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ]
    
    # Add specific test files or default to all
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append("tests/")
    
    logger.info(f"🧪 Running command: {' '.join(cmd)}")
    logger.info("-" * 80)
    
    try:
        # Run pytest
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        logger.info("-" * 80)
        if result.returncode == 0:
            logger.info("✅ All tests passed!")
        else:
            logger.error(f"❌ Tests failed with exit code: {result.returncode}")
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        logger.error("⏰ Tests timed out after 5 minutes")
        return 1
    except Exception as e:
        logger.error(f"💥 Error running tests: {e}")
        return 1

def check_imports():
    """Check if all imports work correctly."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Checking imports...")
    
    import_tests = [
        ("app.schema", "Basic schema"),
        ("app.metrics.centering", "Centering analysis"),
        ("app.metrics.edges_corners", "Edges and corners"),
        ("app.metrics.surface", "Surface analysis"),
        ("app.metrics.glare", "Glare detection"),
        ("app.scoring", "Scoring system"),
        ("app.visualize", "Visualization"),
    ]
    
    failed_imports = []
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            logger.info(f"✅ {description}: {module_name}")
        except Exception as e:
            logger.error(f"❌ {description}: {module_name} - {e}")
            failed_imports.append((module_name, str(e)))
    
    if failed_imports:
        logger.error("💥 Import failures detected:")
        for module, error in failed_imports:
            logger.error(f"  - {module}: {error}")
        return False
    
    logger.info("✅ All imports successful")
    return True

def main():
    """Main test runner."""
    logger = setup_logging()
    
    # Check imports first
    if not check_imports():
        logger.error("❌ Import check failed. Cannot run tests.")
        return 1
    
    # Parse command line arguments
    test_files = sys.argv[1:] if len(sys.argv) > 1 else None
    
    if test_files:
        logger.info(f"🎯 Running specific tests: {test_files}")
    else:
        logger.info("🧪 Running all tests")
    
    # Run tests
    exit_code = run_pytest_with_logging(test_files)
    
    # Final summary
    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("🎉 TEST SUITE COMPLETED SUCCESSFULLY!")
    else:
        logger.error("💥 TEST SUITE FAILED!")
        logger.error("📋 Check the logs above for detailed error information")
        logger.error("🔧 Common fixes:")
        logger.error("  - Check import paths")
        logger.error("  - Verify all required functions exist")
        logger.error("  - Check function signatures match test expectations")
    
    logger.info("=" * 80)
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)