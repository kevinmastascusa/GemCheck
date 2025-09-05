"""
Pytest configuration and fixtures for GemCheck tests.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np

# Configure logging for tests
def pytest_configure():
    """Configure pytest with detailed logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('tests/test_results.log', mode='w')
        ]
    )
    
    # Set specific loggers
    logging.getLogger('app').setLevel(logging.DEBUG)
    logging.getLogger('app.metrics').setLevel(logging.DEBUG)
    logging.getLogger('app.scoring').setLevel(logging.DEBUG)
    
    print("Starting GemCheck tests with detailed logging...")

def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("STARTING GEMCHECK TEST SESSION")
    logger.info("="*80)
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Project root: {project_root}")

def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info(f"TEST SESSION FINISHED - Exit status: {exitstatus}")
    logger.info("="*80)


@pytest.fixture
def sample_card_image():
    """Create a basic sample card image for testing."""
    # Create white border
    image = np.ones((400, 300, 3), dtype=np.uint8) * 255
    
    # Create inner card area (gray)
    margin = 30
    image[margin:-margin, margin:-margin] = [200, 200, 200]
    
    return image


@pytest.fixture
def test_config():
    """Provide test configuration values."""
    return {
        'centering': {
            'max_error_threshold': 0.25
        },
        'edges': {
            'border_width': 6,
            'whitening_threshold': 0.15
        },
        'corners': {
            'corner_size': 50,
            'sharpness_threshold': 0.3
        },
        'surface': {
            'scratch_threshold': 0.02,
            'min_defect_area': 5
        },
        'glare': {
            'highlight_threshold': 0.8,
            'min_region_area': 100,
            'max_penalty': 10.0
        }
    }


@pytest.fixture
def mock_analysis_results():
    """Provide mock analysis results for testing."""
    from app.schema import (
        CenteringFindings, EdgeFindings, CornerFindings, 
        SurfaceFindings, GlareFindings
    )
    
    centering = CenteringFindings(
        left_margin_px=30.0, right_margin_px=32.0,
        top_margin_px=40.0, bottom_margin_px=38.0,
        horizontal_error=0.03, vertical_error=0.02, combined_error=0.036,
        max_error_threshold=0.25, inner_frame_detected=True,
        detection_method="edge_based", centering_score=90.0
    )
    
    edges = EdgeFindings(
        total_perimeter_px=1000, whitened_perimeter_px=30,
        whitening_percentage=3.0, nick_count=1, largest_nick_area_px=5,
        clean_edge_percentage=95.0, whitening_threshold=0.15,
        edge_score=88.0
    )
    
    corners = CornerFindings(
        corner_scores={"top_left": 92.0, "top_right": 89.0, "bottom_right": 91.0, "bottom_left": 90.0},
        corner_sharpness={}, corner_whitening={}, corner_damage_area={},
        minimum_corner_score=89.0, sharpness_threshold=0.3,
        corner_score=90.5
    )
    
    surface = SurfaceFindings(
        total_area_px=100000, defect_area_px=50, defect_percentage=0.05,
        scratch_count=0, print_line_count=0, stain_count=0,
        defect_regions=[], ml_assist_used=False, ml_confidence=None,
        surface_quality_score=0.92
    )
    
    glare = GlareFindings(
        glare_detected=False, glare_area_px=0, glare_percentage=0.0,
        penalty_applied=0.0, affected_regions=[], glare_threshold=0.8
    )
    
    return {
        'centering': centering,
        'edges': edges,
        'corners': corners,
        'surface': surface,
        'glare': glare
    }