"""
Glare and reflection detection for card grading.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
from app.schema import GlareFindings, BoundingBox

logger = logging.getLogger(__name__)


def detect_specular_highlights(image: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Detect specular highlights (glare) using HSV analysis.
    
    Args:
        image: Input image in RGB format
        threshold: Threshold for highlight detection (0-1)
        
    Returns:
        Binary mask of detected highlights
    """
    try:
        # Convert to HSV for better highlight detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # High value (brightness) indicates potential glare
        high_value_mask = v > (threshold * 255)
        
        # Low saturation also indicates glare (desaturated bright regions)
        low_saturation_mask = s < 50  # Very low saturation
        
        # Combine conditions for glare detection
        glare_mask = high_value_mask & low_saturation_mask
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.morphologyEx(glare_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
        
        return glare_mask
        
    except Exception as e:
        logger.error(f"Specular highlight detection failed: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8)


def find_glare_regions(glare_mask: np.ndarray, min_area: int = 100) -> List[BoundingBox]:
    """
    Find individual glare regions and their bounding boxes.
    
    Args:
        glare_mask: Binary mask of glare regions
        min_area: Minimum area for a glare region to be considered
        
    Returns:
        List of bounding boxes for glare regions
    """
    try:
        # Find connected components
        contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        glare_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                glare_regions.append(BoundingBox(x=x, y=y, width=w, height=h))
        
        logger.info(f"Found {len(glare_regions)} significant glare regions")
        return glare_regions
        
    except Exception as e:
        logger.error(f"Glare region finding failed: {e}")
        return []


def calculate_glare_penalty(glare_percentage: float, max_penalty: float = 10.0) -> float:
    """
    Calculate penalty points to deduct from overall score due to glare.
    
    Args:
        glare_percentage: Percentage of image affected by glare
        max_penalty: Maximum penalty points
        
    Returns:
        Penalty points to deduct (0 to max_penalty)
    """
    try:
        # Linear penalty based on glare percentage
        penalty = min(max_penalty, (glare_percentage / 10.0) * max_penalty)
        
        logger.info(f"Glare penalty: {penalty:.2f} points for {glare_percentage:.2f}% glare")
        return penalty
        
    except Exception as e:
        logger.error(f"Glare penalty calculation failed: {e}")
        return max_penalty


def analyze_glare(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> GlareFindings:
    """
    Complete glare and reflection analysis.
    
    Args:
        image: Input image in RGB format
        config: Configuration parameters
        
    Returns:
        GlareFindings object with detailed results
    """
    if config is None:
        config = {
            'glare_threshold': 0.8,
            'min_region_area': 100,
            'max_penalty': 10.0
        }
    
    try:
        glare_threshold = config.get('glare_threshold', 0.8)
        min_region_area = config.get('min_region_area', 100)
        max_penalty = config.get('max_penalty', 10.0)
        
        # Detect glare
        glare_mask = detect_specular_highlights(image, glare_threshold)
        
        # Calculate glare statistics
        total_pixels = image.shape[0] * image.shape[1]
        glare_pixels = np.sum(glare_mask > 0)
        glare_percentage = (glare_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        # Find glare regions
        glare_regions = find_glare_regions(glare_mask, min_region_area)
        
        # Determine if glare is significant
        glare_detected = glare_percentage > 2.0  # More than 2% of image
        
        # Calculate penalty
        penalty = calculate_glare_penalty(glare_percentage, max_penalty) if glare_detected else 0.0
        
        # Create findings object
        findings = GlareFindings(
            glare_detected=glare_detected,
            glare_area_px=int(glare_pixels),
            glare_percentage=glare_percentage,
            penalty_applied=penalty,
            affected_regions=glare_regions,
            glare_threshold=glare_threshold
        )
        
        logger.info(f"Glare analysis completed: {glare_percentage:.2f}% affected, penalty: {penalty:.2f}")
        return findings
        
    except Exception as e:
        logger.error(f"Glare analysis failed: {e}")
        return GlareFindings(
            glare_detected=True,
            glare_area_px=999999,
            glare_percentage=100.0,
            penalty_applied=max_penalty if 'max_penalty' in locals() else 10.0,
            affected_regions=[],
            glare_threshold=0.8
        )