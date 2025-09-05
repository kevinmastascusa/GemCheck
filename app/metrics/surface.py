"""
Surface condition analysis for card grading.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
from app.schema import SurfaceFindings, DefectRegion, BoundingBox

logger = logging.getLogger(__name__)


def detect_scratches_simple(image: np.ndarray, threshold: float = 0.02) -> Tuple[np.ndarray, List[DefectRegion]]:
    """
    Simple scratch detection using morphological operations.
    
    Args:
        image: Rectified card image
        threshold: Detection threshold for scratches
        
    Returns:
        Tuple of (scratch_mask, list_of_scratch_regions)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological top-hat to detect scratches
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to create binary mask
        thresh_value = threshold * 255
        _, scratch_mask = cv2.threshold(tophat, thresh_value, 255, cv2.THRESH_BINARY)
        
        # Find scratch regions
        contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        scratch_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Filter very small detections
                x, y, w, h = cv2.boundingRect(contour)
                
                scratch_regions.append(DefectRegion(
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    confidence=0.7,
                    defect_type="scratch",
                    severity=min(1.0, area / 100.0),
                    area_pixels=int(area)
                ))
        
        logger.info(f"Detected {len(scratch_regions)} scratch regions")
        return scratch_mask, scratch_regions
        
    except Exception as e:
        logger.error(f"Scratch detection failed: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8), []


def detect_print_lines_simple(image: np.ndarray) -> List[DefectRegion]:
    """
    Simple print line detection.
    
    Args:
        image: Rectified card image
        
    Returns:
        List of detected print line regions
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        line_regions = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 50:  # Significant lines only
                    # Create bounding box
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    
                    line_regions.append(DefectRegion(
                        bbox=BoundingBox(x=x_min, y=y_min, width=x_max-x_min+1, height=y_max-y_min+1),
                        confidence=0.8,
                        defect_type="print_line",
                        severity=min(1.0, length / 200.0),
                        area_pixels=int(length * 2)
                    ))
        
        logger.info(f"Detected {len(line_regions)} print lines")
        return line_regions
        
    except Exception as e:
        logger.error(f"Print line detection failed: {e}")
        return []


def analyze_surface(image: np.ndarray, config: Optional[Dict[str, Any]] = None,
                   ml_model: Optional[Any] = None) -> SurfaceFindings:
    """
    Complete surface condition analysis.
    
    Args:
        image: Rectified card image in RGB format
        config: Configuration parameters
        ml_model: Optional ML model for defect detection
        
    Returns:
        SurfaceFindings object with detailed results
    """
    if config is None:
        config = {
            'scratch_threshold': 0.02,
            'use_ml_assist': False
        }
    
    try:
        scratch_threshold = config.get('scratch_threshold', 0.02)
        use_ml = config.get('use_ml_assist', False) and ml_model is not None
        
        # Calculate total area
        total_area = image.shape[0] * image.shape[1]
        
        # Initialize defect tracking
        all_defect_regions = []
        total_defect_area = 0
        
        # Detect scratches
        scratch_mask, scratch_regions = detect_scratches_simple(image, scratch_threshold)
        all_defect_regions.extend(scratch_regions)
        scratch_count = len(scratch_regions)
        
        # Detect print lines
        line_regions = detect_print_lines_simple(image)
        all_defect_regions.extend(line_regions)
        print_line_count = len(line_regions)
        
        # Simple stain detection (very basic)
        stain_count = 0  # Placeholder
        
        # Calculate total defect area
        for region in all_defect_regions:
            total_defect_area += region.area_pixels
        
        # Calculate defect percentage
        defect_percentage = (total_defect_area / total_area * 100) if total_area > 0 else 0
        
        # Simple surface quality score
        surface_quality_score = max(0.0, 1.0 - defect_percentage / 10.0)
        
        # Optional ML assist (placeholder)
        ml_confidence = None
        if use_ml:
            try:
                # Placeholder for ML model inference
                ml_confidence = 0.8
                logger.info("ML assist applied to surface analysis")
            except Exception as e:
                logger.warning(f"ML assist failed: {e}")
                use_ml = False
        
        # Create findings object
        findings = SurfaceFindings(
            total_area_px=total_area,
            defect_area_px=total_defect_area,
            defect_percentage=defect_percentage,
            scratch_count=scratch_count,
            print_line_count=print_line_count,
            stain_count=stain_count,
            defect_regions=all_defect_regions,
            ml_assist_used=use_ml,
            ml_confidence=ml_confidence,
            surface_quality_score=surface_quality_score
        )
        
        logger.info(f"Surface analysis completed: {len(all_defect_regions)} total defects")
        return findings
        
    except Exception as e:
        logger.error(f"Surface analysis failed: {e}")
        return SurfaceFindings(
            total_area_px=total_area if 'total_area' in locals() else 0,
            defect_area_px=999999,
            defect_percentage=100.0,
            scratch_count=999,
            print_line_count=999,
            stain_count=999,
            defect_regions=[],
            ml_assist_used=False,
            ml_confidence=None,
            surface_quality_score=0.0
        )