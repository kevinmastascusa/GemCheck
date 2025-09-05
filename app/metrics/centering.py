"""
Centering analysis for card grading.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from app.schema import CenteringFindings

logger = logging.getLogger(__name__)


def detect_inner_frame_edges(image: np.ndarray, method: str = 'edge_based') -> Optional[np.ndarray]:
    """
    Detect the inner frame/border of the card.
    
    Args:
        image: Rectified card image
        method: Detection method ('edge_based' or 'color_based')
        
    Returns:
        Rectangle coordinates [x, y, width, height] or None if not found
    """
    try:
        if method == 'edge_based':
            return _detect_frame_edges(image)
        elif method == 'color_based':
            return _detect_frame_color(image)
        else:
            logger.warning(f"Unknown detection method: {method}")
            return _detect_frame_edges(image)
    except Exception as e:
        logger.error(f"Inner frame detection failed: {e}")
        return None


def _detect_frame_edges(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect inner frame using edge detection and line detection.
    
    Args:
        image: Rectified card image
        
    Returns:
        Rectangle coordinates [x, y, width, height] or None
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=20)
        
        if lines is None:
            logger.warning("No lines detected for frame detection")
            return None
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            if abs(x2 - x1) > abs(y2 - y1):  # More horizontal
                horizontal_lines.append(line[0])
            else:  # More vertical
                vertical_lines.append(line[0])
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            logger.warning("Insufficient lines detected for frame rectangle")
            return None
        
        # Find top, bottom, left, right boundaries
        h, w = image.shape[:2]
        
        # For horizontal lines, find top and bottom
        horizontal_y = [min(y1, y2) for x1, y1, x2, y2 in horizontal_lines]
        top_y = sorted(horizontal_y)[len(horizontal_y)//4]  # Skip outliers
        bottom_y = sorted(horizontal_y, reverse=True)[len(horizontal_y)//4]
        
        # For vertical lines, find left and right
        vertical_x = [min(x1, x2) for x1, y1, x2, y2 in vertical_lines]
        left_x = sorted(vertical_x)[len(vertical_x)//4]  # Skip outliers
        right_x = sorted(vertical_x, reverse=True)[len(vertical_x)//4]
        
        # Create rectangle
        frame_x = max(0, left_x)
        frame_y = max(0, top_y)
        frame_w = min(w - frame_x, right_x - left_x)
        frame_h = min(h - frame_y, bottom_y - top_y)
        
        if frame_w > 0 and frame_h > 0:
            logger.info(f"Detected inner frame: ({frame_x}, {frame_y}, {frame_w}, {frame_h})")
            return np.array([frame_x, frame_y, frame_w, frame_h])
        
        return None
        
    except Exception as e:
        logger.error(f"Edge-based frame detection failed: {e}")
        return None


def _detect_frame_color(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect inner frame using color-based detection (e.g., yellow border on Pokémon cards).
    
    Args:
        image: Rectified card image
        
    Returns:
        Rectangle coordinates [x, y, width, height] or None
    """
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common card borders
        color_ranges = [
            # Yellow/gold range (common on Pokémon cards)
            (np.array([20, 100, 100]), np.array([30, 255, 255])),
            # Black border range
            (np.array([0, 0, 0]), np.array([180, 255, 50])),
            # White border range  
            (np.array([0, 0, 200]), np.array([180, 30, 255]))
        ]
        
        best_rectangle = None
        best_score = 0
        
        for lower, upper in color_ranges:
            # Create mask for the color range
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            for contour in contours:
                # Approximate contour to rectangle
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    # Calculate bounding rectangle
                    rect = cv2.boundingRect(approx)
                    x, y, w, h = rect
                    
                    # Score based on size and position (should be inner frame)
                    h_img, w_img = image.shape[:2]
                    area_ratio = (w * h) / (w_img * h_img)
                    
                    # Good inner frame should be 20-80% of image area
                    if 0.2 <= area_ratio <= 0.8:
                        score = area_ratio * cv2.contourArea(contour)
                        if score > best_score:
                            best_score = score
                            best_rectangle = rect
        
        if best_rectangle is not None:
            x, y, w, h = best_rectangle
            logger.info(f"Detected color-based frame: ({x}, {y}, {w}, {h})")
            return np.array([x, y, w, h])
        
        return None
        
    except Exception as e:
        logger.error(f"Color-based frame detection failed: {e}")
        return None


def calculate_margins(image: np.ndarray, inner_frame: np.ndarray) -> Dict[str, float]:
    """
    Calculate margins between card edges and inner frame.
    
    Args:
        image: Rectified card image
        inner_frame: Inner frame rectangle [x, y, width, height]
        
    Returns:
        Dictionary with margin measurements
    """
    try:
        h, w = image.shape[:2]
        frame_x, frame_y, frame_w, frame_h = inner_frame
        
        margins = {
            'left_px': frame_x,
            'right_px': w - (frame_x + frame_w),
            'top_px': frame_y,
            'bottom_px': h - (frame_y + frame_h),
            'card_width_px': w,
            'card_height_px': h,
            'frame_width_px': frame_w,
            'frame_height_px': frame_h
        }
        
        # Calculate margins in millimeters (assuming standard card size)
        # Standard trading card: 63mm x 88mm
        mm_per_px_w = 63.0 / w
        mm_per_px_h = 88.0 / h
        
        margins.update({
            'left_mm': margins['left_px'] * mm_per_px_w,
            'right_mm': margins['right_px'] * mm_per_px_w,
            'top_mm': margins['top_px'] * mm_per_px_h,
            'bottom_mm': margins['bottom_px'] * mm_per_px_h
        })
        
        logger.debug(f"Calculated margins: {margins}")
        return margins
        
    except Exception as e:
        logger.error(f"Margin calculation failed: {e}")
        return {}


def calculate_centering_errors(margins: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate centering errors from margin measurements.
    
    Args:
        margins: Margin measurements dictionary
        
    Returns:
        Dictionary with centering error calculations
    """
    try:
        # Horizontal centering error
        left = margins.get('left_px', 0)
        right = margins.get('right_px', 0)
        
        if left + right > 0:
            horizontal_error = abs(left - right) / (left + right)
        else:
            horizontal_error = 0.0
        
        # Vertical centering error
        top = margins.get('top_px', 0)
        bottom = margins.get('bottom_px', 0)
        
        if top + bottom > 0:
            vertical_error = abs(top - bottom) / (top + bottom)
        else:
            vertical_error = 0.0
        
        # Combined error (Euclidean distance normalized by sqrt(2))
        combined_error = np.sqrt(horizontal_error**2 + vertical_error**2) / np.sqrt(2)
        
        errors = {
            'horizontal_error': horizontal_error,
            'vertical_error': vertical_error,
            'combined_error': combined_error
        }
        
        logger.info(f"Calculated centering errors: H={horizontal_error:.4f}, V={vertical_error:.4f}, C={combined_error:.4f}")
        return errors
        
    except Exception as e:
        logger.error(f"Centering error calculation failed: {e}")
        return {'horizontal_error': 0.0, 'vertical_error': 0.0, 'combined_error': 0.0}


def calculate_centering_score(combined_error: float, max_error_threshold: float = 0.25) -> float:
    """
    Convert centering error to a 0-100 score.
    
    Args:
        combined_error: Combined centering error (0-1)
        max_error_threshold: Maximum acceptable error threshold
        
    Returns:
        Centering score (0-100)
    """
    try:
        if combined_error <= 0:
            return 100.0
        
        # Linear scoring: score decreases as error increases
        score = max(0.0, 100.0 * (1.0 - combined_error / max_error_threshold))
        
        logger.info(f"Centering score: {score:.2f} (error: {combined_error:.4f}, threshold: {max_error_threshold})")
        return score
        
    except Exception as e:
        logger.error(f"Centering score calculation failed: {e}")
        return 0.0


def analyze_centering(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> CenteringFindings:
    """
    Complete centering analysis for a rectified card image.
    
    Args:
        image: Rectified card image in RGB format
        config: Configuration parameters
        
    Returns:
        CenteringFindings object with detailed results
    """
    if config is None:
        config = {
            'detection_method': 'edge_based',
            'max_error_threshold': 0.25
        }
    
    try:
        detection_method = config.get('detection_method', 'edge_based')
        max_error_threshold = config.get('max_error_threshold', 0.25)
        
        # Detect inner frame
        inner_frame = detect_inner_frame_edges(image, method=detection_method)
        
        if inner_frame is None:
            # Fallback: assume 10% margin on all sides
            h, w = image.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            inner_frame = np.array([margin_x, margin_y, w - 2*margin_x, h - 2*margin_y])
            frame_detected = False
            logger.warning("Using fallback frame estimation")
        else:
            frame_detected = True
        
        # Calculate margins
        margins = calculate_margins(image, inner_frame)
        
        # Calculate centering errors
        errors = calculate_centering_errors(margins)
        
        # Calculate centering score
        centering_score = calculate_centering_score(errors['combined_error'], max_error_threshold)
        
        # Create findings object
        findings = CenteringFindings(
            left_margin_px=margins.get('left_px', 0),
            right_margin_px=margins.get('right_px', 0),
            top_margin_px=margins.get('top_px', 0),
            bottom_margin_px=margins.get('bottom_px', 0),
            horizontal_error=errors['horizontal_error'],
            vertical_error=errors['vertical_error'],
            combined_error=errors['combined_error'],
            max_error_threshold=max_error_threshold,
            inner_frame_detected=frame_detected,
            detection_method=detection_method,
            centering_score=centering_score
        )
        
        logger.info("Centering analysis completed successfully")
        return findings
        
    except Exception as e:
        logger.error(f"Centering analysis failed: {e}")
        # Return default findings
        return CenteringFindings(
            left_margin_px=0,
            right_margin_px=0,
            top_margin_px=0,
            bottom_margin_px=0,
            horizontal_error=1.0,
            vertical_error=1.0,
            combined_error=1.0,
            max_error_threshold=max_error_threshold,
            inner_frame_detected=False,
            detection_method=detection_method,
            centering_score=0.0
        )