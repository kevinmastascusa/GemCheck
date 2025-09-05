"""
Edge and corner condition analysis for card grading.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
from app.schema import EdgeFindings, CornerFindings, BoundingBox

logger = logging.getLogger(__name__)


def create_edge_mask(image: np.ndarray, border_width: int = 6) -> np.ndarray:
    """
    Create a mask for the border/edge region of the card.
    
    Args:
        image: Rectified card image
        border_width: Width of border region in pixels
        
    Returns:
        Binary mask of edge region
    """
    try:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create border mask
        cv2.rectangle(mask, (0, 0), (w, border_width), 255, -1)  # Top
        cv2.rectangle(mask, (0, h-border_width), (w, h), 255, -1)  # Bottom
        cv2.rectangle(mask, (0, 0), (border_width, h), 255, -1)  # Left
        cv2.rectangle(mask, (w-border_width, 0), (w, h), 255, -1)  # Right
        
        return mask
        
    except Exception as e:
        logger.error(f"Edge mask creation failed: {e}")
        return np.ones(image.shape[:2], dtype=np.uint8) * 255


def detect_edge_whitening(image: np.ndarray, edge_mask: np.ndarray, 
                         threshold: float = 0.15) -> Tuple[np.ndarray, float]:
    """
    Detect whitening (fiber exposure) on card edges.
    
    Args:
        image: Rectified card image
        edge_mask: Mask of edge region
        threshold: Threshold for whitening detection
        
    Returns:
        Tuple of (whitening_mask, whitening_percentage)
    """
    try:
        # Convert to LAB color space for better white detection
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)
        
        # High L* values indicate brightness (potential whitening)
        bright_mask = L > (np.mean(L[edge_mask > 0]) + threshold * 255)
        
        # Low chroma indicates desaturation (whitening)
        chroma = np.sqrt(a.astype(np.float32)**2 + b.astype(np.float32)**2)
        low_chroma_mask = chroma < (np.mean(chroma[edge_mask > 0]) * 0.5)
        
        # Combine conditions for whitening detection
        whitening_mask = bright_mask & low_chroma_mask & (edge_mask > 0)
        
        # Calculate percentage of edge that is whitened
        edge_pixels = np.sum(edge_mask > 0)
        whitened_pixels = np.sum(whitening_mask > 0)
        whitening_percentage = (whitened_pixels / edge_pixels * 100) if edge_pixels > 0 else 0
        
        logger.info(f"Edge whitening: {whitening_percentage:.2f}% of edge affected")
        return whitening_mask.astype(np.uint8) * 255, whitening_percentage
        
    except Exception as e:
        logger.error(f"Edge whitening detection failed: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.0


def analyze_edges(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> EdgeFindings:
    """
    Complete edge condition analysis.
    
    Args:
        image: Rectified card image in RGB format
        config: Configuration parameters
        
    Returns:
        EdgeFindings object with detailed results
    """
    if config is None:
        config = {
            'border_width': 6,
            'whitening_threshold': 0.15,
        }
    
    try:
        border_width = config.get('border_width', 6)
        whitening_threshold = config.get('whitening_threshold', 0.15)
        
        # Create edge mask
        edge_mask = create_edge_mask(image, border_width)
        
        # Calculate total perimeter
        h, w = image.shape[:2]
        total_perimeter = 2 * (h + w) * border_width
        
        # Detect whitening
        whitening_mask, whitening_percentage = detect_edge_whitening(
            image, edge_mask, whitening_threshold
        )
        whitened_pixels = np.sum(whitening_mask > 0)
        
        # Simple nick detection (count dark spots in edge region)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edge_gray = cv2.bitwise_and(gray, gray, mask=edge_mask)
        dark_spots = cv2.threshold(edge_gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nick_count = len([c for c in contours if cv2.contourArea(c) > 10])
        largest_nick_area = max([cv2.contourArea(c) for c in contours], default=0)
        
        # Calculate clean edge percentage
        clean_percentage = max(0.0, 100.0 - whitening_percentage)
        
        # Calculate edge score (simple formula based on whitening and nicks)
        whitening_penalty = whitening_percentage * 2  # 2 points per % whitening
        nick_penalty = nick_count * 5  # 5 points per nick
        edge_score = max(0.0, 100.0 - whitening_penalty - nick_penalty)
        
        # Create findings object
        findings = EdgeFindings(
            total_perimeter_px=total_perimeter,
            whitened_perimeter_px=whitened_pixels,
            whitening_percentage=whitening_percentage,
            nick_count=nick_count,
            largest_nick_area_px=int(largest_nick_area),
            clean_edge_percentage=clean_percentage,
            whitening_threshold=whitening_threshold,
            edge_score=edge_score
        )
        
        logger.info("Edge analysis completed successfully")
        return findings
        
    except Exception as e:
        logger.error(f"Edge analysis failed: {e}")
        return EdgeFindings(
            total_perimeter_px=0,
            whitened_perimeter_px=0,
            whitening_percentage=100.0,
            nick_count=999,
            largest_nick_area_px=999,
            clean_edge_percentage=0.0,
            whitening_threshold=0.15,
            edge_score=0.0
        )


def analyze_corners(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> CornerFindings:
    """
    Complete corner condition analysis.
    
    Args:
        image: Rectified card image in RGB format
        config: Configuration parameters
        
    Returns:
        CornerFindings object with detailed results
    """
    if config is None:
        config = {
            'corner_size': 50,
            'sharpness_threshold': 0.3,
        }
    
    try:
        corner_size = config.get('corner_size', 50)
        sharpness_threshold = config.get('sharpness_threshold', 0.3)
        
        h, w = image.shape[:2]
        
        # Define corner regions
        corners = {
            'top_left': (0, 0, corner_size, corner_size),
            'top_right': (w-corner_size, 0, corner_size, corner_size),
            'bottom_right': (w-corner_size, h-corner_size, corner_size, corner_size),
            'bottom_left': (0, h-corner_size, corner_size, corner_size)
        }
        
        corner_scores = {}
        corner_sharpness = {}
        corner_whitening = {}
        corner_damage_area = {}
        
        # Analyze each corner
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        for corner_name, (x, y, cw, ch) in corners.items():
            # Extract corner region
            corner_region = gray[y:y+ch, x:x+cw]
            
            # Simple sharpness measure using Laplacian variance
            if corner_region.size > 0:
                laplacian = cv2.Laplacian(corner_region, cv2.CV_64F)
                sharpness = laplacian.var() / 10000  # Normalize
                sharpness = min(1.0, sharpness)
            else:
                sharpness = 0.0
            
            # Simple whitening detection
            bright_pixels = np.sum(corner_region > 200)
            total_pixels = corner_region.size
            whitening_pct = (bright_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            # Simple damage area (very dark or very bright pixels)
            damage_pixels = np.sum((corner_region < 50) | (corner_region > 240))
            
            # Calculate corner score
            score = 100.0
            if sharpness < sharpness_threshold:
                score -= 30.0
            score -= whitening_pct * 0.5
            score -= (damage_pixels / total_pixels * 100) * 0.3
            score = max(0.0, score)
            
            corner_scores[corner_name] = score
            corner_sharpness[corner_name] = sharpness
            corner_whitening[corner_name] = whitening_pct
            corner_damage_area[corner_name] = damage_pixels
        
        # Find minimum corner score
        min_score = min(corner_scores.values()) if corner_scores else 0.0
        
        # Calculate overall corner score (average of individual corners)
        overall_corner_score = sum(corner_scores.values()) / len(corner_scores) if corner_scores else 0.0
        
        # Create findings object
        findings = CornerFindings(
            corner_scores=corner_scores,
            corner_sharpness=corner_sharpness,
            corner_whitening=corner_whitening,
            corner_damage_area=corner_damage_area,
            minimum_corner_score=min_score,
            sharpness_threshold=sharpness_threshold,
            corner_score=overall_corner_score
        )
        
        logger.info(f"Corner analysis completed, minimum score: {min_score:.2f}")
        return findings
        
    except Exception as e:
        logger.error(f"Corner analysis failed: {e}")
        corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        return CornerFindings(
            corner_scores={name: 0.0 for name in corner_names},
            corner_sharpness={name: 0.0 for name in corner_names},
            corner_whitening={name: 100.0 for name in corner_names},
            corner_damage_area={name: 999 for name in corner_names},
            minimum_corner_score=0.0,
            sharpness_threshold=0.3,
            corner_score=0.0
        )