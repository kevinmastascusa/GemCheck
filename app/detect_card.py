"""
Card detection and rectification functionality.
Detects trading cards in images and performs perspective correction.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CardBounds:
    """
    Represents the detected bounds of a trading card.
    """
    corners: np.ndarray  # 4x2 array of corner coordinates
    confidence: float    # Detection confidence (0-1)
    area: float         # Area of detected card region
    aspect_ratio: float # Width/height ratio


class CardDetector:
    """
    Detects and extracts trading cards from images using computer vision.
    """
    
    def __init__(self, target_width: int = 750, target_height: int = 1050):
        """
        Initialize card detector.
        
        Args:
            target_width: Target width for rectified cards
            target_height: Target height for rectified cards
        """
        self.target_width = target_width
        self.target_height = target_height
        self.expected_aspect_ratio = target_height / target_width  # ~1.4 for standard cards
    
    def detect_card(self, image: np.ndarray, 
                   min_area_ratio: float = 0.1,
                   max_area_ratio: float = 0.9) -> Optional[CardBounds]:
        """
        Detect a trading card in the image.
        
        Args:
            image: Input image
            min_area_ratio: Minimum card area as fraction of image area
            max_area_ratio: Maximum card area as fraction of image area
            
        Returns:
            CardBounds if card detected, None otherwise
        """
        try:
            logger.debug("Starting card detection")
            
            # Preprocess image
            processed = self._preprocess_for_detection(image)
            
            # Find contours
            contours = self._find_card_contours(processed)
            
            if not contours:
                logger.warning("No contours found for card detection")
                return None
            
            # Filter and rank contours
            candidates = self._filter_card_candidates(
                contours, image.shape, min_area_ratio, max_area_ratio
            )
            
            if not candidates:
                logger.warning("No valid card candidates found")
                return None
            
            # Select best candidate
            best_card = self._select_best_candidate(candidates, image.shape)
            
            if best_card is None:
                logger.warning("No suitable card candidate selected")
                return None
            
            logger.info(f"Card detected with confidence: {best_card.confidence:.3f}")
            return best_card
            
        except Exception as e:
            logger.error(f"Card detection failed: {e}")
            return None
    
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for card edge detection."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize if too large (for performance)
            h, w = gray.shape
            if max(h, w) > 1500:
                scale = 1500 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h))
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding works better than simple thresholding
            # for cards with varying lighting conditions
            adaptive = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def _find_card_contours(self, processed_image: np.ndarray) -> List[np.ndarray]:
        """Find contours that could represent card edges."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Sort by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            return contours[:10]  # Keep top 10 largest contours
            
        except Exception as e:
            logger.error(f"Contour finding failed: {e}")
            return []
    
    def _filter_card_candidates(self, contours: List[np.ndarray], 
                              image_shape: Tuple[int, ...],
                              min_area_ratio: float,
                              max_area_ratio: float) -> List[CardBounds]:
        """Filter contours to find valid card candidates."""
        try:
            h, w = image_shape[:2]
            image_area = h * w
            candidates = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Check area constraints
                area_ratio = area / image_area
                if not (min_area_ratio <= area_ratio <= max_area_ratio):
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for quadrilaterals (4 corners)
                if len(approx) == 4:
                    corners = self._order_corners(approx.reshape(4, 2))
                    
                    # Calculate aspect ratio
                    aspect_ratio = self._calculate_aspect_ratio(corners)
                    
                    # Check if aspect ratio is reasonable for a trading card
                    if 1.2 <= aspect_ratio <= 1.6:  # Standard cards are ~1.4
                        confidence = self._calculate_confidence(
                            corners, area, aspect_ratio, image_shape
                        )
                        
                        candidates.append(CardBounds(
                            corners=corners,
                            confidence=confidence,
                            area=area,
                            aspect_ratio=aspect_ratio
                        ))
            
            # Sort by confidence
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            return candidates
            
        except Exception as e:
            logger.error(f"Candidate filtering failed: {e}")
            return []
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order: top-left, top-right, bottom-right, bottom-left.
        """
        try:
            # Calculate centroid
            centroid = np.mean(corners, axis=0)
            
            # Calculate angles from centroid
            angles = []
            for corner in corners:
                diff = corner - centroid
                angle = np.arctan2(diff[1], diff[0])
                angles.append(angle)
            
            # Sort corners by angle
            sorted_indices = np.argsort(angles)
            
            # Determine which corner is which based on angle quadrants
            ordered = np.zeros_like(corners)
            
            for i, idx in enumerate(sorted_indices):
                corner = corners[idx]
                angle = angles[idx]
                
                # Quadrant-based assignment
                if -np.pi/2 <= angle < 0:  # Top-right quadrant
                    ordered[1] = corner
                elif 0 <= angle < np.pi/2:  # Bottom-right quadrant
                    ordered[2] = corner
                elif np.pi/2 <= angle < np.pi:  # Bottom-left quadrant
                    ordered[3] = corner
                else:  # Top-left quadrant
                    ordered[0] = corner
            
            return ordered.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Corner ordering failed: {e}")
            return corners.astype(np.float32)
    
    def _calculate_aspect_ratio(self, corners: np.ndarray) -> float:
        """Calculate aspect ratio of detected quadrilateral."""
        try:
            # Calculate distances
            top_width = np.linalg.norm(corners[1] - corners[0])
            bottom_width = np.linalg.norm(corners[2] - corners[3])
            left_height = np.linalg.norm(corners[3] - corners[0])
            right_height = np.linalg.norm(corners[2] - corners[1])
            
            # Average dimensions
            width = (top_width + bottom_width) / 2
            height = (left_height + right_height) / 2
            
            return height / width if width > 0 else 0
            
        except Exception as e:
            logger.error(f"Aspect ratio calculation failed: {e}")
            return 0
    
    def _calculate_confidence(self, corners: np.ndarray, area: float, 
                            aspect_ratio: float, image_shape: Tuple[int, ...]) -> float:
        """Calculate confidence score for detected card."""
        try:
            confidence = 0.0
            
            # Area score (prefer medium-sized detections)
            h, w = image_shape[:2]
            area_ratio = area / (h * w)
            if 0.2 <= area_ratio <= 0.8:
                confidence += 0.3
            elif 0.1 <= area_ratio <= 0.9:
                confidence += 0.15
            
            # Aspect ratio score
            aspect_diff = abs(aspect_ratio - self.expected_aspect_ratio)
            if aspect_diff < 0.1:
                confidence += 0.3
            elif aspect_diff < 0.2:
                confidence += 0.15
            
            # Corner regularity score (prefer rectangular shapes)
            regularity = self._calculate_corner_regularity(corners)
            confidence += regularity * 0.2
            
            # Edge straightness score
            straightness = self._calculate_edge_straightness(corners)
            confidence += straightness * 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _calculate_corner_regularity(self, corners: np.ndarray) -> float:
        """Calculate how close corners are to forming a rectangle."""
        try:
            # Calculate all angles
            angles = []
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                p3 = corners[(i + 2) % 4]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
            
            # For a rectangle, all angles should be close to π/2
            angle_diffs = [abs(angle - np.pi/2) for angle in angles]
            regularity = 1.0 - (np.mean(angle_diffs) / (np.pi/4))
            
            return max(0.0, regularity)
            
        except Exception as e:
            logger.error(f"Corner regularity calculation failed: {e}")
            return 0.0
    
    def _calculate_edge_straightness(self, corners: np.ndarray) -> float:
        """Calculate how straight the edges are (should be close to 1.0 for rectangles)."""
        try:
            straightness_scores = []
            
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                
                # Calculate edge length
                edge_length = np.linalg.norm(p2 - p1)
                if edge_length == 0:
                    continue
                
                # For this simplified version, assume perfect straightness
                # In a more advanced implementation, you could check intermediate points
                straightness_scores.append(1.0)
            
            return np.mean(straightness_scores) if straightness_scores else 0.0
            
        except Exception as e:
            logger.error(f"Edge straightness calculation failed: {e}")
            return 0.0
    
    def _select_best_candidate(self, candidates: List[CardBounds], 
                             image_shape: Tuple[int, ...]) -> Optional[CardBounds]:
        """Select the best card candidate from the list."""
        try:
            if not candidates:
                return None
            
            # Additional filtering
            valid_candidates = []
            for candidate in candidates:
                # Minimum confidence threshold
                if candidate.confidence < 0.3:
                    continue
                
                # Check if corners are within image bounds
                h, w = image_shape[:2]
                if self._corners_in_bounds(candidate.corners, w, h):
                    valid_candidates.append(candidate)
            
            return valid_candidates[0] if valid_candidates else None
            
        except Exception as e:
            logger.error(f"Best candidate selection failed: {e}")
            return None
    
    def _corners_in_bounds(self, corners: np.ndarray, width: int, height: int) -> bool:
        """Check if all corners are within image bounds."""
        for corner in corners:
            x, y = corner
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        return True
    
    def rectify_card(self, image: np.ndarray, card_bounds: CardBounds) -> Optional[np.ndarray]:
        """
        Perform perspective correction to rectify the detected card.
        
        Args:
            image: Original image
            card_bounds: Detected card boundaries
            
        Returns:
            Rectified card image, or None if rectification failed
        """
        try:
            logger.debug("Starting card rectification")
            
            # Define destination points for rectification
            dst_corners = np.array([
                [0, 0],                                           # Top-left
                [self.target_width - 1, 0],                      # Top-right
                [self.target_width - 1, self.target_height - 1], # Bottom-right
                [0, self.target_height - 1]                      # Bottom-left
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            transform_matrix = cv2.getPerspectiveTransform(
                card_bounds.corners, dst_corners
            )
            
            # Apply perspective correction
            rectified = cv2.warpPerspective(
                image, transform_matrix, (self.target_width, self.target_height)
            )
            
            logger.info(f"Card rectified to {self.target_width}x{self.target_height}")
            return rectified
            
        except Exception as e:
            logger.error(f"Card rectification failed: {e}")
            return None


def detect_and_rectify_card(image: np.ndarray, 
                          target_width: int = 750,
                          target_height: int = 1050) -> Optional[np.ndarray]:
    """
    Convenience function to detect and rectify a card in one step.
    
    Args:
        image: Input image containing a card
        target_width: Target width for rectified card
        target_height: Target height for rectified card
        
    Returns:
        Rectified card image, or None if detection/rectification failed
    """
    try:
        detector = CardDetector(target_width, target_height)
        
        # Detect card
        card_bounds = detector.detect_card(image)
        if card_bounds is None:
            logger.warning("Card detection failed")
            return None
        
        # Rectify card
        rectified = detector.rectify_card(image, card_bounds)
        if rectified is None:
            logger.warning("Card rectification failed")
            return None
        
        return rectified
        
    except Exception as e:
        logger.error(f"Card detection and rectification failed: {e}")
        return None


def validate_card_detection(image: np.ndarray, 
                          card_bounds: CardBounds,
                          min_confidence: float = 0.5) -> bool:
    """
    Validate that a detected card meets quality requirements.
    
    Args:
        image: Original image
        card_bounds: Detected card bounds
        min_confidence: Minimum confidence threshold
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check confidence
        if card_bounds.confidence < min_confidence:
            logger.warning(f"Card confidence {card_bounds.confidence:.3f} below threshold {min_confidence}")
            return False
        
        # Check aspect ratio
        if not (1.2 <= card_bounds.aspect_ratio <= 1.6):
            logger.warning(f"Card aspect ratio {card_bounds.aspect_ratio:.2f} outside valid range")
            return False
        
        # Check area
        h, w = image.shape[:2]
        area_ratio = card_bounds.area / (h * w)
        if not (0.1 <= area_ratio <= 0.9):
            logger.warning(f"Card area ratio {area_ratio:.3f} outside valid range")
            return False
        
        # Check corner positions are reasonable
        corners = card_bounds.corners
        if not all(0 <= x < w and 0 <= y < h for x, y in corners):
            logger.warning("Card corners outside image bounds")
            return False
        
        logger.info("Card detection validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Card validation failed: {e}")
        return False