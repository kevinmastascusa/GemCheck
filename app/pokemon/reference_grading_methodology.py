"""
Reference Grading Methodology based on the Pikachu Yellow Cheeks template.
Implements systematic grading following the measurement grid approach shown in the reference.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GradingZone(Enum):
    """The four primary grading zones as shown in reference template."""
    EDGES = "edges"
    CORNERS = "corners" 
    CENTERING = "centering"
    SURFACE = "surface"


@dataclass
class MeasurementPoint:
    """A specific measurement point on the card."""
    zone: GradingZone
    coordinates: Tuple[int, int]  # (x, y) in pixels
    measurement_type: str  # What is being measured at this point
    tolerance: float  # Acceptable variance
    weight: float  # Importance weight for overall grade


@dataclass
class GradingGrid:
    """Measurement grid system for systematic card analysis."""
    grid_size: int  # Size of each grid square in pixels
    card_bounds: Tuple[int, int, int, int]  # (left, top, right, bottom) of card area
    measurement_points: List[MeasurementPoint]
    reference_card_era: str  # "vintage", "modern", etc.


@dataclass
class ZoneGradingResult:
    """Grading result for a specific zone."""
    zone: GradingZone
    raw_score: float  # 0-100 raw measurement score
    adjusted_score: float  # Score adjusted for card era/type
    defects_found: List[str]  # List of specific defects
    measurements: Dict[str, float]  # Detailed measurements
    confidence: float  # Confidence in the grading


class ReferenceGradingMethodology:
    """
    Implements systematic Pokemon card grading based on the reference template methodology.
    
    Based on the 1999 Pokémon Game Pikachu #58 Yellow Cheeks reference template which shows:
    - Measurement grid for systematic analysis
    - Four key grading zones: EDGES, CORNERS, CENTERING, SURFACE
    - Specific measurement points for each zone
    """
    
    def __init__(self):
        self.reference_grid = self._create_reference_grid()
        self.era_adjustments = self._define_era_adjustments()
    
    def _create_reference_grid(self) -> GradingGrid:
        """Create the reference measurement grid based on the template."""
        
        # Standard Pokemon card proportions (based on reference template)
        # These are normalized coordinates that can be scaled to any card size
        measurement_points = [
            # EDGES zone measurements
            MeasurementPoint(
                zone=GradingZone.EDGES,
                coordinates=(0.5, 0.05),  # Top edge center (normalized)
                measurement_type="top_edge_wear",
                tolerance=0.02,
                weight=0.25
            ),
            MeasurementPoint(
                zone=GradingZone.EDGES,
                coordinates=(0.5, 0.95),  # Bottom edge center
                measurement_type="bottom_edge_wear", 
                tolerance=0.02,
                weight=0.25
            ),
            MeasurementPoint(
                zone=GradingZone.EDGES,
                coordinates=(0.05, 0.5),  # Left edge center
                measurement_type="left_edge_wear",
                tolerance=0.02,
                weight=0.25
            ),
            MeasurementPoint(
                zone=GradingZone.EDGES,
                coordinates=(0.95, 0.5),  # Right edge center
                measurement_type="right_edge_wear",
                tolerance=0.02,
                weight=0.25
            ),
            
            # CORNERS zone measurements
            MeasurementPoint(
                zone=GradingZone.CORNERS,
                coordinates=(0.05, 0.05),  # Top-left corner
                measurement_type="corner_sharpness",
                tolerance=0.03,
                weight=0.25
            ),
            MeasurementPoint(
                zone=GradingZone.CORNERS,
                coordinates=(0.95, 0.05),  # Top-right corner
                measurement_type="corner_sharpness",
                tolerance=0.03,
                weight=0.25
            ),
            MeasurementPoint(
                zone=GradingZone.CORNERS,
                coordinates=(0.05, 0.95),  # Bottom-left corner
                measurement_type="corner_sharpness",
                tolerance=0.03,
                weight=0.25
            ),
            MeasurementPoint(
                zone=GradingZone.CORNERS,
                coordinates=(0.95, 0.95),  # Bottom-right corner
                measurement_type="corner_sharpness",
                tolerance=0.03,
                weight=0.25
            ),
            
            # CENTERING zone measurements
            MeasurementPoint(
                zone=GradingZone.CENTERING,
                coordinates=(0.95, 0.5),  # Right side centering check
                measurement_type="horizontal_centering",
                tolerance=0.05,  # 5% tolerance for vintage cards
                weight=0.5
            ),
            MeasurementPoint(
                zone=GradingZone.CENTERING,
                coordinates=(0.5, 0.95),  # Bottom centering check
                measurement_type="vertical_centering",
                tolerance=0.05,
                weight=0.5
            ),
            
            # SURFACE zone measurements
            MeasurementPoint(
                zone=GradingZone.SURFACE,
                coordinates=(0.5, 0.35),  # Artwork area (based on Pikachu template)
                measurement_type="artwork_surface_quality",
                tolerance=0.04,
                weight=0.4
            ),
            MeasurementPoint(
                zone=GradingZone.SURFACE,
                coordinates=(0.5, 0.7),  # Text area
                measurement_type="text_area_surface_quality",
                tolerance=0.03,
                weight=0.3
            ),
            MeasurementPoint(
                zone=GradingZone.SURFACE,
                coordinates=(0.2, 0.8),  # Bottom left area (weakness/resistance)
                measurement_type="bottom_surface_quality", 
                tolerance=0.03,
                weight=0.3
            ),
        ]
        
        return GradingGrid(
            grid_size=20,  # Standard grid size from reference
            card_bounds=(0, 0, 1, 1),  # Normalized bounds
            measurement_points=measurement_points,
            reference_card_era="vintage"  # Based on 1999 Pikachu
        )
    
    def _define_era_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Define grading adjustments for different Pokemon card eras."""
        return {
            "vintage": {  # 1998-2001 (Base Set, Jungle, Fossil, etc.)
                "centering_tolerance_multiplier": 1.5,  # More lenient centering
                "edge_wear_tolerance_multiplier": 1.3,  # More edge wear acceptable
                "corner_wear_tolerance_multiplier": 1.4,  # Corner wear more acceptable
                "surface_scratch_tolerance_multiplier": 1.6,  # Surface defects more common
                "print_quality_expectation": 0.7,  # Lower print quality standards
            },
            "e_card": {  # 2001-2003
                "centering_tolerance_multiplier": 1.3,
                "edge_wear_tolerance_multiplier": 1.2,
                "corner_wear_tolerance_multiplier": 1.2,
                "surface_scratch_tolerance_multiplier": 1.3,
                "print_quality_expectation": 0.75,
            },
            "modern": {  # 2014+ (XY series onward)
                "centering_tolerance_multiplier": 0.8,  # Stricter centering standards
                "edge_wear_tolerance_multiplier": 0.7,  # Less edge wear acceptable
                "corner_wear_tolerance_multiplier": 0.7,  # Corners should be sharp
                "surface_scratch_tolerance_multiplier": 0.6,  # Higher surface quality expected
                "print_quality_expectation": 0.95,  # Very high print quality
            }
        }
    
    def grade_card_using_reference_methodology(self, card_image: np.ndarray, 
                                             card_era: str = "vintage") -> Dict[GradingZone, ZoneGradingResult]:
        """
        Grade a Pokemon card using the reference template methodology.
        
        Args:
            card_image: Card image to grade (RGB format)
            card_era: Era of the card ("vintage", "e_card", "modern")
            
        Returns:
            Dictionary of grading results for each zone
        """
        logger.info(f"Grading card using reference methodology (era: {card_era})")
        
        # Scale measurement points to actual image size
        scaled_grid = self._scale_grid_to_image(card_image, self.reference_grid)
        
        # Get era adjustments
        era_adjustments = self.era_adjustments.get(card_era, self.era_adjustments["vintage"])
        
        # Grade each zone
        zone_results = {}
        
        for zone in GradingZone:
            zone_points = [p for p in scaled_grid.measurement_points if p.zone == zone]
            zone_result = self._grade_zone(card_image, zone, zone_points, era_adjustments)
            zone_results[zone] = zone_result
        
        return zone_results
    
    def _scale_grid_to_image(self, image: np.ndarray, reference_grid: GradingGrid) -> GradingGrid:
        """Scale the normalized reference grid to the actual image dimensions."""
        height, width = image.shape[:2]
        
        scaled_points = []
        for point in reference_grid.measurement_points:
            # Convert normalized coordinates to pixel coordinates
            scaled_x = int(point.coordinates[0] * width)
            scaled_y = int(point.coordinates[1] * height)
            
            scaled_point = MeasurementPoint(
                zone=point.zone,
                coordinates=(scaled_x, scaled_y),
                measurement_type=point.measurement_type,
                tolerance=point.tolerance,
                weight=point.weight
            )
            scaled_points.append(scaled_point)
        
        return GradingGrid(
            grid_size=reference_grid.grid_size,
            card_bounds=(0, 0, width, height),
            measurement_points=scaled_points,
            reference_card_era=reference_grid.reference_card_era
        )
    
    def _grade_zone(self, image: np.ndarray, zone: GradingZone, 
                   measurement_points: List[MeasurementPoint], 
                   era_adjustments: Dict[str, float]) -> ZoneGradingResult:
        """Grade a specific zone using its measurement points."""
        
        if zone == GradingZone.CENTERING:
            return self._grade_centering_zone(image, measurement_points, era_adjustments)
        elif zone == GradingZone.EDGES:
            return self._grade_edges_zone(image, measurement_points, era_adjustments)
        elif zone == GradingZone.CORNERS:
            return self._grade_corners_zone(image, measurement_points, era_adjustments)
        elif zone == GradingZone.SURFACE:
            return self._grade_surface_zone(image, measurement_points, era_adjustments)
        else:
            # Fallback
            return ZoneGradingResult(
                zone=zone,
                raw_score=75.0,
                adjusted_score=75.0,
                defects_found=[],
                measurements={},
                confidence=0.5
            )
    
    def _grade_centering_zone(self, image: np.ndarray, points: List[MeasurementPoint], 
                            era_adjustments: Dict[str, float]) -> ZoneGradingResult:
        """Grade centering using systematic margin measurements."""
        height, width = image.shape[:2]
        
        # Improved card border detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use multiple edge detection approaches to find the actual card boundary
        # 1. Try adaptive thresholding first (better for card edges)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 2. Also try standard edge detection as fallback
        edges = cv2.Canny(gray, 30, 100)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine all contours and filter for realistic card rectangles
        all_contours = contours1 + contours2
        
        # Filter contours to find the most likely card boundary
        card_candidates = []
        min_area = (width * height) * 0.4  # Card should cover at least 40% of image
        max_area = (width * height) * 0.95  # But not more than 95%
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Card should have reasonable aspect ratio (not too thin/wide)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 <= aspect_ratio <= 2.0:  # Reasonable card proportions
                    card_candidates.append((contour, area, x, y, w, h))
        
        if card_candidates:
            # Use the largest qualifying contour
            best_contour = max(card_candidates, key=lambda x: x[1])
            _, _, x, y, w, h = best_contour
        elif all_contours:
            # Fallback: use largest contour but with warnings
            largest_contour = max(all_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            logger.warning(f"Card detection uncertain: using largest contour with area {cv2.contourArea(largest_contour)}")
        else:
            # Last resort: assume card fills most of the image with small margins
            margin = min(width, height) * 0.05  # 5% margin
            x, y = int(margin), int(margin)
            w, h = int(width - 2*margin), int(height - 2*margin)
            logger.warning("No contours found: using estimated card boundaries")
        
        # Calculate margins
        left_margin = x
        right_margin = width - (x + w)
        top_margin = y  
        bottom_margin = height - (y + h)
        
        # Log card detection results for debugging
        card_coverage = (w * h) / (width * height) * 100
        logger.info(f"Card detected: {w}x{h} at ({x},{y}), covers {card_coverage:.1f}% of image")
        
        # Calculate centering errors
        horizontal_error = abs(left_margin - right_margin) / (left_margin + right_margin) if (left_margin + right_margin) > 0 else 0
        vertical_error = abs(top_margin - bottom_margin) / (top_margin + bottom_margin) if (top_margin + bottom_margin) > 0 else 0
        
        # Apply era adjustments
        tolerance_multiplier = era_adjustments.get("centering_tolerance_multiplier", 1.0)
        adjusted_tolerance = 0.05 * tolerance_multiplier  # Base 5% tolerance
        
        # Calculate score
        max_error = max(horizontal_error, vertical_error)
        raw_score = max(0, 100 * (1 - max_error / adjusted_tolerance))
        adjusted_score = min(100, raw_score)
        
        defects = []
        if horizontal_error > adjusted_tolerance:
            defects.append(f"Horizontal centering error: {horizontal_error:.1%}")
        if vertical_error > adjusted_tolerance:
            defects.append(f"Vertical centering error: {vertical_error:.1%}")
        
        measurements = {
            "left_margin": left_margin,
            "right_margin": right_margin,
            "top_margin": top_margin,
            "bottom_margin": bottom_margin,
            "horizontal_error": horizontal_error,
            "vertical_error": vertical_error,
            "max_error": max_error,
            "card_coverage_percent": card_coverage
        }
        
        return ZoneGradingResult(
            zone=GradingZone.CENTERING,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            defects_found=defects,
            measurements=measurements,
            confidence=0.8 if card_candidates else 0.5
        )
    
    def _grade_edges_zone(self, image: np.ndarray, points: List[MeasurementPoint], 
                         era_adjustments: Dict[str, float]) -> ZoneGradingResult:
        """Grade edge condition focusing on wear and whitening."""
        height, width = image.shape[:2]
        
        # Analyze edge regions
        edge_thickness = 20  # Pixels to analyze from edge
        
        edge_regions = {
            "top": image[:edge_thickness, :],
            "bottom": image[-edge_thickness:, :],
            "left": image[:, :edge_thickness],
            "right": image[:, -edge_thickness:]
        }
        
        total_whitening = 0
        total_wear_score = 0
        defects = []
        measurements = {}
        
        for edge_name, edge_region in edge_regions.items():
            # Convert to grayscale for analysis
            gray_edge = cv2.cvtColor(edge_region, cv2.COLOR_RGB2GRAY)
            
            # Detect whitening (bright pixels indicating wear)
            white_threshold = 220
            white_pixels = np.sum(gray_edge > white_threshold)
            total_pixels = gray_edge.size
            whitening_percentage = white_pixels / total_pixels
            
            # Detect edge smoothness (wear creates rough edges)
            edges = cv2.Canny(gray_edge, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            total_whitening += whitening_percentage
            total_wear_score += (1 - edge_density)  # Lower edge density = more wear
            
            measurements[f"{edge_name}_whitening"] = whitening_percentage
            measurements[f"{edge_name}_edge_density"] = edge_density
            
            # Apply era-specific thresholds
            wear_tolerance = era_adjustments.get("edge_wear_tolerance_multiplier", 1.0)
            whitening_threshold = 0.15 * wear_tolerance  # Base 15% whitening threshold
            
            if whitening_percentage > whitening_threshold:
                defects.append(f"{edge_name.title()} edge whitening: {whitening_percentage:.1%}")
        
        # Calculate overall edge score
        avg_whitening = total_whitening / 4
        avg_wear = total_wear_score / 4
        
        # Score calculation (lower whitening and wear = higher score)
        whitening_penalty = min(50, avg_whitening * 200)  # Up to 50 point penalty
        wear_penalty = min(30, avg_wear * 100)  # Up to 30 point penalty
        
        raw_score = 100 - whitening_penalty - wear_penalty
        adjusted_score = max(0, raw_score)
        
        measurements["average_whitening"] = avg_whitening
        measurements["average_wear"] = avg_wear
        
        return ZoneGradingResult(
            zone=GradingZone.EDGES,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            defects_found=defects,
            measurements=measurements,
            confidence=0.85
        )
    
    def _grade_corners_zone(self, image: np.ndarray, points: List[MeasurementPoint], 
                          era_adjustments: Dict[str, float]) -> ZoneGradingResult:
        """Grade corner condition focusing on sharpness and wear."""
        height, width = image.shape[:2]
        corner_size = 40  # Size of corner region to analyze
        
        corner_regions = {
            "top_left": image[:corner_size, :corner_size],
            "top_right": image[:corner_size, -corner_size:],
            "bottom_left": image[-corner_size:, :corner_size],
            "bottom_right": image[-corner_size:, -corner_size:]
        }
        
        total_sharpness = 0
        defects = []
        measurements = {}
        
        for corner_name, corner_region in corner_regions.items():
            # Convert to grayscale
            gray_corner = cv2.cvtColor(corner_region, cv2.COLOR_RGB2GRAY)
            
            # Measure corner sharpness using gradient magnitude
            grad_x = cv2.Sobel(gray_corner, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_corner, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Sharpness is the mean gradient in the corner area
            sharpness = np.mean(gradient_magnitude)
            total_sharpness += sharpness
            
            measurements[f"{corner_name}_sharpness"] = sharpness
            
            # Apply era-specific thresholds
            wear_tolerance = era_adjustments.get("corner_wear_tolerance_multiplier", 1.0)
            sharpness_threshold = 25.0 / wear_tolerance  # Base threshold adjusted for era
            
            if sharpness < sharpness_threshold:
                defects.append(f"{corner_name.replace('_', '-').title()} corner worn (sharpness: {sharpness:.1f})")
        
        # Calculate corner score
        avg_sharpness = total_sharpness / 4
        measurements["average_sharpness"] = avg_sharpness
        
        # Score based on sharpness (higher sharpness = better score)
        expected_sharpness = 40.0  # Expected for perfect corners
        sharpness_ratio = min(1.0, avg_sharpness / expected_sharpness)
        raw_score = sharpness_ratio * 100
        adjusted_score = max(0, raw_score)
        
        return ZoneGradingResult(
            zone=GradingZone.CORNERS,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            defects_found=defects,
            measurements=measurements,
            confidence=0.8
        )
    
    def _grade_surface_zone(self, image: np.ndarray, points: List[MeasurementPoint], 
                          era_adjustments: Dict[str, float]) -> ZoneGradingResult:
        """Grade surface condition focusing on scratches, print lines, and overall quality."""
        # Create analysis regions based on measurement points
        surface_regions = []
        region_size = 60
        
        for point in points:
            x, y = point.coordinates
            x1 = max(0, x - region_size // 2)
            y1 = max(0, y - region_size // 2)
            x2 = min(image.shape[1], x + region_size // 2)
            y2 = min(image.shape[0], y + region_size // 2)
            
            region = image[y1:y2, x1:x2]
            surface_regions.append((point.measurement_type, region))
        
        total_defect_score = 0
        defects = []
        measurements = {}
        
        for region_name, region in surface_regions:
            if region.size == 0:
                continue
                
            # Convert to grayscale for defect detection
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            
            # Detect scratches using line detection
            edges = cv2.Canny(gray_region, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
            
            scratch_count = len(lines) if lines is not None else 0
            
            # Detect print lines/quality issues using texture analysis
            # Calculate local standard deviation as texture measure
            kernel = np.ones((5, 5), np.float32) / 25
            mean = cv2.filter2D(gray_region.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray_region.astype(np.float32))**2, -1, kernel)
            texture_variance = sqr_mean - mean**2
            texture_score = np.mean(texture_variance)
            
            # Calculate defect score for this region
            scratch_penalty = min(30, scratch_count * 3)  # Up to 30 points for scratches
            texture_penalty = min(20, max(0, (50 - texture_score) * 0.4))  # Texture quality penalty
            
            region_defect_score = scratch_penalty + texture_penalty
            total_defect_score += region_defect_score
            
            measurements[f"{region_name}_scratches"] = scratch_count
            measurements[f"{region_name}_texture_score"] = texture_score
            
            # Apply era adjustments
            scratch_tolerance = era_adjustments.get("surface_scratch_tolerance_multiplier", 1.0)
            adjusted_scratch_threshold = 3 * scratch_tolerance
            
            if scratch_count > adjusted_scratch_threshold:
                defects.append(f"{region_name}: {scratch_count} scratches detected")
        
        # Calculate overall surface score
        avg_defect_score = total_defect_score / len(surface_regions) if surface_regions else 0
        raw_score = max(0, 100 - avg_defect_score)
        
        # Apply era adjustments to final score
        print_quality_expectation = era_adjustments.get("print_quality_expectation", 0.8)
        adjusted_score = raw_score * print_quality_expectation
        
        measurements["total_defect_score"] = total_defect_score
        measurements["average_defect_score"] = avg_defect_score
        
        return ZoneGradingResult(
            zone=GradingZone.SURFACE,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            defects_found=defects,
            measurements=measurements,
            confidence=0.75
        )
    
    def calculate_overall_grade(self, zone_results: Dict[GradingZone, ZoneGradingResult]) -> Dict[str, Any]:
        """Calculate overall card grade from zone results."""
        
        # PSA zone weights (approximate based on grading standards)
        zone_weights = {
            GradingZone.CENTERING: 0.35,  # 35% - Most important for PSA
            GradingZone.SURFACE: 0.30,    # 30% - Surface quality crucial
            GradingZone.CORNERS: 0.20,    # 20% - Corner condition
            GradingZone.EDGES: 0.15       # 15% - Edge condition
        }
        
        weighted_score = 0
        total_weight = 0
        zone_scores = {}
        all_defects = []
        
        for zone, result in zone_results.items():
            weight = zone_weights.get(zone, 0.25)
            weighted_score += result.adjusted_score * weight
            total_weight += weight
            zone_scores[zone.value] = result.adjusted_score
            all_defects.extend(result.defects_found)
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0
        
        # Convert to PSA 1-10 scale
        if overall_score >= 98:
            psa_grade = 10
            grade_label = "Gem Mint"
        elif overall_score >= 92:
            psa_grade = 9  
            grade_label = "Mint"
        elif overall_score >= 85:
            psa_grade = 8
            grade_label = "NM-Mint"
        elif overall_score >= 78:
            psa_grade = 7
            grade_label = "Near Mint"
        elif overall_score >= 70:
            psa_grade = 6
            grade_label = "Excellent"
        elif overall_score >= 60:
            psa_grade = 5
            grade_label = "VG-EX"
        elif overall_score >= 50:
            psa_grade = 4
            grade_label = "Good"
        elif overall_score >= 40:
            psa_grade = 3
            grade_label = "Fair"
        elif overall_score >= 20:
            psa_grade = 2
            grade_label = "Poor"
        else:
            psa_grade = 1
            grade_label = "Authentic"
        
        return {
            "overall_score": overall_score,
            "psa_grade": psa_grade,
            "grade_label": grade_label,
            "zone_scores": zone_scores,
            "defects_found": all_defects,
            "grading_methodology": "Reference Template Based",
            "confidence": np.mean([result.confidence for result in zone_results.values()])
        }