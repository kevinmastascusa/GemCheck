"""
Template Abstractor for Pokemon Card Reference Templates.
Analyzes reference template images to extract measurement grids and key calibration points.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class GridAnalysis:
    """Analysis results from measurement grid detection."""
    grid_detected: bool
    grid_size: int  # Size of grid squares in pixels
    grid_lines_horizontal: List[int]  # Y coordinates of horizontal lines
    grid_lines_vertical: List[int]    # X coordinates of vertical lines
    card_bounds: Tuple[int, int, int, int]  # (left, top, right, bottom)
    confidence: float


@dataclass
class AnnotationPoint:
    """A detected annotation point from the reference template."""
    label: str
    location: Tuple[int, int]
    region_type: str  # 'centering', 'edges', 'corners', 'surface'
    confidence: float


@dataclass
class AbstractedTemplate:
    """Abstracted template data extracted from reference image."""
    template_name: str
    card_dimensions: Tuple[int, int]  # (width, height)
    grid_analysis: GridAnalysis
    annotation_points: List[AnnotationPoint]
    measurement_zones: Dict[str, List[Tuple[int, int, int, int]]]  # region -> list of (x,y,w,h) zones
    calibration_standards: Dict[str, float]


class ReferenceTemplateAbstractor:
    """Abstracts measurement grids and annotations from reference template images."""
    
    def __init__(self):
        self.debug_mode = False
    
    def analyze_reference_template(self, template_image: np.ndarray) -> AbstractedTemplate:
        """
        Analyze a reference template image to extract measurement grid and annotations.
        
        Args:
            template_image: Reference template image (RGB format)
            
        Returns:
            AbstractedTemplate with extracted measurement data
        """
        logger.info(f"Analyzing reference template: {template_image.shape}")
        
        # Analyze the measurement grid
        grid_analysis = self._detect_measurement_grid(template_image)
        
        # Detect annotation points and labels
        annotation_points = self._detect_annotation_points(template_image, grid_analysis)
        
        # Extract measurement zones
        measurement_zones = self._extract_measurement_zones(template_image, grid_analysis, annotation_points)
        
        # Calculate card dimensions from grid
        if grid_analysis.grid_detected:
            card_width = grid_analysis.card_bounds[2] - grid_analysis.card_bounds[0]
            card_height = grid_analysis.card_bounds[3] - grid_analysis.card_bounds[1]
        else:
            # Fallback: estimate from image
            card_width = template_image.shape[1] // 2
            card_height = template_image.shape[0] // 2
        
        # Define calibration standards based on detected features
        calibration_standards = self._calculate_calibration_standards(
            grid_analysis, annotation_points, measurement_zones
        )
        
        abstracted_template = AbstractedTemplate(
            template_name="Pikachu_Yellow_Cheeks_Reference",
            card_dimensions=(card_width, card_height),
            grid_analysis=grid_analysis,
            annotation_points=annotation_points,
            measurement_zones=measurement_zones,
            calibration_standards=calibration_standards
        )
        
        logger.info(f"Template abstraction completed: {len(annotation_points)} points, "
                   f"{len(measurement_zones)} zones, grid_detected={grid_analysis.grid_detected}")
        
        return abstracted_template
    
    def _detect_measurement_grid(self, template_image: np.ndarray) -> GridAnalysis:
        """Detect the measurement grid overlay in the template."""
        height, width = template_image.shape[:2]
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for rho, theta in lines[:, 0]:
                # Classify lines as horizontal or vertical
                angle = np.degrees(theta)
                
                if abs(angle) < 10 or abs(angle - 180) < 10:  # Horizontal lines
                    y = int(rho / np.sin(theta)) if np.sin(theta) != 0 else 0
                    if 0 <= y <= height:
                        horizontal_lines.append(y)
                
                elif abs(angle - 90) < 10:  # Vertical lines
                    x = int(rho / np.cos(theta)) if np.cos(theta) != 0 else 0
                    if 0 <= x <= width:
                        vertical_lines.append(x)
        
        # Remove duplicates and sort
        horizontal_lines = sorted(list(set(horizontal_lines)))
        vertical_lines = sorted(list(set(vertical_lines)))
        
        # Estimate grid size from line spacing
        grid_size = 20  # Default fallback
        if len(horizontal_lines) >= 2:
            spacings = [horizontal_lines[i+1] - horizontal_lines[i] for i in range(len(horizontal_lines)-1)]
            grid_size = int(np.median(spacings)) if spacings else grid_size
        
        # Estimate card bounds from the template
        # The card should be in the central area surrounded by grid
        card_left = width // 4 if len(vertical_lines) < 2 else vertical_lines[len(vertical_lines)//4]
        card_right = 3 * width // 4 if len(vertical_lines) < 2 else vertical_lines[3*len(vertical_lines)//4]
        card_top = height // 4 if len(horizontal_lines) < 2 else horizontal_lines[len(horizontal_lines)//4]
        card_bottom = 3 * height // 4 if len(horizontal_lines) < 2 else horizontal_lines[3*len(horizontal_lines)//4]
        
        # Calculate confidence based on grid regularity
        confidence = 0.5  # Default
        if len(horizontal_lines) >= 4 and len(vertical_lines) >= 4:
            confidence = 0.8
        elif len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            confidence = 0.6
        
        grid_detected = confidence > 0.5
        
        return GridAnalysis(
            grid_detected=grid_detected,
            grid_size=grid_size,
            grid_lines_horizontal=horizontal_lines,
            grid_lines_vertical=vertical_lines,
            card_bounds=(card_left, card_top, card_right, card_bottom),
            confidence=confidence
        )
    
    def _detect_annotation_points(self, template_image: np.ndarray, grid_analysis: GridAnalysis) -> List[AnnotationPoint]:
        """Detect annotation points and labels in the template."""
        # For the Pikachu template, we know the approximate locations based on the image
        # This would normally use OCR or template matching, but for now we'll use known positions
        
        height, width = template_image.shape[:2]
        card_bounds = grid_analysis.card_bounds
        
        annotation_points = []
        
        # Based on the reference template image, define key annotation points
        # These are estimated from the visual analysis of the template
        
        # Centering points (margins)
        annotation_points.extend([
            AnnotationPoint("left_margin", (card_bounds[0], height//2), "centering", 0.9),
            AnnotationPoint("right_margin", (card_bounds[2], height//2), "centering", 0.9),
            AnnotationPoint("top_margin", (width//2, card_bounds[1]), "centering", 0.9),
            AnnotationPoint("bottom_margin", (width//2, card_bounds[3]), "centering", 0.9),
        ])
        
        # Edge points
        annotation_points.extend([
            AnnotationPoint("top_edge", (width//2, card_bounds[1]), "edges", 0.8),
            AnnotationPoint("bottom_edge", (width//2, card_bounds[3]), "edges", 0.8),
            AnnotationPoint("left_edge", (card_bounds[0], height//2), "edges", 0.8),
            AnnotationPoint("right_edge", (card_bounds[2], height//2), "edges", 0.8),
        ])
        
        # Corner points
        annotation_points.extend([
            AnnotationPoint("top_left_corner", (card_bounds[0], card_bounds[1]), "corners", 0.9),
            AnnotationPoint("top_right_corner", (card_bounds[2], card_bounds[1]), "corners", 0.9),
            AnnotationPoint("bottom_left_corner", (card_bounds[0], card_bounds[3]), "corners", 0.9),
            AnnotationPoint("bottom_right_corner", (card_bounds[2], card_bounds[3]), "corners", 0.9),
        ])
        
        # Surface analysis points
        card_center_x = (card_bounds[0] + card_bounds[2]) // 2
        card_center_y = (card_bounds[1] + card_bounds[3]) // 2
        
        annotation_points.extend([
            AnnotationPoint("artwork_surface", (card_center_x, card_bounds[1] + (card_bounds[3] - card_bounds[1]) // 3), "surface", 0.8),
            AnnotationPoint("text_surface", (card_center_x, card_bounds[3] - (card_bounds[3] - card_bounds[1]) // 4), "surface", 0.8),
            AnnotationPoint("border_surface", (card_bounds[0] + 20, card_bounds[1] + 20), "surface", 0.7),
        ])
        
        return annotation_points
    
    def _extract_measurement_zones(self, template_image: np.ndarray, grid_analysis: GridAnalysis, 
                                 annotation_points: List[AnnotationPoint]) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Extract measurement zones for different analysis types."""
        measurement_zones = {
            "centering": [],
            "edges": [],
            "corners": [],
            "surface": []
        }
        
        card_bounds = grid_analysis.card_bounds
        card_width = card_bounds[2] - card_bounds[0]
        card_height = card_bounds[3] - card_bounds[1]
        
        # Define zones based on the annotation points
        zone_size = max(20, grid_analysis.grid_size)
        
        for point in annotation_points:
            x, y = point.location
            region_type = point.region_type
            
            # Create a measurement zone around each point
            zone_x = max(0, x - zone_size // 2)
            zone_y = max(0, y - zone_size // 2)
            zone_w = min(zone_size, template_image.shape[1] - zone_x)
            zone_h = min(zone_size, template_image.shape[0] - zone_y)
            
            measurement_zones[region_type].append((zone_x, zone_y, zone_w, zone_h))
        
        # Add additional surface zones for comprehensive analysis
        surface_grid_size = card_width // 4
        for i in range(2):
            for j in range(2):
                surface_x = card_bounds[0] + i * surface_grid_size
                surface_y = card_bounds[1] + j * surface_grid_size
                measurement_zones["surface"].append((surface_x, surface_y, surface_grid_size, surface_grid_size))
        
        return measurement_zones
    
    def _calculate_calibration_standards(self, grid_analysis: GridAnalysis, 
                                       annotation_points: List[AnnotationPoint],
                                       measurement_zones: Dict[str, List[Tuple[int, int, int, int]]]) -> Dict[str, float]:
        """Calculate calibration standards based on the reference template."""
        
        # Base standards for vintage Pokemon cards (based on the Pikachu template)
        standards = {
            # Centering standards (more lenient for vintage)
            "centering_horizontal_tolerance": 0.06,  # 6% tolerance
            "centering_vertical_tolerance": 0.06,
            "centering_max_error": 0.18,  # 18% maximum error
            
            # Edge standards
            "edge_whitening_threshold": 0.20,  # 20% whitening acceptable for vintage
            "edge_wear_threshold": 0.15,
            "edge_acceptable_wear_percent": 25.0,
            
            # Corner standards
            "corner_sharpness_threshold": 0.30,  # Lower sharpness requirements for vintage
            "corner_wear_tolerance": 0.35,
            "corner_damage_threshold": 0.45,
            
            # Surface standards
            "surface_scratch_threshold": 0.035,  # More lenient scratch detection
            "surface_print_line_threshold": 0.30,
            "surface_overall_condition_minimum": 0.55,
            
            # Grid-based adjustments
            "grid_confidence_factor": grid_analysis.confidence,
            "grid_size_factor": grid_analysis.grid_size / 20.0,  # Normalize to expected grid size
        }
        
        # Adjust standards based on grid quality
        if grid_analysis.confidence > 0.8:
            # High confidence grid allows more precise standards
            standards["centering_horizontal_tolerance"] *= 0.9
            standards["centering_vertical_tolerance"] *= 0.9
        elif grid_analysis.confidence < 0.6:
            # Low confidence grid requires more lenient standards
            standards["centering_horizontal_tolerance"] *= 1.2
            standards["centering_vertical_tolerance"] *= 1.2
        
        return standards
    
    def save_abstracted_template(self, abstracted_template: AbstractedTemplate, output_path: str):
        """Save the abstracted template to a JSON file."""
        template_data = {
            "template_name": abstracted_template.template_name,
            "card_dimensions": abstracted_template.card_dimensions,
            "grid_analysis": {
                "grid_detected": abstracted_template.grid_analysis.grid_detected,
                "grid_size": abstracted_template.grid_analysis.grid_size,
                "grid_lines_horizontal": abstracted_template.grid_analysis.grid_lines_horizontal,
                "grid_lines_vertical": abstracted_template.grid_analysis.grid_lines_vertical,
                "card_bounds": abstracted_template.grid_analysis.card_bounds,
                "confidence": abstracted_template.grid_analysis.confidence
            },
            "annotation_points": [
                {
                    "label": point.label,
                    "location": point.location,
                    "region_type": point.region_type,
                    "confidence": point.confidence
                }
                for point in abstracted_template.annotation_points
            ],
            "measurement_zones": abstracted_template.measurement_zones,
            "calibration_standards": abstracted_template.calibration_standards
        }
        
        with open(output_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Abstracted template saved to: {output_path}")
    
    def load_abstracted_template(self, template_path: str) -> Optional[AbstractedTemplate]:
        """Load an abstracted template from a JSON file."""
        try:
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            grid_data = template_data["grid_analysis"]
            grid_analysis = GridAnalysis(
                grid_detected=grid_data["grid_detected"],
                grid_size=grid_data["grid_size"],
                grid_lines_horizontal=grid_data["grid_lines_horizontal"],
                grid_lines_vertical=grid_data["grid_lines_vertical"],
                card_bounds=tuple(grid_data["card_bounds"]),
                confidence=grid_data["confidence"]
            )
            
            annotation_points = [
                AnnotationPoint(
                    label=point["label"],
                    location=tuple(point["location"]),
                    region_type=point["region_type"],
                    confidence=point["confidence"]
                )
                for point in template_data["annotation_points"]
            ]
            
            abstracted_template = AbstractedTemplate(
                template_name=template_data["template_name"],
                card_dimensions=tuple(template_data["card_dimensions"]),
                grid_analysis=grid_analysis,
                annotation_points=annotation_points,
                measurement_zones=template_data["measurement_zones"],
                calibration_standards=template_data["calibration_standards"]
            )
            
            logger.info(f"Abstracted template loaded from: {template_path}")
            return abstracted_template
            
        except Exception as e:
            logger.error(f"Failed to load abstracted template: {e}")
            return None