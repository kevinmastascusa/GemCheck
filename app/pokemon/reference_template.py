"""
Reference Card Template Processor for calibrated Pokemon card analysis.
Uses the reference template to establish baseline measurements and grading standards.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json

from .card_types import PokemonCardType, PokemonRarity, PokemonCardEra
from .template_processor import CardPartType, CardPartOutline

logger = logging.getLogger(__name__)


@dataclass
class ReferencePoint:
    """A reference point for analysis calibration."""
    name: str
    location: Tuple[int, int]  # (x, y) coordinates
    purpose: str  # What this point is used for
    tolerance: float = 0.02  # Acceptable variance from ideal


@dataclass
class ReferenceTemplate:
    """Complete reference template with calibration points."""
    card_name: str
    card_set: str
    card_era: PokemonCardEra
    template_image: np.ndarray
    
    # Reference measurements in pixels
    card_width: int
    card_height: int
    
    # Grading reference points
    centering_points: List[ReferencePoint]
    edge_points: List[ReferencePoint] 
    corner_points: List[ReferencePoint]
    surface_points: List[ReferencePoint]
    
    # Grid system for measurements
    grid_size: int  # Size of measurement grid squares
    grid_origin: Tuple[int, int]  # Top-left corner of grid
    
    # Ideal measurements for this template
    ideal_centering: Dict[str, float]  # Expected centering values
    edge_tolerances: Dict[str, float]  # Acceptable edge variances
    corner_standards: Dict[str, float]  # Corner quality thresholds
    surface_thresholds: Dict[str, float]  # Surface defect limits


class ReferenceTemplateProcessor:
    """Processes reference templates for calibrated grading analysis."""
    
    def __init__(self, template_path: str = "resources/ML TEMPLATES/reference.webp"):
        self.template_path = Path(template_path)
        self.reference_template = None
        self.calibration_data = {}
        
        # Load the reference template
        self.load_reference_template()
    
    def load_reference_template(self) -> bool:
        """Load and process the reference template image."""
        try:
            if not self.template_path.exists():
                logger.error(f"Reference template not found: {self.template_path}")
                return False
            
            # Load the template image
            template_image = cv2.imread(str(self.template_path))
            template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"Loaded reference template: {template_image.shape}")
            
            # Create the reference template based on the Pikachu card
            self.reference_template = self._create_pikachu_reference_template(template_image)
            
            logger.info("Reference template processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reference template: {e}")
            return False
    
    def _create_pikachu_reference_template(self, template_image: np.ndarray) -> ReferenceTemplate:
        """Create reference template from the Pikachu Yellow Cheeks card."""
        height, width = template_image.shape[:2]
        
        # Analyze the grid overlay to determine measurements
        grid_info = self._analyze_measurement_grid(template_image)
        
        # Define reference points based on the template annotations
        centering_points = [
            ReferencePoint("left_margin", (grid_info["card_left"], height//2), "Left margin measurement"),
            ReferencePoint("right_margin", (grid_info["card_right"], height//2), "Right margin measurement"),
            ReferencePoint("top_margin", (width//2, grid_info["card_top"]), "Top margin measurement"),
            ReferencePoint("bottom_margin", (width//2, grid_info["card_bottom"]), "Bottom margin measurement"),
        ]
        
        edge_points = [
            ReferencePoint("top_edge", (width//2, grid_info["card_top"]), "Top edge analysis point"),
            ReferencePoint("bottom_edge", (width//2, grid_info["card_bottom"]), "Bottom edge analysis point"),
            ReferencePoint("left_edge", (grid_info["card_left"], height//2), "Left edge analysis point"),
            ReferencePoint("right_edge", (grid_info["card_right"], height//2), "Right edge analysis point"),
        ]
        
        corner_points = [
            ReferencePoint("top_left", (grid_info["card_left"], grid_info["card_top"]), "Top-left corner"),
            ReferencePoint("top_right", (grid_info["card_right"], grid_info["card_top"]), "Top-right corner"),
            ReferencePoint("bottom_left", (grid_info["card_left"], grid_info["card_bottom"]), "Bottom-left corner"),
            ReferencePoint("bottom_right", (grid_info["card_right"], grid_info["card_bottom"]), "Bottom-right corner"),
        ]
        
        surface_points = [
            ReferencePoint("artwork_center", (grid_info["artwork_x"], grid_info["artwork_y"]), "Artwork surface analysis"),
            ReferencePoint("text_area", (grid_info["text_x"], grid_info["text_y"]), "Text area surface analysis"),
            ReferencePoint("border_area", (grid_info["border_x"], grid_info["border_y"]), "Border surface analysis"),
        ]
        
        # Calculate ideal measurements for this vintage card
        card_width = grid_info["card_right"] - grid_info["card_left"]
        card_height = grid_info["card_bottom"] - grid_info["card_top"]
        
        ideal_centering = {
            "horizontal_tolerance": 0.05,  # 5% tolerance for vintage cards
            "vertical_tolerance": 0.05,
            "max_error": 0.15  # Maximum acceptable centering error
        }
        
        edge_tolerances = {
            "whitening_threshold": 0.20,  # Higher tolerance for vintage cards
            "wear_threshold": 0.15,
            "acceptable_wear_percent": 20.0
        }
        
        corner_standards = {
            "sharpness_threshold": 0.25,  # Lower standards for vintage
            "wear_tolerance": 0.30,
            "damage_threshold": 0.40
        }
        
        surface_thresholds = {
            "scratch_threshold": 0.03,  # More lenient for vintage
            "print_line_threshold": 0.25,
            "overall_condition_minimum": 0.60
        }
        
        return ReferenceTemplate(
            card_name="Pikachu Yellow Cheeks",
            card_set="Pokemon Game",
            card_era=PokemonCardEra.VINTAGE,
            template_image=template_image,
            card_width=card_width,
            card_height=card_height,
            centering_points=centering_points,
            edge_points=edge_points,
            corner_points=corner_points,
            surface_points=surface_points,
            grid_size=grid_info["grid_size"],
            grid_origin=grid_info["grid_origin"],
            ideal_centering=ideal_centering,
            edge_tolerances=edge_tolerances,
            corner_standards=corner_standards,
            surface_thresholds=surface_thresholds
        )
    
    def _analyze_measurement_grid(self, template_image: np.ndarray) -> Dict[str, int]:
        """Analyze the measurement grid overlay to extract reference measurements."""
        height, width = template_image.shape[:2]
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)
        
        # Detect the grid lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # Find the card boundaries within the grid
        # Based on the template, estimate these values
        grid_size = 20  # Approximate grid square size in pixels
        
        # Estimate card boundaries (these would be detected from the actual template)
        card_left = width // 6  # Approximate based on template
        card_right = width - width // 6
        card_top = height // 6
        card_bottom = height - height // 6
        
        # Define key analysis areas
        artwork_x = (card_left + card_right) // 2
        artwork_y = card_top + (card_bottom - card_top) // 3
        
        text_x = (card_left + card_right) // 2 
        text_y = card_bottom - (card_bottom - card_top) // 4
        
        border_x = card_left + (card_right - card_left) // 10
        border_y = card_top + (card_bottom - card_top) // 10
        
        return {
            "grid_size": grid_size,
            "grid_origin": (card_left - 100, card_top - 100),
            "card_left": card_left,
            "card_right": card_right,
            "card_top": card_top,
            "card_bottom": card_bottom,
            "artwork_x": artwork_x,
            "artwork_y": artwork_y,
            "text_x": text_x,
            "text_y": text_y,
            "border_x": border_x,
            "border_y": border_y
        }
    
    def calibrate_analysis(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Calibrate analysis parameters using the reference template."""
        if not self.reference_template:
            logger.error("No reference template loaded")
            return {}
        
        try:
            # Resize test image to match template if needed
            template_height, template_width = self.reference_template.template_image.shape[:2]
            test_height, test_width = test_image.shape[:2]
            
            # Calculate scaling factors
            scale_x = test_width / template_width
            scale_y = test_height / template_height
            
            # Apply reference template standards with scaling
            calibration = {
                "scale_factors": {"x": scale_x, "y": scale_y},
                "centering_standards": self._scale_centering_standards(scale_x, scale_y),
                "edge_standards": self._scale_edge_standards(scale_x, scale_y),
                "corner_standards": self._scale_corner_standards(scale_x, scale_y),
                "surface_standards": self._scale_surface_standards(scale_x, scale_y),
                "reference_points": self._scale_reference_points(scale_x, scale_y)
            }
            
            logger.info(f"Calibration completed with scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
            return calibration
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return {}
    
    def _scale_centering_standards(self, scale_x: float, scale_y: float) -> Dict[str, float]:
        """Scale centering standards based on image size."""
        standards = self.reference_template.ideal_centering.copy()
        
        # Centering tolerances are percentages, so they don't need scaling
        # But we can adjust them based on image quality/resolution
        resolution_factor = min(scale_x, scale_y)
        
        if resolution_factor < 0.5:
            # Lower resolution, be more lenient
            standards["horizontal_tolerance"] *= 1.5
            standards["vertical_tolerance"] *= 1.5
            standards["max_error"] *= 1.3
        elif resolution_factor > 2.0:
            # Higher resolution, be more strict
            standards["horizontal_tolerance"] *= 0.8
            standards["vertical_tolerance"] *= 0.8
            standards["max_error"] *= 0.9
        
        return standards
    
    def _scale_edge_standards(self, scale_x: float, scale_y: float) -> Dict[str, float]:
        """Scale edge analysis standards."""
        standards = self.reference_template.edge_tolerances.copy()
        
        # Adjust for resolution
        resolution_factor = min(scale_x, scale_y)
        
        if resolution_factor < 0.5:
            standards["whitening_threshold"] *= 1.4
            standards["wear_threshold"] *= 1.3
        elif resolution_factor > 2.0:
            standards["whitening_threshold"] *= 0.7
            standards["wear_threshold"] *= 0.8
        
        return standards
    
    def _scale_corner_standards(self, scale_x: float, scale_y: float) -> Dict[str, float]:
        """Scale corner analysis standards."""
        standards = self.reference_template.corner_standards.copy()
        
        # Adjust for resolution
        resolution_factor = min(scale_x, scale_y)
        
        if resolution_factor < 0.5:
            standards["sharpness_threshold"] *= 1.5
            standards["wear_tolerance"] *= 1.4
        elif resolution_factor > 2.0:
            standards["sharpness_threshold"] *= 0.6
            standards["wear_tolerance"] *= 0.7
        
        return standards
    
    def _scale_surface_standards(self, scale_x: float, scale_y: float) -> Dict[str, float]:
        """Scale surface analysis standards."""
        standards = self.reference_template.surface_thresholds.copy()
        
        # Adjust for resolution
        resolution_factor = min(scale_x, scale_y)
        
        if resolution_factor < 0.5:
            standards["scratch_threshold"] *= 1.6
            standards["print_line_threshold"] *= 1.3
        elif resolution_factor > 2.0:
            standards["scratch_threshold"] *= 0.5
            standards["print_line_threshold"] *= 0.8
        
        return standards
    
    def _scale_reference_points(self, scale_x: float, scale_y: float) -> Dict[str, List[Tuple[int, int]]]:
        """Scale reference points to match test image size."""
        scaled_points = {}
        
        for category, points in [
            ("centering", self.reference_template.centering_points),
            ("edges", self.reference_template.edge_points),
            ("corners", self.reference_template.corner_points),
            ("surface", self.reference_template.surface_points)
        ]:
            scaled_points[category] = [
                (int(point.location[0] * scale_x), int(point.location[1] * scale_y))
                for point in points
            ]
        
        return scaled_points
    
    def create_calibrated_overlay(self, test_image: np.ndarray, calibration_data: Dict[str, Any]) -> np.ndarray:
        """Create an overlay showing calibrated reference points on the test image."""
        overlay = test_image.copy()
        
        if not calibration_data:
            return overlay
        
        reference_points = calibration_data.get("reference_points", {})
        
        # Colors for different analysis areas
        colors = {
            "centering": (255, 0, 0),    # Red
            "edges": (0, 255, 0),        # Green  
            "corners": (0, 0, 255),      # Blue
            "surface": (255, 255, 0)     # Yellow
        }
        
        # Draw reference points
        for category, points in reference_points.items():
            color = colors.get(category, (255, 255, 255))
            
            for point in points:
                cv2.circle(overlay, point, 5, color, -1)
                cv2.circle(overlay, point, 8, (0, 0, 0), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        if "centering" in reference_points and len(reference_points["centering"]) >= 4:
            points = reference_points["centering"]
            cv2.putText(overlay, "CENTERING", (points[0][0] - 50, points[0][1] - 10), 
                       font, font_scale, (255, 0, 0), thickness)
        
        if "edges" in reference_points and len(reference_points["edges"]) >= 2:
            points = reference_points["edges"]
            cv2.putText(overlay, "EDGES", (points[0][0] - 30, points[0][1] - 10), 
                       font, font_scale, (0, 255, 0), thickness)
        
        if "corners" in reference_points and len(reference_points["corners"]) >= 1:
            points = reference_points["corners"]
            cv2.putText(overlay, "CORNERS", (points[0][0] - 40, points[0][1] - 10), 
                       font, font_scale, (0, 0, 255), thickness)
        
        if "surface" in reference_points and len(reference_points["surface"]) >= 1:
            points = reference_points["surface"]
            cv2.putText(overlay, "SURFACE", (points[0][0] - 40, points[0][1] + 25), 
                       font, font_scale, (255, 255, 0), thickness)
        
        return overlay
    
    def get_reference_info(self) -> Dict[str, Any]:
        """Get information about the loaded reference template."""
        if not self.reference_template:
            return {}
        
        return {
            "card_name": self.reference_template.card_name,
            "card_set": self.reference_template.card_set,
            "card_era": self.reference_template.card_era.value,
            "dimensions": {
                "width": self.reference_template.card_width,
                "height": self.reference_template.card_height
            },
            "standards": {
                "centering": self.reference_template.ideal_centering,
                "edges": self.reference_template.edge_tolerances,
                "corners": self.reference_template.corner_standards,
                "surface": self.reference_template.surface_thresholds
            },
            "reference_points_count": {
                "centering": len(self.reference_template.centering_points),
                "edges": len(self.reference_template.edge_points),
                "corners": len(self.reference_template.corner_points),
                "surface": len(self.reference_template.surface_points)
            }
        }