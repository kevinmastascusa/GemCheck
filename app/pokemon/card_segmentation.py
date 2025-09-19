"""
Pokemon Card Segmentation System
Segments card images into their key components for detailed analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class CardComponent(Enum):
    """Different components of a Pokemon card."""
    OUTER_BORDER = "outer_border"           # Card edge/border
    INNER_FRAME = "inner_frame"             # Yellow/colored frame around content
    ARTWORK = "artwork"                     # Pokemon illustration area
    NAME_BAR = "name_bar"                   # Pokemon name section
    HP_SECTION = "hp_section"               # HP and type indicators
    ATTACK_TEXT = "attack_text"             # Attack descriptions
    FLAVOR_TEXT = "flavor_text"             # Flavor text area
    BOTTOM_BAR = "bottom_bar"               # Bottom info (rarity, set, etc.)
    ENERGY_SYMBOLS = "energy_symbols"       # Energy cost symbols
    WEAKNESS_RESISTANCE = "weakness_resistance"  # W/R section
    RETREAT_COST = "retreat_cost"           # Retreat cost area
    SET_SYMBOL = "set_symbol"               # Set symbol location
    RARITY_SYMBOL = "rarity_symbol"         # Rarity symbol
    CARD_NUMBER = "card_number"             # Card number text
    HOLOGRAPHIC_AREA = "holographic_area"   # Holographic pattern area


@dataclass
class ComponentMask:
    """A segmented component with its mask and metadata."""
    component: CardComponent
    mask: np.ndarray                        # Binary mask for the component
    bounding_box: Tuple[int, int, int, int] # (x, y, width, height)
    confidence: float                       # Confidence in segmentation
    area_percentage: float                  # Percentage of card area
    properties: Dict[str, Any]              # Component-specific properties


@dataclass
class SegmentationResult:
    """Complete card segmentation result."""
    original_image: np.ndarray
    components: Dict[CardComponent, ComponentMask]
    card_bounds: Tuple[int, int, int, int]  # Overall card boundaries
    preprocessing_info: Dict[str, Any]
    segmentation_quality: float            # Overall segmentation confidence


class PokemonCardSegmenter:
    """
    Segments Pokemon cards into their constituent components using computer vision.
    Handles different card eras, types, and layouts.
    """
    
    def __init__(self):
        self.era_templates = self._initialize_era_templates()
        self.color_ranges = self._initialize_color_ranges()
        
    def _initialize_era_templates(self) -> Dict[str, Dict]:
        """Initialize templates for different Pokemon card eras."""
        return {
            "vintage": {
                "outer_border_thickness": 0.02,     # 2% of card width
                "inner_frame_thickness": 0.015,     # 1.5% of card width
                "name_bar_height": 0.08,            # 8% of card height
                "artwork_ratio": 0.35,              # 35% of card height
                "bottom_bar_height": 0.25,          # 25% of card height
                "expected_colors": {
                    "yellow_frame": ([20, 100, 100], [30, 255, 255]),  # HSV ranges
                    "blue_frame": ([100, 100, 100], [120, 255, 255]),
                    "red_frame": ([0, 100, 100], [10, 255, 255])
                }
            },
            "modern": {
                "outer_border_thickness": 0.015,
                "inner_frame_thickness": 0.012,
                "name_bar_height": 0.07,
                "artwork_ratio": 0.38,
                "bottom_bar_height": 0.22,
                "expected_colors": {
                    "silver_frame": ([0, 0, 180], [180, 30, 255]),
                    "gold_frame": ([15, 100, 100], [25, 255, 255])
                }
            }
        }
    
    def _initialize_color_ranges(self) -> Dict[str, Tuple]:
        """Initialize color ranges for different card elements."""
        return {
            # HSV color ranges for common Pokemon card elements
            "yellow_frame": ([20, 100, 100], [30, 255, 255]),
            "blue_water": ([100, 100, 50], [120, 255, 255]),
            "red_fire": ([0, 100, 100], [10, 255, 255]),
            "green_grass": ([40, 100, 50], [80, 255, 255]),
            "purple_psychic": ([120, 100, 50], [150, 255, 255]),
            "brown_fighting": ([10, 100, 50], [20, 255, 255]),
            "white_text": ([0, 0, 200], [180, 30, 255]),
            "black_text": ([0, 0, 0], [180, 30, 50]),
            "silver_frame": ([0, 0, 150], [180, 30, 200])
        }
    
    def segment_card(self, image: np.ndarray, card_era: str = "vintage") -> SegmentationResult:
        """
        Segment a Pokemon card into its components.
        
        Args:
            image: Card image (RGB format)
            card_era: Era of the card ("vintage", "modern", etc.)
            
        Returns:
            SegmentationResult with all identified components
        """
        logger.info(f"Segmenting Pokemon card (era: {card_era})")
        
        # Preprocess the image
        preprocessed_image, preprocessing_info = self._preprocess_image(image)
        
        # Detect overall card boundaries
        card_bounds = self._detect_card_bounds(preprocessed_image)
        
        # Extract card region
        card_image = self._extract_card_region(preprocessed_image, card_bounds)
        
        # Segment components based on era
        era_template = self.era_templates.get(card_era, self.era_templates["vintage"])
        components = self._segment_components(card_image, era_template, card_era)
        
        # Calculate segmentation quality
        segmentation_quality = self._calculate_segmentation_quality(components)
        
        logger.info(f"Segmentation completed: {len(components)} components, quality: {segmentation_quality:.2f}")
        
        return SegmentationResult(
            original_image=image,
            components=components,
            card_bounds=card_bounds,
            preprocessing_info=preprocessing_info,
            segmentation_quality=segmentation_quality
        )
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess the image for better segmentation."""
        preprocessing_info = {}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # Noise reduction
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Edge enhancement
        edges = cv2.Canny(enhanced_gray, 50, 150)
        
        preprocessing_info.update({
            "original_shape": image.shape,
            "enhancement_applied": True,
            "noise_reduction": True,
            "edge_detection": True
        })
        
        # Combine enhanced image
        enhanced_image = denoised.copy()
        
        return enhanced_image, preprocessing_info
    
    def _detect_card_bounds(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect the overall boundaries of the card."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to find card outline
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest rectangular contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Validate aspect ratio (Pokemon cards are roughly 2.5:3.5)
            aspect_ratio = w / h
            if 0.6 < aspect_ratio < 0.8:  # Valid card aspect ratio
                return (x, y, w, h)
        
        # Fallback: use most of the image
        h, w = image.shape[:2]
        margin = min(w, h) // 20  # 5% margin
        return (margin, margin, w - 2*margin, h - 2*margin)
    
    def _extract_card_region(self, image: np.ndarray, card_bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract and normalize the card region."""
        x, y, w, h = card_bounds
        card_region = image[y:y+h, x:x+w]
        
        # Standardize size for consistent analysis
        standard_height = 600
        standard_width = int(standard_height * 0.72)  # Pokemon card aspect ratio
        
        card_normalized = cv2.resize(card_region, (standard_width, standard_height))
        
        return card_normalized
    
    def _segment_components(self, card_image: np.ndarray, era_template: Dict, card_era: str) -> Dict[CardComponent, ComponentMask]:
        """Segment the card into its components."""
        components = {}
        h, w = card_image.shape[:2]
        
        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(card_image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(card_image, cv2.COLOR_RGB2GRAY)
        
        # 1. Segment outer border
        components[CardComponent.OUTER_BORDER] = self._segment_outer_border(card_image, era_template)
        
        # 2. Segment inner frame (colored frame around content)
        components[CardComponent.INNER_FRAME] = self._segment_inner_frame(card_image, hsv, era_template)
        
        # 3. Segment artwork area
        components[CardComponent.ARTWORK] = self._segment_artwork_area(card_image, era_template)
        
        # 4. Segment name bar
        components[CardComponent.NAME_BAR] = self._segment_name_bar(card_image, era_template)
        
        # 5. Segment HP section
        components[CardComponent.HP_SECTION] = self._segment_hp_section(card_image, era_template)
        
        # 6. Segment text areas
        components[CardComponent.ATTACK_TEXT] = self._segment_attack_text(card_image, era_template)
        components[CardComponent.FLAVOR_TEXT] = self._segment_flavor_text(card_image, era_template)
        
        # 7. Segment bottom information bar
        components[CardComponent.BOTTOM_BAR] = self._segment_bottom_bar(card_image, era_template)
        
        # 8. Segment symbols and small elements
        components.update(self._segment_symbols_and_elements(card_image, era_template))
        
        # 9. Detect holographic areas (if applicable)
        components[CardComponent.HOLOGRAPHIC_AREA] = self._segment_holographic_area(card_image)
        
        return components
    
    def _segment_outer_border(self, image: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the outer border of the card."""
        h, w = image.shape[:2]
        border_thickness = int(era_template["outer_border_thickness"] * w)
        
        # Create border mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Border regions: top, bottom, left, right
        mask[:border_thickness, :] = 255  # Top
        mask[-border_thickness:, :] = 255  # Bottom
        mask[:, :border_thickness] = 255  # Left
        mask[:, -border_thickness:] = 255  # Right
        
        # Calculate bounding box
        bbox = (0, 0, w, h)
        
        return ComponentMask(
            component=CardComponent.OUTER_BORDER,
            mask=mask,
            bounding_box=bbox,
            confidence=0.95,
            area_percentage=(np.sum(mask > 0) / (h * w)) * 100,
            properties={"border_thickness": border_thickness}
        )
    
    def _segment_inner_frame(self, image: np.ndarray, hsv: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the colored inner frame."""
        h, w = image.shape[:2]
        
        # Look for colored frames based on era
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        detected_color = None
        
        for color_name, (lower, upper) in era_template["expected_colors"].items():
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
            # Check if this color forms a frame-like structure
            if self._is_frame_like(color_mask):
                frame_mask = cv2.bitwise_or(frame_mask, color_mask)
                detected_color = color_name
                break
        
        # If no colored frame detected, estimate based on position
        if np.sum(frame_mask) < (h * w * 0.05):  # Less than 5% of image
            frame_thickness = int(era_template["inner_frame_thickness"] * w)
            
            # Create estimated frame mask
            frame_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(frame_mask, (frame_thickness, frame_thickness), 
                         (w - frame_thickness, h - frame_thickness), 255, frame_thickness)
        
        # Find bounding box
        contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, fw, fh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = (x, y, fw, fh)
        else:
            bbox = (0, 0, w, h)
        
        return ComponentMask(
            component=CardComponent.INNER_FRAME,
            mask=frame_mask,
            bounding_box=bbox,
            confidence=0.8 if detected_color else 0.6,
            area_percentage=(np.sum(frame_mask > 0) / (h * w)) * 100,
            properties={"detected_color": detected_color}
        )
    
    def _is_frame_like(self, mask: np.ndarray) -> bool:
        """Check if a mask represents a frame-like structure."""
        h, w = mask.shape
        
        # Check if mask forms a rectangular frame
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is roughly rectangular and along edges
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        
        # Frame should be near edges and have reasonable size
        edge_proximity = min(x, y, w - (x + cw), h - (y + ch))
        size_ratio = (cw * ch) / (w * h)
        
        return edge_proximity < min(w, h) * 0.1 and 0.1 < size_ratio < 0.9
    
    def _segment_artwork_area(self, image: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the Pokemon artwork/illustration area."""
        h, w = image.shape[:2]
        
        # Artwork is typically in the upper portion of the card
        artwork_height_ratio = era_template["artwork_ratio"]
        name_bar_height = int(era_template["name_bar_height"] * h)
        
        # Define artwork region
        artwork_top = name_bar_height
        artwork_bottom = artwork_top + int(artwork_height_ratio * h)
        artwork_left = int(0.1 * w)  # 10% margin from sides
        artwork_right = int(0.9 * w)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[artwork_top:artwork_bottom, artwork_left:artwork_right] = 255
        
        # Refine using color and texture analysis
        artwork_region = image[artwork_top:artwork_bottom, artwork_left:artwork_right]
        refined_mask = self._refine_artwork_mask(artwork_region)
        
        # Apply refined mask
        mask[artwork_top:artwork_bottom, artwork_left:artwork_right] = refined_mask
        
        bbox = (artwork_left, artwork_top, artwork_right - artwork_left, artwork_bottom - artwork_top)
        
        return ComponentMask(
            component=CardComponent.ARTWORK,
            mask=mask,
            bounding_box=bbox,
            confidence=0.85,
            area_percentage=(np.sum(mask > 0) / (h * w)) * 100,
            properties={"artwork_region": "upper_card"}
        )
    
    def _refine_artwork_mask(self, artwork_region: np.ndarray) -> np.ndarray:
        """Refine the artwork mask using texture and color analysis."""
        if artwork_region.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        
        h, w = artwork_region.shape[:2]
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(artwork_region, cv2.COLOR_RGB2GRAY)
        
        # Use edge density to identify artwork vs frame
        edges = cv2.Canny(gray, 30, 100)
        
        # Create initial mask based on edge density
        kernel = np.ones((5, 5), np.uint8)
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel / 25)
        
        # Areas with higher edge density are likely artwork
        artwork_threshold = np.percentile(edge_density, 70)  # Top 30% edge density
        mask = (edge_density > artwork_threshold).astype(np.uint8) * 255
        
        # Clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _segment_name_bar(self, image: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the Pokemon name bar area."""
        h, w = image.shape[:2]
        
        # Name bar is typically at the top of the card content
        name_bar_height = int(era_template["name_bar_height"] * h)
        
        # Define name bar region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:name_bar_height, :] = 255
        
        bbox = (0, 0, w, name_bar_height)
        
        return ComponentMask(
            component=CardComponent.NAME_BAR,
            mask=mask,
            bounding_box=bbox,
            confidence=0.9,
            area_percentage=(np.sum(mask > 0) / (h * w)) * 100,
            properties={"position": "top"}
        )
    
    def _segment_hp_section(self, image: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the HP and type indicator section."""
        h, w = image.shape[:2]
        
        # HP section is typically in the upper right
        name_bar_height = int(era_template["name_bar_height"] * h)
        hp_width = int(0.3 * w)  # Approximately 30% of card width
        
        # Define HP section region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:name_bar_height, -hp_width:] = 255
        
        bbox = (w - hp_width, 0, hp_width, name_bar_height)
        
        return ComponentMask(
            component=CardComponent.HP_SECTION,
            mask=mask,
            bounding_box=bbox,
            confidence=0.85,
            area_percentage=(np.sum(mask > 0) / (h * w)) * 100,
            properties={"position": "upper_right"}
        )
    
    def _segment_attack_text(self, image: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the attack text area."""
        h, w = image.shape[:2]
        
        # Attack text is in the middle section of the card
        artwork_height = int(era_template["artwork_ratio"] * h)
        name_bar_height = int(era_template["name_bar_height"] * h)
        bottom_bar_height = int(era_template["bottom_bar_height"] * h)
        
        attack_top = name_bar_height + artwork_height
        attack_bottom = h - bottom_bar_height
        
        # Define attack text region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[attack_top:attack_bottom, :] = 255
        
        bbox = (0, attack_top, w, attack_bottom - attack_top)
        
        return ComponentMask(
            component=CardComponent.ATTACK_TEXT,
            mask=mask,
            bounding_box=bbox,
            confidence=0.8,
            area_percentage=(np.sum(mask > 0) / (h * w)) * 100,
            properties={"text_type": "attacks_abilities"}
        )
    
    def _segment_flavor_text(self, image: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the flavor text area (if present)."""
        h, w = image.shape[:2]
        
        # Flavor text is typically in the lower portion of attack text area
        bottom_bar_height = int(era_template["bottom_bar_height"] * h)
        flavor_height = int(0.08 * h)  # Approximately 8% of card height
        
        flavor_top = h - bottom_bar_height - flavor_height
        flavor_bottom = h - bottom_bar_height
        
        # Define flavor text region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[flavor_top:flavor_bottom, :] = 255
        
        bbox = (0, flavor_top, w, flavor_height)
        
        return ComponentMask(
            component=CardComponent.FLAVOR_TEXT,
            mask=mask,
            bounding_box=bbox,
            confidence=0.7,  # Lower confidence as not all cards have flavor text
            area_percentage=(np.sum(mask > 0) / (h * w)) * 100,
            properties={"text_type": "flavor", "optional": True}
        )
    
    def _segment_bottom_bar(self, image: np.ndarray, era_template: Dict) -> ComponentMask:
        """Segment the bottom information bar."""
        h, w = image.shape[:2]
        
        # Bottom bar contains weakness, resistance, retreat cost, etc.
        bottom_bar_height = int(era_template["bottom_bar_height"] * h)
        
        # Define bottom bar region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[-bottom_bar_height:, :] = 255
        
        bbox = (0, h - bottom_bar_height, w, bottom_bar_height)
        
        return ComponentMask(
            component=CardComponent.BOTTOM_BAR,
            mask=mask,
            bounding_box=bbox,
            confidence=0.9,
            area_percentage=(np.sum(mask > 0) / (h * w)) * 100,
            properties={"contains": "weakness_resistance_retreat_info"}
        )
    
    def _segment_symbols_and_elements(self, image: np.ndarray, era_template: Dict) -> Dict[CardComponent, ComponentMask]:
        """Segment small symbols and elements."""
        h, w = image.shape[:2]
        components = {}
        
        # Convert to grayscale for symbol detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use template matching and size filtering for small elements
        # For now, create placeholder regions based on typical positions
        
        # Set symbol (typically bottom right)
        set_symbol_size = int(0.04 * min(w, h))  # 4% of smaller dimension
        set_x = int(0.85 * w)
        set_y = int(0.9 * h)
        
        set_mask = np.zeros((h, w), dtype=np.uint8)
        set_mask[set_y:set_y+set_symbol_size, set_x:set_x+set_symbol_size] = 255
        
        components[CardComponent.SET_SYMBOL] = ComponentMask(
            component=CardComponent.SET_SYMBOL,
            mask=set_mask,
            bounding_box=(set_x, set_y, set_symbol_size, set_symbol_size),
            confidence=0.6,
            area_percentage=(np.sum(set_mask > 0) / (h * w)) * 100,
            properties={"position": "bottom_right", "symbol_type": "set"}
        )
        
        # Rarity symbol (typically bottom right, near set symbol)
        rarity_x = int(0.75 * w)
        rarity_y = int(0.9 * h)
        
        rarity_mask = np.zeros((h, w), dtype=np.uint8)
        rarity_mask[rarity_y:rarity_y+set_symbol_size, rarity_x:rarity_x+set_symbol_size] = 255
        
        components[CardComponent.RARITY_SYMBOL] = ComponentMask(
            component=CardComponent.RARITY_SYMBOL,
            mask=rarity_mask,
            bounding_box=(rarity_x, rarity_y, set_symbol_size, set_symbol_size),
            confidence=0.6,
            area_percentage=(np.sum(rarity_mask > 0) / (h * w)) * 100,
            properties={"position": "bottom_right", "symbol_type": "rarity"}
        )
        
        return components
    
    def _segment_holographic_area(self, image: np.ndarray) -> ComponentMask:
        """Segment holographic/foil areas."""
        h, w = image.shape[:2]
        
        # Convert to HSV for better holo detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Holographic areas typically have high saturation and varied hues
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Create holo mask based on saturation and brightness patterns
        holo_mask = np.zeros((h, w), dtype=np.uint8)
        
        # High saturation areas
        high_sat_mask = saturation > 150
        
        # Areas with high variance in hue (rainbow effect)
        hue = hsv[:, :, 0].astype(np.float32)
        kernel = np.ones((5, 5), np.float32) / 25
        hue_mean = cv2.filter2D(hue, -1, kernel)
        hue_variance = cv2.filter2D((hue - hue_mean)**2, -1, kernel)
        high_variance_mask = hue_variance > 500
        
        # Combine conditions
        holo_candidate_mask = high_sat_mask & high_variance_mask
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        holo_mask = cv2.morphologyEx(holo_candidate_mask.astype(np.uint8) * 255, 
                                    cv2.MORPH_CLOSE, kernel)
        
        # Find bounding box
        contours, _ = cv2.findContours(holo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, hw, hh = cv2.boundingRect(largest_contour)
            bbox = (x, y, hw, hh)
            confidence = 0.7
        else:
            bbox = (0, 0, w, h)
            confidence = 0.3
        
        return ComponentMask(
            component=CardComponent.HOLOGRAPHIC_AREA,
            mask=holo_mask,
            bounding_box=bbox,
            confidence=confidence,
            area_percentage=(np.sum(holo_mask > 0) / (h * w)) * 100,
            properties={"detection_method": "saturation_variance"}
        )
    
    def _calculate_segmentation_quality(self, components: Dict[CardComponent, ComponentMask]) -> float:
        """Calculate overall segmentation quality."""
        if not components:
            return 0.0
        
        # Calculate average confidence
        confidences = [comp.confidence for comp in components.values()]
        avg_confidence = np.mean(confidences)
        
        # Check coverage (components should cover most of the card)
        total_coverage = sum(comp.area_percentage for comp in components.values())
        coverage_score = min(1.0, total_coverage / 80.0)  # Expect ~80% coverage
        
        # Check for essential components
        essential_components = [
            CardComponent.OUTER_BORDER,
            CardComponent.ARTWORK,
            CardComponent.NAME_BAR,
            CardComponent.BOTTOM_BAR
        ]
        
        essential_found = sum(1 for comp in essential_components if comp in components)
        essential_score = essential_found / len(essential_components)
        
        # Combined quality score
        quality = (avg_confidence * 0.4 + coverage_score * 0.3 + essential_score * 0.3)
        
        return quality
    
    def visualize_segmentation(self, segmentation_result: SegmentationResult) -> np.ndarray:
        """Create a visualization of the segmentation result."""
        image = segmentation_result.original_image.copy()
        h, w = image.shape[:2]
        
        # Create overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color mapping for different components
        component_colors = {
            CardComponent.OUTER_BORDER: (255, 0, 0),      # Red
            CardComponent.INNER_FRAME: (0, 255, 0),       # Green
            CardComponent.ARTWORK: (0, 0, 255),           # Blue
            CardComponent.NAME_BAR: (255, 255, 0),        # Yellow
            CardComponent.HP_SECTION: (255, 0, 255),      # Magenta
            CardComponent.ATTACK_TEXT: (0, 255, 255),     # Cyan
            CardComponent.BOTTOM_BAR: (128, 128, 128),    # Gray
            CardComponent.HOLOGRAPHIC_AREA: (255, 128, 0) # Orange
        }
        
        # Apply colored masks
        for component, component_mask in segmentation_result.components.items():
            if component in component_colors:
                color = component_colors[component]
                mask_3d = np.stack([component_mask.mask] * 3, axis=-1)
                colored_mask = (mask_3d > 0) * np.array(color)
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask.astype(np.uint8), 0.3, 0)
        
        # Combine with original image
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        return result
    
    def save_component_masks(self, segmentation_result: SegmentationResult, output_dir: str):
        """Save individual component masks as separate images."""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for component, component_mask in segmentation_result.components.items():
            mask_filename = f"{component.value}_mask.png"
            mask_path = output_path / mask_filename
            
            cv2.imwrite(str(mask_path), component_mask.mask)
            
            # Also save the component region
            if np.sum(component_mask.mask) > 0:
                x, y, w, h = component_mask.bounding_box
                component_region = segmentation_result.original_image[y:y+h, x:x+w]
                region_filename = f"{component.value}_region.png"
                region_path = output_path / region_filename
                
                component_region_bgr = cv2.cvtColor(component_region, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(region_path), component_region_bgr)
        
        logger.info(f"Component masks saved to: {output_path}")
    
    def extract_component_features(self, segmentation_result: SegmentationResult) -> Dict[str, Any]:
        """Extract features from each segmented component."""
        features = {}
        
        for component, component_mask in segmentation_result.components.items():
            if np.sum(component_mask.mask) == 0:
                continue
            
            # Extract region
            x, y, w, h = component_mask.bounding_box
            region = segmentation_result.original_image[y:y+h, x:x+w]
            region_mask = component_mask.mask[y:y+h, x:x+w]
            
            if region.size == 0:
                continue
            
            # Calculate features
            component_features = {
                "area": np.sum(region_mask > 0),
                "area_percentage": component_mask.area_percentage,
                "bounding_box": component_mask.bounding_box,
                "confidence": component_mask.confidence,
                "mean_color": np.mean(region[region_mask > 0], axis=0).tolist() if np.sum(region_mask > 0) > 0 else [0, 0, 0],
                "color_std": np.std(region[region_mask > 0], axis=0).tolist() if np.sum(region_mask > 0) > 0 else [0, 0, 0]
            }
            
            # Add component-specific features
            if component == CardComponent.ARTWORK:
                component_features.update(self._extract_artwork_features(region, region_mask))
            elif component == CardComponent.HOLOGRAPHIC_AREA:
                component_features.update(self._extract_holo_features(region, region_mask))
            
            features[component.value] = component_features
        
        return features
    
    def _extract_artwork_features(self, region: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Extract specific features from artwork region."""
        if region.size == 0 or np.sum(mask) == 0:
            return {}
        
        # Convert to different color spaces
        hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # Texture analysis
        edges = cv2.Canny(gray_region, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color distribution
        hue_values = hsv_region[mask > 0, 0]
        saturation_values = hsv_region[mask > 0, 1]
        
        return {
            "edge_density": edge_density,
            "dominant_hue": np.median(hue_values) if len(hue_values) > 0 else 0,
            "saturation_mean": np.mean(saturation_values) if len(saturation_values) > 0 else 0,
            "color_variety": len(np.unique(hue_values)) if len(hue_values) > 0 else 0
        }
    
    def _extract_holo_features(self, region: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Extract specific features from holographic region."""
        if region.size == 0 or np.sum(mask) == 0:
            return {}
        
        hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        # Holographic pattern analysis
        hue_values = hsv_region[mask > 0, 0]
        saturation_values = hsv_region[mask > 0, 1]
        value_values = hsv_region[mask > 0, 2]
        
        return {
            "hue_variance": np.var(hue_values) if len(hue_values) > 0 else 0,
            "saturation_mean": np.mean(saturation_values) if len(saturation_values) > 0 else 0,
            "rainbow_effect": len(np.unique(hue_values)) / len(hue_values) if len(hue_values) > 0 else 0,
            "brightness_variation": np.std(value_values) if len(value_values) > 0 else 0
        }