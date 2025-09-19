"""
Pokémon Card Template Processor - Isolates and outlines each card part for precise analysis.
Creates detailed templates and masks for every component of Pokémon cards.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json

from .card_types import PokemonCardType, PokemonRarity, PokemonCardEra

logger = logging.getLogger(__name__)


class CardPartType(Enum):
    """Types of card parts that can be isolated."""
    # Main content areas
    ARTWORK = "artwork"
    NAME_BAR = "name_bar"
    HP_SECTION = "hp_section"
    TYPE_SYMBOLS = "type_symbols"
    TEXT_BOX = "text_box"
    FLAVOR_TEXT = "flavor_text"
    
    # Borders and frames
    OUTER_BORDER = "outer_border"
    INNER_FRAME = "inner_frame"
    YELLOW_BORDER = "yellow_border"  # Vintage cards
    SILVER_BORDER = "silver_border"  # Modern cards
    
    # Bottom section
    RARITY_SYMBOL = "rarity_symbol"
    SET_SYMBOL = "set_symbol"
    CARD_NUMBER = "card_number"
    COPYRIGHT_TEXT = "copyright_text"
    REGULATION_MARK = "regulation_mark"  # Modern cards
    
    # Game mechanics
    ENERGY_COST = "energy_cost"
    ATTACK_SECTIONS = "attack_sections"
    ABILITY_SECTION = "ability_section"
    WEAKNESS_RESISTANCE = "weakness_resistance"
    RETREAT_COST = "retreat_cost"
    
    # Special effects
    HOLOGRAPHIC_AREA = "holographic_area"
    TEXTURE_AREA = "texture_area"
    FOIL_AREA = "foil_area"
    
    # Quality assessment areas
    CORNERS = "corners"
    EDGES = "edges"
    SURFACE_AREAS = "surface_areas"


@dataclass
class CardPartOutline:
    """Detailed outline information for a card part."""
    part_type: CardPartType
    contour: np.ndarray  # OpenCV contour points
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    mask: np.ndarray  # Binary mask for the part
    area: float
    perimeter: float
    confidence: float  # How confident we are in this detection
    
    # Part-specific metadata
    metadata: Dict[str, Any] = None
    
    # Visual characteristics
    average_color: Tuple[int, int, int] = (0, 0, 0)
    color_variance: float = 0.0
    texture_metric: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CardTemplate:
    """Complete template with all isolated card parts."""
    card_type: PokemonCardType
    rarity: PokemonRarity
    era: PokemonCardEra
    
    # Template dimensions
    width: int
    height: int
    
    # All detected parts
    parts: Dict[CardPartType, CardPartOutline]
    
    # Template metadata
    detection_confidence: float
    processing_time: float
    
    # Quality assessment
    template_quality: float  # How well the template matches expected structure
    missing_parts: List[CardPartType]
    extra_parts: List[CardPartType]
    
    def __post_init__(self):
        if not hasattr(self, 'parts'):
            self.parts = {}
        if not hasattr(self, 'missing_parts'):
            self.missing_parts = []
        if not hasattr(self, 'extra_parts'):
            self.extra_parts = []


class PokemonCardTemplateProcessor:
    """Advanced processor for creating detailed Pokémon card templates."""
    
    def __init__(self):
        # Standard Pokémon card dimensions
        self.standard_width = 750
        self.standard_height = 1050
        
        # Template definitions for different card types
        self.templates = self._load_card_templates()
        
        # Color ranges for different card elements (HSV)
        self.color_ranges = {
            "yellow_border": (np.array([20, 100, 100]), np.array([30, 255, 255])),
            "silver_border": (np.array([0, 0, 180]), np.array([180, 30, 255])),
            "energy_fire": (np.array([0, 120, 120]), np.array([10, 255, 255])),
            "energy_water": (np.array([100, 120, 120]), np.array([130, 255, 255])),
            "energy_grass": (np.array([35, 120, 120]), np.array([85, 255, 255])),
            "energy_electric": (np.array([20, 120, 120]), np.array([35, 255, 255])),
            "energy_psychic": (np.array([140, 120, 120]), np.array([160, 255, 255])),
            "energy_fighting": (np.array([10, 120, 120]), np.array([20, 255, 255])),
            "energy_darkness": (np.array([0, 0, 0]), np.array([180, 255, 50])),
            "energy_metal": (np.array([0, 0, 180]), np.array([180, 30, 255])),
            "holo_rainbow": (np.array([0, 50, 150]), np.array([180, 255, 255])),
        }
        
        # Part detection parameters
        self.detection_params = {
            "contour_min_area": 100,
            "contour_max_area": 50000,
            "approximation_epsilon": 0.02,
            "corner_detection_threshold": 50,
            "edge_detection_threshold": (50, 150),
            "holo_detection_threshold": 0.05
        }

    def process_card(self, image: np.ndarray, card_type: Optional[PokemonCardType] = None,
                    rarity: Optional[PokemonRarity] = None, era: Optional[PokemonCardEra] = None) -> CardTemplate:
        """
        Process a card image and create a detailed template with all parts outlined.
        
        Args:
            image: Card image in RGB format
            card_type: Optional pre-detected card type
            rarity: Optional pre-detected rarity
            era: Optional pre-detected era
            
        Returns:
            Complete card template with isolated parts
        """
        import time
        start_time = time.time()
        
        try:
            # Auto-detect card characteristics if not provided
            if card_type is None:
                card_type = self._detect_card_type(image)
            if rarity is None:
                rarity = self._detect_rarity(image)
            if era is None:
                era = self._detect_era(image)
            
            # Create template
            template = CardTemplate(
                card_type=card_type,
                rarity=rarity,
                era=era,
                width=image.shape[1],
                height=image.shape[0],
                parts={},
                detection_confidence=0.0,
                processing_time=0.0,
                template_quality=0.0
            )
            
            # Process each part type
            self._detect_artwork(image, template)
            self._detect_name_bar(image, template)
            self._detect_hp_section(image, template)
            self._detect_text_box(image, template)
            self._detect_borders_and_frames(image, template)
            self._detect_bottom_section(image, template)
            self._detect_game_mechanics(image, template)
            self._detect_special_effects(image, template, rarity)
            self._detect_quality_areas(image, template)
            
            # Calculate template quality
            template.template_quality = self._calculate_template_quality(template)
            template.processing_time = time.time() - start_time
            
            logger.info(f"Processed card template with {len(template.parts)} parts in {template.processing_time:.2f}s")
            return template
            
        except Exception as e:
            logger.error(f"Card template processing failed: {e}")
            return CardTemplate(
                card_type=card_type or PokemonCardType.POKEMON,
                rarity=rarity or PokemonRarity.COMMON,
                era=era or PokemonCardEra.SWORD_SHIELD,
                width=image.shape[1],
                height=image.shape[0],
                parts={},
                detection_confidence=0.0,
                processing_time=time.time() - start_time,
                template_quality=0.0
            )

    def _detect_artwork(self, image: np.ndarray, template: CardTemplate):
        """Detect and outline the main artwork area."""
        try:
            h, w = image.shape[:2]
            
            # Get expected artwork region based on card type
            if template.card_type == PokemonCardType.POKEMON:
                # Pokémon cards have artwork in upper portion
                expected_region = (int(w * 0.08), int(h * 0.12), int(w * 0.84), int(h * 0.42))
            elif template.card_type in [PokemonCardType.TRAINER, PokemonCardType.SUPPORTER]:
                # Trainer cards have different artwork placement
                expected_region = (int(w * 0.08), int(h * 0.18), int(w * 0.84), int(h * 0.35))
            else:
                # Energy cards have centered artwork
                expected_region = (int(w * 0.15), int(h * 0.15), int(w * 0.70), int(h * 0.70))
            
            x, y, reg_w, reg_h = expected_region
            artwork_region = image[y:y+reg_h, x:x+reg_w]
            
            # Create artwork outline using edge detection
            gray = cv2.cvtColor(artwork_region, cv2.COLOR_RGB2GRAY)
            
            # Use adaptive threshold to handle varying lighting
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find the main artwork contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (likely the artwork frame)
                main_contour = max(contours, key=cv2.contourArea)
                
                # Adjust contour coordinates to full image
                main_contour[:, 0, 0] += x
                main_contour[:, 0, 1] += y
                
                # Create mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [main_contour], 255)
                
                # Calculate properties
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                bounding_box = cv2.boundingRect(main_contour)
                
                # Calculate visual characteristics
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                avg_color = cv2.mean(image, mask=mask)[:3]
                color_variance = np.var(masked_image[mask > 0])
                
                # Texture analysis
                gray_masked = cv2.bitwise_and(gray, gray, mask=mask[y:y+reg_h, x:x+reg_w])
                texture_metric = np.std(gray_masked[gray_masked > 0]) if np.any(gray_masked > 0) else 0
                
                artwork_outline = CardPartOutline(
                    part_type=CardPartType.ARTWORK,
                    contour=main_contour,
                    bounding_box=bounding_box,
                    mask=mask,
                    area=area,
                    perimeter=perimeter,
                    confidence=0.8,  # High confidence for main artwork
                    average_color=tuple(map(int, avg_color)),
                    color_variance=color_variance,
                    texture_metric=texture_metric,
                    metadata={
                        "expected_region": expected_region,
                        "detection_method": "edge_detection"
                    }
                )
                
                template.parts[CardPartType.ARTWORK] = artwork_outline
                
        except Exception as e:
            logger.error(f"Artwork detection failed: {e}")

    def _detect_name_bar(self, image: np.ndarray, template: CardTemplate):
        """Detect and outline the Pokémon name area."""
        try:
            h, w = image.shape[:2]
            
            # Name bar is typically at the top
            name_region = (int(w * 0.08), int(h * 0.04), int(w * 0.84), int(h * 0.08))
            x, y, reg_w, reg_h = name_region
            
            name_area = image[y:y+reg_h, x:x+reg_w]
            gray = cv2.cvtColor(name_area, cv2.COLOR_RGB2GRAY)
            
            # Use text detection approach
            # Names are usually on contrasting backgrounds
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find text-like contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create overall name bar outline
            name_contour = np.array([
                [x, y], [x + reg_w, y], [x + reg_w, y + reg_h], [x, y + reg_h]
            ])
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [name_contour], 255)
            
            avg_color = cv2.mean(image, mask=mask)[:3]
            
            name_outline = CardPartOutline(
                part_type=CardPartType.NAME_BAR,
                contour=name_contour,
                bounding_box=(x, y, reg_w, reg_h),
                mask=mask,
                area=reg_w * reg_h,
                perimeter=2 * (reg_w + reg_h),
                confidence=0.9,
                average_color=tuple(map(int, avg_color)),
                metadata={
                    "text_contours": len(contours),
                    "detection_method": "region_based"
                }
            )
            
            template.parts[CardPartType.NAME_BAR] = name_outline
            
        except Exception as e:
            logger.error(f"Name bar detection failed: {e}")

    def _detect_hp_section(self, image: np.ndarray, template: CardTemplate):
        """Detect HP section (for Pokémon cards)."""
        if template.card_type != PokemonCardType.POKEMON:
            return
            
        try:
            h, w = image.shape[:2]
            
            # HP is typically in top right
            hp_region = (int(w * 0.70), int(h * 0.04), int(w * 0.25), int(h * 0.08))
            x, y, reg_w, reg_h = hp_region
            
            hp_contour = np.array([
                [x, y], [x + reg_w, y], [x + reg_w, y + reg_h], [x, y + reg_h]
            ])
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [hp_contour], 255)
            
            avg_color = cv2.mean(image, mask=mask)[:3]
            
            hp_outline = CardPartOutline(
                part_type=CardPartType.HP_SECTION,
                contour=hp_contour,
                bounding_box=(x, y, reg_w, reg_h),
                mask=mask,
                area=reg_w * reg_h,
                perimeter=2 * (reg_w + reg_h),
                confidence=0.8,
                average_color=tuple(map(int, avg_color)),
                metadata={"card_type_specific": True}
            )
            
            template.parts[CardPartType.HP_SECTION] = hp_outline
            
        except Exception as e:
            logger.error(f"HP section detection failed: {e}")

    def _detect_text_box(self, image: np.ndarray, template: CardTemplate):
        """Detect the main text box area."""
        try:
            h, w = image.shape[:2]
            
            # Text box location varies by card type
            if template.card_type == PokemonCardType.POKEMON:
                text_region = (int(w * 0.08), int(h * 0.55), int(w * 0.84), int(h * 0.30))
            else:
                text_region = (int(w * 0.08), int(h * 0.55), int(w * 0.84), int(h * 0.35))
            
            x, y, reg_w, reg_h = text_region
            
            text_contour = np.array([
                [x, y], [x + reg_w, y], [x + reg_w, y + reg_h], [x, y + reg_h]
            ])
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [text_contour], 255)
            
            # Analyze text area for readability
            text_area = image[y:y+reg_h, x:x+reg_w]
            gray_text = cv2.cvtColor(text_area, cv2.COLOR_RGB2GRAY)
            text_contrast = np.std(gray_text)
            
            avg_color = cv2.mean(image, mask=mask)[:3]
            
            text_outline = CardPartOutline(
                part_type=CardPartType.TEXT_BOX,
                contour=text_contour,
                bounding_box=(x, y, reg_w, reg_h),
                mask=mask,
                area=reg_w * reg_h,
                perimeter=2 * (reg_w + reg_h),
                confidence=0.9,
                average_color=tuple(map(int, avg_color)),
                metadata={
                    "text_contrast": text_contrast,
                    "readability_score": min(text_contrast / 50, 1.0)
                }
            )
            
            template.parts[CardPartType.TEXT_BOX] = text_outline
            
        except Exception as e:
            logger.error(f"Text box detection failed: {e}")

    def _detect_borders_and_frames(self, image: np.ndarray, template: CardTemplate):
        """Detect card borders and inner frames."""
        try:
            h, w = image.shape[:2]
            
            # Outer border (entire card edge)
            border_thickness = max(2, min(w, h) // 100)
            outer_border_contour = np.array([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
            ])
            
            outer_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(outer_mask, [outer_border_contour], 255)
            
            outer_outline = CardPartOutline(
                part_type=CardPartType.OUTER_BORDER,
                contour=outer_border_contour,
                bounding_box=(0, 0, w, h),
                mask=outer_mask,
                area=w * h,
                perimeter=2 * (w + h),
                confidence=1.0,
                average_color=(0, 0, 0),
                metadata={"border_thickness": border_thickness}
            )
            
            template.parts[CardPartType.OUTER_BORDER] = outer_outline
            
            # Detect yellow border for vintage cards
            if template.era == PokemonCardEra.VINTAGE:
                self._detect_yellow_border(image, template)
            
            # Detect inner frame
            self._detect_inner_frame(image, template)
            
        except Exception as e:
            logger.error(f"Border detection failed: {e}")

    def _detect_yellow_border(self, image: np.ndarray, template: CardTemplate):
        """Detect yellow border on vintage cards."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Yellow color range
            lower_yellow, upper_yellow = self.color_ranges["yellow_border"]
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Find yellow border contours
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest yellow region (should be the border)
                main_contour = max(contours, key=cv2.contourArea)
                
                yellow_outline = CardPartOutline(
                    part_type=CardPartType.YELLOW_BORDER,
                    contour=main_contour,
                    bounding_box=cv2.boundingRect(main_contour),
                    mask=yellow_mask,
                    area=cv2.contourArea(main_contour),
                    perimeter=cv2.arcLength(main_contour, True),
                    confidence=0.8,
                    average_color=(255, 255, 0),
                    metadata={"era_specific": True, "vintage_indicator": True}
                )
                
                template.parts[CardPartType.YELLOW_BORDER] = yellow_outline
                
        except Exception as e:
            logger.error(f"Yellow border detection failed: {e}")

    def _detect_inner_frame(self, image: np.ndarray, template: CardTemplate):
        """Detect the inner frame that surrounds the artwork."""
        try:
            h, w = image.shape[:2]
            
            # Estimate inner frame based on card type and era
            if template.card_type == PokemonCardType.POKEMON:
                if template.era == PokemonCardEra.VINTAGE:
                    # Vintage cards have more pronounced frames
                    frame_margin = 0.06
                else:
                    frame_margin = 0.04
            else:
                frame_margin = 0.05
            
            frame_x = int(w * frame_margin)
            frame_y = int(h * frame_margin)
            frame_w = int(w * (1 - 2 * frame_margin))
            frame_h = int(h * (1 - 2 * frame_margin))
            
            frame_contour = np.array([
                [frame_x, frame_y], 
                [frame_x + frame_w, frame_y], 
                [frame_x + frame_w, frame_y + frame_h], 
                [frame_x, frame_y + frame_h]
            ])
            
            # Create mask for frame
            frame_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(frame_mask, [frame_contour], True, 255, 2)
            
            frame_outline = CardPartOutline(
                part_type=CardPartType.INNER_FRAME,
                contour=frame_contour,
                bounding_box=(frame_x, frame_y, frame_w, frame_h),
                mask=frame_mask,
                area=frame_w * frame_h,
                perimeter=2 * (frame_w + frame_h),
                confidence=0.7,
                average_color=(128, 128, 128),
                metadata={
                    "frame_margin": frame_margin,
                    "detection_method": "estimated"
                }
            )
            
            template.parts[CardPartType.INNER_FRAME] = frame_outline
            
        except Exception as e:
            logger.error(f"Inner frame detection failed: {e}")

    def _detect_bottom_section(self, image: np.ndarray, template: CardTemplate):
        """Detect bottom section elements (rarity symbol, set symbol, etc.)."""
        try:
            h, w = image.shape[:2]
            
            # Rarity symbol (bottom right)
            rarity_region = (int(w * 0.85), int(h * 0.87), int(w * 0.12), int(h * 0.10))
            self._create_region_outline(image, template, CardPartType.RARITY_SYMBOL, rarity_region)
            
            # Set symbol (near rarity symbol)
            set_region = (int(w * 0.75), int(h * 0.87), int(w * 0.08), int(h * 0.10))
            self._create_region_outline(image, template, CardPartType.SET_SYMBOL, set_region)
            
            # Card number (bottom center/right)
            number_region = (int(w * 0.60), int(h * 0.90), int(w * 0.15), int(h * 0.08))
            self._create_region_outline(image, template, CardPartType.CARD_NUMBER, number_region)
            
            # Copyright (bottom left)
            copyright_region = (int(w * 0.05), int(h * 0.90), int(w * 0.50), int(h * 0.08))
            self._create_region_outline(image, template, CardPartType.COPYRIGHT_TEXT, copyright_region)
            
            # Regulation mark (modern cards, bottom left)
            if template.era in [PokemonCardEra.SWORD_SHIELD, PokemonCardEra.SCARLET_VIOLET]:
                reg_region = (int(w * 0.05), int(h * 0.85), int(w * 0.06), int(h * 0.06))
                self._create_region_outline(image, template, CardPartType.REGULATION_MARK, reg_region)
            
        except Exception as e:
            logger.error(f"Bottom section detection failed: {e}")

    def _detect_game_mechanics(self, image: np.ndarray, template: CardTemplate):
        """Detect game mechanic elements (energy costs, attacks, etc.)."""
        if template.card_type != PokemonCardType.POKEMON:
            return
            
        try:
            h, w = image.shape[:2]
            
            # Energy cost symbols (left side of attacks)
            energy_regions = []
            for i in range(3):  # Up to 3 attacks typically
                attack_y = int(h * (0.58 + i * 0.08))
                energy_region = (int(w * 0.10), attack_y, int(w * 0.15), int(h * 0.06))
                energy_regions.append(energy_region)
            
            # Detect energy symbols in these regions
            for i, region in enumerate(energy_regions):
                self._detect_energy_symbols_in_region(image, template, region, f"attack_{i+1}")
            
            # Weakness/Resistance (bottom of text box)
            wr_region = (int(w * 0.08), int(h * 0.78), int(w * 0.40), int(h * 0.08))
            self._create_region_outline(image, template, CardPartType.WEAKNESS_RESISTANCE, wr_region)
            
            # Retreat cost (bottom right of text box)
            retreat_region = (int(w * 0.75), int(h * 0.78), int(w * 0.17), int(h * 0.08))
            self._create_region_outline(image, template, CardPartType.RETREAT_COST, retreat_region)
            
        except Exception as e:
            logger.error(f"Game mechanics detection failed: {e}")

    def _detect_special_effects(self, image: np.ndarray, template: CardTemplate, rarity: PokemonRarity):
        """Detect holographic, texture, and other special effects."""
        try:
            # Only detect holo effects for relevant rarities
            if rarity in [PokemonRarity.RARE_HOLO, PokemonRarity.REVERSE_HOLO, 
                         PokemonRarity.ULTRA_RARE, PokemonRarity.SECRET_RARE, PokemonRarity.RAINBOW_RARE]:
                self._detect_holographic_areas(image, template, rarity)
            
            # Texture detection for textured cards
            if rarity in [PokemonRarity.ULTRA_RARE, PokemonRarity.SECRET_RARE]:
                self._detect_texture_areas(image, template)
            
        except Exception as e:
            logger.error(f"Special effects detection failed: {e}")

    def _detect_holographic_areas(self, image: np.ndarray, template: CardTemplate, rarity: PokemonRarity):
        """Detect holographic foil areas."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Detect high saturation, varying hue areas (characteristic of holo)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            holo_mask = cv2.bitwise_and(saturation > 120, value > 150)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            holo_mask = cv2.morphologyEx(holo_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find holographic contours
            contours, _ = cv2.findContours(holo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Combine all holo areas into one outline
                all_points = np.vstack(contours)
                hull = cv2.convexHull(all_points)
                
                holo_outline = CardPartOutline(
                    part_type=CardPartType.HOLOGRAPHIC_AREA,
                    contour=hull,
                    bounding_box=cv2.boundingRect(hull),
                    mask=holo_mask,
                    area=cv2.contourArea(hull),
                    perimeter=cv2.arcLength(hull, True),
                    confidence=0.7,
                    average_color=(255, 255, 255),
                    metadata={
                        "rarity": rarity.value,
                        "holo_regions": len(contours),
                        "coverage_percent": np.sum(holo_mask) / holo_mask.size * 100
                    }
                )
                
                template.parts[CardPartType.HOLOGRAPHIC_AREA] = holo_outline
                
        except Exception as e:
            logger.error(f"Holographic area detection failed: {e}")

    def _detect_texture_areas(self, image: np.ndarray, template: CardTemplate):
        """Detect textured surface areas."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate local standard deviation (texture indicator)
            kernel = np.ones((5, 5), np.float32) / 25
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_diff = (gray.astype(np.float32) - mean) ** 2
            local_std = cv2.filter2D(sqr_diff, -1, kernel) ** 0.5
            
            # High standard deviation indicates texture
            texture_mask = local_std > np.percentile(local_std, 90)
            
            # Find texture contours
            contours, _ = cv2.findContours(texture_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_texture = max(contours, key=cv2.contourArea)
                
                texture_outline = CardPartOutline(
                    part_type=CardPartType.TEXTURE_AREA,
                    contour=main_texture,
                    bounding_box=cv2.boundingRect(main_texture),
                    mask=texture_mask.astype(np.uint8) * 255,
                    area=cv2.contourArea(main_texture),
                    perimeter=cv2.arcLength(main_texture, True),
                    confidence=0.6,
                    texture_metric=np.mean(local_std),
                    metadata={
                        "texture_strength": np.mean(local_std),
                        "texture_coverage": np.sum(texture_mask) / texture_mask.size * 100
                    }
                )
                
                template.parts[CardPartType.TEXTURE_AREA] = texture_outline
                
        except Exception as e:
            logger.error(f"Texture area detection failed: {e}")

    def _detect_quality_areas(self, image: np.ndarray, template: CardTemplate):
        """Detect areas important for quality assessment."""
        try:
            h, w = image.shape[:2]
            
            # Corner regions for corner wear assessment
            corner_size = min(w, h) // 8
            corners = {
                "top_left": (0, 0, corner_size, corner_size),
                "top_right": (w - corner_size, 0, corner_size, corner_size),
                "bottom_left": (0, h - corner_size, corner_size, corner_size),
                "bottom_right": (w - corner_size, h - corner_size, corner_size, corner_size)
            }
            
            corner_contours = []
            for corner_name, (x, y, cw, ch) in corners.items():
                corner_contour = np.array([
                    [x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]
                ])
                corner_contours.append(corner_contour)
            
            # Combine all corners into one mask
            corner_mask = np.zeros((h, w), dtype=np.uint8)
            for contour in corner_contours:
                cv2.fillPoly(corner_mask, [contour], 255)
            
            all_corners = np.vstack(corner_contours)
            
            corner_outline = CardPartOutline(
                part_type=CardPartType.CORNERS,
                contour=all_corners,
                bounding_box=(0, 0, w, h),
                mask=corner_mask,
                area=4 * corner_size * corner_size,
                perimeter=4 * 4 * corner_size,
                confidence=1.0,
                metadata={
                    "corner_size": corner_size,
                    "individual_corners": corners
                }
            )
            
            template.parts[CardPartType.CORNERS] = corner_outline
            
            # Edge regions for edge wear assessment
            edge_thickness = max(3, min(w, h) // 50)
            edge_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Top and bottom edges
            edge_mask[:edge_thickness, :] = 255
            edge_mask[-edge_thickness:, :] = 255
            # Left and right edges
            edge_mask[:, :edge_thickness] = 255
            edge_mask[:, -edge_thickness:] = 255
            
            edge_contour = np.array([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
            ])
            
            edge_outline = CardPartOutline(
                part_type=CardPartType.EDGES,
                contour=edge_contour,
                bounding_box=(0, 0, w, h),
                mask=edge_mask,
                area=2 * w * edge_thickness + 2 * h * edge_thickness,
                perimeter=2 * (w + h),
                confidence=1.0,
                metadata={"edge_thickness": edge_thickness}
            )
            
            template.parts[CardPartType.EDGES] = edge_outline
            
        except Exception as e:
            logger.error(f"Quality areas detection failed: {e}")

    def _create_region_outline(self, image: np.ndarray, template: CardTemplate, 
                             part_type: CardPartType, region: Tuple[int, int, int, int]):
        """Create an outline for a simple rectangular region."""
        try:
            x, y, w, h = region
            img_h, img_w = image.shape[:2]
            
            # Ensure region is within image bounds
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            contour = np.array([
                [x, y], [x + w, y], [x + w, y + h], [x, y + h]
            ])
            
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            avg_color = cv2.mean(image, mask=mask)[:3]
            
            outline = CardPartOutline(
                part_type=part_type,
                contour=contour,
                bounding_box=(x, y, w, h),
                mask=mask,
                area=w * h,
                perimeter=2 * (w + h),
                confidence=0.8,
                average_color=tuple(map(int, avg_color))
            )
            
            template.parts[part_type] = outline
            
        except Exception as e:
            logger.error(f"Region outline creation failed for {part_type}: {e}")

    def _detect_energy_symbols_in_region(self, image: np.ndarray, template: CardTemplate, 
                                       region: Tuple[int, int, int, int], attack_name: str):
        """Detect energy symbols in a specific region."""
        try:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            
            energy_found = []
            
            # Check for each energy type
            for energy_type, (lower, upper) in self.color_ranges.items():
                if energy_type.startswith("energy_"):
                    mask = cv2.inRange(hsv_roi, lower, upper)
                    
                    # Find energy symbol contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if 20 < area < 300:  # Reasonable size for energy symbol
                            # Adjust coordinates to full image
                            adjusted_contour = contour.copy()
                            adjusted_contour[:, 0, 0] += x
                            adjusted_contour[:, 0, 1] += y
                            
                            energy_found.append({
                                "type": energy_type,
                                "contour": adjusted_contour,
                                "area": area
                            })
            
            # Create combined energy cost outline if symbols found
            if energy_found:
                all_contours = [e["contour"] for e in energy_found]
                all_points = np.vstack(all_contours)
                hull = cv2.convexHull(all_points)
                
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [hull], 255)
                
                energy_outline = CardPartOutline(
                    part_type=CardPartType.ENERGY_COST,
                    contour=hull,
                    bounding_box=cv2.boundingRect(hull),
                    mask=mask,
                    area=cv2.contourArea(hull),
                    perimeter=cv2.arcLength(hull, True),
                    confidence=0.7,
                    metadata={
                        "attack": attack_name,
                        "energy_symbols": energy_found,
                        "symbol_count": len(energy_found)
                    }
                )
                
                # Use unique key for multiple energy cost regions
                key = f"{CardPartType.ENERGY_COST.value}_{attack_name}"
                template.parts[key] = energy_outline
                
        except Exception as e:
            logger.error(f"Energy symbol detection failed: {e}")

    def _calculate_template_quality(self, template: CardTemplate) -> float:
        """Calculate overall template quality score."""
        try:
            # Expected parts for different card types
            expected_parts = {
                PokemonCardType.POKEMON: [
                    CardPartType.ARTWORK, CardPartType.NAME_BAR, CardPartType.HP_SECTION,
                    CardPartType.TEXT_BOX, CardPartType.OUTER_BORDER, CardPartType.RARITY_SYMBOL
                ],
                PokemonCardType.TRAINER: [
                    CardPartType.ARTWORK, CardPartType.NAME_BAR, CardPartType.TEXT_BOX,
                    CardPartType.OUTER_BORDER, CardPartType.RARITY_SYMBOL
                ],
                PokemonCardType.ENERGY: [
                    CardPartType.ARTWORK, CardPartType.NAME_BAR, CardPartType.OUTER_BORDER,
                    CardPartType.RARITY_SYMBOL
                ]
            }
            
            expected = expected_parts.get(template.card_type, expected_parts[PokemonCardType.POKEMON])
            
            # Calculate detection rate
            detected = sum(1 for part in expected if part in template.parts)
            detection_rate = detected / len(expected)
            
            # Calculate average confidence
            confidences = [part.confidence for part in template.parts.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Overall quality score
            quality = (detection_rate * 0.7 + avg_confidence * 0.3)
            
            # Update missing parts
            template.missing_parts = [part for part in expected if part not in template.parts]
            
            return quality
            
        except Exception as e:
            logger.error(f"Template quality calculation failed: {e}")
            return 0.5

    def _load_card_templates(self) -> Dict:
        """Load pre-defined card templates (simplified for demo)."""
        return {
            "pokemon": {
                "required_parts": ["artwork", "name_bar", "hp_section", "text_box"],
                "optional_parts": ["energy_cost", "weakness_resistance", "retreat_cost"]
            },
            "trainer": {
                "required_parts": ["artwork", "name_bar", "text_box"],
                "optional_parts": []
            },
            "energy": {
                "required_parts": ["artwork", "name_bar"],
                "optional_parts": []
            }
        }

    def _detect_card_type(self, image: np.ndarray) -> PokemonCardType:
        """Simple card type detection (placeholder)."""
        # In practice, would use OCR and visual analysis
        return PokemonCardType.POKEMON

    def _detect_rarity(self, image: np.ndarray) -> PokemonRarity:
        """Simple rarity detection (placeholder)."""
        # In practice, would use symbol detection and visual analysis
        return PokemonRarity.COMMON

    def _detect_era(self, image: np.ndarray) -> PokemonCardEra:
        """Simple era detection (placeholder)."""
        # In practice, would analyze border color and design elements
        return PokemonCardEra.SWORD_SHIELD

    def export_template(self, template: CardTemplate, output_path: str):
        """Export template with all outlines to a file."""
        try:
            template_data = {
                "card_type": template.card_type.value,
                "rarity": template.rarity.value,
                "era": template.era.value,
                "dimensions": {"width": template.width, "height": template.height},
                "parts": {},
                "quality": template.template_quality,
                "processing_time": template.processing_time
            }
            
            for part_type, outline in template.parts.items():
                if isinstance(part_type, CardPartType):
                    part_key = part_type.value
                else:
                    part_key = str(part_type)
                    
                template_data["parts"][part_key] = {
                    "bounding_box": outline.bounding_box,
                    "area": outline.area,
                    "perimeter": outline.perimeter,
                    "confidence": outline.confidence,
                    "average_color": outline.average_color,
                    "metadata": outline.metadata
                }
            
            with open(output_path, 'w') as f:
                json.dump(template_data, f, indent=2)
                
            logger.info(f"Template exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Template export failed: {e}")

    def visualize_template(self, image: np.ndarray, template: CardTemplate) -> np.ndarray:
        """Create a visualization of the template with all outlines."""
        try:
            visualization = image.copy()
            
            # Color map for different part types
            colors = {
                CardPartType.ARTWORK: (255, 0, 0),      # Red
                CardPartType.NAME_BAR: (0, 255, 0),     # Green
                CardPartType.HP_SECTION: (0, 0, 255),   # Blue
                CardPartType.TEXT_BOX: (255, 255, 0),   # Yellow
                CardPartType.RARITY_SYMBOL: (255, 0, 255), # Magenta
                CardPartType.ENERGY_COST: (0, 255, 255),   # Cyan
                CardPartType.HOLOGRAPHIC_AREA: (255, 165, 0), # Orange
                CardPartType.CORNERS: (128, 0, 128),     # Purple
                CardPartType.EDGES: (255, 192, 203)      # Pink
            }
            
            # Draw all part outlines
            for part_type, outline in template.parts.items():
                if isinstance(part_type, CardPartType):
                    color = colors.get(part_type, (128, 128, 128))
                else:
                    color = (128, 128, 128)  # Default gray
                
                # Draw contour
                cv2.drawContours(visualization, [outline.contour], -1, color, 2)
                
                # Draw bounding box
                x, y, w, h = outline.bounding_box
                cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 1)
                
                # Add label
                label = part_type.value if isinstance(part_type, CardPartType) else str(part_type)
                cv2.putText(visualization, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, color, 1)
            
            return visualization
            
        except Exception as e:
            logger.error(f"Template visualization failed: {e}")
            return image


# Example usage and utility functions
def process_pokemon_card(image_path: str) -> CardTemplate:
    """Process a Pokémon card image and return detailed template."""
    processor = PokemonCardTemplateProcessor()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process card
    template = processor.process_card(image_rgb)
    
    return template


def create_part_isolation_masks(template: CardTemplate) -> Dict[str, np.ndarray]:
    """Create individual masks for each detected part."""
    masks = {}
    
    for part_type, outline in template.parts.items():
        part_name = part_type.value if isinstance(part_type, CardPartType) else str(part_type)
        masks[part_name] = outline.mask
    
    return masks