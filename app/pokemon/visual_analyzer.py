"""
Pokémon-specific visual analysis for card parts, defects, and grading factors.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .card_types import PokemonCardType, PokemonRarity, PokemonCardEra, PokemonCardParts

logger = logging.getLogger(__name__)


@dataclass
class PokemonCardRegions:
    """Detected regions of a Pokémon card for targeted analysis."""
    # Main regions
    artwork: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    name_area: Optional[Tuple[int, int, int, int]] = None
    hp_area: Optional[Tuple[int, int, int, int]] = None
    text_box: Optional[Tuple[int, int, int, int]] = None
    
    # Border and frame regions
    yellow_border: Optional[Tuple[int, int, int, int]] = None
    card_frame: Optional[Tuple[int, int, int, int]] = None
    
    # Bottom info regions
    rarity_symbol: Optional[Tuple[int, int, int, int]] = None
    set_symbol: Optional[Tuple[int, int, int, int]] = None
    card_number: Optional[Tuple[int, int, int, int]] = None
    copyright_area: Optional[Tuple[int, int, int, int]] = None
    
    # Special regions
    energy_symbols: List[Tuple[int, int, int, int]] = None
    weakness_resistance: Optional[Tuple[int, int, int, int]] = None
    retreat_cost: Optional[Tuple[int, int, int, int]] = None
    
    def __post_init__(self):
        if self.energy_symbols is None:
            self.energy_symbols = []


@dataclass
class PokemonDefectAnalysis:
    """Pokémon-specific defect analysis results."""
    # Holographic defects
    holo_scratches: List[Dict[str, Any]] = None
    holo_wear: float = 0.0
    foil_peeling: List[Tuple[int, int, int, int]] = None
    
    # Edge defects specific to Pokémon cards
    edge_whitening: Dict[str, float] = None  # Per edge
    corner_peeling: Dict[str, float] = None  # Per corner
    
    # Print defects
    print_lines: List[Dict[str, Any]] = None
    color_misalignment: float = 0.0
    ink_spots: List[Tuple[int, int, int, int]] = None
    
    # Surface defects
    surface_scratches: List[Dict[str, Any]] = None
    indentations: List[Dict[str, Any]] = None
    staining: List[Tuple[int, int, int, int]] = None
    
    # Pokémon-specific issues
    artwork_damage: float = 0.0
    text_legibility: float = 1.0
    symbol_clarity: float = 1.0
    
    def __post_init__(self):
        if self.holo_scratches is None:
            self.holo_scratches = []
        if self.foil_peeling is None:
            self.foil_peeling = []
        if self.edge_whitening is None:
            self.edge_whitening = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
        if self.corner_peeling is None:
            self.corner_peeling = {"top_left": 0.0, "top_right": 0.0, "bottom_left": 0.0, "bottom_right": 0.0}
        if self.print_lines is None:
            self.print_lines = []
        if self.ink_spots is None:
            self.ink_spots = []
        if self.surface_scratches is None:
            self.surface_scratches = []
        if self.indentations is None:
            self.indentations = []
        if self.staining is None:
            self.staining = []


class PokemonVisualAnalyzer:
    """Specialized visual analyzer for Pokémon cards."""
    
    def __init__(self):
        # Pokémon card standard dimensions (in pixels for analysis)
        self.standard_width = 750
        self.standard_height = 1050
        
        # Region templates for different card types
        self.region_templates = {
            PokemonCardType.POKEMON: {
                "artwork": (0.1, 0.15, 0.8, 0.4),  # Relative coordinates (x, y, w, h)
                "name_area": (0.1, 0.05, 0.8, 0.1),
                "hp_area": (0.7, 0.05, 0.25, 0.1),
                "text_box": (0.1, 0.6, 0.8, 0.25),
                "rarity_symbol": (0.85, 0.88, 0.1, 0.08),
                "set_symbol": (0.75, 0.88, 0.08, 0.08)
            },
            PokemonCardType.TRAINER: {
                "artwork": (0.1, 0.2, 0.8, 0.35),
                "name_area": (0.1, 0.05, 0.8, 0.1),
                "text_box": (0.1, 0.6, 0.8, 0.25),
                "rarity_symbol": (0.85, 0.88, 0.1, 0.08),
                "set_symbol": (0.75, 0.88, 0.08, 0.08)
            },
            PokemonCardType.ENERGY: {
                "artwork": (0.2, 0.2, 0.6, 0.6),
                "name_area": (0.1, 0.05, 0.8, 0.1),
                "rarity_symbol": (0.85, 0.88, 0.1, 0.08),
                "set_symbol": (0.75, 0.88, 0.08, 0.08)
            }
        }
        
        # Era-specific characteristics
        self.era_characteristics = {
            PokemonCardEra.VINTAGE: {
                "has_yellow_border": True,
                "copyright_text": "1998-1999 Wizards",
                "set_symbol_position": "bottom_right"
            },
            PokemonCardEra.SCARLET_VIOLET: {
                "has_yellow_border": False,
                "has_regulation_mark": True,
                "has_qr_code": True
            }
        }

    def analyze_pokemon_card(self, image: np.ndarray, card_type: PokemonCardType,
                           rarity: PokemonRarity, era: PokemonCardEra) -> Tuple[PokemonCardRegions, PokemonDefectAnalysis]:
        """
        Comprehensive analysis of a Pokémon card.
        
        Args:
            image: Card image in RGB format
            card_type: Detected card type
            rarity: Detected rarity
            era: Detected era
            
        Returns:
            Tuple of detected regions and defect analysis
        """
        try:
            # Detect card regions
            regions = self._detect_card_regions(image, card_type)
            
            # Analyze defects
            defects = self._analyze_defects(image, regions, rarity, era)
            
            return regions, defects
            
        except Exception as e:
            logger.error(f"Pokémon card analysis failed: {e}")
            return PokemonCardRegions(), PokemonDefectAnalysis()

    def _detect_card_regions(self, image: np.ndarray, card_type: PokemonCardType) -> PokemonCardRegions:
        """Detect and extract specific regions of the Pokémon card."""
        regions = PokemonCardRegions()
        h, w = image.shape[:2]
        
        # Get template for card type
        template = self.region_templates.get(card_type, self.region_templates[PokemonCardType.POKEMON])
        
        # Convert relative coordinates to absolute
        for region_name, (rel_x, rel_y, rel_w, rel_h) in template.items():
            x = int(rel_x * w)
            y = int(rel_y * h)
            region_w = int(rel_w * w)
            region_h = int(rel_h * h)
            
            setattr(regions, region_name, (x, y, region_w, region_h))
        
        # Detect energy symbols (variable positions)
        regions.energy_symbols = self._detect_energy_symbols(image)
        
        # Detect yellow border for vintage cards
        if self._has_yellow_border(image):
            border_thickness = int(min(w, h) * 0.02)  # 2% of smaller dimension
            regions.yellow_border = (0, 0, w, border_thickness)  # Top border as representative
        
        return regions

    def _analyze_defects(self, image: np.ndarray, regions: PokemonCardRegions,
                        rarity: PokemonRarity, era: PokemonCardEra) -> PokemonDefectAnalysis:
        """Analyze Pokémon-specific defects and condition issues."""
        defects = PokemonDefectAnalysis()
        
        # Holographic analysis for applicable rarities
        if rarity in [PokemonRarity.RARE_HOLO, PokemonRarity.REVERSE_HOLO, 
                     PokemonRarity.ULTRA_RARE, PokemonRarity.SECRET_RARE]:
            self._analyze_holographic_defects(image, regions, defects)
        
        # Edge whitening analysis
        self._analyze_edge_whitening(image, defects)
        
        # Corner analysis
        self._analyze_corner_condition(image, defects)
        
        # Print defect analysis
        self._analyze_print_defects(image, regions, defects, era)
        
        # Surface condition analysis
        self._analyze_surface_condition(image, regions, defects)
        
        # Pokémon-specific quality checks
        self._analyze_pokemon_specific_quality(image, regions, defects)
        
        return defects

    def _analyze_holographic_defects(self, image: np.ndarray, regions: PokemonCardRegions,
                                   defects: PokemonDefectAnalysis):
        """Analyze defects specific to holographic cards with enhanced holo awareness."""
        try:
            # Convert to multiple color spaces for comprehensive holo analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Enhanced holographic area detection
            holo_mask = self._detect_holographic_areas(image)
            
            # Detect scratches specifically in holographic areas
            self._detect_holo_scratches(gray, holo_mask, defects, regions)
            
            # Analyze foil peeling and delamination
            self._detect_foil_peeling(image, holo_mask, defects)
            
            # Assess holographic pattern integrity
            self._assess_holo_pattern_integrity(hsv, holo_mask, defects)
            
            # Detect rainbow/prism effects disruption
            self._detect_rainbow_disruption(lab, holo_mask, defects)
            
            # Calculate overall holographic wear with enhanced metrics
            if defects.holo_scratches or defects.foil_peeling:
                total_scratch_length = sum(s["length"] for s in defects.holo_scratches)
                foil_damage_area = sum(w * h for x, y, w, h in defects.foil_peeling)
                
                # Enhanced wear calculation considering multiple factors
                scratch_factor = min(total_scratch_length / 1000, 1.0)
                peeling_factor = min(foil_damage_area / (image.shape[0] * image.shape[1] * 0.1), 1.0)
                
                defects.holo_wear = min((scratch_factor + peeling_factor) / 2, 1.0)
                
        except Exception as e:
            logger.error(f"Enhanced holographic defect analysis failed: {e}")

    def _detect_holographic_areas(self, image: np.ndarray) -> np.ndarray:
        """Enhanced detection of holographic foil areas."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Holographic areas typically have:
        # 1. High saturation variations
        # 2. Metallic/reflective properties
        # 3. Rainbow/iridescent effects
        
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Create base holo mask for highly saturated, bright areas
        holo_mask = cv2.bitwise_and(
            saturation > 80,   # High saturation
            value > 120        # Good brightness
        )
        
        # Enhance with texture analysis for foil patterns
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local standard deviation (foil creates texture variation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_CLOSE, kernel)
        sqr_diff = (gray.astype(np.float32) - mean) ** 2
        local_std = cv2.morphologyEx(sqr_diff, cv2.MORPH_CLOSE, kernel) ** 0.5
        
        # High texture variation indicates foil
        texture_mask = local_std > 15
        
        # Combine masks
        enhanced_holo_mask = cv2.bitwise_or(holo_mask, texture_mask)
        
        # Clean up the mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        enhanced_holo_mask = cv2.morphologyEx(enhanced_holo_mask.astype(np.uint8), 
                                            cv2.MORPH_CLOSE, kernel_clean)
        
        return enhanced_holo_mask

    def _detect_holo_scratches(self, gray: np.ndarray, holo_mask: np.ndarray, 
                              defects: PokemonDefectAnalysis, regions: PokemonCardRegions):
        """Detect scratches specifically in holographic areas."""
        # Apply holo mask to focus on foil areas
        masked_gray = cv2.bitwise_and(gray, gray, mask=holo_mask)
        
        # Enhanced edge detection for holo scratches
        # Holographic scratches often appear as sharp linear disruptions
        edges = cv2.Canny(masked_gray, 20, 80, apertureSize=3)
        
        # Use probabilistic Hough transform to detect scratch lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                              minLineLength=15, maxLineGap=3)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Filter for significant scratches in holo areas
                if length > 20:
                    # Check if scratch is in holographic area
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    if holo_mask[center_y, center_x] > 0:
                        
                        # Calculate scratch severity based on multiple factors
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        severity = self._calculate_holo_scratch_severity(
                            length, angle, center_x, center_y, masked_gray)
                        
                        scratch_info = {
                            "start": (x1, y1),
                            "end": (x2, y2),
                            "length": length,
                            "angle": angle,
                            "severity": severity,
                            "location": self._get_scratch_location(x1, y1, x2, y2, regions),
                            "in_holo_area": True
                        }
                        defects.holo_scratches.append(scratch_info)

    def _detect_foil_peeling(self, image: np.ndarray, holo_mask: np.ndarray, 
                            defects: PokemonDefectAnalysis):
        """Detect areas where holographic foil is peeling or damaged."""
        # Foil peeling often appears as areas with:
        # 1. Sudden loss of holographic properties
        # 2. Color discontinuities
        # 3. Texture changes
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find edges of holographic areas (potential peeling zones)
        holo_edges = cv2.Canny(holo_mask, 50, 150)
        
        # Dilate to find areas near holo boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge_zones = cv2.dilate(holo_edges, kernel, iterations=2)
        
        # Look for areas with abrupt brightness changes (indicating delamination)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=3)
        gradient_mag = np.abs(gradient)
        
        # High gradient areas near holo edges indicate potential peeling
        peeling_candidates = cv2.bitwise_and(
            gradient_mag > np.percentile(gradient_mag, 85),
            edge_zones > 0
        )
        
        # Find contours of potential peeling areas
        contours, _ = cv2.findContours(peeling_candidates.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Significant peeling area
                x, y, w, h = cv2.boundingRect(contour)
                defects.foil_peeling.append((x, y, w, h))

    def _assess_holo_pattern_integrity(self, hsv: np.ndarray, holo_mask: np.ndarray, 
                                     defects: PokemonDefectAnalysis):
        """Assess the integrity of holographic patterns."""
        # Extract holographic regions
        holo_hsv = cv2.bitwise_and(hsv, hsv, mask=holo_mask)
        
        # Analyze hue distribution in holographic areas
        hue_channel = holo_hsv[:, :, 0]
        hue_values = hue_channel[holo_mask > 0]
        
        if len(hue_values) > 0:
            # Healthy holographic patterns should show good hue diversity
            unique_hues = len(np.unique(hue_values))
            hue_std = np.std(hue_values)
            
            # Calculate pattern integrity score
            # Good holo should have diverse hues (rainbow effect)
            expected_diversity = 30  # Expected number of distinct hues
            diversity_score = min(unique_hues / expected_diversity, 1.0)
            
            expected_variation = 20  # Expected hue standard deviation
            variation_score = min(hue_std / expected_variation, 1.0)
            
            pattern_integrity = (diversity_score + variation_score) / 2
            
            # Store as inverse (damage level)
            defects.holo_wear = max(defects.holo_wear, 1.0 - pattern_integrity)

    def _detect_rainbow_disruption(self, lab: np.ndarray, holo_mask: np.ndarray, 
                                 defects: PokemonDefectAnalysis):
        """Detect disruptions in rainbow/prism effects of holographic foil."""
        # LAB color space is better for analyzing color uniformity
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Apply holo mask
        holo_l = l_channel * (holo_mask / 255.0)
        holo_a = a_channel * (holo_mask / 255.0)
        holo_b = b_channel * (holo_mask / 255.0)
        
        # Calculate color gradients (rainbow effects create smooth gradients)
        grad_a = cv2.Sobel(holo_a, cv2.CV_64F, 1, 1, ksize=3)
        grad_b = cv2.Sobel(holo_b, cv2.CV_64F, 1, 1, ksize=3)
        
        # Sudden changes in color gradients indicate disrupted rainbow effects
        gradient_magnitude = np.sqrt(grad_a**2 + grad_b**2)
        
        # Find areas with abnormally high gradient changes
        threshold = np.percentile(gradient_magnitude[holo_mask > 0], 90)
        disruption_mask = (gradient_magnitude > threshold) & (holo_mask > 0)
        
        # Count disrupted pixels as additional wear factor
        disrupted_pixels = np.sum(disruption_mask)
        total_holo_pixels = np.sum(holo_mask > 0)
        
        if total_holo_pixels > 0:
            disruption_ratio = disrupted_pixels / total_holo_pixels
            defects.holo_wear = max(defects.holo_wear, disruption_ratio)

    def _calculate_holo_scratch_severity(self, length: float, angle: float, 
                                       center_x: int, center_y: int, 
                                       masked_gray: np.ndarray) -> float:
        """Calculate severity of a holographic scratch based on multiple factors."""
        # Base severity from length
        length_severity = min(length / 100, 1.0)
        
        # Angle factor: scratches perpendicular to foil patterns are more visible
        angle_factor = abs(np.sin(np.radians(angle))) + 0.5  # 0.5 to 1.5 range
        
        # Depth factor: analyze local contrast around scratch
        y, x = center_y, center_x
        if 5 <= y < masked_gray.shape[0] - 5 and 5 <= x < masked_gray.shape[1] - 5:
            local_region = masked_gray[y-5:y+5, x-5:x+5]
            if local_region.size > 0:
                contrast = np.std(local_region)
                depth_factor = min(contrast / 30, 1.0)  # Normalize contrast
            else:
                depth_factor = 0.5
        else:
            depth_factor = 0.5
        
        # Combine factors
        severity = length_severity * angle_factor * depth_factor
        return min(severity, 1.0)

    def _analyze_edge_whitening(self, image: np.ndarray, defects: PokemonDefectAnalysis):
        """Analyze edge whitening (white showing on card edges)."""
        try:
            h, w = image.shape[:2]
            edge_thickness = max(2, min(w, h) // 100)  # Adaptive edge thickness
            
            # Define edge regions
            edges = {
                "top": image[:edge_thickness, :],
                "bottom": image[-edge_thickness:, :],
                "left": image[:, :edge_thickness],
                "right": image[:, -edge_thickness:]
            }
            
            for edge_name, edge_region in edges.items():
                # Convert to grayscale
                gray_edge = cv2.cvtColor(edge_region, cv2.COLOR_RGB2GRAY)
                
                # Calculate whiteness (high brightness values indicate whitening)
                white_pixels = np.sum(gray_edge > 200)  # Threshold for "white"
                total_pixels = gray_edge.size
                whitening_ratio = white_pixels / total_pixels
                
                defects.edge_whitening[edge_name] = whitening_ratio
                
        except Exception as e:
            logger.error(f"Edge whitening analysis failed: {e}")

    def _analyze_corner_condition(self, image: np.ndarray, defects: PokemonDefectAnalysis):
        """Analyze corner condition and peeling."""
        try:
            h, w = image.shape[:2]
            corner_size = min(w, h) // 10  # 10% of smaller dimension
            
            # Define corner regions
            corners = {
                "top_left": image[:corner_size, :corner_size],
                "top_right": image[:corner_size, -corner_size:],
                "bottom_left": image[-corner_size:, :corner_size],
                "bottom_right": image[-corner_size:, -corner_size:]
            }
            
            for corner_name, corner_region in corners.items():
                # Analyze corner sharpness and integrity
                gray_corner = cv2.cvtColor(corner_region, cv2.COLOR_RGB2GRAY)
                
                # Edge detection to check corner sharpness
                edges = cv2.Canny(gray_corner, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Sharp corners should have clear edges
                # Damaged corners will have fuzzy or missing edges
                corner_quality = min(edge_density * 5, 1.0)  # Normalize to 0-1
                defects.corner_peeling[corner_name] = 1.0 - corner_quality
                
        except Exception as e:
            logger.error(f"Corner analysis failed: {e}")

    def _analyze_print_defects(self, image: np.ndarray, regions: PokemonCardRegions,
                             defects: PokemonDefectAnalysis, era: PokemonCardEra):
        """Analyze print quality defects."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect horizontal print lines
            # These are common printing defects in Pokémon cards
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Find contours of potential print lines
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Significant size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Print lines are typically long and thin
                    if aspect_ratio > 10:
                        line_info = {
                            "bbox": (x, y, w, h),
                            "severity": min(area / 1000, 1.0),
                            "type": "horizontal_line"
                        }
                        defects.print_lines.append(line_info)
            
            # Color misalignment detection (simplified)
            # Split into color channels and look for misalignment
            b, g, r = cv2.split(image)
            
            # Calculate correlation between color channels
            # Misalignment shows as low correlation
            corr_rg = np.corrcoef(r.flatten(), g.flatten())[0, 1]
            corr_rb = np.corrcoef(r.flatten(), b.flatten())[0, 1]
            corr_gb = np.corrcoef(g.flatten(), b.flatten())[0, 1]
            
            avg_correlation = (corr_rg + corr_rb + corr_gb) / 3
            defects.color_misalignment = max(0, 1.0 - avg_correlation)
            
        except Exception as e:
            logger.error(f"Print defect analysis failed: {e}")

    def _analyze_surface_condition(self, image: np.ndarray, regions: PokemonCardRegions,
                                 defects: PokemonDefectAnalysis):
        """Analyze surface scratches, indentations, and staining."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Surface scratch detection using edge analysis
            edges = cv2.Canny(gray, 20, 60)  # Lower threshold for subtle scratches
            
            # Find small line-like features that could be scratches
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                  minLineLength=10, maxLineGap=3)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if 10 < length < 100:  # Surface scratches are typically smaller
                        scratch_info = {
                            "start": (x1, y1),
                            "end": (x2, y2),
                            "length": length,
                            "severity": min(length / 50, 1.0)
                        }
                        defects.surface_scratches.append(scratch_info)
            
            # Indentation detection using local variance
            kernel = np.ones((5, 5), np.float32) / 25
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_diff = (gray.astype(np.float32) - mean) ** 2
            variance = cv2.filter2D(sqr_diff, -1, kernel)
            
            # High variance areas might indicate indentations
            high_variance_mask = variance > np.percentile(variance, 95)
            contours, _ = cv2.findContours(high_variance_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 500:  # Reasonable size for indentations
                    x, y, w, h = cv2.boundingRect(contour)
                    indent_info = {
                        "bbox": (x, y, w, h),
                        "area": area,
                        "severity": min(area / 100, 1.0)
                    }
                    defects.indentations.append(indent_info)
                    
        except Exception as e:
            logger.error(f"Surface condition analysis failed: {e}")

    def _analyze_pokemon_specific_quality(self, image: np.ndarray, regions: PokemonCardRegions,
                                        defects: PokemonDefectAnalysis):
        """Analyze Pokémon-specific quality factors."""
        try:
            # Artwork damage assessment
            if regions.artwork:
                x, y, w, h = regions.artwork
                artwork_region = image[y:y+h, x:x+w]
                
                # Calculate artwork quality based on variance and edge density
                gray_artwork = cv2.cvtColor(artwork_region, cv2.COLOR_RGB2GRAY)
                artwork_variance = np.var(gray_artwork)
                
                edges = cv2.Canny(gray_artwork, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Good artwork should have reasonable variance and edge detail
                expected_variance = 2000  # Typical for Pokémon artwork
                expected_edge_density = 0.1
                
                variance_score = min(artwork_variance / expected_variance, 1.0)
                edge_score = min(edge_density / expected_edge_density, 1.0)
                
                artwork_quality = (variance_score + edge_score) / 2
                defects.artwork_damage = 1.0 - artwork_quality
            
            # Text legibility assessment
            if regions.text_box:
                x, y, w, h = regions.text_box
                text_region = image[y:y+h, x:x+w]
                
                # Convert to grayscale and assess text clarity
                gray_text = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
                
                # Good text should have high contrast
                text_contrast = np.std(gray_text)
                expected_contrast = 50  # Typical for readable text
                
                defects.text_legibility = min(text_contrast / expected_contrast, 1.0)
            
            # Symbol clarity assessment
            if regions.rarity_symbol:
                x, y, w, h = regions.rarity_symbol
                symbol_region = image[y:y+h, x:x+w]
                
                # Assess symbol sharpness
                gray_symbol = cv2.cvtColor(symbol_region, cv2.COLOR_RGB2GRAY)
                
                # Sharp symbols should have clear edges
                edges = cv2.Canny(gray_symbol, 50, 150)
                edge_ratio = np.sum(edges > 0) / edges.size
                
                defects.symbol_clarity = min(edge_ratio * 10, 1.0)  # Scale appropriately
                
        except Exception as e:
            logger.error(f"Pokémon-specific quality analysis failed: {e}")

    def _detect_energy_symbols(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect energy symbol locations on the card."""
        # Simplified energy symbol detection
        # In practice, would use template matching for specific energy types
        
        energy_symbols = []
        h, w = image.shape[:2]
        
        # Energy symbols are typically in the attack cost area
        search_region = image[int(h*0.6):int(h*0.85), int(w*0.1):int(w*0.9)]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(search_region, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common energy types (simplified)
        energy_colors = {
            "fire": (np.array([0, 100, 100]), np.array([10, 255, 255])),
            "water": (np.array([100, 100, 100]), np.array([130, 255, 255])),
            "grass": (np.array([35, 100, 100]), np.array([85, 255, 255])),
            "electric": (np.array([20, 100, 100]), np.array([35, 255, 255])),
            "psychic": (np.array([140, 100, 100]), np.array([160, 255, 255])),
            "fighting": (np.array([10, 100, 100]), np.array([20, 255, 255])),
            "darkness": (np.array([0, 0, 0]), np.array([180, 255, 50])),
            "metal": (np.array([0, 0, 180]), np.array([180, 30, 255]))
        }
        
        for energy_type, (lower, upper) in energy_colors.items():
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours of potential energy symbols
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # Reasonable size for energy symbols
                    x, y, w_sym, h_sym = cv2.boundingRect(contour)
                    # Adjust coordinates to full image
                    x += int(w * 0.1)
                    y += int(h * 0.6)
                    energy_symbols.append((x, y, w_sym, h_sym))
        
        return energy_symbols

    def _has_yellow_border(self, image: np.ndarray) -> bool:
        """Check if the card has a yellow border (vintage cards)."""
        h, w = image.shape[:2]
        border_thickness = max(2, min(w, h) // 50)
        
        # Sample border regions
        top_border = image[:border_thickness, :]
        
        # Convert to HSV and check for yellow color
        hsv_border = cv2.cvtColor(top_border, cv2.COLOR_RGB2HSV)
        
        # Yellow range in HSV
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        yellow_mask = cv2.inRange(hsv_border, yellow_lower, yellow_upper)
        yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
        
        return yellow_ratio > 0.3  # 30% threshold for yellow border

    def _get_scratch_location(self, x1: int, y1: int, x2: int, y2: int, 
                            regions: PokemonCardRegions) -> str:
        """Determine which card region a scratch is located in."""
        # Calculate scratch center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check which region contains the scratch center
        if regions.artwork:
            ax, ay, aw, ah = regions.artwork
            if ax <= center_x <= ax + aw and ay <= center_y <= ay + ah:
                return "artwork"
        
        if regions.text_box:
            tx, ty, tw, th = regions.text_box
            if tx <= center_x <= tx + tw and ty <= center_y <= ty + th:
                return "text_box"
        
        if regions.name_area:
            nx, ny, nw, nh = regions.name_area
            if nx <= center_x <= nx + nw and ny <= center_y <= ny + nh:
                return "name_area"
        
        return "border"

    def get_quality_score(self, defects: PokemonDefectAnalysis, 
                         card_type: PokemonCardType, rarity: PokemonRarity) -> float:
        """Calculate overall quality score based on defect analysis."""
        score = 100.0
        
        # Holographic defects (more severe for holo cards)
        if defects.holo_scratches:
            holo_penalty = defects.holo_wear * 30  # Up to 30 point penalty
            if rarity in [PokemonRarity.RARE_HOLO, PokemonRarity.ULTRA_RARE]:
                holo_penalty *= 1.5  # More severe for valuable holos
            score -= holo_penalty
        
        # Edge whitening
        avg_edge_whitening = sum(defects.edge_whitening.values()) / 4
        score -= avg_edge_whitening * 20  # Up to 20 point penalty
        
        # Corner peeling
        avg_corner_damage = sum(defects.corner_peeling.values()) / 4
        score -= avg_corner_damage * 25  # Up to 25 point penalty
        
        # Print defects
        print_penalty = len(defects.print_lines) * 5  # 5 points per print line
        print_penalty += defects.color_misalignment * 15  # Up to 15 for misalignment
        score -= min(print_penalty, 30)  # Cap at 30 points
        
        # Surface condition
        surface_penalty = len(defects.surface_scratches) * 2  # 2 points per scratch
        surface_penalty += len(defects.indentations) * 3  # 3 points per indentation
        score -= min(surface_penalty, 20)  # Cap at 20 points
        
        # Pokémon-specific quality
        score -= defects.artwork_damage * 15  # Up to 15 for artwork damage
        score -= (1.0 - defects.text_legibility) * 10  # Up to 10 for text issues
        score -= (1.0 - defects.symbol_clarity) * 5  # Up to 5 for symbol issues
        
        return max(0.0, score)  # Ensure non-negative score