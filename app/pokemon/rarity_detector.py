"""
Advanced Pokémon card rarity detection using visual analysis and pattern recognition.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import re

from .card_types import PokemonRarity, PokemonCardEra, PokemonCardType

logger = logging.getLogger(__name__)


@dataclass
class RarityFeatures:
    """Visual features used for rarity detection."""
    # Holographic patterns
    has_holographic: bool = False
    holo_intensity: float = 0.0
    holo_pattern_type: str = "none"
    
    # Foil characteristics
    has_reverse_holo: bool = False
    has_rainbow_foil: bool = False
    has_gold_foil: bool = False
    has_texture: bool = False
    
    # Symbol detection
    rarity_symbol_detected: bool = False
    rarity_symbol_type: str = "unknown"
    rarity_symbol_confidence: float = 0.0
    
    # Card numbering
    card_number: Optional[str] = None
    is_secret_rare_number: bool = False
    total_cards_in_set: Optional[int] = None
    
    # Visual characteristics
    border_type: str = "silver"  # silver, gold, yellow, etc.
    has_special_effects: bool = False
    shine_intensity: float = 0.0
    
    # Text indicators
    text_indicators: List[str] = None
    
    def __post_init__(self):
        if self.text_indicators is None:
            self.text_indicators = []


class PokemonRarityDetector:
    """Advanced detector for Pokémon card rarities using computer vision."""
    
    def __init__(self):
        # Rarity symbol templates (simplified representation)
        self.rarity_symbols = {
            "common": {"shape": "circle", "filled": True, "color": "black"},
            "uncommon": {"shape": "diamond", "filled": True, "color": "black"},
            "rare": {"shape": "star", "filled": False, "color": "black"},
            "rare_holo": {"shape": "star", "filled": True, "color": "black"},
            "ultra_rare": {"shape": "star", "filled": True, "color": "white"},
            "secret_rare": {"shape": "star", "filled": True, "color": "gold"}
        }
        
        # Holographic pattern characteristics
        self.holo_patterns = {
            "cosmos": {"description": "Cosmos holo pattern (stars/sparkles)"},
            "linear": {"description": "Linear holo pattern (parallel lines)"},
            "crosshatch": {"description": "Crosshatch holo pattern"},
            "pokeball": {"description": "Pokéball holo pattern"},
            "lightning": {"description": "Lightning holo pattern"},
            "leaf": {"description": "Leaf holo pattern"},
            "rainbow": {"description": "Rainbow holo pattern"}
        }
        
        # Era-specific rarity information
        self.era_rarities = {
            PokemonCardEra.VINTAGE: {
                "available_rarities": [
                    PokemonRarity.COMMON, PokemonRarity.UNCOMMON, 
                    PokemonRarity.RARE, PokemonRarity.RARE_HOLO,
                    PokemonRarity.FIRST_EDITION, PokemonRarity.SHADOWLESS
                ],
                "special_characteristics": ["yellow_border", "shadowless_text"]
            },
            PokemonCardEra.SCARLET_VIOLET: {
                "available_rarities": [
                    PokemonRarity.COMMON, PokemonRarity.UNCOMMON, PokemonRarity.RARE,
                    PokemonRarity.RARE_HOLO, PokemonRarity.ULTRA_RARE, 
                    PokemonRarity.SECRET_RARE, PokemonRarity.RAINBOW_RARE,
                    PokemonRarity.ALTERNATE_ART, PokemonRarity.FULL_ART
                ],
                "special_characteristics": ["texture", "rainbow_foil", "gold_foil"]
            }
        }

    def detect_rarity(self, image: np.ndarray, text_content: str = "", 
                     era: Optional[PokemonCardEra] = None) -> Tuple[PokemonRarity, RarityFeatures]:
        """
        Detect the rarity of a Pokémon card using visual and text analysis.
        
        Args:
            image: Card image in RGB format
            text_content: OCR text from the card
            era: Detected card era for context
            
        Returns:
            Tuple of detected rarity and feature analysis
        """
        try:
            features = RarityFeatures()
            
            # Extract visual features
            self._analyze_holographic_patterns(image, features)
            self._detect_rarity_symbol(image, features)
            self._analyze_foil_characteristics(image, features)
            self._detect_card_number(text_content, features)
            self._analyze_text_indicators(text_content, features)
            
            # Determine rarity based on features
            rarity = self._classify_rarity(features, era)
            
            logger.info(f"Detected rarity: {rarity.value} with confidence: {features.rarity_symbol_confidence:.2f}")
            return rarity, features
            
        except Exception as e:
            logger.error(f"Rarity detection failed: {e}")
            return PokemonRarity.COMMON, RarityFeatures()

    def _analyze_holographic_patterns(self, image: np.ndarray, features: RarityFeatures):
        """Analyze holographic foil patterns in the card with enhanced detection."""
        try:
            # Convert to multiple color spaces for comprehensive analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Enhanced holographic detection using multiple approaches
            holo_mask = self._detect_advanced_holographic_areas(image)
            
            # Calculate holographic coverage
            total_pixels = image.shape[0] * image.shape[1]
            holo_pixels = np.sum(holo_mask > 0)
            holo_coverage = holo_pixels / total_pixels
            
            features.holo_intensity = holo_coverage
            features.has_holographic = holo_coverage > 0.03  # Lower threshold for better detection
            
            if features.has_holographic:
                # Enhanced pattern analysis
                features.holo_pattern_type = self._identify_advanced_holo_pattern(image, holo_mask)
                
                # Analyze foil characteristics
                self._analyze_foil_properties(image, holo_mask, features)
                
                # Check for reverse holo with improved detection
                artwork_region = self._extract_artwork_region(image)
                if artwork_region is not None:
                    artwork_holo = self._calculate_advanced_holo_in_region(artwork_region)
                    background_holo = self._calculate_background_holo(image, artwork_region)
                    
                    # Reverse holo: high background foil, low artwork foil
                    features.has_reverse_holo = (background_holo > 0.08 and artwork_holo < 0.02) or \
                                               (holo_coverage > 0.15 and artwork_holo < 0.03)
                
                # Detect specific foil types
                self._detect_special_foil_types(image, holo_mask, features)
            
        except Exception as e:
            logger.error(f"Enhanced holographic analysis failed: {e}")

    def _detect_advanced_holographic_areas(self, image: np.ndarray) -> np.ndarray:
        """Advanced detection of holographic areas using multiple techniques."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Method 1: High saturation and brightness
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        sat_mask = cv2.bitwise_and(
            saturation > 80,   # Slightly lower threshold
            value > 120        # Slightly lower threshold
        )
        
        # Method 2: Texture-based detection (foil creates unique textures)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_CLOSE, kernel)
        sqr_diff = (gray.astype(np.float32) - mean) ** 2
        local_std = cv2.morphologyEx(sqr_diff, cv2.MORPH_CLOSE, kernel) ** 0.5
        
        texture_mask = local_std > 12  # Foil creates texture variation
        
        # Method 3: Edge density (foil patterns create many edges)
        edges = cv2.Canny(gray, 30, 90)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
        
        # Calculate local edge density
        edge_kernel = np.ones((7, 7), np.float32) / 49
        edge_density = cv2.filter2D(edges_dilated.astype(np.float32), -1, edge_kernel)
        edge_mask = edge_density > 0.15
        
        # Method 4: Color variance (holographic areas have high color variation)
        b, g, r = cv2.split(image)
        color_variance = np.var([b, g, r], axis=0)
        variance_mask = color_variance > np.percentile(color_variance, 75)
        
        # Combine all methods (ensure all masks are uint8)
        combined_mask = cv2.bitwise_or(sat_mask.astype(np.uint8), texture_mask.astype(np.uint8))
        combined_mask = cv2.bitwise_or(combined_mask, edge_mask.astype(np.uint8))
        combined_mask = cv2.bitwise_or(combined_mask, variance_mask.astype(np.uint8))
        
        # Clean up the mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_clean)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_clean)
        
        return cleaned_mask

    def _analyze_foil_properties(self, image: np.ndarray, holo_mask: np.ndarray, 
                                features: RarityFeatures):
        """Analyze specific properties of holographic foil."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract holo regions
        holo_hsv = cv2.bitwise_and(hsv, hsv, mask=holo_mask)
        hue_values = holo_hsv[:, :, 0][holo_mask > 0]
        
        if len(hue_values) > 0:
            # Calculate shine intensity based on brightness in holo areas
            value_channel = hsv[:, :, 2]
            holo_brightness = value_channel[holo_mask > 0]
            features.shine_intensity = np.mean(holo_brightness) / 255.0
            
            # Analyze hue diversity (indicator of rainbow effects)
            unique_hues = len(np.unique(hue_values))
            hue_diversity = min(unique_hues / 50, 1.0)  # Normalize
            
            # High diversity suggests rainbow foil
            if hue_diversity > 0.6:
                features.has_rainbow_foil = True

    def _identify_advanced_holo_pattern(self, image: np.ndarray, holo_mask: np.ndarray) -> str:
        """Identify holographic pattern type using advanced analysis."""
        # Extract holo regions for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        holo_gray = cv2.bitwise_and(gray, gray, mask=holo_mask)
        
        # Analyze pattern characteristics
        # 1. Line detection for linear patterns
        edges = cv2.Canny(holo_gray, 30, 90)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                               minLineLength=30, maxLineGap=5)
        
        line_count = len(lines) if lines is not None else 0
        
        # 2. Circle detection for cosmos/star patterns
        circles = cv2.HoughCircles(holo_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                  param1=50, param2=30, minRadius=5, maxRadius=50)
        
        circle_count = len(circles[0]) if circles is not None else 0
        
        # 3. Connected components analysis
        num_labels, _ = cv2.connectedComponents(holo_mask)
        
        # 4. Texture analysis using simplified LBP-like method
        try:
            from skimage.feature import local_binary_pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(holo_gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                     range=(0, n_points + 2), density=True)
            texture_uniformity = np.max(lbp_hist)
        except ImportError:
            # Fallback texture analysis without scikit-image
            kernel = np.ones((3, 3), np.float32) / 9
            smoothed = cv2.filter2D(holo_gray.astype(np.float32), -1, kernel)
            texture_var = np.var(holo_gray.astype(np.float32) - smoothed)
            texture_uniformity = min(texture_var / 100, 1.0)
        except:
            texture_uniformity = 0.5
        
        # Pattern classification based on characteristics
        if line_count > 20 and texture_uniformity > 0.3:
            return "linear"
        elif circle_count > 5:
            return "cosmos"
        elif num_labels > 50:
            return "crosshatch"
        elif texture_uniformity < 0.2:
            return "rainbow"
        elif line_count > 10:
            return "lightning"
        else:
            return "cosmos"  # Default for complex patterns

    def _detect_special_foil_types(self, image: np.ndarray, holo_mask: np.ndarray,
                                  features: RarityFeatures):
        """Detect special foil types like gold, rainbow, etc."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Gold foil detection (enhanced)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Gold characteristics: yellow/orange hue with high saturation and brightness
        hue_mask = ((hue >= 10) & (hue <= 40)).astype(np.uint8)
        sat_val_mask = ((saturation > 120) & (value > 180)).astype(np.uint8)
        gold_mask = cv2.bitwise_and(hue_mask, sat_val_mask)
        
        gold_in_holo = cv2.bitwise_and(gold_mask, holo_mask)
        gold_ratio = np.sum(gold_in_holo) / max(np.sum(holo_mask), 1)
        
        features.has_gold_foil = gold_ratio > 0.3
        
        # Rainbow foil detection (enhanced)
        holo_hue = hue[holo_mask > 0]
        if len(holo_hue) > 0:
            unique_hues = len(np.unique(holo_hue))
            hue_range = np.max(holo_hue) - np.min(holo_hue)
            
            # Rainbow foil has wide hue range and many unique hues
            features.has_rainbow_foil = (unique_hues > 25 and hue_range > 60) or \
                                       features.has_rainbow_foil

    def _calculate_advanced_holo_in_region(self, region: np.ndarray) -> float:
        """Calculate holographic coverage in a region using advanced detection."""
        holo_mask = self._detect_advanced_holographic_areas(region)
        return np.sum(holo_mask > 0) / (region.shape[0] * region.shape[1])

    def _calculate_background_holo(self, image: np.ndarray, artwork_region: np.ndarray) -> float:
        """Calculate holographic coverage in non-artwork areas."""
        # Create mask for entire image
        full_holo_mask = self._detect_advanced_holographic_areas(image)
        
        # Calculate artwork position
        h, w = image.shape[:2]
        ah, aw = artwork_region.shape[:2]
        
        # Estimate artwork position (center region typically)
        start_y = int(h * 0.15)
        end_y = start_y + ah
        start_x = int(w * 0.1)
        end_x = start_x + aw
        
        # Create background mask (everything except artwork)
        background_mask = np.ones((h, w), dtype=np.uint8)
        if end_y <= h and end_x <= w:
            background_mask[start_y:end_y, start_x:end_x] = 0
        
        # Calculate holo in background
        background_holo = cv2.bitwise_and(full_holo_mask, background_mask)
        background_pixels = np.sum(background_mask > 0)
        
        if background_pixels > 0:
            return np.sum(background_holo > 0) / background_pixels
        return 0.0

    def _analyze_foil_characteristics(self, image: np.ndarray, features: RarityFeatures):
        """Analyze special foil characteristics like rainbow, gold, etc."""
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            saturation = hsv[:, :, 1]
            
            # Rainbow foil detection (multiple hues with high saturation)
            unique_hues = len(np.unique(hue[saturation > 150]))
            if unique_hues > 30:  # Many different hues
                features.has_rainbow_foil = True
            
            # Gold foil detection (yellow/orange hues with high brightness)
            hue_mask = ((hue >= 15) & (hue <= 45)).astype(np.uint8)
            sat_mask = (saturation > 100).astype(np.uint8)
            gold_mask = cv2.bitwise_and(hue_mask, sat_mask)
            gold_coverage = np.sum(gold_mask) / (image.shape[0] * image.shape[1])
            features.has_gold_foil = gold_coverage > 0.1
            
            # Texture detection (look for surface irregularities)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate local standard deviation (texture indicator)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_CLOSE, kernel)
            sqr_diff = (gray.astype(np.float32) - mean) ** 2
            local_std = cv2.morphologyEx(sqr_diff, cv2.MORPH_CLOSE, kernel) ** 0.5
            
            texture_metric = np.mean(local_std)
            features.has_texture = texture_metric > 15  # Threshold for texture
            
        except Exception as e:
            logger.error(f"Foil analysis failed: {e}")

    def _detect_rarity_symbol(self, image: np.ndarray, features: RarityFeatures):
        """Detect and classify the rarity symbol."""
        try:
            # Convert to grayscale for symbol detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Look for symbols in the bottom right area (typical location)
            h, w = gray.shape
            symbol_region = gray[int(h * 0.7):, int(w * 0.7):]
            
            # Edge detection to find symbol shapes
            edges = cv2.Canny(symbol_region, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # Reasonable size for rarity symbol
                    # Analyze shape
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    
                    if len(approx) >= 8:  # Likely circular (common)
                        features.rarity_symbol_type = "common"
                        features.rarity_symbol_confidence = 0.8
                    elif len(approx) == 4:  # Diamond (uncommon)
                        features.rarity_symbol_type = "uncommon"
                        features.rarity_symbol_confidence = 0.7
                    elif 5 <= len(approx) <= 7:  # Star-like (rare)
                        features.rarity_symbol_type = "rare"
                        features.rarity_symbol_confidence = 0.6
                    
                    features.rarity_symbol_detected = True
                    break
                    
        except Exception as e:
            logger.error(f"Symbol detection failed: {e}")

    def _detect_card_number(self, text_content: str, features: RarityFeatures):
        """Extract and analyze card numbering for rarity hints."""
        try:
            # Look for card number pattern (e.g., "150/150", "151/150")
            number_pattern = r'(\d+)/(\d+)'
            matches = re.findall(number_pattern, text_content)
            
            if matches:
                card_num, total_cards = map(int, matches[0])
                features.card_number = f"{card_num}/{total_cards}"
                features.total_cards_in_set = total_cards
                
                # Secret rare detection (card number exceeds set total)
                if card_num > total_cards:
                    features.is_secret_rare_number = True
                    
        except Exception as e:
            logger.error(f"Card number detection failed: {e}")

    def _analyze_text_indicators(self, text_content: str, features: RarityFeatures):
        """Analyze text for rarity indicators."""
        text_lower = text_content.lower()
        
        # Common text indicators
        indicators = {
            "ultra rare": ["ultra rare", "ur"],
            "secret rare": ["secret rare", "sr"],
            "full art": ["full art"],
            "alternate art": ["alternate art", "alt art"],
            "rainbow rare": ["rainbow rare", "rr"],
            "gold rare": ["gold rare", "golden"],
            "promo": ["promo", "promotional", "black star"],
            "first edition": ["1st edition", "first edition"],
            "shadowless": ["shadowless"]
        }
        
        for rarity_type, keywords in indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                features.text_indicators.append(rarity_type)

    def _classify_rarity(self, features: RarityFeatures, era: Optional[PokemonCardEra]) -> PokemonRarity:
        """Classify rarity based on analyzed features."""
        # Text indicators take precedence
        if "secret rare" in features.text_indicators or features.is_secret_rare_number:
            if features.has_rainbow_foil:
                return PokemonRarity.RAINBOW_RARE
            elif features.has_gold_foil:
                return PokemonRarity.GOLD_RARE
            else:
                return PokemonRarity.SECRET_RARE
        
        if "ultra rare" in features.text_indicators:
            return PokemonRarity.ULTRA_RARE
            
        if "full art" in features.text_indicators:
            return PokemonRarity.FULL_ART
            
        if "alternate art" in features.text_indicators:
            return PokemonRarity.ALTERNATE_ART
            
        if "first edition" in features.text_indicators:
            return PokemonRarity.FIRST_EDITION
            
        if "shadowless" in features.text_indicators:
            return PokemonRarity.SHADOWLESS
            
        if "promo" in features.text_indicators:
            return PokemonRarity.PROMO
        
        # Visual feature classification
        if features.has_holographic:
            if features.has_reverse_holo:
                return PokemonRarity.REVERSE_HOLO
            else:
                return PokemonRarity.RARE_HOLO
        
        # Symbol-based classification
        if features.rarity_symbol_detected:
            symbol_to_rarity = {
                "common": PokemonRarity.COMMON,
                "uncommon": PokemonRarity.UNCOMMON,
                "rare": PokemonRarity.RARE
            }
            return symbol_to_rarity.get(features.rarity_symbol_type, PokemonRarity.COMMON)
        
        # Default classification
        return PokemonRarity.COMMON

    def _identify_holo_pattern(self, holo_mask: np.ndarray) -> str:
        """Identify the type of holographic pattern."""
        # Simplified pattern detection - in practice, would use more sophisticated analysis
        
        # Count connected components to identify pattern type
        num_labels, _ = cv2.connectedComponents(holo_mask.astype(np.uint8))
        
        if num_labels > 100:
            return "cosmos"  # Many small sparkly areas
        elif num_labels > 20:
            return "crosshatch"  # Medium complexity
        else:
            return "linear"  # Simple pattern
    
    def _extract_artwork_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract the main artwork region of the card."""
        try:
            h, w = image.shape[:2]
            # Typical artwork region for Pokémon cards
            artwork_region = image[int(h * 0.15):int(h * 0.55), int(w * 0.1):int(w * 0.9)]
            return artwork_region
        except:
            return None
    
    def _calculate_holo_in_region(self, region: np.ndarray) -> float:
        """Calculate holographic coverage in a specific region."""
        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            holo_mask = ((saturation > 100) & (value > 150)).astype(np.uint8)
            return np.sum(holo_mask) / (region.shape[0] * region.shape[1])
        except:
            return 0.0

    def get_rarity_description(self, rarity: PokemonRarity, features: RarityFeatures) -> str:
        """Get a detailed description of the detected rarity."""
        descriptions = {
            PokemonRarity.COMMON: "Common card with solid circle rarity symbol",
            PokemonRarity.UNCOMMON: "Uncommon card with solid diamond rarity symbol",
            PokemonRarity.RARE: "Rare card with hollow star rarity symbol",
            PokemonRarity.RARE_HOLO: "Rare holographic card with foil artwork",
            PokemonRarity.REVERSE_HOLO: "Reverse holographic with foil background",
            PokemonRarity.ULTRA_RARE: "Ultra Rare card (EX, GX, V, etc.)",
            PokemonRarity.SECRET_RARE: "Secret Rare with card number exceeding set total",
            PokemonRarity.RAINBOW_RARE: "Rainbow Rare with multicolored foil pattern",
            PokemonRarity.GOLD_RARE: "Gold Rare with golden foil treatment",
            PokemonRarity.FIRST_EDITION: "First Edition vintage card",
            PokemonRarity.SHADOWLESS: "Shadowless Base Set card",
            PokemonRarity.PROMO: "Promotional card"
        }
        
        base_desc = descriptions.get(rarity, "Unknown rarity")
        
        # Add feature details
        details = []
        if features.has_holographic:
            details.append(f"holographic pattern: {features.holo_pattern_type}")
        if features.has_texture:
            details.append("textured surface")
        if features.card_number:
            details.append(f"card number: {features.card_number}")
            
        if details:
            return f"{base_desc} ({', '.join(details)})"
        return base_desc