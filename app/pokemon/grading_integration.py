"""
Integration of holographic awareness into the main PSA grading pipeline.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .card_types import PokemonCardType, PokemonRarity, PokemonCardEra, GradingCriteria
from .rarity_detector import PokemonRarityDetector, RarityFeatures
from .visual_analyzer import PokemonVisualAnalyzer, PokemonDefectAnalysis, PokemonCardRegions
from .holo_visualizer import HolographicVisualizer, HolographicOverlaySettings

logger = logging.getLogger(__name__)


@dataclass
class HolographicGradingFactors:
    """Holographic-specific grading factors."""
    holo_pattern_quality: float = 1.0      # 0-1 scale
    holo_scratch_penalty: float = 0.0      # 0-1 scale  
    foil_peeling_penalty: float = 0.0      # 0-1 scale
    rainbow_integrity: float = 1.0         # 0-1 scale
    shine_consistency: float = 1.0         # 0-1 scale
    reverse_holo_alignment: float = 1.0    # For reverse holos
    
    # Overall holographic condition score
    holo_condition_score: float = 10.0     # PSA 1-10 scale


@dataclass
class EnhancedGradingResult:
    """Enhanced grading result with holographic awareness."""
    # Standard PSA factors
    centering_grade: float = 10.0
    corners_grade: float = 10.0
    edges_grade: float = 10.0
    surface_grade: float = 10.0
    
    # Holographic factors
    holo_factors: Optional[HolographicGradingFactors] = None
    
    # Final grades
    overall_grade: float = 10.0
    holo_adjusted_grade: float = 10.0
    
    # Analysis metadata
    card_type: Optional[PokemonCardType] = None
    rarity: Optional[PokemonRarity] = None
    era: Optional[PokemonCardEra] = None
    has_holographic: bool = False


class HolographicGradingIntegrator:
    """Integrates holographic analysis into PSA grading system."""
    
    def __init__(self):
        self.rarity_detector = PokemonRarityDetector()
        self.visual_analyzer = PokemonVisualAnalyzer()
        self.holo_visualizer = HolographicVisualizer()
        
        # Holographic grading weights by rarity
        self.holo_weights = {
            PokemonRarity.RARE_HOLO: {
                'holo_pattern_quality': 0.25,
                'holo_scratch_penalty': 0.30,
                'foil_peeling_penalty': 0.25,
                'rainbow_integrity': 0.20
            },
            PokemonRarity.REVERSE_HOLO: {
                'holo_pattern_quality': 0.20,
                'holo_scratch_penalty': 0.25,
                'foil_peeling_penalty': 0.20,
                'reverse_holo_alignment': 0.35
            },
            PokemonRarity.ULTRA_RARE: {
                'holo_pattern_quality': 0.30,
                'holo_scratch_penalty': 0.35,
                'foil_peeling_penalty': 0.25,
                'rainbow_integrity': 0.10
            },
            PokemonRarity.SECRET_RARE: {
                'holo_pattern_quality': 0.35,
                'holo_scratch_penalty': 0.40,
                'foil_peeling_penalty': 0.25
            },
            PokemonRarity.RAINBOW_RARE: {
                'rainbow_integrity': 0.50,
                'holo_scratch_penalty': 0.30,
                'foil_peeling_penalty': 0.20
            },
            PokemonRarity.GOLD_RARE: {
                'holo_pattern_quality': 0.40,
                'holo_scratch_penalty': 0.35,
                'foil_peeling_penalty': 0.25
            }
        }
    
    def grade_card_with_holo_awareness(self, image: np.ndarray, 
                                     base_grades: Optional[Dict[str, float]] = None,
                                     text_content: str = "") -> EnhancedGradingResult:
        """
        Grade a Pokémon card with full holographic awareness.
        
        Args:
            image: Card image in RGB format
            base_grades: Optional base grades from standard PSA analysis
            text_content: OCR text from the card
            
        Returns:
            Enhanced grading result with holographic factors
        """
        try:
            result = EnhancedGradingResult()
            
            # Step 1: Detect rarity and holographic features
            rarity, rarity_features = self.rarity_detector.detect_rarity(image, text_content)
            result.rarity = rarity
            result.has_holographic = rarity_features.has_holographic
            
            # Step 2: Determine card type and era (simplified for demo)
            card_type = self._determine_card_type(text_content)
            era = self._determine_era(text_content, image)
            result.card_type = card_type
            result.era = era
            
            # Step 3: Analyze card regions and defects
            regions, defects = self.visual_analyzer.analyze_pokemon_card(
                image, card_type, rarity, era)
            
            # Step 4: Apply base grades or calculate them
            if base_grades:
                result.centering_grade = base_grades.get('centering', 10.0)
                result.corners_grade = base_grades.get('corners', 10.0)
                result.edges_grade = base_grades.get('edges', 10.0)
                result.surface_grade = base_grades.get('surface', 10.0)
            else:
                # Calculate base grades using visual analyzer
                result.centering_grade = self._calculate_centering_grade(regions, image)
                result.corners_grade = self._calculate_corners_grade(defects)
                result.edges_grade = self._calculate_edges_grade(defects)
                result.surface_grade = self._calculate_surface_grade(defects)
            
            # Step 5: Calculate holographic factors if applicable
            if result.has_holographic:
                result.holo_factors = self._calculate_holographic_factors(
                    rarity_features, defects, regions, image)
            
            # Step 6: Calculate final grades
            result.overall_grade = self._calculate_overall_grade(result)
            result.holo_adjusted_grade = self._calculate_holo_adjusted_grade(result)
            
            logger.info(f"Graded {rarity.value} card: Overall={result.overall_grade:.1f}, "
                       f"Holo-Adjusted={result.holo_adjusted_grade:.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Holographic grading failed: {e}")
            return EnhancedGradingResult()
    
    def _calculate_holographic_factors(self, rarity_features: RarityFeatures,
                                     defects: PokemonDefectAnalysis,
                                     regions: PokemonCardRegions,
                                     image: np.ndarray) -> HolographicGradingFactors:
        """Calculate holographic-specific grading factors."""
        factors = HolographicGradingFactors()
        
        # 1. Holographic pattern quality
        factors.holo_pattern_quality = self._assess_pattern_quality(
            rarity_features, image, regions)
        
        # 2. Holographic scratch penalty
        factors.holo_scratch_penalty = self._calculate_scratch_penalty(defects)
        
        # 3. Foil peeling penalty  
        factors.foil_peeling_penalty = self._calculate_peeling_penalty(defects, image)
        
        # 4. Rainbow integrity (for rainbow cards)
        if rarity_features.has_rainbow_foil:
            factors.rainbow_integrity = self._assess_rainbow_integrity(
                rarity_features, image)
        
        # 5. Shine consistency
        factors.shine_consistency = self._assess_shine_consistency(
            rarity_features, image)
        
        # 6. Reverse holo alignment (for reverse holos)
        if rarity_features.has_reverse_holo:
            factors.reverse_holo_alignment = self._assess_reverse_holo_alignment(
                rarity_features, regions, image)
        
        # 7. Calculate overall holographic condition score
        factors.holo_condition_score = self._calculate_holo_condition_score(factors)
        
        return factors
    
    def _assess_pattern_quality(self, features: RarityFeatures, 
                               image: np.ndarray, regions: PokemonCardRegions) -> float:
        """Assess the quality of holographic patterns."""
        # Base quality from pattern type and intensity
        pattern_quality = min(features.holo_intensity * 2, 1.0)
        
        # Bonus for well-defined patterns
        pattern_bonuses = {
            'cosmos': 0.1,
            'linear': 0.05,
            'crosshatch': 0.08,
            'rainbow': 0.15,
            'lightning': 0.12,
            'leaf': 0.10
        }
        
        bonus = pattern_bonuses.get(features.holo_pattern_type, 0.0)
        pattern_quality = min(pattern_quality + bonus, 1.0)
        
        # Penalty for inconsistent patterns
        if features.shine_intensity < 0.3:
            pattern_quality *= 0.8  # Dim foil penalty
        
        return pattern_quality
    
    def _calculate_scratch_penalty(self, defects: PokemonDefectAnalysis) -> float:
        """Calculate penalty for holographic scratches."""
        if not defects.holo_scratches:
            return 0.0
        
        # Calculate total scratch impact
        total_length = sum(s.get('length', 0) for s in defects.holo_scratches)
        avg_severity = np.mean([s.get('severity', 0.5) for s in defects.holo_scratches])
        
        # Scratch penalty based on quantity and severity
        quantity_penalty = min(len(defects.holo_scratches) * 0.05, 0.4)
        severity_penalty = avg_severity * 0.3
        length_penalty = min(total_length / 1000, 0.3)
        
        total_penalty = quantity_penalty + severity_penalty + length_penalty
        return min(total_penalty, 1.0)
    
    def _calculate_peeling_penalty(self, defects: PokemonDefectAnalysis, 
                                 image: np.ndarray) -> float:
        """Calculate penalty for foil peeling."""
        if not defects.foil_peeling:
            return 0.0
        
        # Calculate total peeling area
        total_area = sum(w * h for x, y, w, h in defects.foil_peeling)
        image_area = image.shape[0] * image.shape[1]
        
        # Peeling penalty based on coverage
        coverage_ratio = total_area / image_area
        penalty = min(coverage_ratio * 5, 1.0)  # Up to 100% penalty
        
        # Additional penalty for multiple peeling areas
        if len(defects.foil_peeling) > 3:
            penalty += 0.1
        
        return min(penalty, 1.0)
    
    def _assess_rainbow_integrity(self, features: RarityFeatures, 
                                 image: np.ndarray) -> float:
        """Assess integrity of rainbow effects."""
        if not features.has_rainbow_foil:
            return 1.0
        
        # Analyze color distribution in rainbow areas
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for high saturation areas (rainbow regions)
        saturation = hsv[:, :, 1]
        rainbow_mask = saturation > 100
        
        if np.sum(rainbow_mask) == 0:
            return 0.5  # No rainbow areas detected
        
        # Analyze hue distribution
        hue_values = hsv[:, :, 0][rainbow_mask]
        unique_hues = len(np.unique(hue_values))
        hue_range = np.max(hue_values) - np.min(hue_values)
        hue_std = np.std(hue_values)
        
        # Good rainbow should have wide hue range and good distribution
        range_score = min(hue_range / 180, 1.0)  # Normalize to full hue range
        diversity_score = min(unique_hues / 60, 1.0)  # Many different hues
        distribution_score = min(hue_std / 30, 1.0)  # Good variation
        
        integrity = (range_score + diversity_score + distribution_score) / 3
        return integrity
    
    def _assess_shine_consistency(self, features: RarityFeatures, 
                                 image: np.ndarray) -> float:
        """Assess consistency of holographic shine."""
        # Analyze brightness variation in holographic areas
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        value = hsv[:, :, 2]
        
        # Create holo mask
        saturation = hsv[:, :, 1]
        holo_mask = saturation > 80
        
        if np.sum(holo_mask) == 0:
            return 1.0  # No holo areas
        
        # Analyze brightness consistency
        holo_brightness = value[holo_mask]
        brightness_std = np.std(holo_brightness)
        brightness_mean = np.mean(holo_brightness)
        
        # Good consistency means reasonable variation but not too chaotic
        # Coefficient of variation
        cv = brightness_std / max(brightness_mean, 1)
        
        # Optimal CV is around 0.2-0.4 for good holographic shine
        if 0.2 <= cv <= 0.4:
            consistency = 1.0
        elif cv < 0.2:
            consistency = 0.8  # Too uniform (might be damaged)
        else:
            consistency = max(0.5, 1.0 - (cv - 0.4) * 2)  # Too chaotic
        
        return consistency
    
    def _assess_reverse_holo_alignment(self, features: RarityFeatures,
                                     regions: PokemonCardRegions,
                                     image: np.ndarray) -> float:
        """Assess alignment quality of reverse holographic foil."""
        if not regions.artwork:
            return 1.0
        
        # Extract artwork and background regions
        ax, ay, aw, ah = regions.artwork
        artwork_region = image[ay:ay+ah, ax:ax+aw]
        
        # Create background mask
        h, w = image.shape[:2]
        background_mask = np.ones((h, w), dtype=np.uint8)
        background_mask[ay:ay+ah, ax:ax+aw] = 0
        
        # Analyze holographic coverage in each region
        artwork_holo = self._get_holo_coverage(artwork_region)
        background_holo = self._get_holo_coverage_with_mask(image, background_mask)
        
        # Good reverse holo: low artwork holo, high background holo
        if background_holo > 0.1 and artwork_holo < 0.05:
            alignment = 1.0
        elif background_holo > 0.05:
            alignment = 0.8 - (artwork_holo * 4)  # Penalty for artwork holo
        else:
            alignment = 0.5  # Poor reverse holo
        
        return max(0.0, alignment)
    
    def _get_holo_coverage(self, region: np.ndarray) -> float:
        """Get holographic coverage in a region."""
        if region.size == 0:
            return 0.0
        
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        holo_mask = cv2.bitwise_and(saturation > 80, value > 120)
        return np.sum(holo_mask) / (region.shape[0] * region.shape[1])
    
    def _get_holo_coverage_with_mask(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Get holographic coverage in masked area."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        holo_mask = cv2.bitwise_and(saturation > 80, value > 120)
        masked_holo = cv2.bitwise_and(holo_mask, mask)
        
        mask_area = np.sum(mask > 0)
        if mask_area == 0:
            return 0.0
        
        return np.sum(masked_holo) / mask_area
    
    def _calculate_holo_condition_score(self, factors: HolographicGradingFactors) -> float:
        """Calculate overall holographic condition score (1-10 scale)."""
        # Start with perfect score
        score = 10.0
        
        # Apply penalties
        score -= factors.holo_scratch_penalty * 3.0      # Up to 3 points
        score -= factors.foil_peeling_penalty * 2.5      # Up to 2.5 points
        score -= (1.0 - factors.holo_pattern_quality) * 2.0  # Up to 2 points
        score -= (1.0 - factors.rainbow_integrity) * 1.5     # Up to 1.5 points
        score -= (1.0 - factors.shine_consistency) * 1.0     # Up to 1 point
        
        return max(1.0, score)
    
    def _calculate_overall_grade(self, result: EnhancedGradingResult) -> float:
        """Calculate overall grade using standard PSA methodology."""
        grades = [
            result.centering_grade,
            result.corners_grade, 
            result.edges_grade,
            result.surface_grade
        ]
        
        # PSA uses lowest grade methodology with some averaging
        lowest_grade = min(grades)
        average_grade = np.mean(grades)
        
        # Weighted combination favoring the lowest grade
        overall = (lowest_grade * 0.6) + (average_grade * 0.4)
        return round(overall * 2) / 2  # Round to nearest 0.5
    
    def _calculate_holo_adjusted_grade(self, result: EnhancedGradingResult) -> float:
        """Calculate grade adjusted for holographic factors."""
        base_grade = result.overall_grade
        
        if not result.has_holographic or not result.holo_factors:
            return base_grade
        
        # Get weights for this rarity
        weights = self.holo_weights.get(result.rarity, {})
        
        if not weights:
            return base_grade
        
        # Calculate holographic adjustment
        holo_penalty = 0.0
        
        for factor_name, weight in weights.items():
            factor_value = getattr(result.holo_factors, factor_name, 0.0)
            
            if 'penalty' in factor_name:
                holo_penalty += factor_value * weight
            else:
                # For quality factors, penalty is (1 - quality)
                holo_penalty += (1.0 - factor_value) * weight
        
        # Apply holographic adjustment (can reduce grade by up to 2 points)
        adjustment = min(holo_penalty * 2.0, 2.0)
        adjusted_grade = max(1.0, base_grade - adjustment)
        
        return round(adjusted_grade * 2) / 2  # Round to nearest 0.5
    
    def _determine_card_type(self, text_content: str) -> PokemonCardType:
        """Determine card type from text content (simplified)."""
        text_lower = text_content.lower()
        
        if any(word in text_lower for word in ['trainer', 'supporter', 'item', 'stadium']):
            return PokemonCardType.TRAINER
        elif 'energy' in text_lower:
            return PokemonCardType.ENERGY
        else:
            return PokemonCardType.POKEMON
    
    def _determine_era(self, text_content: str, image: np.ndarray) -> PokemonCardEra:
        """Determine card era (simplified)."""
        text_lower = text_content.lower()
        
        # Look for era indicators
        if any(indicator in text_lower for indicator in ['1998', '1999', 'wizards']):
            return PokemonCardEra.VINTAGE
        elif any(indicator in text_lower for indicator in ['2003', '2004', '2005']):
            return PokemonCardEra.E_CARD
        else:
            return PokemonCardEra.SCARLET_VIOLET
    
    def _calculate_centering_grade(self, regions: PokemonCardRegions, 
                                 image: np.ndarray) -> float:
        """Calculate centering grade (simplified)."""
        # This would integrate with existing centering analysis
        return 9.0  # Placeholder
    
    def _calculate_corners_grade(self, defects: PokemonDefectAnalysis) -> float:
        """Calculate corners grade based on defect analysis."""
        avg_corner_damage = np.mean(list(defects.corner_peeling.values()))
        grade = 10.0 - (avg_corner_damage * 5.0)
        return max(1.0, grade)
    
    def _calculate_edges_grade(self, defects: PokemonDefectAnalysis) -> float:
        """Calculate edges grade based on defect analysis."""
        avg_edge_whitening = np.mean(list(defects.edge_whitening.values()))
        grade = 10.0 - (avg_edge_whitening * 4.0)
        return max(1.0, grade)
    
    def _calculate_surface_grade(self, defects: PokemonDefectAnalysis) -> float:
        """Calculate surface grade based on defect analysis."""
        surface_penalty = len(defects.surface_scratches) * 0.5
        surface_penalty += len(defects.indentations) * 0.3
        surface_penalty += defects.artwork_damage * 2.0
        
        grade = 10.0 - min(surface_penalty, 8.0)
        return max(1.0, grade)
    
    def create_comprehensive_overlay(self, image: np.ndarray, 
                                   result: EnhancedGradingResult,
                                   rarity_features: RarityFeatures,
                                   defects: PokemonDefectAnalysis,
                                   regions: PokemonCardRegions) -> np.ndarray:
        """Create comprehensive grading overlay with holographic analysis."""
        if result.has_holographic:
            settings = HolographicOverlaySettings(
                show_holo_mask=True,
                show_scratch_lines=True,
                show_peeling_areas=True,
                show_pattern_analysis=True,
                overlay_alpha=0.4
            )
            
            overlay = self.holo_visualizer.create_holographic_overlay(
                image, rarity_features, defects, regions, settings)
        else:
            overlay = image.copy()
        
        # Add grading information
        self._add_grading_info_overlay(overlay, result)
        
        return overlay
    
    def _add_grading_info_overlay(self, overlay: np.ndarray, 
                                result: EnhancedGradingResult):
        """Add grading information to overlay."""
        h, w = overlay.shape[:2]
        
        # Create info panel
        info_height = 200
        info_width = 300
        info_panel = np.zeros((info_height, info_width, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)  # Dark gray background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Grade information
        lines = [
            f"Overall Grade: {result.overall_grade:.1f}",
            f"Centering: {result.centering_grade:.1f}",
            f"Corners: {result.corners_grade:.1f}",
            f"Edges: {result.edges_grade:.1f}",
            f"Surface: {result.surface_grade:.1f}",
            "",
            f"Rarity: {result.rarity.value if result.rarity else 'Unknown'}",
        ]
        
        if result.has_holographic and result.holo_factors:
            lines.extend([
                "",
                f"Holo-Adjusted: {result.holo_adjusted_grade:.1f}",
                f"Holo Condition: {result.holo_factors.holo_condition_score:.1f}",
                f"Pattern Quality: {result.holo_factors.holo_pattern_quality:.2f}",
                f"Scratch Penalty: {result.holo_factors.holo_scratch_penalty:.2f}"
            ])
        
        # Draw text
        y_offset = 20
        for line in lines:
            if line:  # Skip empty lines
                cv2.putText(info_panel, line, (10, y_offset), font, font_scale, 
                          (255, 255, 255), thickness)
            y_offset += 15
        
        # Place info panel in top-right corner
        if h > info_height and w > info_width:
            overlay[0:info_height, w-info_width:w] = info_panel