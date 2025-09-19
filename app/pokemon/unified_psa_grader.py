"""
Unified PSA Grading System for Pokemon Cards
Comprehensive grading system that mimics human PSA grader behavior.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import time
from pathlib import Path

from .preprocessing_pipeline import PokemonCardPreprocessor, PreprocessingConfig, PreprocessingLevel, PreprocessingResult
from .card_segmentation import PokemonCardSegmenter, SegmentationResult, CardComponent
from .reference_grading_methodology import ReferenceGradingMethodology, GradingZone
from .grading_integration import HolographicGradingIntegrator, EnhancedGradingResult
from .card_types import PokemonRarity, PokemonCardEra

logger = logging.getLogger(__name__)


class PSAGradeLevel(Enum):
    """PSA Grade levels with human descriptions."""
    GEM_MINT_10 = (10, "Gem Mint", "Perfect card with no visible flaws")
    MINT_9 = (9, "Mint", "Near perfect with very minor flaws")
    NEAR_MINT_8 = (8, "Near Mint", "Excellent condition with minor wear")
    EXCELLENT_7 = (7, "Excellent", "Light wear but still attractive")
    NEAR_EXCELLENT_6 = (6, "Near Excellent", "Moderate wear visible")
    EXCELLENT_MINUS_5 = (5, "Excellent-", "More noticeable wear")
    VERY_GOOD_4 = (4, "Very Good", "Obvious wear but structurally sound")
    GOOD_3 = (3, "Good", "Significant wear and handling")
    FAIR_2 = (2, "Fair", "Heavy wear and damage")
    POOR_1 = (1, "Poor", "Extreme wear and damage")
    
    def __init__(self, grade: int, description: str, condition: str):
        self.grade = grade
        self.description = description
        self.condition = condition


@dataclass
class PSAGradingFactors:
    """Comprehensive PSA grading factors."""
    # Primary grading factors (traditional PSA)
    centering: float = 10.0
    corners: float = 10.0
    edges: float = 10.0
    surface: float = 10.0
    
    # Detailed component grades
    artwork_condition: float = 10.0
    text_clarity: float = 10.0
    borders_condition: float = 10.0
    
    # Holographic factors (if applicable)
    holo_pattern_quality: float = 10.0
    holo_scratch_penalty: float = 0.0
    foil_condition: float = 10.0
    
    # Human grader considerations
    eye_appeal: float = 10.0          # Overall visual appeal
    print_quality: float = 10.0       # Print registration and quality
    structural_integrity: float = 10.0 # Overall card structure
    
    # Calculated grades
    component_average: float = 10.0
    lowest_component: float = 10.0
    final_grade: float = 10.0
    
    # Metadata
    has_holographic: bool = False
    card_era: Optional[PokemonCardEra] = None
    rarity: Optional[PokemonRarity] = None


@dataclass
class PSAGradingReport:
    """Comprehensive PSA grading report."""
    final_grade: PSAGradeLevel
    grading_factors: PSAGradingFactors
    detailed_analysis: Dict[str, Any]
    processing_time: float
    confidence_score: float
    
    # Human grader notes
    key_observations: List[str]
    grade_limiting_factors: List[str]
    positives: List[str]
    areas_of_concern: List[str]
    
    # Technical data
    image_quality_score: float
    segmentation_quality: float
    preprocessing_steps: List[str]


class UnifiedPSAGrader:
    """
    Unified PSA grading system that mimics human grader behavior.
    
    This system combines preprocessing, segmentation, holographic analysis,
    and reference methodology to provide comprehensive PSA-style grading
    that closely mimics how human graders evaluate Pokemon cards.
    """
    
    def __init__(self):
        self.preprocessor = PokemonCardPreprocessor()
        self.segmenter = PokemonCardSegmenter()
        self.reference_grader = ReferenceGradingMethodology()
        self.holo_grader = HolographicGradingIntegrator()
        
        # Human grader decision weights
        self.grading_weights = self._initialize_grading_weights()
        
        # Grade threshold mappings (how humans mentally categorize grades)
        self.grade_thresholds = self._initialize_grade_thresholds()
        
    def _initialize_grading_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize grading weights that mimic human grader priorities."""
        return {
            "vintage": {
                "centering": 0.30,      # Very important for vintage
                "corners": 0.25,        # Critical for vintage cards
                "edges": 0.20,          # Important
                "surface": 0.25,        # Includes print quality issues
            },
            "modern": {
                "centering": 0.25,      # Important but less forgiving
                "corners": 0.30,        # Critical for modern standards
                "edges": 0.25,          # Higher expectations
                "surface": 0.20,        # Better print quality expected
            },
            "holographic": {
                "centering": 0.20,      # Still important
                "corners": 0.25,        # Critical for holo cards
                "edges": 0.20,          # Important
                "surface": 0.15,        # Base surface
                "holo_condition": 0.20, # Holographic specific
            }
        }
    
    def _initialize_grade_thresholds(self) -> Dict[int, Dict[str, float]]:
        """Initialize grade thresholds based on human grader standards."""
        return {
            10: {"min_centering": 9.5, "min_corners": 9.8, "min_edges": 9.5, "min_surface": 9.5},
            9:  {"min_centering": 8.5, "min_corners": 9.0, "min_edges": 8.5, "min_surface": 8.5},
            8:  {"min_centering": 7.5, "min_corners": 8.0, "min_edges": 7.5, "min_surface": 7.5},
            7:  {"min_centering": 6.5, "min_corners": 7.0, "min_edges": 6.5, "min_surface": 6.5},
            6:  {"min_centering": 5.5, "min_corners": 6.0, "min_edges": 5.5, "min_surface": 5.5},
            5:  {"min_centering": 4.5, "min_corners": 5.0, "min_edges": 4.5, "min_surface": 4.5},
            4:  {"min_centering": 3.5, "min_corners": 4.0, "min_edges": 3.5, "min_surface": 3.5},
            3:  {"min_centering": 2.5, "min_corners": 3.0, "min_edges": 2.5, "min_surface": 2.5},
            2:  {"min_centering": 1.5, "min_corners": 2.0, "min_edges": 1.5, "min_surface": 1.5},
            1:  {"min_centering": 0.0, "min_corners": 0.0, "min_edges": 0.0, "min_surface": 0.0},
        }
    
    def grade_card(self, image: Union[np.ndarray, str], 
                   card_era: str = "vintage",
                   processing_level: PreprocessingLevel = PreprocessingLevel.STANDARD,
                   ocr_text: str = "") -> PSAGradingReport:
        """
        Grade a Pokemon card using comprehensive PSA methodology.
        
        Args:
            image: Card image (numpy array or file path)
            card_era: Era of the card for appropriate grading standards
            processing_level: Level of image preprocessing
            ocr_text: Extracted text from the card (optional)
            
        Returns:
            Comprehensive PSA grading report
        """
        start_time = time.time()
        
        logger.info(f"Starting PSA grading analysis for {card_era} era card")
        
        try:
            # Step 1: Preprocess the card image
            preprocessing_config = PreprocessingConfig(
                level=processing_level,
                segment_components=True,
                extract_features=True,
                correct_perspective=True,
                enhance_contrast=True
            )
            
            preprocessing_result = self.preprocessor.process_card_image(
                image, preprocessing_config, card_era)
            
            # Step 2: Apply reference grading methodology
            zone_results = self.reference_grader.grade_card_using_reference_methodology(
                preprocessing_result.processed_image, card_era)
            reference_result = self.reference_grader.calculate_overall_grade(zone_results)
            
            # Step 3: Apply holographic grading if applicable
            zone_scores = reference_result.get('zone_scores', {})
            holo_result = self.holo_grader.grade_card_with_holo_awareness(
                preprocessing_result.processed_image,
                base_grades={
                    'centering': zone_scores.get(GradingZone.CENTERING.value),
                    'corners': zone_scores.get(GradingZone.CORNERS.value),
                    'edges': zone_scores.get(GradingZone.EDGES.value),
                    'surface': zone_scores.get(GradingZone.SURFACE.value)
                },
                text_content=ocr_text
            )
            
            # Step 4: Apply human-like grading logic
            grading_factors = self._calculate_human_like_grades(
                preprocessing_result, reference_result, holo_result, card_era)
            
            # Step 5: Generate comprehensive report
            report = self._generate_grading_report(
                grading_factors, preprocessing_result, reference_result, 
                holo_result, start_time)
            
            logger.info(f"PSA grading completed: Grade {report.final_grade.grade}")
            return report
            
        except Exception as e:
            logger.error(f"PSA grading failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return default report with error information
            return self._create_error_report(str(e), start_time)
    
    def _calculate_human_like_grades(self, preprocessing_result: PreprocessingResult,
                                   reference_result: Any, holo_result: EnhancedGradingResult,
                                   card_era: str) -> PSAGradingFactors:
        """Calculate grades using human-like decision making."""
        factors = PSAGradingFactors()
        
        # Extract era information
        if card_era in ["vintage", "classic"]:
            era = PokemonCardEra.VINTAGE
            weight_category = "vintage"
        else:
            era = PokemonCardEra.SCARLET_VIOLET
            weight_category = "modern"
        
        factors.card_era = era
        factors.rarity = holo_result.rarity
        factors.has_holographic = holo_result.has_holographic
        
        # Base grades from reference methodology (human-like systematic approach)
        zone_scores = reference_result.get('zone_scores', {})
        factors.centering = zone_scores.get(GradingZone.CENTERING.value)
        factors.corners = zone_scores.get(GradingZone.CORNERS.value)
        factors.edges = zone_scores.get(GradingZone.EDGES.value)
        factors.surface = zone_scores.get(GradingZone.SURFACE.value)
        
        # Apply holographic adjustments if applicable
        if factors.has_holographic and holo_result.holo_factors:
            factors.holo_pattern_quality = holo_result.holo_factors.holo_pattern_quality * 10
            factors.holo_scratch_penalty = holo_result.holo_factors.holo_scratch_penalty
            factors.foil_condition = max(1.0, 10.0 - (holo_result.holo_factors.holo_scratch_penalty * 5))
            
            # Adjust surface grade for holographic factors
            factors.surface = min(factors.surface, holo_result.holo_factors.holo_condition_score)
            weight_category = "holographic"
        
        # Component-specific grading (human graders look at these)
        if preprocessing_result.segmentation_result:
            factors = self._grade_card_components(factors, preprocessing_result.segmentation_result)
        
        # Human grader eye appeal assessment
        factors.eye_appeal = self._assess_eye_appeal(factors, preprocessing_result.quality_metrics)
        
        # Print quality assessment
        factors.print_quality = self._assess_print_quality(preprocessing_result.quality_metrics)
        
        # Structural integrity (human graders check card structure)
        factors.structural_integrity = self._assess_structural_integrity(factors)
        
        # Calculate final grade using human-like logic
        factors.component_average = np.mean([factors.centering, factors.corners, 
                                           factors.edges, factors.surface])
        factors.lowest_component = min(factors.centering, factors.corners, 
                                     factors.edges, factors.surface)
        
        # Human graders use a combination of lowest grade and weighted average
        factors.final_grade = self._calculate_final_grade_human_style(factors, weight_category)
        
        return factors
    
    def _grade_card_components(self, factors: PSAGradingFactors, 
                             segmentation: SegmentationResult) -> PSAGradingFactors:
        """Grade individual card components like human graders do."""
        
        # Artwork condition assessment
        artwork_quality = 10.0
        if CardComponent.ARTWORK in segmentation.components:
            artwork_info = segmentation.components[CardComponent.ARTWORK]
            if artwork_info.confidence < 0.8:
                artwork_quality -= 1.0  # Poor segmentation indicates damage
            # Additional artwork damage assessment would go here
        
        # Text clarity assessment
        text_quality = 10.0
        text_components = [CardComponent.NAME_BAR, CardComponent.ATTACK_TEXT, CardComponent.FLAVOR_TEXT]
        for component in text_components:
            if component in segmentation.components:
                component_info = segmentation.components[component]
                if component_info.confidence < 0.7:
                    text_quality -= 0.5  # Text damage
        
        # Border condition assessment
        border_quality = 10.0
        if CardComponent.OUTER_BORDER in segmentation.components:
            border_info = segmentation.components[CardComponent.OUTER_BORDER]
            if border_info.confidence < 0.9:
                border_quality -= 1.0  # Border wear
        
        factors.artwork_condition = artwork_quality
        factors.text_clarity = text_quality
        factors.borders_condition = border_quality
        
        return factors
    
    def _assess_eye_appeal(self, factors: PSAGradingFactors, 
                          quality_metrics: Dict[str, Any]) -> float:
        """Assess overall eye appeal like human graders do."""
        # Base eye appeal from image quality
        initial_quality = quality_metrics.get("initial_quality", {})
        final_quality = quality_metrics.get("final_quality", {})
        
        base_appeal = min(final_quality.get("overall_quality", 80) / 10, 10.0)
        
        # Adjust for specific factors
        if factors.has_holographic:
            # Holographic cards get bonus for working foil
            if factors.holo_pattern_quality > 8.0:
                base_appeal = min(base_appeal + 0.5, 10.0)
            
            # Penalty for damaged foil
            base_appeal -= factors.holo_scratch_penalty * 2.0
        
        # Centering affects eye appeal significantly
        if factors.centering < 6.0:
            base_appeal -= 1.0  # Poor centering hurts eye appeal
        elif factors.centering > 9.0:
            base_appeal = min(base_appeal + 0.3, 10.0)  # Great centering helps
        
        return max(1.0, base_appeal)
    
    def _assess_print_quality(self, quality_metrics: Dict[str, Any]) -> float:
        """Assess print quality like human graders examine it."""
        final_quality = quality_metrics.get("final_quality", {})
        
        # Base print quality from contrast and sharpness
        contrast = final_quality.get("contrast", 25)
        sharpness = final_quality.get("sharpness", 50)
        
        # Human graders look for crisp, clear printing
        contrast_score = min(contrast / 30, 1.0) * 5  # 0-5 scale
        sharpness_score = min(sharpness / 60, 1.0) * 5  # 0-5 scale
        
        print_quality = contrast_score + sharpness_score  # 0-10 scale
        
        # Noise penalty (print defects)
        noise_level = final_quality.get("noise_level", 0)
        if noise_level > 20:
            print_quality -= (noise_level - 20) / 10
        
        return max(1.0, min(print_quality, 10.0))
    
    def _assess_structural_integrity(self, factors: PSAGradingFactors) -> float:
        """Assess structural integrity like human graders check card structure."""
        # Base structural score from corners and edges
        structure_score = (factors.corners * 0.6) + (factors.edges * 0.4)
        
        # Severe corner damage affects structure more
        if factors.corners < 5.0:
            structure_score -= 1.0
        
        # Edge damage affects structural integrity
        if factors.edges < 4.0:
            structure_score -= 0.5
        
        return max(1.0, structure_score)
    
    def _calculate_final_grade_human_style(self, factors: PSAGradingFactors, 
                                         weight_category: str) -> float:
        """Calculate final grade using human grader decision logic."""
        weights = self.grading_weights[weight_category]
        
        # Human graders start with component grades
        base_components = [factors.centering, factors.corners, factors.edges, factors.surface]
        
        # Check if card meets minimum thresholds for each grade level
        for grade in range(10, 0, -1):
            thresholds = self.grade_thresholds[grade]
            
            # Check if card meets minimum requirements
            meets_requirements = (
                factors.centering >= thresholds["min_centering"] and
                factors.corners >= thresholds["min_corners"] and
                factors.edges >= thresholds["min_edges"] and
                factors.surface >= thresholds["min_surface"]
            )
            
            if meets_requirements:
                # Apply weighted calculation within the grade band
                weighted_score = (
                    factors.centering * weights["centering"] +
                    factors.corners * weights["corners"] +
                    factors.edges * weights["edges"] +
                    factors.surface * weights["surface"]
                )
                
                # Add holographic factors if applicable
                if "holo_condition" in weights and factors.has_holographic:
                    holo_score = max(1.0, 10.0 - factors.holo_scratch_penalty * 5)
                    weighted_score += holo_score * weights["holo_condition"]
                
                # Human graders also consider eye appeal
                eye_appeal_adjustment = (factors.eye_appeal - 7.0) * 0.1  # ±0.3 adjustment
                weighted_score += eye_appeal_adjustment
                
                # Ensure grade doesn't exceed the threshold band
                final_grade = min(grade + 0.5, weighted_score)
                
                # Round to nearest 0.5 (human grader style)
                return round(final_grade * 2) / 2
        
        # Fallback to grade 1 if no thresholds met
        return 1.0
    
    def _generate_grading_report(self, factors: PSAGradingFactors,
                               preprocessing_result: PreprocessingResult,
                               reference_result: Any, holo_result: EnhancedGradingResult,
                               start_time: float) -> PSAGradingReport:
        """Generate comprehensive grading report."""
        processing_time = time.time() - start_time
        
        # Determine PSA grade level
        grade_level = None
        for level in PSAGradeLevel:
            if level.grade == int(factors.final_grade):
                grade_level = level
                break
        
        if grade_level is None:
            grade_level = PSAGradeLevel.POOR_1
        
        # Generate human-like observations
        key_observations = self._generate_key_observations(factors, preprocessing_result)
        grade_limiting_factors = self._identify_grade_limiting_factors(factors)
        positives = self._identify_positives(factors)
        areas_of_concern = self._identify_concerns(factors)
        
        # Calculate confidence based on image quality and segmentation
        confidence = self._calculate_confidence(preprocessing_result, factors)
        
        # Detailed analysis
        detailed_analysis = {
            "preprocessing_quality": preprocessing_result.quality_metrics,
            "segmentation_quality": preprocessing_result.segmentation_result.segmentation_quality if preprocessing_result.segmentation_result else 0.0,
            "reference_methodology": reference_result.get('zone_scores', {}),
            "holographic_analysis": asdict(holo_result.holo_factors) if holo_result.holo_factors else {},
            "component_grades": {
                "artwork": factors.artwork_condition,
                "text": factors.text_clarity,
                "borders": factors.borders_condition
            }
        }
        
        return PSAGradingReport(
            final_grade=grade_level,
            grading_factors=factors,
            detailed_analysis=detailed_analysis,
            processing_time=processing_time,
            confidence_score=confidence,
            key_observations=key_observations,
            grade_limiting_factors=grade_limiting_factors,
            positives=positives,
            areas_of_concern=areas_of_concern,
            image_quality_score=preprocessing_result.quality_metrics["final_quality"]["overall_quality"],
            segmentation_quality=preprocessing_result.segmentation_result.segmentation_quality if preprocessing_result.segmentation_result else 0.0,
            preprocessing_steps=preprocessing_result.preprocessing_steps
        )
    
    def _generate_key_observations(self, factors: PSAGradingFactors, 
                                 preprocessing_result: PreprocessingResult) -> List[str]:
        """Generate key observations like a human grader would note."""
        observations = []
        
        # Centering observations
        if factors.centering >= 9.0:
            observations.append("Excellent centering with minimal border variation")
        elif factors.centering >= 7.0:
            observations.append("Good centering with minor border inconsistencies")
        elif factors.centering >= 5.0:
            observations.append("Fair centering with noticeable border variation")
        else:
            observations.append("Poor centering with significant border inconsistencies")
        
        # Corner observations
        if factors.corners >= 9.0:
            observations.append("Sharp, crisp corners with no visible wear")
        elif factors.corners >= 7.0:
            observations.append("Minor corner wear visible under magnification")
        else:
            observations.append("Noticeable corner wear and rounding")
        
        # Edge observations
        if factors.edges >= 8.0:
            observations.append("Clean edges with minimal whitening")
        else:
            observations.append("Edge wear and whitening present")
        
        # Surface observations
        if factors.surface >= 9.0:
            observations.append("Pristine surface with no visible defects")
        elif factors.surface >= 7.0:
            observations.append("Minor surface imperfections present")
        else:
            observations.append("Surface damage affecting card appearance")
        
        # Holographic observations
        if factors.has_holographic:
            if factors.holo_pattern_quality >= 8.0:
                observations.append("Holographic foil shows excellent pattern integrity")
            else:
                observations.append("Holographic foil shows wear or damage")
        
        # Quality improvements from preprocessing
        quality_improvement = preprocessing_result.quality_metrics["improvement"]["overall_quality"]
        if quality_improvement > 10:
            observations.append("Significant image enhancement improved analysis quality")
        
        return observations
    
    def _identify_grade_limiting_factors(self, factors: PSAGradingFactors) -> List[str]:
        """Identify factors limiting the grade like a human grader would."""
        limiting_factors = []
        
        # Find the lowest scoring components
        component_scores = {
            "Centering": factors.centering,
            "Corners": factors.corners,
            "Edges": factors.edges,
            "Surface": factors.surface
        }
        
        lowest_score = min(component_scores.values())
        
        for component, score in component_scores.items():
            if score == lowest_score and score < 8.0:
                limiting_factors.append(f"{component} condition limits overall grade")
        
        # Holographic specific limiting factors
        if factors.has_holographic and factors.holo_scratch_penalty > 0.2:
            limiting_factors.append("Holographic foil damage impacts grade")
        
        # Eye appeal factors
        if factors.eye_appeal < 6.0:
            limiting_factors.append("Poor overall eye appeal affects grade")
        
        return limiting_factors
    
    def _identify_positives(self, factors: PSAGradingFactors) -> List[str]:
        """Identify positive aspects like a human grader would note."""
        positives = []
        
        if factors.centering >= 9.0:
            positives.append("Exceptional centering quality")
        
        if factors.corners >= 9.0:
            positives.append("Pristine corner condition")
        
        if factors.surface >= 9.0:
            positives.append("Excellent surface preservation")
        
        if factors.has_holographic and factors.holo_pattern_quality >= 8.0:
            positives.append("Strong holographic foil condition")
        
        if factors.eye_appeal >= 8.5:
            positives.append("Strong overall eye appeal")
        
        if factors.print_quality >= 9.0:
            positives.append("Excellent print quality and registration")
        
        return positives
    
    def _identify_concerns(self, factors: PSAGradingFactors) -> List[str]:
        """Identify areas of concern like a human grader would note."""
        concerns = []
        
        if factors.centering < 6.0:
            concerns.append("Centering issues significantly impact appearance")
        
        if factors.corners < 5.0:
            concerns.append("Corner damage affects structural integrity")
        
        if factors.surface < 6.0:
            concerns.append("Surface damage impacts card presentation")
        
        if factors.has_holographic and factors.holo_scratch_penalty > 0.3:
            concerns.append("Holographic foil damage is readily apparent")
        
        if factors.structural_integrity < 7.0:
            concerns.append("Overall structural integrity compromised")
        
        return concerns
    
    def _calculate_confidence(self, preprocessing_result: PreprocessingResult, 
                            factors: PSAGradingFactors) -> float:
        """Calculate confidence in the grading assessment."""
        # Base confidence from image quality
        image_quality = preprocessing_result.quality_metrics["final_quality"]["overall_quality"]
        base_confidence = min(image_quality / 100, 1.0)
        
        # Segmentation quality contributes to confidence
        if preprocessing_result.segmentation_result:
            seg_quality = preprocessing_result.segmentation_result.segmentation_quality
            base_confidence = (base_confidence + seg_quality) / 2
        
        # Consistent grades across components increase confidence
        component_grades = [factors.centering, factors.corners, factors.edges, factors.surface]
        grade_std = np.std(component_grades)
        
        # Lower standard deviation means more consistent assessment
        consistency_bonus = max(0, (3.0 - grade_std) / 3.0) * 0.2
        
        final_confidence = min(base_confidence + consistency_bonus, 1.0)
        return final_confidence
    
    def _create_error_report(self, error_message: str, start_time: float) -> PSAGradingReport:
        """Create error report when grading fails."""
        processing_time = time.time() - start_time
        
        return PSAGradingReport(
            final_grade=PSAGradeLevel.POOR_1,
            grading_factors=PSAGradingFactors(final_grade=1.0),
            detailed_analysis={"error": error_message},
            processing_time=processing_time,
            confidence_score=0.0,
            key_observations=[f"Grading failed: {error_message}"],
            grade_limiting_factors=["Analysis error prevented proper grading"],
            positives=[],
            areas_of_concern=["Unable to complete grading analysis"],
            image_quality_score=0.0,
            segmentation_quality=0.0,
            preprocessing_steps=[]
        )
    
    def create_grading_visualization(self, image: np.ndarray, 
                                   report: PSAGradingReport) -> np.ndarray:
        """Create comprehensive grading visualization."""
        visualization = image.copy()
        h, w = visualization.shape[:2]
        
        # Create overlay panel
        panel_height = 300
        panel_width = 400
        
        if h > panel_height and w > panel_width:
            # Create info panel
            panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)  # Dark background
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Grade information
            y_pos = 30
            
            # Main grade
            grade_text = f"PSA GRADE: {report.final_grade.grade}"
            cv2.putText(panel, grade_text, (10, y_pos), font, font_scale + 0.2, 
                       (0, 255, 0) if report.final_grade.grade >= 8 else (255, 255, 0) if report.final_grade.grade >= 6 else (255, 0, 0), 
                       thickness)
            y_pos += 35
            
            # Grade description
            desc_text = report.final_grade.description
            cv2.putText(panel, desc_text, (10, y_pos), font, font_scale - 0.1, (255, 255, 255), 1)
            y_pos += 30
            
            # Component grades
            components = [
                ("Centering", report.grading_factors.centering),
                ("Corners", report.grading_factors.corners),
                ("Edges", report.grading_factors.edges),
                ("Surface", report.grading_factors.surface)
            ]
            
            for comp_name, comp_grade in components:
                comp_text = f"{comp_name}: {comp_grade:.1f}"
                cv2.putText(panel, comp_text, (10, y_pos), font, font_scale - 0.1, (255, 255, 255), 1)
                y_pos += 20
            
            # Holographic info if applicable
            if report.grading_factors.has_holographic:
                y_pos += 10
                holo_text = f"Holo Condition: {report.grading_factors.foil_condition:.1f}"
                cv2.putText(panel, holo_text, (10, y_pos), font, font_scale - 0.1, (255, 255, 0), 1)
                y_pos += 20
            
            # Confidence
            y_pos += 10
            conf_text = f"Confidence: {report.confidence_score:.2f}"
            cv2.putText(panel, conf_text, (10, y_pos), font, font_scale - 0.1, (200, 200, 200), 1)
            
            # Place panel on image
            visualization[h-panel_height:h, w-panel_width:w] = panel
        
        return visualization
    
    def save_grading_report(self, report: PSAGradingReport, output_path: str):
        """Save detailed grading report to file."""
        report_data = {
            "psa_grade": {
                "numerical_grade": report.final_grade.grade,
                "description": report.final_grade.description,
                "condition": report.final_grade.condition
            },
            "component_grades": {
                "centering": report.grading_factors.centering,
                "corners": report.grading_factors.corners,
                "edges": report.grading_factors.edges,
                "surface": report.grading_factors.surface
            },
            "additional_factors": {
                "artwork_condition": report.grading_factors.artwork_condition,
                "text_clarity": report.grading_factors.text_clarity,
                "eye_appeal": report.grading_factors.eye_appeal,
                "print_quality": report.grading_factors.print_quality
            },
            "holographic_analysis": {
                "has_holographic": report.grading_factors.has_holographic,
                "pattern_quality": report.grading_factors.holo_pattern_quality,
                "foil_condition": report.grading_factors.foil_condition
            } if report.grading_factors.has_holographic else None,
            "assessment_notes": {
                "key_observations": report.key_observations,
                "grade_limiting_factors": report.grade_limiting_factors,
                "positives": report.positives,
                "areas_of_concern": report.areas_of_concern
            },
            "technical_analysis": {
                "confidence_score": report.confidence_score,
                "image_quality_score": report.image_quality_score,
                "segmentation_quality": report.segmentation_quality,
                "processing_time": report.processing_time,
                "preprocessing_steps": report.preprocessing_steps
            },
            "detailed_analysis": report.detailed_analysis
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Grading report saved to {output_path}")