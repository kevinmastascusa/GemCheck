"""
Pokémon card analysis package with comprehensive holographic awareness.

Data Attribution:
This package uses official Pokémon TCG data from the PokemonTCG/pokemon-tcg-data repository
(https://github.com/PokemonTCG/pokemon-tcg-data) under MIT License for card identification,
set information, and game mechanics data.
"""

from .card_types import (
    PokemonCardType,
    PokemonRarity,
    PokemonCardEra,
    PokemonCardParts,
    PokemonCardDetector,
    PokemonCardAnalysis,
    GradingCriteria,
    get_pokemon_card_parts,
    get_era_specific_considerations,
    POKEMON_DEFECT_PATTERNS
)

from .rarity_detector import (
    PokemonRarityDetector, RarityFeatures
)

from .visual_analyzer import (
    PokemonVisualAnalyzer, PokemonDefectAnalysis, PokemonCardRegions
)

from .template_processor import (
    PokemonCardTemplateProcessor, CardTemplate, CardPartOutline
)

from .tcg_data_integration import (
    PokemonTCGDataManager
)

from .pdf_report_generator import (
    PokemonCardPDFReportGenerator, GradingReportData
)

from .holo_visualizer import (
    HolographicVisualizer, HolographicOverlaySettings
)

from .grading_integration import (
    HolographicGradingIntegrator, EnhancedGradingResult, HolographicGradingFactors
)

from .reference_template import (
    ReferenceTemplateProcessor, ReferenceTemplate, ReferencePoint
)

__version__ = "1.0.0"
__author__ = "PSA Pregrader Team"

# Main interface functions
def analyze_pokemon_card(image, text_content="", include_holo_analysis=True, use_reference_template=True):
    """
    Comprehensive Pokémon card analysis with holographic awareness and reference template calibration.
    
    Args:
        image: Card image (numpy array or file path)
        text_content: OCR text from the card
        include_holo_analysis: Whether to include holographic analysis
        use_reference_template: Whether to use reference template for calibrated grading
        
    Returns:
        Complete analysis results including rarity, defects, and grading
    """
    from .grading_integration import HolographicGradingIntegrator
    from .reference_template import ReferenceTemplateProcessor
    
    # Load image if path provided
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize grading integrator
    integrator = HolographicGradingIntegrator()
    
    # Apply reference template calibration if requested
    calibration_data = {}
    if use_reference_template:
        try:
            reference_processor = ReferenceTemplateProcessor()
            calibration_data = reference_processor.calibrate_analysis(image)
            if calibration_data:
                print(f"[OK] Reference template calibration applied (scale: x={calibration_data['scale_factors']['x']:.2f}, y={calibration_data['scale_factors']['y']:.2f})")
            else:
                print("[WARNING] Reference template calibration failed, using default standards")
        except Exception as e:
            print(f"[WARNING] Reference template error: {e}")
    
    # Run analysis with calibration data
    result = integrator.grade_card_with_holo_awareness(
        image, 
        text_content=text_content,
        calibration_data=calibration_data
    )
    
    return result

def create_reference_overlay(image, show_reference_points=True):
    """
    Create reference template overlay showing calibration points.
    
    Args:
        image: Card image
        show_reference_points: Whether to show reference measurement points
        
    Returns:
        Image with reference template overlay
    """
    from .reference_template import ReferenceTemplateProcessor
    
    try:
        reference_processor = ReferenceTemplateProcessor()
        calibration_data = reference_processor.calibrate_analysis(image)
        
        if calibration_data and show_reference_points:
            return reference_processor.create_calibrated_overlay(image, calibration_data)
        else:
            return image
    except Exception as e:
        print(f"Reference overlay error: {e}")
        return image

def create_holographic_overlay(image, rarity_features, defect_analysis, regions, settings=None):
    """
    Create holographic analysis overlay for visualization.
    
    Args:
        image: Card image
        rarity_features: Detected rarity features
        defect_analysis: Defect analysis results
        regions: Card regions
        settings: Overlay settings
        
    Returns:
        Image with holographic overlay
    """
    from .holo_visualizer import HolographicVisualizer
    
    visualizer = HolographicVisualizer()
    return visualizer.create_holographic_overlay(
        image, rarity_features, defect_analysis, regions, settings)

def generate_pokemon_report(image, analysis_result, output_path):
    """
    Generate comprehensive PDF report for Pokémon card analysis.
    
    Args:
        image: Card image
        analysis_result: Analysis results
        output_path: Output PDF file path
        
    Returns:
        Success status
    """
    try:
        from .pdf_report_generator import PokemonCardPDFReportGenerator, GradingReportData
        from .template_processor import CardTemplate, CardPartOutline
        from .visual_analyzer import PokemonDefectAnalysis
        from datetime import datetime
        
        # Create simplified template and defect analysis for demo
        template = CardTemplate(
            card_type=analysis_result.card_type or PokemonCardType.POKEMON,
            parts=[],
            confidence=0.9
        )
        
        defects = PokemonDefectAnalysis()
        
        # Convert analysis result to report data
        report_data = GradingReportData(
            card_name="Test Card",
            set_name="Demo Set",
            card_number="001/001",
            rarity=analysis_result.rarity or PokemonRarity.COMMON,
            era=analysis_result.era or PokemonCardEra.SCARLET_VIOLET,
            template=template,
            defect_analysis=defects,
            overall_grade=analysis_result.overall_grade,
            centering_score=analysis_result.centering_grade,
            surface_score=analysis_result.surface_grade,
            edges_score=analysis_result.edges_grade,
            corners_score=analysis_result.corners_grade,
            grade_reasoning="Automated analysis results",
            improvement_suggestions=["Sample improvement suggestion"],
            original_image=image,
            overlay_images={},
            analysis_date=datetime.now(),
            processing_time=1.0
        )
        
        generator = PokemonCardPDFReportGenerator()
        return generator.generate_report(image, report_data, output_path)
    except Exception as e:
        print(f"Report generation failed: {e}")
        return False

__all__ = [
    # Core types
    'PokemonCardType', 'PokemonRarity', 'PokemonCardEra', 'PokemonCardParts',
    'PokemonCardDetector', 'PokemonCardAnalysis', 'GradingCriteria',
    'get_pokemon_card_parts', 'get_era_specific_considerations', 'POKEMON_DEFECT_PATTERNS',
    
    # Analysis components  
    'PokemonRarityDetector', 'RarityFeatures',
    'PokemonVisualAnalyzer', 'PokemonDefectAnalysis', 'PokemonCardRegions',
    'PokemonCardTemplateProcessor', 'CardTemplate', 'CardPartOutline',
    'PokemonTCGDataManager',
    
    # Holographic components
    'HolographicVisualizer', 'HolographicOverlaySettings',
    'HolographicGradingIntegrator', 'EnhancedGradingResult', 'HolographicGradingFactors',
    
    # Reference template components
    'ReferenceTemplateProcessor', 'ReferenceTemplate', 'ReferencePoint',
    
    # Report generation
    'PokemonCardPDFReportGenerator', 'GradingReportData',
    
    # Main interface functions
    'analyze_pokemon_card',
    'create_holographic_overlay',
    'create_reference_overlay',
    'generate_pokemon_report'
]