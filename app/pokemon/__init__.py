"""
Pokémon card analysis package with comprehensive holographic awareness.
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

__version__ = "1.0.0"
__author__ = "PSA Pregrader Team"

# Main interface functions
def analyze_pokemon_card(image, text_content="", include_holo_analysis=True):
    """
    Comprehensive Pokémon card analysis with holographic awareness.
    
    Args:
        image: Card image (numpy array or file path)
        text_content: OCR text from the card
        include_holo_analysis: Whether to include holographic analysis
        
    Returns:
        Complete analysis results including rarity, defects, and grading
    """
    from .grading_integration import HolographicGradingIntegrator
    
    integrator = HolographicGradingIntegrator()
    
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = integrator.grade_card_with_holo_awareness(image, text_content=text_content)
    return result

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
    
    # Report generation
    'PokemonCardPDFReportGenerator', 'GradingReportData',
    
    # Main interface functions
    'analyze_pokemon_card',
    'create_holographic_overlay', 
    'generate_pokemon_report'
]