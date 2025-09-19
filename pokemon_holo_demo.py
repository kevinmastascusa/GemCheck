"""
Demonstration of the enhanced holographic-aware Pokémon card grading system.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

# Import the enhanced Pokémon analysis system
from app.pokemon import (
    analyze_pokemon_card,
    create_holographic_overlay,
    generate_pokemon_report,
    HolographicVisualizer,
    HolographicOverlaySettings,
    HolographicGradingIntegrator,
    PokemonRarityDetector,
    PokemonVisualAnalyzer
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_holographic_analysis(image_path: str, output_dir: str = "demo_output"):
    """
    Demonstrate comprehensive holographic card analysis.
    
    Args:
        image_path: Path to card image
        output_dir: Directory for output files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Load and prepare image
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Simulate OCR text (in real usage, this would come from OCR)
        sample_text = "Charizard HP 120 Fire Type Pokemon"
        
        print("Performing comprehensive holographic analysis...")
        
        # 1. Complete analysis with holographic awareness
        result = analyze_pokemon_card(image_rgb, text_content=sample_text)
        
        # 2. Display results
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Card Type: {result.card_type.value if result.card_type else 'Unknown'}")
        print(f"Rarity: {result.rarity.value if result.rarity else 'Unknown'}")
        print(f"Era: {result.era.value if result.era else 'Unknown'}")
        print(f"Has Holographic: {result.has_holographic}")
        
        print(f"\n=== GRADING RESULTS ===")
        print(f"Overall Grade: {result.overall_grade:.1f}")
        print(f"Centering: {result.centering_grade:.1f}")
        print(f"Corners: {result.corners_grade:.1f}")
        print(f"Edges: {result.edges_grade:.1f}")
        print(f"Surface: {result.surface_grade:.1f}")
        
        if result.has_holographic and result.holo_factors:
            print(f"\n=== HOLOGRAPHIC ANALYSIS ===")
            print(f"Holo-Adjusted Grade: {result.holo_adjusted_grade:.1f}")
            print(f"Holo Condition Score: {result.holo_factors.holo_condition_score:.1f}")
            print(f"Pattern Quality: {result.holo_factors.holo_pattern_quality:.2f}")
            print(f"Scratch Penalty: {result.holo_factors.holo_scratch_penalty:.2f}")
            print(f"Foil Peeling Penalty: {result.holo_factors.foil_peeling_penalty:.2f}")
            print(f"Rainbow Integrity: {result.holo_factors.rainbow_integrity:.2f}")
            print(f"Shine Consistency: {result.holo_factors.shine_consistency:.2f}")
        
        # 3. Generate detailed analysis components
        print("\nGenerating detailed analysis components...")
        
        integrator = HolographicGradingIntegrator()
        rarity_detector = PokemonRarityDetector()
        visual_analyzer = PokemonVisualAnalyzer()
        
        # Get detailed analysis components
        rarity, rarity_features = rarity_detector.detect_rarity(image_rgb, sample_text)
        regions, defects = visual_analyzer.analyze_pokemon_card(
            image_rgb, result.card_type, rarity, result.era)
        
        # 4. Create visualizations
        print("Creating holographic visualizations...")
        
        # Standard holographic overlay
        overlay_settings = HolographicOverlaySettings(
            show_holo_mask=True,
            show_scratch_lines=True,
            show_peeling_areas=True,
            show_pattern_analysis=True,
            show_rainbow_disruption=True,
            overlay_alpha=0.5,
            highlight_severity=True
        )
        
        holo_overlay = create_holographic_overlay(
            image_rgb, rarity_features, defects, regions, overlay_settings)
        
        # Comprehensive grading overlay
        comprehensive_overlay = integrator.create_comprehensive_overlay(
            image_rgb, result, rarity_features, defects, regions)
        
        # Holographic severity heatmap
        visualizer = HolographicVisualizer()
        severity_heatmap = visualizer.create_holo_severity_heatmap(image_rgb, defects)
        
        # Comparison view
        comparison_view = visualizer.create_holo_comparison_overlay(
            image_rgb, holo_overlay, rarity_features)
        
        # 5. Save visualizations
        print("Saving visualization outputs...")
        
        cv2.imwrite(str(output_path / "01_original.jpg"), 
                   cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / "02_holo_overlay.jpg"), 
                   cv2.cvtColor(holo_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / "03_comprehensive_overlay.jpg"), 
                   cv2.cvtColor(comprehensive_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / "04_severity_heatmap.jpg"), 
                   cv2.cvtColor(severity_heatmap, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / "05_comparison_view.jpg"), 
                   cv2.cvtColor(comparison_view, cv2.COLOR_RGB2BGR))
        
        # 6. Generate PDF report
        print("Generating comprehensive PDF report...")
        
        report_path = output_path / "holographic_analysis_report.pdf"
        success = generate_pokemon_report(image_rgb, result, str(report_path))
        
        if success:
            print(f"PDF report generated: {report_path}")
        else:
            print("Failed to generate PDF report")
        
        # 7. Create analysis summary
        create_analysis_summary(result, rarity_features, defects, output_path)
        
        print(f"\n=== DEMO COMPLETE ===")
        print(f"All outputs saved to: {output_path}")
        print("\nGenerated files:")
        for file in sorted(output_path.iterdir()):
            print(f"  - {file.name}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error during analysis: {e}")

def create_analysis_summary(result, rarity_features, defects, output_path):
    """Create a text summary of the analysis."""
    summary_path = output_path / "analysis_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("POKEMON CARD HOLOGRAPHIC ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CARD IDENTIFICATION:\n")
        f.write(f"  Type: {result.card_type.value if result.card_type else 'Unknown'}\n")
        f.write(f"  Rarity: {result.rarity.value if result.rarity else 'Unknown'}\n")
        f.write(f"  Era: {result.era.value if result.era else 'Unknown'}\n")
        f.write(f"  Holographic: {'Yes' if result.has_holographic else 'No'}\n\n")
        
        f.write("GRADING SCORES:\n")
        f.write(f"  Overall Grade: {result.overall_grade:.1f}/10\n")
        f.write(f"  Centering: {result.centering_grade:.1f}/10\n")
        f.write(f"  Corners: {result.corners_grade:.1f}/10\n")
        f.write(f"  Edges: {result.edges_grade:.1f}/10\n")
        f.write(f"  Surface: {result.surface_grade:.1f}/10\n")
        
        if result.has_holographic and result.holo_factors:
            f.write(f"  Holo-Adjusted Grade: {result.holo_adjusted_grade:.1f}/10\n\n")
            
            f.write("HOLOGRAPHIC ANALYSIS:\n")
            f.write(f"  Holo Condition: {result.holo_factors.holo_condition_score:.1f}/10\n")
            f.write(f"  Pattern Quality: {result.holo_factors.holo_pattern_quality:.2f}\n")
            f.write(f"  Scratch Penalty: {result.holo_factors.holo_scratch_penalty:.2f}\n")
            f.write(f"  Foil Peeling: {result.holo_factors.foil_peeling_penalty:.2f}\n")
            f.write(f"  Rainbow Integrity: {result.holo_factors.rainbow_integrity:.2f}\n")
            f.write(f"  Shine Consistency: {result.holo_factors.shine_consistency:.2f}\n\n")
        
        if result.has_holographic:
            f.write("HOLOGRAPHIC FEATURES:\n")
            f.write(f"  Pattern Type: {rarity_features.holo_pattern_type}\n")
            f.write(f"  Intensity: {rarity_features.holo_intensity:.1%}\n")
            f.write(f"  Shine Intensity: {rarity_features.shine_intensity:.2f}\n")
            f.write(f"  Reverse Holo: {'Yes' if rarity_features.has_reverse_holo else 'No'}\n")
            f.write(f"  Rainbow Foil: {'Yes' if rarity_features.has_rainbow_foil else 'No'}\n")
            f.write(f"  Gold Foil: {'Yes' if rarity_features.has_gold_foil else 'No'}\n")
            f.write(f"  Texture: {'Yes' if rarity_features.has_texture else 'No'}\n\n")
        
        f.write("DEFECT ANALYSIS:\n")
        f.write(f"  Holo Scratches: {len(defects.holo_scratches)}\n")
        f.write(f"  Foil Peeling Areas: {len(defects.foil_peeling)}\n")
        f.write(f"  Surface Scratches: {len(defects.surface_scratches)}\n")
        f.write(f"  Print Lines: {len(defects.print_lines)}\n")
        f.write(f"  Indentations: {len(defects.indentations)}\n")
        f.write(f"  Overall Holo Wear: {defects.holo_wear:.1%}\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        if result.holo_adjusted_grade >= 9.0:
            f.write("  Excellent condition holographic card with minimal defects.\n")
        elif result.holo_adjusted_grade >= 7.0:
            f.write("  Good condition with minor holographic imperfections.\n")
        elif result.holo_adjusted_grade >= 5.0:
            f.write("  Moderate condition with visible holographic wear.\n")
        else:
            f.write("  Poor condition with significant holographic damage.\n")

def test_with_sample_images():
    """Test the system with sample/placeholder images."""
    print("Creating sample test images for demonstration...")
    
    # Create sample images for testing
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create a sample holographic-style image
    sample_image = np.zeros((800, 600, 3), dtype=np.uint8)
    
    # Add some colorful patterns to simulate holographic effects
    for y in range(0, 800, 20):
        for x in range(0, 600, 20):
            hue = (x + y) % 180
            color = cv2.cvtColor(np.array([[[hue, 255, 200]]], dtype=np.uint8), 
                               cv2.COLOR_HSV2RGB)[0, 0]
            cv2.rectangle(sample_image, (x, y), (x+15, y+15), color.tolist(), -1)
    
    # Add some simulated scratches
    cv2.line(sample_image, (100, 100), (500, 150), (255, 255, 255), 2)
    cv2.line(sample_image, (200, 300), (400, 280), (255, 255, 255), 1)
    
    # Save sample image
    sample_path = output_dir / "sample_holo_card.jpg"
    cv2.imwrite(str(sample_path), cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    
    print(f"Created sample image: {sample_path}")
    
    # Run demo with sample image
    demo_holographic_analysis(str(sample_path))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use provided image path
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "demo_output"
        demo_holographic_analysis(image_path, output_dir)
    else:
        # Use sample images for testing
        print("No image path provided. Running with sample test images...")
        test_with_sample_images()