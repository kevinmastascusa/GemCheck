#!/usr/bin/env python3
"""
Test script for reference template functionality.
Demonstrates calibrated grading using the Pikachu Yellow Cheeks reference template.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_reference_template():
    """Test the reference template system with real Pokemon cards."""
    print("Testing Reference Template System...")
    
    try:
        # Import the Pokemon analysis functions
        from app.pokemon import analyze_pokemon_card, create_reference_overlay
        from app.pokemon.reference_template import ReferenceTemplateProcessor
        
        # Load the reference template
        reference_processor = ReferenceTemplateProcessor()
        
        print("Reference Template Info:")
        ref_info = reference_processor.get_reference_info()
        if ref_info:
            print(f"  Card: {ref_info['card_name']} ({ref_info['card_set']})")
            print(f"  Era: {ref_info['card_era']}")
            print(f"  Dimensions: {ref_info['dimensions']['width']}x{ref_info['dimensions']['height']}")
            print(f"  Reference Points: {ref_info['reference_points_count']}")
            print(f"  Standards: Centering tolerance {ref_info['standards']['centering']['horizontal_tolerance']:.1%}")
        else:
            print("  [ERROR] Failed to load reference template")
            return False
        
        # Test with the downloaded Charizard card
        test_image_path = project_root / "test_charizard.png"
        
        if not test_image_path.exists():
            print("📥 Downloading test card for comparison...")
            import urllib.request
            card_url = "https://images.pokemontcg.io/base1/4_hires.png"
            urllib.request.urlretrieve(card_url, test_image_path)
            print(f"  Downloaded: {test_image_path}")
        
        # Load the test card
        print("🃏 Loading test card...")
        test_image = cv2.imread(str(test_image_path))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"  Test card loaded: {test_image.shape}")
        
        # Test calibration
        print("⚙️ Testing calibration...")
        calibration_data = reference_processor.calibrate_analysis(test_image)
        
        if calibration_data:
            scale_x = calibration_data['scale_factors']['x']
            scale_y = calibration_data['scale_factors']['y']
            print(f"  ✅ Calibration successful!")
            print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
            
            # Show adjusted standards
            centering_std = calibration_data['centering_standards']
            print(f"  Adjusted centering tolerance: {centering_std['horizontal_tolerance']:.1%}")
            print(f"  Adjusted max error: {centering_std['max_error']:.1%}")
        else:
            print("  ❌ Calibration failed")
            return False
        
        # Create reference overlay
        print("🎨 Creating reference overlay...")
        overlay_image = create_reference_overlay(test_image)
        
        # Save overlay image
        overlay_path = project_root / "test_reference_overlay.png"
        overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        print(f"  Reference overlay saved: {overlay_path}")
        
        # Test calibrated analysis
        print("🔬 Running calibrated analysis...")
        analysis_result = analyze_pokemon_card(
            test_image, 
            text_content="Charizard Base Set", 
            use_reference_template=True
        )
        
        if analysis_result:
            print(f"  ✅ Analysis completed!")
            print(f"  Overall grade: {analysis_result.overall_grade}")
            print(f"  Centering: {getattr(analysis_result, 'centering_grade', 'N/A')}")
            print(f"  Surface: {getattr(analysis_result, 'surface_grade', 'N/A')}")
            print(f"  Edges: {getattr(analysis_result, 'edges_grade', 'N/A')}")
            print(f"  Corners: {getattr(analysis_result, 'corners_grade', 'N/A')}")
        else:
            print("  ❌ Analysis failed")
            return False
        
        # Test without reference template for comparison
        print("🔬 Running analysis without reference template...")
        uncalibrated_result = analyze_pokemon_card(
            test_image,
            text_content="Charizard Base Set", 
            use_reference_template=False
        )
        
        if uncalibrated_result:
            print(f"  Without calibration - Overall grade: {uncalibrated_result.overall_grade}")
            
            grade_diff = analysis_result.overall_grade - uncalibrated_result.overall_grade
            print(f"  Grade difference with calibration: {grade_diff:+.1f}")
            
            if abs(grade_diff) > 0.1:
                print(f"  📊 Reference template made a {abs(grade_diff):.1f} point difference!")
            else:
                print(f"  📊 Reference template provided minor adjustment")
        
        print("\n🎉 Reference template test completed successfully!")
        print(f"\nFiles created:")
        print(f"  - Reference overlay: {overlay_path}")
        print(f"  - Test card: {test_image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_cards():
    """Test reference template with multiple card types."""
    print("\n🔍 Testing with multiple card types...")
    
    try:
        from app.pokemon import analyze_pokemon_card, create_reference_overlay
        
        # Test cards with different characteristics
        test_cards = [
            {
                "name": "Charizard Base Set",
                "url": "https://images.pokemontcg.io/base1/4_hires.png",
                "expected_era": "vintage"
            },
            {
                "name": "Pikachu Promo",
                "url": "https://images.pokemontcg.io/smp/SM04_hires.png", 
                "expected_era": "modern"
            }
        ]
        
        results = []
        
        for i, card_info in enumerate(test_cards):
            print(f"\n📋 Testing card {i+1}: {card_info['name']}")
            
            # Download card
            card_path = project_root / f"test_card_{i+1}.png"
            if not card_path.exists():
                print(f"  📥 Downloading {card_info['name']}...")
                import urllib.request
                urllib.request.urlretrieve(card_info['url'], card_path)
            
            # Load and analyze
            image = cv2.imread(str(card_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Analyze with reference template
            result = analyze_pokemon_card(
                image,
                text_content=card_info['name'],
                use_reference_template=True
            )
            
            if result:
                results.append({
                    "name": card_info['name'],
                    "grade": result.overall_grade,
                    "era": card_info['expected_era']
                })
                
                print(f"    Grade: {result.overall_grade}")
            else:
                print(f"    ❌ Analysis failed")
        
        # Compare results
        if len(results) >= 2:
            vintage_grade = next((r["grade"] for r in results if r["era"] == "vintage"), None)
            modern_grade = next((r["grade"] for r in results if r["era"] == "modern"), None)
            
            if vintage_grade and modern_grade:
                print(f"\n📊 Era Comparison:")
                print(f"  Vintage card grade: {vintage_grade}")
                print(f"  Modern card grade: {modern_grade}")
                print(f"  Difference: {modern_grade - vintage_grade:+.1f} (modern vs vintage)")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple card test failed: {e}")
        return False

if __name__ == "__main__":
    print("PSA Pre-Grader Reference Template Testing")
    print("=" * 50)
    
    # Test basic reference template functionality
    success = test_reference_template()
    
    if success:
        # Test with multiple cards
        test_multiple_cards()
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!" if success else "❌ Testing failed!")