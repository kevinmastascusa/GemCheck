#!/usr/bin/env python3
"""
Test the reference grading methodology based on the Pikachu template.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_reference_grading():
    """Test the reference-based grading methodology."""
    print("Testing Reference-Based Grading Methodology")
    print("=" * 50)
    
    try:
        from app.pokemon.reference_grading_methodology import ReferenceGradingMethodology, GradingZone
        
        # Initialize the grading system
        grader = ReferenceGradingMethodology()
        print("Reference grading methodology initialized")
        
        # Test with Charizard card
        test_image_path = project_root / "test_charizard.png"
        
        if not test_image_path.exists():
            print("Downloading test Charizard card...")
            import urllib.request
            card_url = "https://images.pokemontcg.io/base1/4_hires.png"
            urllib.request.urlretrieve(card_url, test_image_path)
            print(f"Downloaded: {test_image_path}")
        
        # Load and analyze the card
        print(f"Loading card image: {test_image_path}")
        test_image = cv2.imread(str(test_image_path))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"Image loaded: {test_image.shape}")
        
        # Grade the card using reference methodology
        print("\nGrading card using reference template methodology...")
        zone_results = grader.grade_card_using_reference_methodology(
            test_image, 
            card_era="vintage"  # Charizard Base Set is vintage
        )
        
        # Display zone results
        print("\nZone Grading Results:")
        print("-" * 30)
        
        for zone, result in zone_results.items():
            print(f"\n{zone.value.upper()}:")
            print(f"  Raw Score: {result.raw_score:.1f}")
            print(f"  Adjusted Score: {result.adjusted_score:.1f}")
            print(f"  Confidence: {result.confidence:.2f}")
            
            if result.defects_found:
                print(f"  Defects:")
                for defect in result.defects_found[:3]:  # Show first 3 defects
                    print(f"    - {defect}")
            
            if result.measurements:
                print(f"  Key Measurements:")
                for key, value in list(result.measurements.items())[:3]:  # Show first 3 measurements
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.3f}")
                    else:
                        print(f"    - {key}: {value}")
        
        # Calculate overall grade
        print("\nCalculating overall grade...")
        overall_result = grader.calculate_overall_grade(zone_results)
        
        print("\nOVERALL GRADING RESULT:")
        print("=" * 30)
        print(f"PSA Grade: {overall_result['psa_grade']}")
        print(f"Grade Label: {overall_result['grade_label']}")
        print(f"Overall Score: {overall_result['overall_score']:.1f}/100")
        print(f"Confidence: {overall_result['confidence']:.2f}")
        print(f"Methodology: {overall_result['grading_methodology']}")
        
        print(f"\nZone Breakdown:")
        for zone, score in overall_result['zone_scores'].items():
            print(f"  {zone.title()}: {score:.1f}")
        
        if overall_result['defects_found']:
            print(f"\nDefects Found ({len(overall_result['defects_found'])}):")
            for i, defect in enumerate(overall_result['defects_found'][:5]):  # Show first 5
                print(f"  {i+1}. {defect}")
            if len(overall_result['defects_found']) > 5:
                print(f"  ... and {len(overall_result['defects_found']) - 5} more")
        
        # Test with different eras
        print(f"\n" + "=" * 50)
        print("Testing Era Adjustments:")
        
        eras_to_test = ["vintage", "modern"]
        for era in eras_to_test:
            print(f"\nTesting with {era} era standards...")
            era_zone_results = grader.grade_card_using_reference_methodology(test_image, card_era=era)
            era_overall = grader.calculate_overall_grade(era_zone_results)
            
            print(f"  {era.title()} Era - PSA Grade: {era_overall['psa_grade']} ({era_overall['grade_label']})")
            print(f"  Overall Score: {era_overall['overall_score']:.1f}")
        
        print(f"\n" + "=" * 50)
        print("SUCCESS: Reference grading methodology test completed!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reference_grading()
    
    print("\n" + "=" * 50)
    print("Test Result:", "PASSED" if success else "FAILED")