#!/usr/bin/env python3
"""
Test script for Unified PSA Grading System.
Demonstrates comprehensive human-like grading of Pokemon cards.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_unified_psa_grading():
    """Test the unified PSA grading system."""
    print("Testing Unified PSA Grading System")
    print("=" * 50)
    
    try:
        from app.pokemon.unified_psa_grader import UnifiedPSAGrader, PSAGradeLevel
        from app.pokemon.preprocessing_pipeline import PreprocessingLevel
        
        # Initialize the unified grader
        grader = UnifiedPSAGrader()
        print("Unified PSA grading system initialized")
        
        # Test with Charizard card
        test_image_path = project_root / "test_charizard.png"
        
        if not test_image_path.exists():
            print("Downloading test Charizard card...")
            import urllib.request
            card_url = "https://images.pokemontcg.io/base1/4_hires.png"
            urllib.request.urlretrieve(card_url, test_image_path)
            print(f"Downloaded: {test_image_path}")
        
        # Load test image
        print(f"\\nLoading card image: {test_image_path}")
        test_image = cv2.imread(str(test_image_path))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"Image loaded: {test_image.shape}")
        
        # Test different processing levels
        processing_levels = [
            (PreprocessingLevel.BASIC, "Basic Processing"),
            (PreprocessingLevel.STANDARD, "Standard Processing"),
            (PreprocessingLevel.ENHANCED, "Enhanced Processing")
        ]
        
        print(f"\\nTesting Unified PSA Grading with different processing levels...")
        
        for level, level_name in processing_levels:
            print(f"\\n{'-' * 30}")
            print(f"Testing {level_name}")
            print(f"{'-' * 30}")
            
            # Grade the card
            grading_report = grader.grade_card(
                image=test_image,
                card_era="vintage",
                processing_level=level,
                ocr_text="Charizard 120 HP Fire Pokemon"
            )
            
            # Display results
            print(f"\\nPSA GRADING RESULTS:")
            print(f"Final Grade: {grading_report.final_grade.grade} - {grading_report.final_grade.description}")
            print(f"Condition: {grading_report.final_grade.condition}")
            print(f"Confidence: {grading_report.confidence_score:.3f}")
            print(f"Processing Time: {grading_report.processing_time:.2f}s")
            
            print(f"\\nCOMPONENT GRADES:")
            factors = grading_report.grading_factors
            print(f"  Centering: {factors.centering:.1f}")
            print(f"  Corners: {factors.corners:.1f}")
            print(f"  Edges: {factors.edges:.1f}")
            print(f"  Surface: {factors.surface:.1f}")
            
            print(f"\\nADDITIONAL FACTORS:")
            print(f"  Artwork Condition: {factors.artwork_condition:.1f}")
            print(f"  Text Clarity: {factors.text_clarity:.1f}")
            print(f"  Eye Appeal: {factors.eye_appeal:.1f}")
            print(f"  Print Quality: {factors.print_quality:.1f}")
            print(f"  Structural Integrity: {factors.structural_integrity:.1f}")
            
            if factors.has_holographic:
                print(f"\\nHOLOGRAPHIC ANALYSIS:")
                print(f"  Pattern Quality: {factors.holo_pattern_quality:.1f}")
                print(f"  Foil Condition: {factors.foil_condition:.1f}")
                print(f"  Scratch Penalty: {factors.holo_scratch_penalty:.3f}")
            
            print(f"\\nKEY OBSERVATIONS:")
            for observation in grading_report.key_observations:
                print(f"  - {observation}")
            
            if grading_report.grade_limiting_factors:
                print(f"\\nGRADE LIMITING FACTORS:")
                for factor in grading_report.grade_limiting_factors:
                    print(f"  - {factor}")
            
            if grading_report.positives:
                print(f"\\nPOSITIVE ASPECTS:")
                for positive in grading_report.positives:
                    print(f"  - {positive}")
            
            if grading_report.areas_of_concern:
                print(f"\\nAREAS OF CONCERN:")
                for concern in grading_report.areas_of_concern:
                    print(f"  - {concern}")
            
            print(f"\\nTECHNICAL ANALYSIS:")
            print(f"  Image Quality Score: {grading_report.image_quality_score:.1f}")
            print(f"  Segmentation Quality: {grading_report.segmentation_quality:.3f}")
            print(f"  Preprocessing Steps: {len(grading_report.preprocessing_steps)}")
            
            # Create visualization
            visualization = grader.create_grading_visualization(test_image, grading_report)
            
            # Save visualization
            viz_filename = f"charizard_psa_grading_{level.value}.png"
            viz_path = project_root / viz_filename
            viz_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(viz_path), viz_bgr)
            print(f"\\nGrading visualization saved: {viz_path}")
            
            # Save detailed report
            report_filename = f"charizard_psa_report_{level.value}.json"
            report_path = project_root / report_filename
            grader.save_grading_report(grading_report, str(report_path))
            print(f"Detailed report saved: {report_path}")
        
        # Test with different eras
        print(f"\\n{'=' * 60}")
        print("TESTING DIFFERENT CARD ERAS")
        print(f"{'=' * 60}")
        
        eras_to_test = [
            ("vintage", "Vintage Era (Base Set)"),
            ("modern", "Modern Era (Current)")
        ]
        
        for era, era_description in eras_to_test:
            print(f"\\nTesting {era_description}:")
            
            grading_report = grader.grade_card(
                image=test_image,
                card_era=era,
                processing_level=PreprocessingLevel.STANDARD,
                ocr_text="Charizard 120 HP Fire Pokemon"
            )
            
            print(f"  Era: {era}")
            print(f"  Final Grade: {grading_report.final_grade.grade}")
            print(f"  Key Factors: Centering={grading_report.grading_factors.centering:.1f}, "
                  f"Corners={grading_report.grading_factors.corners:.1f}, "
                  f"Edges={grading_report.grading_factors.edges:.1f}, "
                  f"Surface={grading_report.grading_factors.surface:.1f}")
            print(f"  Confidence: {grading_report.confidence_score:.3f}")
        
        # Grade comparison summary
        print(f"\\n{'=' * 60}")
        print("GRADING SYSTEM SUMMARY")
        print(f"{'=' * 60}")
        
        # Final test with standard settings
        final_report = grader.grade_card(
            image=test_image,
            card_era="vintage",
            processing_level=PreprocessingLevel.STANDARD,
            ocr_text="Charizard 120 HP Fire Pokemon"
        )
        
        print(f"\\nFINAL PSA GRADE ASSESSMENT:")
        print(f"Card: 1998 Pokemon Base Set Charizard")
        print(f"Grade: PSA {final_report.final_grade.grade} - {final_report.final_grade.description}")
        print(f"Condition: {final_report.final_grade.condition}")
        
        print(f"\\nGRADING BREAKDOWN:")
        factors = final_report.grading_factors
        print(f"  Primary Components:")
        print(f"    Centering: {factors.centering:.1f}/10")
        print(f"    Corners: {factors.corners:.1f}/10")
        print(f"    Edges: {factors.edges:.1f}/10")
        print(f"    Surface: {factors.surface:.1f}/10")
        
        print(f"\\n  Secondary Factors:")
        print(f"    Eye Appeal: {factors.eye_appeal:.1f}/10")
        print(f"    Print Quality: {factors.print_quality:.1f}/10")
        print(f"    Structural Integrity: {factors.structural_integrity:.1f}/10")
        
        print(f"\\n  Calculated Values:")
        print(f"    Component Average: {factors.component_average:.1f}")
        print(f"    Lowest Component: {factors.lowest_component:.1f}")
        print(f"    Final Grade: {factors.final_grade:.1f}")
        
        print(f"\\nASSESSMENT CONFIDENCE: {final_report.confidence_score:.1%}")
        print(f"PROCESSING TIME: {final_report.processing_time:.2f} seconds")
        
        # Human-like grading summary
        print(f"\\n{'=' * 60}")
        print("HUMAN GRADER STYLE ASSESSMENT")
        print(f"{'=' * 60}")
        
        print(f"\\nThis card would likely receive a PSA {final_report.final_grade.grade} grade based on:")
        
        print(f"\\nStrengths:")
        for positive in final_report.positives[:3]:  # Top 3 positives
            print(f"  + {positive}")
        
        print(f"\\nWeaknesses:")
        for concern in final_report.areas_of_concern[:3]:  # Top 3 concerns
            print(f"  - {concern}")
        
        if final_report.grade_limiting_factors:
            print(f"\\nPrimary Grade Limiters:")
            for limiter in final_report.grade_limiting_factors[:2]:  # Top 2 limiters
                print(f"  ! {limiter}")
        
        print(f"\\n" + "=" * 60)
        print("SUCCESS: Unified PSA grading system test completed!")
        
        print(f"\\nFiles created:")
        print(f"  - PSA grading visualizations: charizard_psa_grading_*.png")
        print(f"  - Detailed reports: charizard_psa_report_*.json")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grade_level_accuracy():
    """Test grade level accuracy and thresholds."""
    print(f"\\nTesting Grade Level Accuracy...")
    
    try:
        from app.pokemon.unified_psa_grader import UnifiedPSAGrader, PSAGradeLevel
        
        grader = UnifiedPSAGrader()
        
        # Test grade thresholds
        print(f"\\nGrade Threshold Analysis:")
        
        test_scores = [
            (10.0, 10.0, 10.0, 10.0, "Perfect card"),
            (9.5, 9.8, 9.5, 9.5, "Near perfect"),
            (8.5, 9.0, 8.5, 8.5, "Excellent condition"),
            (7.0, 8.0, 7.5, 7.5, "Good condition"),
            (5.0, 6.0, 5.5, 6.0, "Fair condition"),
            (3.0, 4.0, 3.5, 4.0, "Poor condition")
        ]
        
        for centering, corners, edges, surface, description in test_scores:
            # Create mock factors for testing
            from app.pokemon.unified_psa_grader import PSAGradingFactors
            
            test_factors = PSAGradingFactors(
                centering=centering,
                corners=corners,
                edges=edges,
                surface=surface
            )
            
            # Test grade calculation
            final_grade = grader._calculate_final_grade_human_style(test_factors, "vintage")
            
            print(f"  {description}: C={centering:.1f}, Co={corners:.1f}, E={edges:.1f}, S={surface:.1f} -> Grade {final_grade:.1f}")
        
        return True
        
    except Exception as e:
        print(f"Grade level test failed: {e}")
        return False

if __name__ == "__main__":
    print("Unified PSA Grading System Test")
    print("=" * 60)
    
    # Test main grading system
    success = test_unified_psa_grading()
    
    if success:
        # Test grade level accuracy
        test_grade_level_accuracy()
    
    print(f"\\n" + "=" * 60)
    print("Test Result:", "PASSED" if success else "FAILED")