#!/usr/bin/env python3
"""
Test script for Pokemon card segmentation and preprocessing.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_card_segmentation():
    """Test the card segmentation system."""
    print("Testing Pokemon Card Segmentation System")
    print("=" * 50)
    
    try:
        from app.pokemon.card_segmentation import PokemonCardSegmenter, CardComponent
        from app.pokemon.preprocessing_pipeline import PokemonCardPreprocessor, PreprocessingConfig, PreprocessingLevel
        
        # Initialize systems
        segmenter = PokemonCardSegmenter()
        preprocessor = PokemonCardPreprocessor()
        
        print("Segmentation and preprocessing systems initialized")
        
        # Test with Charizard card
        test_image_path = project_root / "test_charizard.png"
        
        if not test_image_path.exists():
            print("Downloading test Charizard card...")
            import urllib.request
            card_url = "https://images.pokemontcg.io/base1/4_hires.png"
            urllib.request.urlretrieve(card_url, test_image_path)
            print(f"Downloaded: {test_image_path}")
        
        # Load test image
        print(f"Loading card image: {test_image_path}")
        test_image = cv2.imread(str(test_image_path))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"Image loaded: {test_image.shape}")
        
        # Test preprocessing pipeline
        print(f"\nTesting Preprocessing Pipeline...")
        config = PreprocessingConfig(
            level=PreprocessingLevel.STANDARD,
            segment_components=True,
            extract_features=True
        )
        
        preprocessing_result = preprocessor.process_card_image(test_image, config, card_era="vintage")
        
        print(f"Preprocessing completed in {preprocessing_result.processing_time:.2f}s")
        print(f"Steps performed: {', '.join(preprocessing_result.preprocessing_steps)}")
        
        # Display quality improvements
        quality_metrics = preprocessing_result.quality_metrics
        print(f"\nQuality Improvements:")
        for metric, improvement in quality_metrics["improvement"].items():
            if improvement != 0:
                print(f"  {metric}: {improvement:+.1f}")
        
        print(f"Overall quality: {quality_metrics['initial_quality']['overall_quality']:.1f} -> {quality_metrics['final_quality']['overall_quality']:.1f}")
        
        # Test segmentation results
        if preprocessing_result.segmentation_result:
            segmentation = preprocessing_result.segmentation_result
            print(f"\nSegmentation Results:")
            print(f"  Components found: {len(segmentation.components)}")
            print(f"  Segmentation quality: {segmentation.segmentation_quality:.2f}")
            
            print(f"\nComponent Analysis:")
            for component, mask_info in segmentation.components.items():
                print(f"  {component.value}:")
                print(f"    Area: {mask_info.area_percentage:.1f}% of card")
                print(f"    Confidence: {mask_info.confidence:.2f}")
                print(f"    Bounding box: {mask_info.bounding_box}")
                
                # Show any component properties
                if mask_info.properties:
                    for key, value in list(mask_info.properties.items())[:2]:  # Show first 2 properties
                        print(f"    {key}: {value}")
        
        # Test feature extraction
        if preprocessing_result.extracted_features:
            print(f"\nExtracted Features:")
            feature_count = 0
            for component, features in preprocessing_result.extracted_features.items():
                print(f"  {component}:")
                for feature_name, feature_value in list(features.items())[:3]:  # Show first 3 features
                    if isinstance(feature_value, float):
                        print(f"    {feature_name}: {feature_value:.3f}")
                    elif isinstance(feature_value, list) and len(feature_value) <= 3:
                        print(f"    {feature_name}: {feature_value}")
                    else:
                        print(f"    {feature_name}: {str(feature_value)[:50]}...")
                feature_count += len(features)
            
            print(f"  Total features extracted: {feature_count}")
        
        # Save visualization
        print(f"\nCreating segmentation visualization...")
        if preprocessing_result.segmentation_result:
            segmentation_viz = segmenter.visualize_segmentation(preprocessing_result.segmentation_result)
            
            # Save visualization
            viz_path = project_root / "charizard_segmentation_visualization.png"
            viz_bgr = cv2.cvtColor(segmentation_viz, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(viz_path), viz_bgr)
            print(f"Segmentation visualization saved: {viz_path}")
            
            # Save component masks
            masks_dir = project_root / "charizard_component_masks"
            segmenter.save_component_masks(preprocessing_result.segmentation_result, str(masks_dir))
            print(f"Component masks saved to: {masks_dir}")
        
        # Save processed image
        processed_path = project_root / "charizard_processed.png"
        processed_bgr = cv2.cvtColor(preprocessing_result.processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(processed_path), processed_bgr)
        print(f"Processed image saved: {processed_path}")
        
        # Generate comprehensive report
        print(f"\nGenerating preprocessing report...")
        report = preprocessor.create_preprocessing_report(preprocessing_result)
        
        print(f"Preprocessing Report Summary:")
        print(f"  Processing level: {report['processing_summary']['processing_level']}")
        print(f"  Steps: {len(report['processing_summary']['steps_performed'])}")
        print(f"  Time: {report['processing_summary']['processing_time']:.2f}s")
        
        if 'segmentation' in report:
            print(f"  Segmentation quality: {report['segmentation']['segmentation_quality']:.2f}")
            print(f"  Components: {report['segmentation']['components_found']}")
        
        # Test with different eras
        print(f"\nTesting with different card eras...")
        eras_to_test = ["vintage", "modern"]
        
        for era in eras_to_test:
            print(f"\n  Testing {era} era processing...")
            era_result = preprocessor.process_card_image(
                test_image, 
                PreprocessingConfig(level=PreprocessingLevel.BASIC, segment_components=True),
                card_era=era
            )
            
            if era_result.segmentation_result:
                era_components = len(era_result.segmentation_result.components)
                era_quality = era_result.segmentation_result.segmentation_quality
                print(f"    {era} era - Components: {era_components}, Quality: {era_quality:.2f}")
        
        print(f"\n" + "=" * 50)
        print("SUCCESS: Card segmentation and preprocessing test completed!")
        
        print(f"\nFiles created:")
        print(f"  - Processed image: {processed_path}")
        print(f"  - Segmentation visualization: {viz_path}")
        print(f"  - Component masks: {masks_dir}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing capabilities."""
    print(f"\nTesting Batch Processing...")
    
    try:
        from app.pokemon.preprocessing_pipeline import PokemonCardPreprocessor, PreprocessingConfig, PreprocessingLevel
        
        # Create test images list (using the same image multiple times for demo)
        test_image_path = project_root / "test_charizard.png"
        
        if test_image_path.exists():
            test_images = [str(test_image_path)]  # In real use, this would be multiple different images
            
            preprocessor = PokemonCardPreprocessor()
            
            # Test batch processing
            batch_results = preprocessor.process_batch(
                test_images,
                PreprocessingConfig(level=PreprocessingLevel.BASIC),
                output_dir=str(project_root / "batch_output")
            )
            
            print(f"Batch processing completed: {len(batch_results)} images processed")
            
            if batch_results:
                avg_processing_time = np.mean([r.processing_time for r in batch_results])
                print(f"Average processing time: {avg_processing_time:.2f}s per image")
        
        return True
        
    except Exception as e:
        print(f"Batch processing test failed: {e}")
        return False

if __name__ == "__main__":
    print("Pokemon Card Segmentation & Preprocessing Test")
    print("=" * 55)
    
    # Test main segmentation and preprocessing
    success = test_card_segmentation()
    
    if success:
        # Test batch processing
        test_batch_processing()
    
    print(f"\n" + "=" * 55)
    print("Test Result:", "PASSED" if success else "FAILED")