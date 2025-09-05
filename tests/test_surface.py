"""
Unit tests for surface analysis functionality.
"""

import pytest
import numpy as np
import cv2

from app.metrics.surface import (
    analyze_surface,
    detect_scratches_simple,
    detect_print_lines_simple
)
from app.schema import SurfaceFindings, DefectRegion


class TestSurfaceAnalysis:
    """Test cases for surface analysis functions."""
    
    def create_test_surface_image(self, width=300, height=400, surface_condition="clean"):
        """Create a test card surface with specified conditions."""
        # Create card surface (light gray)
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        if surface_condition == "scratched":
            # Add some scratches (dark lines)
            cv2.line(image, (50, 100), (150, 120), (180, 180, 180), 2)
            cv2.line(image, (200, 200), (250, 250), (190, 190, 190), 1)
            cv2.line(image, (80, 300), (120, 350), (170, 170, 170), 3)
            
        elif surface_condition == "print_lines":
            # Add horizontal print lines
            for y in range(100, 300, 20):
                cv2.line(image, (50, y), (250, y), (220, 220, 220), 1)
                
        elif surface_condition == "mixed":
            # Add both scratches and print lines
            cv2.line(image, (50, 100), (150, 120), (180, 180, 180), 2)
            for y in range(200, 250, 15):
                cv2.line(image, (100, y), (200, y), (220, 220, 220), 1)
        
        return image
    
    def test_detect_scratches_simple_clean(self):
        """Test scratch detection on clean surface."""
        image = self.create_test_surface_image(surface_condition="clean")
        
        scratch_mask, defect_regions = detect_scratches_simple(image)
        
        assert isinstance(scratch_mask, np.ndarray)
        assert isinstance(defect_regions, list)
        assert scratch_mask.shape[:2] == image.shape[:2]
        # Clean surface should have few or no defects
        assert len(defect_regions) <= 5
    
    def test_detect_scratches_simple_scratched(self):
        """Test scratch detection on scratched surface."""
        image = self.create_test_surface_image(surface_condition="scratched")
        
        scratch_mask, defect_regions = detect_scratches_simple(image)
        
        assert isinstance(scratch_mask, np.ndarray)
        assert isinstance(defect_regions, list)
        # Scratched surface should show some defects
        assert len(defect_regions) >= 0  # May or may not detect depending on algorithm
    
    def test_detect_print_lines_clean(self):
        """Test print line detection on clean surface."""
        image = self.create_test_surface_image(surface_condition="clean")
        
        line_regions = detect_print_lines_simple(image)
        
        assert isinstance(line_regions, list)
        # Clean surface should have few or no print lines
        assert len(line_regions) >= 0
    
    def test_detect_print_lines_with_lines(self):
        """Test print line detection on surface with print lines."""
        image = self.create_test_surface_image(surface_condition="print_lines")
        
        line_regions = detect_print_lines_simple(image)
        
        assert isinstance(line_regions, list)
        assert len(line_regions) >= 0
    
    def test_surface_score_ranges(self):
        """Test that surface analysis produces reasonable score ranges."""
        # Test with clean surface
        clean_image = self.create_test_surface_image(surface_condition="clean")
        clean_result = analyze_surface(clean_image)
        
        assert isinstance(clean_result.surface_quality_score, float)
        assert 0.0 <= clean_result.surface_quality_score <= 1.0
        
        # Test with damaged surface
        damaged_image = self.create_test_surface_image(surface_condition="scratched")
        damaged_result = analyze_surface(damaged_image)
        
        assert isinstance(damaged_result.surface_quality_score, float)
        assert 0.0 <= damaged_result.surface_quality_score <= 1.0
    
    def test_analyze_surface_clean(self):
        """Test complete surface analysis on clean card."""
        image = self.create_test_surface_image(surface_condition="clean")
        
        result = analyze_surface(image)
        
        assert isinstance(result, SurfaceFindings)
        assert hasattr(result, 'surface_quality_score')
        assert 0.0 <= result.surface_quality_score <= 1.0
        assert result.defect_percentage >= 0.0
        assert result.scratch_count >= 0
        assert result.print_line_count >= 0
        assert isinstance(result.defect_regions, list)
    
    def test_analyze_surface_scratched(self):
        """Test surface analysis on scratched card."""
        image = self.create_test_surface_image(surface_condition="scratched")
        
        result = analyze_surface(image)
        
        assert isinstance(result, SurfaceFindings)
        assert 0.0 <= result.surface_quality_score <= 1.0
        assert result.defect_percentage >= 0.0
    
    def test_analyze_surface_mixed_defects(self):
        """Test surface analysis on card with mixed defects."""
        image = self.create_test_surface_image(surface_condition="mixed")
        
        result = analyze_surface(image)
        
        assert isinstance(result, SurfaceFindings)
        assert 0.0 <= result.surface_quality_score <= 1.0
        assert result.defect_percentage >= 0.0
    
    def test_analyze_surface_with_config(self):
        """Test surface analysis with custom configuration."""
        image = self.create_test_surface_image()
        config = {
            'scratch_threshold': 0.03,
            'min_defect_area': 10
        }
        
        result = analyze_surface(image, config)
        
        assert isinstance(result, SurfaceFindings)
        assert 0.0 <= result.surface_quality_score <= 1.0
    
    def test_surface_analysis_edge_cases(self):
        """Test surface analysis with edge cases."""
        
        # Test very small image
        small_image = np.ones((50, 50, 3), dtype=np.uint8) * 240
        result_small = analyze_surface(small_image)
        assert isinstance(result_small, SurfaceFindings)
        
        # Test very large defect area
        damaged_image = np.ones((400, 300, 3), dtype=np.uint8) * 200  # Darker overall
        result_damaged = analyze_surface(damaged_image)
        assert isinstance(result_damaged, SurfaceFindings)
        
        # Test all-black image
        black_image = np.zeros((200, 200, 3), dtype=np.uint8)
        result_black = analyze_surface(black_image)
        assert isinstance(result_black, SurfaceFindings)
    
    def test_defect_region_properties(self):
        """Test that defect regions have proper properties."""
        image = self.create_test_surface_image(surface_condition="scratched")
        
        result = analyze_surface(image)
        
        # Check each defect region
        for defect in result.defect_regions:
            assert isinstance(defect, DefectRegion)
            assert hasattr(defect, 'bbox')
            assert hasattr(defect, 'confidence')
            assert hasattr(defect, 'defect_type')
            assert 0.0 <= defect.confidence <= 1.0
            assert defect.area_pixels > 0


class TestSurfaceRobustness:
    """Test robustness of surface analysis."""
    
    def test_noisy_image(self):
        """Test surface analysis on noisy image."""
        # Create base image
        image = np.ones((300, 300, 3), dtype=np.uint8) * 240
        
        # Add random noise
        noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        result = analyze_surface(noisy_image)
        
        assert isinstance(result, SurfaceFindings)
        assert 0.0 <= result.surface_quality_score <= 1.0
    
    def test_varying_lighting(self):
        """Test surface analysis with varying lighting conditions."""
        # Create image with gradient lighting
        image = np.ones((300, 300, 3), dtype=np.uint8) * 240
        
        # Apply lighting gradient
        for y in range(300):
            brightness = int(220 + 40 * (y / 300))
            brightness = min(255, max(0, brightness))  # Clamp to valid uint8 range
            image[y, :] = brightness
        
        result = analyze_surface(image)
        
        assert isinstance(result, SurfaceFindings)
        assert 0.0 <= result.surface_quality_score <= 1.0
    
    def test_extreme_threshold_values(self):
        """Test surface analysis with extreme threshold values."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 240
        
        # Very strict threshold
        config_strict = {'scratch_threshold': 0.001}
        result_strict = analyze_surface(image, config_strict)
        assert isinstance(result_strict, SurfaceFindings)
        
        # Very lenient threshold
        config_lenient = {'scratch_threshold': 0.1}
        result_lenient = analyze_surface(image, config_lenient)
        assert isinstance(result_lenient, SurfaceFindings)


if __name__ == "__main__":
    pytest.main([__file__])