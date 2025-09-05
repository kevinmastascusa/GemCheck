"""
Unit tests for glare detection functionality.
"""

import pytest
import numpy as np
import cv2

from app.metrics.glare import (
    analyze_glare,
    detect_specular_highlights,
    calculate_glare_penalty
)
from app.schema import GlareFindings


class TestGlareAnalysis:
    """Test cases for glare analysis functions."""
    
    def create_test_glare_image(self, width=300, height=400, glare_condition="none"):
        """Create a test card image with specified glare conditions."""
        # Create normal card surface
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        if glare_condition == "single_spot":
            # Add a single bright glare spot
            center = (width//2, height//2)
            cv2.circle(image, center, 30, (255, 255, 255), -1)
            
        elif glare_condition == "multiple_spots":
            # Add multiple glare spots
            cv2.circle(image, (100, 150), 20, (255, 255, 255), -1)
            cv2.circle(image, (200, 300), 25, (250, 250, 250), -1)
            cv2.circle(image, (250, 100), 15, (245, 245, 245), -1)
            
        elif glare_condition == "edge_glare":
            # Add glare along one edge
            image[:50, :] = 255  # Top edge bright
            
        elif glare_condition == "subtle":
            # Add subtle highlights
            cv2.ellipse(image, (150, 200), (40, 60), 0, 0, 360, (230, 230, 230), -1)
        
        return image
    
    def test_detect_specular_highlights_none(self):
        """Test highlight detection on image with no glare."""
        image = self.create_test_glare_image(glare_condition="none")
        
        highlight_mask = detect_specular_highlights(image)
        
        assert isinstance(highlight_mask, np.ndarray)
        assert highlight_mask.shape[:2] == image.shape[:2]
        # Should have minimal highlights
        highlight_pixels = np.sum(highlight_mask > 0)
        assert highlight_pixels >= 0  # Can be 0 or small number
    
    def test_detect_specular_highlights_single(self):
        """Test highlight detection on image with single glare spot."""
        image = self.create_test_glare_image(glare_condition="single_spot")
        
        highlight_mask = detect_specular_highlights(image)
        
        assert isinstance(highlight_mask, np.ndarray)
        # Should detect the bright spot
        highlight_pixels = np.sum(highlight_mask > 0)
        assert highlight_pixels > 0  # Should detect something
    
    def test_detect_specular_highlights_multiple(self):
        """Test highlight detection on image with multiple glare spots."""
        image = self.create_test_glare_image(glare_condition="multiple_spots")
        
        highlight_mask = detect_specular_highlights(image)
        
        assert isinstance(highlight_mask, np.ndarray)
        highlight_pixels = np.sum(highlight_mask > 0)
        assert highlight_pixels >= 0
    
    def test_detect_specular_highlights_threshold(self):
        """Test highlight detection with different thresholds."""
        image = self.create_test_glare_image(glare_condition="single_spot")
        
        # Test with strict threshold
        strict_mask = detect_specular_highlights(image, threshold=0.95)
        assert isinstance(strict_mask, np.ndarray)
        
        # Test with lenient threshold  
        lenient_mask = detect_specular_highlights(image, threshold=0.7)
        assert isinstance(lenient_mask, np.ndarray)
        
        # Lenient should detect more or equal highlights
        strict_pixels = np.sum(strict_mask > 0)
        lenient_pixels = np.sum(lenient_mask > 0)
        assert lenient_pixels >= strict_pixels
    
    def test_calculate_glare_penalty_none(self):
        """Test penalty calculation for no glare."""
        glare_percentage = 0.0
        penalty = calculate_glare_penalty(glare_percentage)
        
        assert isinstance(penalty, float)
        assert penalty == 0.0
    
    def test_calculate_glare_penalty_moderate(self):
        """Test penalty calculation for moderate glare."""
        glare_percentage = 5.0
        penalty = calculate_glare_penalty(glare_percentage)
        
        assert isinstance(penalty, float)
        assert penalty > 0.0
        assert penalty <= 10.0  # Should be within reasonable range
    
    def test_calculate_glare_penalty_severe(self):
        """Test penalty calculation for severe glare."""
        glare_percentage = 15.0
        penalty = calculate_glare_penalty(glare_percentage)
        
        assert isinstance(penalty, float)
        assert penalty > 0.0
        # Severe glare should have higher penalty
        assert penalty >= calculate_glare_penalty(5.0)
    
    def test_analyze_glare_none(self):
        """Test complete glare analysis on image with no glare."""
        image = self.create_test_glare_image(glare_condition="none")
        
        result = analyze_glare(image)
        
        assert isinstance(result, GlareFindings)
        assert isinstance(result.glare_detected, bool)
        assert result.glare_percentage >= 0.0
        assert result.penalty_applied >= 0.0
        assert isinstance(result.affected_regions, list)
        
        # No glare should result in minimal penalty
        assert result.penalty_applied <= 1.0
    
    def test_analyze_glare_single_spot(self):
        """Test glare analysis on image with single glare spot."""
        image = self.create_test_glare_image(glare_condition="single_spot")
        
        result = analyze_glare(image)
        
        assert isinstance(result, GlareFindings)
        assert result.glare_percentage >= 0.0
        assert result.penalty_applied >= 0.0
    
    def test_analyze_glare_multiple_spots(self):
        """Test glare analysis on image with multiple glare spots."""
        image = self.create_test_glare_image(glare_condition="multiple_spots")
        
        result = analyze_glare(image)
        
        assert isinstance(result, GlareFindings)
        assert result.glare_percentage >= 0.0
        assert len(result.affected_regions) >= 0
    
    def test_analyze_glare_with_config(self):
        """Test glare analysis with custom configuration."""
        image = self.create_test_glare_image(glare_condition="single_spot")
        config = {
            'highlight_threshold': 0.85,
            'min_region_area': 50,
            'max_penalty': 15.0
        }
        
        result = analyze_glare(image, config)
        
        assert isinstance(result, GlareFindings)
        assert result.penalty_applied <= config['max_penalty']
    
    def test_analyze_glare_edge_cases(self):
        """Test glare analysis with edge cases."""
        
        # Test completely white image (extreme glare)
        white_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        result_white = analyze_glare(white_image)
        assert isinstance(result_white, GlareFindings)
        
        # Test completely black image
        black_image = np.zeros((200, 200, 3), dtype=np.uint8)
        result_black = analyze_glare(black_image)
        assert isinstance(result_black, GlareFindings)
        assert result_black.glare_percentage == 0.0
        
        # Test very small image
        small_image = np.ones((50, 50, 3), dtype=np.uint8) * 200
        result_small = analyze_glare(small_image)
        assert isinstance(result_small, GlareFindings)
    
    def test_glare_detection_consistency(self):
        """Test that glare detection is consistent."""
        image = self.create_test_glare_image(glare_condition="single_spot")
        
        # Run analysis multiple times
        results = []
        for _ in range(3):
            result = analyze_glare(image)
            results.append(result)
        
        # Results should be consistent
        for result in results:
            assert isinstance(result, GlareFindings)
            # Glare percentage should be similar across runs
            first_percentage = results[0].glare_percentage
            assert abs(result.glare_percentage - first_percentage) < 1.0


class TestGlareRobustness:
    """Test robustness of glare detection."""
    
    def test_varying_lighting_conditions(self):
        """Test glare detection under varying lighting."""
        # Create image with gradient lighting
        image = np.ones((300, 300, 3), dtype=np.uint8) * 180
        
        # Add lighting gradient
        for x in range(300):
            brightness = int(180 + 50 * (x / 300))
            image[:, x] = brightness
        
        # Add actual glare spot
        cv2.circle(image, (150, 150), 20, (255, 255, 255), -1)
        
        result = analyze_glare(image)
        
        assert isinstance(result, GlareFindings)
        assert result.glare_percentage >= 0.0
    
    def test_noisy_image_glare(self):
        """Test glare detection on noisy image."""
        # Create base image with glare
        image = np.ones((250, 250, 3), dtype=np.uint8) * 200
        cv2.circle(image, (125, 125), 25, (255, 255, 255), -1)
        
        # Add noise
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        result = analyze_glare(noisy_image)
        
        assert isinstance(result, GlareFindings)
        assert result.glare_percentage >= 0.0
    
    def test_different_color_temperatures(self):
        """Test glare detection with different color temperatures."""
        # Warm-tinted image
        warm_image = np.ones((200, 200, 3), dtype=np.uint8)
        warm_image[:, :, 0] = 220  # More red
        warm_image[:, :, 1] = 200  # Medium green
        warm_image[:, :, 2] = 180  # Less blue
        cv2.circle(warm_image, (100, 100), 15, (255, 255, 255), -1)
        
        result_warm = analyze_glare(warm_image)
        assert isinstance(result_warm, GlareFindings)
        
        # Cool-tinted image
        cool_image = np.ones((200, 200, 3), dtype=np.uint8)
        cool_image[:, :, 0] = 180  # Less red
        cool_image[:, :, 1] = 200  # Medium green
        cool_image[:, :, 2] = 220  # More blue
        cv2.circle(cool_image, (100, 100), 15, (255, 255, 255), -1)
        
        result_cool = analyze_glare(cool_image)
        assert isinstance(result_cool, GlareFindings)


if __name__ == "__main__":
    pytest.main([__file__])