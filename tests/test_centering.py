"""
Unit tests for centering analysis functionality.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from app.metrics.centering import (
    analyze_centering, 
    calculate_centering_score,
    detect_card_frame_edges,
    detect_card_frame_color
)
from app.schema import CenteringFindings


class TestCenteringAnalysis:
    """Test cases for centering analysis functions."""
    
    def test_calculate_centering_score_perfect(self):
        """Test centering score calculation with perfect centering."""
        score = calculate_centering_score(0.0)
        assert score == 100.0
    
    def test_calculate_centering_score_maximum_error(self):
        """Test centering score with maximum acceptable error."""
        score = calculate_centering_score(0.25, max_error_threshold=0.25)
        assert score == 0.0
    
    def test_calculate_centering_score_half_error(self):
        """Test centering score with half maximum error."""
        score = calculate_centering_score(0.125, max_error_threshold=0.25)
        assert score == 50.0
    
    def test_calculate_centering_score_exceeds_threshold(self):
        """Test centering score when error exceeds threshold."""
        score = calculate_centering_score(0.5, max_error_threshold=0.25)
        assert score == 0.0
    
    def create_test_card_image(self, width=300, height=400, 
                             left_margin=30, right_margin=30, 
                             top_margin=40, bottom_margin=40):
        """Create a synthetic card image for testing."""
        # Create white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Create inner colored rectangle (card content)
        inner_left = left_margin
        inner_top = top_margin
        inner_right = width - right_margin
        inner_bottom = height - bottom_margin
        
        # Fill inner area with gray color
        image[inner_top:inner_bottom, inner_left:inner_right] = [128, 128, 128]
        
        # Add some noise to make it more realistic
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def test_detect_card_frame_edges_perfect_centering(self):
        """Test edge detection with perfectly centered card."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=30, right_margin=30,
            top_margin=40, bottom_margin=40
        )
        
        left, right, top, bottom = detect_card_frame_edges(image)
        
        # Should detect margins with some tolerance
        assert 25 <= left <= 35
        assert 25 <= right <= 35
        assert 35 <= top <= 45
        assert 35 <= bottom <= 45
    
    def test_detect_card_frame_edges_off_center(self):
        """Test edge detection with off-center card."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=20, right_margin=40,  # Off-center horizontally
            top_margin=30, bottom_margin=50   # Off-center vertically
        )
        
        left, right, top, bottom = detect_card_frame_edges(image)
        
        # Should detect the asymmetric margins
        assert left < right  # Left margin smaller than right
        assert top < bottom  # Top margin smaller than bottom
    
    def test_analyze_centering_perfect(self):
        """Test complete centering analysis with perfect centering."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=30, right_margin=30,
            top_margin=40, bottom_margin=40
        )
        
        result = analyze_centering(image)
        
        assert isinstance(result, CenteringFindings)
        assert result.centering_score >= 90.0  # Should be high for perfect centering
        assert abs(result.horizontal_error) < 0.1
        assert abs(result.vertical_error) < 0.1
        assert result.combined_error < 0.1
    
    def test_analyze_centering_off_center(self):
        """Test centering analysis with significantly off-center card."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=10, right_margin=50,   # 40px difference
            top_margin=20, bottom_margin=60    # 40px difference
        )
        
        result = analyze_centering(image)
        
        assert isinstance(result, CenteringFindings)
        assert result.centering_score < 90.0  # Should be lower for off-center card
        assert result.horizontal_error > 0.1
        assert result.vertical_error > 0.1
        assert result.combined_error > 0.1
    
    def test_analyze_centering_with_detection_failure(self):
        """Test centering analysis when frame detection fails."""
        # Create image with no clear frame (all same color)
        image = np.ones((400, 300, 3), dtype=np.uint8) * 128
        
        result = analyze_centering(image)
        
        assert isinstance(result, CenteringFindings)
        # Should fall back to default margins
        assert result.left_margin_px > 0
        assert result.right_margin_px > 0
        assert result.top_margin_px > 0
        assert result.bottom_margin_px > 0
    
    def test_centering_with_custom_threshold(self):
        """Test centering analysis with custom error threshold."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=20, right_margin=40,
            top_margin=30, bottom_margin=50
        )
        
        # Test with stricter threshold
        result_strict = analyze_centering(image, max_error_threshold=0.1)
        
        # Test with more lenient threshold
        result_lenient = analyze_centering(image, max_error_threshold=0.5)
        
        # Stricter threshold should result in lower score
        assert result_strict.centering_score <= result_lenient.centering_score
    
    def test_detect_card_frame_color_method(self):
        """Test color-based frame detection method."""
        # Create image with distinct border color
        image = np.ones((400, 300, 3), dtype=np.uint8) * 255  # White border
        
        # Inner card area in different color
        image[50:350, 40:260] = [100, 100, 100]  # Gray inner area
        
        left, right, top, bottom = detect_card_frame_color(image)
        
        # Should detect the color transition
        assert 35 <= left <= 45
        assert 35 <= right <= 45
        assert 45 <= top <= 55
        assert 45 <= bottom <= 55
    
    def test_centering_edge_cases(self):
        """Test centering analysis with edge cases."""
        
        # Test with very small image
        small_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        small_image[10:40, 10:40] = [128, 128, 128]
        
        result_small = analyze_centering(small_image)
        assert isinstance(result_small, CenteringFindings)
        
        # Test with very large margins (card takes small portion)
        large_margin_image = self.create_test_card_image(
            width=400, height=500,
            left_margin=150, right_margin=150,
            top_margin=200, bottom_margin=200
        )
        
        result_large = analyze_centering(large_margin_image)
        assert isinstance(result_large, CenteringFindings)
        
        # Test with minimal margins
        minimal_image = self.create_test_card_image(
            width=300, height=400,
            left_margin=2, right_margin=2,
            top_margin=2, bottom_margin=2
        )
        
        result_minimal = analyze_centering(minimal_image)
        assert isinstance(result_minimal, CenteringFindings)
    
    def test_centering_analysis_robustness(self):
        """Test robustness of centering analysis to noise and artifacts."""
        
        # Create base image
        image = self.create_test_card_image(width=300, height=400)
        
        # Add significant noise
        noise = np.random.randint(-50, 50, image.shape, dtype=np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        result = analyze_centering(noisy_image)
        assert isinstance(result, CenteringFindings)
        assert 0 <= result.centering_score <= 100
        
        # Add some artificial artifacts (lines, spots)
        artifact_image = image.copy()
        cv2.line(artifact_image, (0, 200), (300, 200), (0, 0, 0), 2)  # Horizontal line
        cv2.circle(artifact_image, (150, 200), 10, (255, 255, 255), -1)  # White circle
        
        result_artifact = analyze_centering(artifact_image)
        assert isinstance(result_artifact, CenteringFindings)
        assert 0 <= result_artifact.centering_score <= 100


@pytest.fixture
def sample_card_images():
    """Provide sample card images for testing."""
    images = {}
    
    # Perfect centering
    images['perfect'] = np.ones((400, 300, 3), dtype=np.uint8) * 255
    images['perfect'][40:360, 30:270] = [128, 128, 128]
    
    # Off-center
    images['off_center'] = np.ones((400, 300, 3), dtype=np.uint8) * 255
    images['off_center'][20:340, 10:250] = [128, 128, 128]
    
    # Heavily off-center
    images['heavy_off'] = np.ones((400, 300, 3), dtype=np.uint8) * 255
    images['heavy_off'][10:300, 5:200] = [128, 128, 128]
    
    return images


def test_centering_analysis_batch(sample_card_images):
    """Test centering analysis on batch of images."""
    results = {}
    
    for name, image in sample_card_images.items():
        result = analyze_centering(image)
        results[name] = result
        
        # Verify all results are valid
        assert isinstance(result, CenteringFindings)
        assert 0 <= result.centering_score <= 100
        assert result.combined_error >= 0
    
    # Verify score ordering makes sense
    assert results['perfect'].centering_score >= results['off_center'].centering_score
    assert results['off_center'].centering_score >= results['heavy_off'].centering_score


if __name__ == "__main__":
    pytest.main([__file__])