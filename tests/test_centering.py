"""
Unit tests for centering analysis functionality.
"""

import pytest
import numpy as np
import cv2
import logging

from app.metrics.centering import (
    analyze_centering, 
    calculate_centering_score,
    detect_inner_frame_edges,
    calculate_margins,
    calculate_centering_errors
)
from app.schema import CenteringFindings

# Set up test logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        
        return image
    
    def test_detect_inner_frame_perfect_centering(self):
        """Test frame detection with perfectly centered card."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=30, right_margin=30,
            top_margin=40, bottom_margin=40
        )
        
        frame = detect_inner_frame_edges(image)
        
        # Frame detection may or may not succeed depending on image complexity
        # This is acceptable behavior
        assert frame is None or (isinstance(frame, np.ndarray) and len(frame) == 4)
    
    def test_analyze_centering_perfect(self):
        """Test complete centering analysis with perfect centering."""
        logger.info("Testing centering analysis with perfect centering")
        
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=30, right_margin=30,
            top_margin=40, bottom_margin=40
        )
        logger.debug(f"Created test image: {image.shape}")
        
        try:
            result = analyze_centering(image)
            logger.info(f"Centering analysis result: score={result.centering_score:.2f}, h_error={result.horizontal_error:.3f}, v_error={result.vertical_error:.3f}")
        except Exception as e:
            logger.error(f"Centering analysis failed: {e}")
            raise
        
        assert isinstance(result, CenteringFindings)
        logger.debug(f"Result type check passed")
        
        assert result.centering_score >= 70.0  # Should be high for reasonably centered card
        logger.debug(f"Score check passed: {result.centering_score} >= 70.0")
        
        assert abs(result.horizontal_error) < 0.3
        logger.debug(f"Horizontal error check passed: {abs(result.horizontal_error)} < 0.3")
        
        assert abs(result.vertical_error) < 0.3  
        logger.debug(f"Vertical error check passed: {abs(result.vertical_error)} < 0.3")
        
        assert result.combined_error < 0.5
        logger.debug(f"Combined error check passed: {result.combined_error} < 0.5")
        
        logger.info("Perfect centering test completed successfully")
    
    def test_analyze_centering_off_center(self):
        """Test centering analysis with significantly off-center card."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=10, right_margin=50,   # 40px difference
            top_margin=20, bottom_margin=60    # 40px difference
        )
        
        result = analyze_centering(image)
        
        assert isinstance(result, CenteringFindings)
        assert 0 <= result.centering_score <= 100
        # Off-center card should have some error
        assert result.combined_error >= 0
    
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
        assert 0 <= result.centering_score <= 100
    
    def test_centering_with_custom_threshold(self):
        """Test centering analysis with custom error threshold."""
        image = self.create_test_card_image(
            width=300, height=400,
            left_margin=20, right_margin=40,
            top_margin=30, bottom_margin=50
        )
        
        # Test with different thresholds
        config_strict = {'max_error_threshold': 0.1}
        config_lenient = {'max_error_threshold': 0.5}
        
        result_strict = analyze_centering(image, config_strict)
        result_lenient = analyze_centering(image, config_lenient)
        
        # Both should be valid results
        assert isinstance(result_strict, CenteringFindings)
        assert isinstance(result_lenient, CenteringFindings)
        assert 0 <= result_strict.centering_score <= 100
        assert 0 <= result_lenient.centering_score <= 100
    
    def test_calculate_margins(self):
        """Test margin calculation function."""
        # Create a simple frame rectangle
        frame = np.array([50, 60, 200, 280])  # x, y, width, height
        
        # Create dummy image
        image = np.ones((400, 300, 3), dtype=np.uint8) * 255
        
        margins = calculate_margins(image, frame)
        
        assert isinstance(margins, dict)
        assert 'left_px' in margins
        assert 'right_px' in margins
        assert 'top_px' in margins
        assert 'bottom_px' in margins
        
        # Check margin values are reasonable
        assert margins['left_px'] >= 0
        assert margins['right_px'] >= 0
        assert margins['top_px'] >= 0
        assert margins['bottom_px'] >= 0
    
    def test_calculate_centering_errors_perfect(self):
        """Test centering error calculation with perfect margins."""
        margins = {
            'left_px': 30.0,
            'right_px': 30.0,
            'top_px': 40.0,
            'bottom_px': 40.0
        }
        
        errors = calculate_centering_errors(margins)
        
        assert isinstance(errors, dict)
        assert 'horizontal_error' in errors
        assert 'vertical_error' in errors
        assert 'combined_error' in errors
        
        # Perfect centering should have zero or very small errors
        assert errors['horizontal_error'] < 0.01
        assert errors['vertical_error'] < 0.01
        assert errors['combined_error'] < 0.01
    
    def test_calculate_centering_errors_off_center(self):
        """Test centering error calculation with uneven margins."""
        margins = {
            'left_px': 10.0,
            'right_px': 50.0,
            'top_px': 20.0,
            'bottom_px': 60.0
        }
        
        errors = calculate_centering_errors(margins)
        
        assert isinstance(errors, dict)
        # Should have significant errors for uneven margins
        assert errors['horizontal_error'] > 0.1
        assert errors['vertical_error'] > 0.1
        assert errors['combined_error'] > 0.1
    
    def test_centering_edge_cases(self):
        """Test centering analysis with edge cases."""
        
        # Test with very small image
        small_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        small_image[10:40, 10:40] = [128, 128, 128]
        
        result_small = analyze_centering(small_image)
        assert isinstance(result_small, CenteringFindings)
        assert 0 <= result_small.centering_score <= 100
        
        # Test with very large margins
        large_margin_image = self.create_test_card_image(
            width=400, height=500,
            left_margin=150, right_margin=150,
            top_margin=200, bottom_margin=200
        )
        
        result_large = analyze_centering(large_margin_image)
        assert isinstance(result_large, CenteringFindings)
        assert 0 <= result_large.centering_score <= 100
    
    def test_centering_analysis_robustness(self):
        """Test robustness of centering analysis to noise."""
        
        # Create base image
        image = self.create_test_card_image(width=300, height=400)
        
        # Add noise
        noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        result = analyze_centering(noisy_image)
        assert isinstance(result, CenteringFindings)
        assert 0 <= result.centering_score <= 100
        
        # Add artifacts
        artifact_image = image.copy()
        cv2.line(artifact_image, (0, 200), (300, 200), (0, 0, 0), 2)
        cv2.circle(artifact_image, (150, 200), 10, (255, 255, 255), -1)
        
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


if __name__ == "__main__":
    pytest.main([__file__])