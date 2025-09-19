"""
Preprocessing Pipeline for Pokemon Card Analysis
Comprehensive preprocessing system to prepare card images for analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .card_segmentation import PokemonCardSegmenter, SegmentationResult, CardComponent

logger = logging.getLogger(__name__)


class PreprocessingLevel(Enum):
    """Different levels of preprocessing intensity."""
    BASIC = "basic"           # Basic cleanup and normalization
    STANDARD = "standard"     # Standard processing for grading
    ENHANCED = "enhanced"     # Enhanced processing for difficult images
    RESEARCH = "research"     # Maximum processing for research/training


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    level: PreprocessingLevel = PreprocessingLevel.STANDARD
    target_size: Tuple[int, int] = (600, 432)  # Target size (height, width)
    normalize_lighting: bool = True
    enhance_contrast: bool = True
    reduce_noise: bool = True
    sharpen_image: bool = False
    correct_perspective: bool = True
    segment_components: bool = True
    extract_features: bool = True
    preserve_colors: bool = True


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline."""
    original_image: np.ndarray
    processed_image: np.ndarray
    segmentation_result: Optional[SegmentationResult]
    preprocessing_steps: List[str]
    quality_metrics: Dict[str, float]
    extracted_features: Dict[str, Any]
    processing_config: PreprocessingConfig
    processing_time: float


class PokemonCardPreprocessor:
    """
    Comprehensive preprocessing pipeline for Pokemon card images.
    Handles image normalization, enhancement, segmentation, and feature extraction.
    """
    
    def __init__(self):
        self.segmenter = PokemonCardSegmenter()
        self.quality_thresholds = self._initialize_quality_thresholds()
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality assessment thresholds."""
        return {
            "min_brightness": 30,        # Minimum average brightness
            "max_brightness": 220,       # Maximum average brightness
            "min_contrast": 20,          # Minimum contrast level
            "min_sharpness": 10,         # Minimum sharpness score
            "max_noise_level": 50,       # Maximum acceptable noise
            "min_resolution": 300,       # Minimum width/height
            "max_blur": 30               # Maximum blur detection score
        }
    
    def process_card_image(self, image: Union[np.ndarray, str], 
                          config: Optional[PreprocessingConfig] = None,
                          card_era: str = "vintage") -> PreprocessingResult:
        """
        Process a Pokemon card image through the complete preprocessing pipeline.
        
        Args:
            image: Input image (numpy array or file path)
            config: Preprocessing configuration
            card_era: Era of the card for appropriate processing
            
        Returns:
            PreprocessingResult with processed image and analysis
        """
        import time
        start_time = time.time()
        
        if config is None:
            config = PreprocessingConfig()
        
        logger.info(f"Starting preprocessing pipeline (level: {config.level.value})")
        
        # Load image if path provided
        if isinstance(image, str):
            original_image = self._load_image(image)
        else:
            original_image = image.copy()
        
        processing_steps = []
        
        # Step 1: Initial quality assessment
        initial_quality = self._assess_image_quality(original_image)
        logger.info(f"Initial quality metrics: {initial_quality}")
        
        # Step 2: Basic preprocessing
        processed_image = original_image.copy()
        
        if config.normalize_lighting:
            processed_image = self._normalize_lighting(processed_image)
            processing_steps.append("lighting_normalization")
        
        if config.reduce_noise:
            processed_image = self._reduce_noise(processed_image, config.level)
            processing_steps.append("noise_reduction")
        
        if config.enhance_contrast:
            processed_image = self._enhance_contrast(processed_image, config.level)
            processing_steps.append("contrast_enhancement")
        
        if config.sharpen_image and config.level in [PreprocessingLevel.ENHANCED, PreprocessingLevel.RESEARCH]:
            processed_image = self._sharpen_image(processed_image)
            processing_steps.append("sharpening")
        
        # Step 3: Geometric corrections
        if config.correct_perspective:
            processed_image = self._correct_perspective(processed_image)
            processing_steps.append("perspective_correction")
        
        # Step 4: Size normalization
        processed_image = self._normalize_size(processed_image, config.target_size)
        processing_steps.append("size_normalization")
        
        # Step 5: Color preservation and enhancement
        if config.preserve_colors:
            processed_image = self._preserve_colors(processed_image, original_image)
            processing_steps.append("color_preservation")
        
        # Step 6: Component segmentation
        segmentation_result = None
        if config.segment_components:
            segmentation_result = self.segmenter.segment_card(processed_image, card_era)
            processing_steps.append("component_segmentation")
        
        # Step 7: Feature extraction
        extracted_features = {}
        if config.extract_features and segmentation_result:
            extracted_features = self.segmenter.extract_component_features(segmentation_result)
            processing_steps.append("feature_extraction")
        
        # Step 8: Final quality assessment
        final_quality = self._assess_image_quality(processed_image)
        
        # Combine quality metrics
        quality_metrics = {
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "improvement": {
                key: final_quality.get(key, 0) - initial_quality.get(key, 0)
                for key in initial_quality.keys()
            }
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {processing_time:.2f}s, {len(processing_steps)} steps")
        
        return PreprocessingResult(
            original_image=original_image,
            processed_image=processed_image,
            segmentation_result=segmentation_result,
            preprocessing_steps=processing_steps,
            quality_metrics=quality_metrics,
            extracted_features=extracted_features,
            processing_config=config,
            processing_time=processing_time
        )
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image from file path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess various quality metrics of the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Brightness assessment
        brightness = np.mean(gray)
        
        # Contrast assessment
        contrast = np.std(gray)
        
        # Sharpness assessment (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Noise assessment (high frequency content)
        noise_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise_response = cv2.filter2D(gray, cv2.CV_64F, noise_kernel)
        noise_level = np.std(noise_response)
        
        # Resolution assessment
        height, width = image.shape[:2]
        min_dimension = min(height, width)
        
        # Blur assessment (edge strength)
        edges = cv2.Canny(gray, 50, 150)
        edge_strength = np.sum(edges > 0) / edges.size
        blur_score = 100 - (edge_strength * 1000)  # Lower is less blurry
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "noise_level": noise_level,
            "resolution": min_dimension,
            "blur_score": max(0, blur_score),
            "overall_quality": self._calculate_overall_quality(brightness, contrast, sharpness, noise_level, min_dimension, blur_score)
        }
    
    def _calculate_overall_quality(self, brightness: float, contrast: float, sharpness: float, 
                                 noise_level: float, resolution: float, blur_score: float) -> float:
        """Calculate overall image quality score."""
        thresholds = self.quality_thresholds
        
        # Normalize each metric (0-100 scale)
        brightness_score = 100 * max(0, min(1, (brightness - thresholds["min_brightness"]) / 
                                           (thresholds["max_brightness"] - thresholds["min_brightness"])))
        
        contrast_score = 100 * min(1, contrast / 50)  # Good contrast around 50
        
        sharpness_score = 100 * min(1, sharpness / 100)  # Good sharpness around 100
        
        noise_score = 100 * max(0, 1 - noise_level / thresholds["max_noise_level"])
        
        resolution_score = 100 * min(1, resolution / thresholds["min_resolution"])
        
        blur_score_normalized = 100 * max(0, 1 - blur_score / thresholds["max_blur"])
        
        # Weighted average
        overall = (brightness_score * 0.15 + contrast_score * 0.2 + sharpness_score * 0.25 + 
                  noise_score * 0.15 + resolution_score * 0.15 + blur_score_normalized * 0.1)
        
        return overall
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """Normalize lighting conditions across the image."""
        # Convert to LAB color space for better lighting control
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_eq = clahe.apply(l_channel)
        
        # Replace L channel and convert back
        lab[:, :, 0] = l_channel_eq
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return normalized
    
    def _reduce_noise(self, image: np.ndarray, level: PreprocessingLevel) -> np.ndarray:
        """Reduce noise while preserving important details."""
        if level == PreprocessingLevel.BASIC:
            # Simple Gaussian blur
            return cv2.GaussianBlur(image, (3, 3), 0.5)
        
        elif level == PreprocessingLevel.STANDARD:
            # Bilateral filter - preserves edges while reducing noise
            return cv2.bilateralFilter(image, 9, 75, 75)
        
        elif level in [PreprocessingLevel.ENHANCED, PreprocessingLevel.RESEARCH]:
            # Non-local means denoising - best quality but slower
            return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray, level: PreprocessingLevel) -> np.ndarray:
        """Enhance image contrast based on processing level."""
        if level == PreprocessingLevel.BASIC:
            # Simple histogram equalization on brightness
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        else:
            # CLAHE on each color channel
            enhanced = image.copy()
            for i in range(3):  # For each RGB channel
                enhanced[:, :, i] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(enhanced[:, :, i])
            
            return enhanced
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter to enhance details."""
        # Unsharp mask sharpening
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion to make card rectangular."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find card edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (presumably the card)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we found a quadrilateral, apply perspective correction
        if len(approx) == 4:
            # Order points: top-left, top-right, bottom-right, bottom-left
            points = approx.reshape(4, 2).astype(np.float32)
            
            # Sort points
            sorted_points = self._order_points(points)
            
            # Define destination points (rectangular card)
            h, w = image.shape[:2]
            card_ratio = 0.72  # Pokemon card aspect ratio (width/height)
            
            if w / h > card_ratio:
                # Width is limiting factor
                dst_w = w
                dst_h = int(w / card_ratio)
            else:
                # Height is limiting factor
                dst_h = h
                dst_w = int(h * card_ratio)
            
            dst_points = np.array([
                [0, 0],
                [dst_w - 1, 0],
                [dst_w - 1, dst_h - 1],
                [0, dst_h - 1]
            ], dtype=np.float32)
            
            # Apply perspective transform
            matrix = cv2.getPerspectiveTransform(sorted_points, dst_points)
            corrected = cv2.warpPerspective(image, matrix, (dst_w, dst_h))
            
            return corrected
        
        return image
    
    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Sort by angle from centroid
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        
        sorted_points = sorted(points, key=angle_from_centroid)
        
        return np.array(sorted_points, dtype=np.float32)
    
    def _normalize_size(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Normalize image to target size while preserving aspect ratio."""
        target_height, target_width = target_size
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit target size
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create target-sized image with padding if necessary
        if new_w != target_width or new_h != target_height:
            # Create black background
            normalized = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate padding offsets
            y_offset = (target_height - new_h) // 2
            x_offset = (target_width - new_w) // 2
            
            # Place resized image in center
            normalized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return normalized
        
        return resized
    
    def _preserve_colors(self, processed_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """Preserve original color characteristics while keeping enhancements."""
        # Resize original to match processed if needed
        if original_image.shape[:2] != processed_image.shape[:2]:
            original_resized = cv2.resize(original_image, 
                                        (processed_image.shape[1], processed_image.shape[0]),
                                        interpolation=cv2.INTER_LANCZOS4)
        else:
            original_resized = original_image
        
        # Convert both to HSV
        processed_hsv = cv2.cvtColor(processed_image, cv2.COLOR_RGB2HSV)
        original_hsv = cv2.cvtColor(original_resized, cv2.COLOR_RGB2HSV)
        
        # Keep enhanced brightness and saturation, restore original hue
        result_hsv = processed_hsv.copy()
        result_hsv[:, :, 0] = original_hsv[:, :, 0]  # Restore original hue
        
        # Blend saturation (70% processed, 30% original)
        result_hsv[:, :, 1] = (0.7 * processed_hsv[:, :, 1] + 0.3 * original_hsv[:, :, 1]).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)
        
        return result
    
    def process_batch(self, image_paths: List[str], 
                     config: Optional[PreprocessingConfig] = None,
                     output_dir: Optional[str] = None) -> List[PreprocessingResult]:
        """Process multiple images in batch."""
        if config is None:
            config = PreprocessingConfig()
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.process_card_image(image_path, config)
                results.append(result)
                
                # Save processed image if output directory specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    filename = Path(image_path).stem + "_processed.png"
                    save_path = output_path / filename
                    
                    processed_bgr = cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(save_path), processed_bgr)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        logger.info(f"Batch processing completed: {len(results)}/{len(image_paths)} successful")
        return results
    
    def create_preprocessing_report(self, result: PreprocessingResult) -> Dict[str, Any]:
        """Create a comprehensive report of preprocessing results."""
        report = {
            "processing_summary": {
                "steps_performed": result.preprocessing_steps,
                "processing_time": result.processing_time,
                "processing_level": result.processing_config.level.value
            },
            "quality_improvement": result.quality_metrics,
            "image_properties": {
                "original_size": result.original_image.shape,
                "processed_size": result.processed_image.shape,
                "target_size": result.processing_config.target_size
            }
        }
        
        if result.segmentation_result:
            report["segmentation"] = {
                "components_found": len(result.segmentation_result.components),
                "segmentation_quality": result.segmentation_result.segmentation_quality,
                "component_list": [comp.value for comp in result.segmentation_result.components.keys()]
            }
        
        if result.extracted_features:
            report["extracted_features"] = {
                "feature_count": len(result.extracted_features),
                "component_features": list(result.extracted_features.keys())
            }
        
        return report