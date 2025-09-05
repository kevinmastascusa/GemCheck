"""
Image preprocessing pipeline for card analysis.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def apply_white_balance(image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
    """
    Apply white balance correction to the image.
    
    Args:
        image: Input image in RGB format
        method: White balance method ('gray_world' or 'white_patch')
        
    Returns:
        White balanced image
    """
    try:
        if method == 'gray_world':
            # Gray world assumption: average of scene should be gray
            result = image.astype(np.float32)
            
            # Calculate channel means
            r_mean = np.mean(result[:, :, 0])
            g_mean = np.mean(result[:, :, 1])
            b_mean = np.mean(result[:, :, 2])
            
            # Calculate gray reference
            gray_mean = (r_mean + g_mean + b_mean) / 3
            
            # Apply correction factors
            if r_mean > 0:
                result[:, :, 0] *= gray_mean / r_mean
            if g_mean > 0:
                result[:, :, 1] *= gray_mean / g_mean
            if b_mean > 0:
                result[:, :, 2] *= gray_mean / b_mean
                
            # Clip values
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        
    except Exception as e:
        logger.warning(f"White balance failed: {e}, returning original image")
        return image


def preprocess_card_image(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for card analysis.
    
    Args:
        image: Input image in RGB format
        config: Optional configuration parameters
        
    Returns:
        Dictionary containing all preprocessing results
    """
    if config is None:
        config = {
            'max_size': 2000,
            'white_balance': True,
            'target_width': 750,
            'target_height': 1050
        }
    
    results = {
        'original_image': image.copy(),
        'processing_steps': []
    }
    
    # Step 1: Resize if needed
    from .io_utils import resize_image_if_needed
    current_image, scale_factor = resize_image_if_needed(image, config.get('max_size', 2000))
    results['resized_image'] = current_image
    results['scale_factor'] = scale_factor
    results['processing_steps'].append(f"Resized with scale factor: {scale_factor:.3f}")
    
    # Step 2: White balance
    if config.get('white_balance', True):
        current_image = apply_white_balance(current_image)
        results['white_balanced_image'] = current_image
        results['processing_steps'].append("Applied white balance correction")
    
    # Step 3: Simple rectification (resize to target dimensions)
    target_w = config.get('target_width', 750)
    target_h = config.get('target_height', 1050)
    results['rectified_image'] = cv2.resize(current_image, (target_w, target_h))
    results['homography_matrix'] = np.eye(3, dtype=np.float32)
    results['processing_steps'].append(f"Resized to canonical {target_w}x{target_h}")
    
    logger.info(f"Preprocessing completed with {len(results['processing_steps'])} steps")
    return results