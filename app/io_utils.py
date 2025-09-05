"""
Input/output utilities for the PSA-style card grading application.
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
from typing import Optional, Tuple, Union
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_image_orientation(image: np.ndarray, exif_dict: Optional[dict] = None) -> np.ndarray:
    """
    Fix image orientation based on EXIF data.
    
    Args:
        image: Input image array
        exif_dict: EXIF data dictionary
        
    Returns:
        Oriented image array
    """
    if exif_dict is None:
        return image
        
    orientation_key = None
    for key in ExifTags.TAGS.keys():
        if ExifTags.TAGS[key] == 'Orientation':
            orientation_key = key
            break
            
    if orientation_key is None or orientation_key not in exif_dict:
        return image
        
    orientation = exif_dict[orientation_key]
    
    # Apply rotation based on orientation value
    if orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return image


def safe_read_image(image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Safely read an image file with EXIF data extraction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (image_array, exif_dict) or (None, None) if failed
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image file does not exist: {image_path}")
            return None, None
            
        # Read EXIF data using PIL
        exif_dict = None
        try:
            with Image.open(image_path) as pil_image:
                if hasattr(pil_image, '_getexif'):
                    exif_data = pil_image._getexif()
                    if exif_data is not None:
                        exif_dict = dict(exif_data.items())
        except Exception as e:
            logger.warning(f"Could not read EXIF data from {image_path}: {e}")
        
        # Read image using OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None, None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Fix orientation if EXIF data available
        image = fix_image_orientation(image, exif_dict)
        
        logger.info(f"Successfully loaded image: {image_path}, shape: {image.shape}")
        return image, exif_dict
        
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return None, None


def safe_write_image(image: np.ndarray, output_path: Union[str, Path], 
                    quality: int = 95) -> bool:
    """
    Safely write an image file.
    
    Args:
        image: Image array (RGB format)
        output_path: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Set quality for JPEG
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:
            encode_params = []
            
        success = cv2.imwrite(str(output_path), image_bgr, encode_params)
        
        if success:
            logger.info(f"Successfully saved image: {output_path}")
        else:
            logger.error(f"Failed to save image: {output_path}")
            
        return success
        
    except Exception as e:
        logger.error(f"Error writing image {output_path}: {e}")
        return False


def get_supported_image_extensions() -> list:
    """Return list of supported image file extensions."""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


def find_images_in_folder(folder_path: Union[str, Path]) -> list:
    """
    Find all supported image files in a folder.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        List of image file paths
    """
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        return []
        
    supported_extensions = get_supported_image_extensions()
    image_files = []
    
    for ext in supported_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # Sort by name
    image_files.sort(key=lambda x: x.name.lower())
    
    logger.info(f"Found {len(image_files)} image files in {folder_path}")
    return image_files


def resize_image_if_needed(image: np.ndarray, max_size: int = 2000) -> Tuple[np.ndarray, float]:
    """
    Resize image if it's larger than max_size on longest side.
    
    Args:
        image: Input image array
        max_size: Maximum size for longest side
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)
    
    if longest_side <= max_size:
        return image, 1.0
        
    scale_factor = max_size / longest_side
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}, scale: {scale_factor:.3f}")
    return resized, scale_factor


def validate_image_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if file is a supported image format.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        True if valid image file, False otherwise
    """
    file_path = Path(file_path)
    
    # Check extension
    if file_path.suffix.lower() not in get_supported_image_extensions():
        return False
        
    # Try to read the file
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False