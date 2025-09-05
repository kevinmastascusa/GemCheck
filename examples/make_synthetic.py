"""
Synthetic card image generator for testing and calibration.
Creates realistic-looking trading card images with known defects and grades.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random
import logging
from dataclasses import dataclass
import argparse
import json

logger = logging.getLogger(__name__)


@dataclass
class SyntheticCardConfig:
    """Configuration for generating a synthetic card."""
    width: int = 750
    height: int = 1050
    target_grade: int = 10
    
    # Centering parameters
    centering_error: float = 0.0  # 0.0 = perfect, higher = worse
    
    # Edge parameters
    edge_whitening: float = 0.0   # 0.0-1.0, percentage of edge whitening
    nick_count: int = 0           # Number of edge nicks
    
    # Corner parameters
    corner_wear: Dict[str, float] = None  # Corner wear levels (0.0-1.0)
    
    # Surface parameters
    scratch_count: int = 0        # Number of scratches
    surface_noise: float = 0.0    # General surface noise level
    
    # Glare parameters
    glare_regions: int = 0        # Number of glare regions
    
    # Color and style
    border_color: Tuple[int, int, int] = (255, 255, 255)  # White border
    card_color: Tuple[int, int, int] = (240, 240, 240)    # Light gray card
    
    def __post_init__(self):
        if self.corner_wear is None:
            self.corner_wear = {
                'top_left': 0.0,
                'top_right': 0.0, 
                'bottom_right': 0.0,
                'bottom_left': 0.0
            }


class SyntheticCardGenerator:
    """
    Generates synthetic trading card images with controllable defects.
    """
    
    def __init__(self, output_dir: str = "examples/synthetic_cards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_card(self, config: SyntheticCardConfig, 
                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Generate a synthetic card image based on configuration.
        
        Args:
            config: Card generation configuration
            save_path: Optional path to save the image
            
        Returns:
            Generated card image as numpy array
        """
        try:
            logger.info(f"Generating synthetic card (grade {config.target_grade})")
            
            # Create base card structure
            image = self._create_base_card(config)
            
            # Add centering error
            if config.centering_error > 0:
                image = self._add_centering_error(image, config)
            
            # Add edge defects
            if config.edge_whitening > 0 or config.nick_count > 0:
                image = self._add_edge_defects(image, config)
            
            # Add corner wear
            if any(wear > 0 for wear in config.corner_wear.values()):
                image = self._add_corner_wear(image, config)
            
            # Add surface defects
            if config.scratch_count > 0 or config.surface_noise > 0:
                image = self._add_surface_defects(image, config)
            
            # Add glare effects
            if config.glare_regions > 0:
                image = self._add_glare_effects(image, config)
            
            # Save image if path provided
            if save_path:
                cv2.imwrite(save_path, image)
                logger.info(f"Synthetic card saved to {save_path}")
            
            return image
            
        except Exception as e:
            logger.error(f"Synthetic card generation failed: {e}")
            raise
    
    def _create_base_card(self, config: SyntheticCardConfig) -> np.ndarray:
        """Create the base card structure."""
        # Create image with border color
        image = np.full((config.height, config.width, 3), 
                       config.border_color, dtype=np.uint8)
        
        # Calculate margins for perfect centering
        margin_x = int(config.width * 0.08)  # 8% margin
        margin_y = int(config.height * 0.08)  # 8% margin
        
        # Create inner card area
        inner_left = margin_x
        inner_right = config.width - margin_x
        inner_top = margin_y
        inner_bottom = config.height - margin_y
        
        image[inner_top:inner_bottom, inner_left:inner_right] = config.card_color
        
        # Add some basic card artwork (simple patterns)
        self._add_card_artwork(image, inner_left, inner_right, inner_top, inner_bottom)
        
        return image
    
    def _add_card_artwork(self, image: np.ndarray, left: int, right: int, 
                         top: int, bottom: int):
        """Add basic artwork to make the card look more realistic."""
        # Add a simple title bar
        title_height = int((bottom - top) * 0.15)
        cv2.rectangle(image, (left + 10, top + 10), 
                     (right - 10, top + title_height), (200, 220, 240), -1)
        
        # Add artwork area
        art_top = top + title_height + 20
        art_bottom = art_top + int((bottom - top) * 0.4)
        cv2.rectangle(image, (left + 20, art_top), 
                     (right - 20, art_bottom), (180, 200, 220), -1)
        
        # Add text area
        text_top = art_bottom + 10
        for i in range(5):
            y = text_top + i * 15
            if y < bottom - 20:
                cv2.rectangle(image, (left + 30, y), 
                             (right - 30, y + 8), (160, 160, 160), -1)
    
    def _add_centering_error(self, image: np.ndarray, 
                           config: SyntheticCardConfig) -> np.ndarray:
        """Add centering error by shifting the card content."""
        if config.centering_error == 0:
            return image
        
        # Calculate shift amounts
        max_shift_x = int(config.width * config.centering_error * 0.3)
        max_shift_y = int(config.height * config.centering_error * 0.3)
        
        shift_x = random.randint(-max_shift_x, max_shift_x)
        shift_y = random.randint(-max_shift_y, max_shift_y)
        
        # Create shifted version
        h, w = image.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h), borderValue=config.border_color)
        
        return shifted
    
    def _add_edge_defects(self, image: np.ndarray, 
                         config: SyntheticCardConfig) -> np.ndarray:
        """Add edge whitening and nicks."""
        h, w = image.shape[:2]
        
        # Add edge whitening
        if config.edge_whitening > 0:
            whitening_width = max(1, int(w * 0.01))  # 1% of width
            whitening_intensity = int(255 * config.edge_whitening)
            
            # Whiten edges randomly
            edges = ['top', 'bottom', 'left', 'right']
            num_edges = max(1, int(len(edges) * config.edge_whitening))
            selected_edges = random.sample(edges, num_edges)
            
            for edge in selected_edges:
                if edge == 'top':
                    image[:whitening_width, :] = [255, 255, 255]
                elif edge == 'bottom':
                    image[-whitening_width:, :] = [255, 255, 255]
                elif edge == 'left':
                    image[:, :whitening_width] = [255, 255, 255]
                elif edge == 'right':
                    image[:, -whitening_width:] = [255, 255, 255]
        
        # Add nicks
        for _ in range(config.nick_count):
            # Random edge position
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            nick_size = random.randint(2, 8)
            
            if edge in ['top', 'bottom']:
                x = random.randint(nick_size, w - nick_size)
                y = 0 if edge == 'top' else h - 1
                # Create small triangular nick
                pts = np.array([[x, y], [x - nick_size, y + nick_size if edge == 'top' else y - nick_size],
                               [x + nick_size, y + nick_size if edge == 'top' else y - nick_size]], np.int32)
            else:
                x = 0 if edge == 'left' else w - 1
                y = random.randint(nick_size, h - nick_size)
                pts = np.array([[x, y], [x + nick_size if edge == 'left' else x - nick_size, y - nick_size],
                               [x + nick_size if edge == 'left' else x - nick_size, y + nick_size]], np.int32)
            
            cv2.fillPoly(image, [pts], (200, 200, 200))
        
        return image
    
    def _add_corner_wear(self, image: np.ndarray, 
                        config: SyntheticCardConfig) -> np.ndarray:
        """Add corner wear/rounding."""
        h, w = image.shape[:2]
        
        corners = {
            'top_left': (0, 0),
            'top_right': (w - 1, 0),
            'bottom_right': (w - 1, h - 1),
            'bottom_left': (0, h - 1)
        }
        
        for corner_name, (cx, cy) in corners.items():
            wear = config.corner_wear.get(corner_name, 0.0)
            if wear == 0:
                continue
            
            # Create corner wear effect
            radius = int(20 * wear)  # Max radius of 20 pixels
            
            # Create mask for corner area
            y, x = np.ogrid[:h, :w]
            
            if corner_name == 'top_left':
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
                mask = mask & (x <= radius) & (y <= radius)
            elif corner_name == 'top_right':
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
                mask = mask & (x >= w - radius) & (y <= radius)
            elif corner_name == 'bottom_right':
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
                mask = mask & (x >= w - radius) & (y >= h - radius)
            else:  # bottom_left
                mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
                mask = mask & (x <= radius) & (y >= h - radius)
            
            # Apply wear color (grayish)
            wear_color = [200 - int(55 * wear), 200 - int(55 * wear), 200 - int(55 * wear)]
            image[mask] = wear_color
        
        return image
    
    def _add_surface_defects(self, image: np.ndarray, 
                           config: SyntheticCardConfig) -> np.ndarray:
        """Add scratches and surface noise."""
        h, w = image.shape[:2]
        
        # Add scratches
        for _ in range(config.scratch_count):
            # Random scratch parameters
            start_x = random.randint(0, w)
            start_y = random.randint(0, h)
            length = random.randint(20, 100)
            angle = random.uniform(0, 2 * np.pi)
            thickness = random.randint(1, 3)
            
            # Calculate end point
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Ensure end point is within bounds
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            # Draw scratch
            scratch_color = [150, 150, 150]  # Dark gray scratch
            cv2.line(image, (start_x, start_y), (end_x, end_y), scratch_color, thickness)
        
        # Add surface noise
        if config.surface_noise > 0:
            noise = np.random.randint(-int(20 * config.surface_noise), 
                                    int(20 * config.surface_noise), 
                                    image.shape, dtype=np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_glare_effects(self, image: np.ndarray, 
                          config: SyntheticCardConfig) -> np.ndarray:
        """Add glare/reflection effects."""
        h, w = image.shape[:2]
        
        for _ in range(config.glare_regions):
            # Random glare parameters
            center_x = random.randint(w // 4, 3 * w // 4)
            center_y = random.randint(h // 4, 3 * h // 4)
            radius_x = random.randint(30, 80)
            radius_y = random.randint(20, 60)
            
            # Create elliptical glare region
            overlay = image.copy()
            cv2.ellipse(overlay, (center_x, center_y), (radius_x, radius_y), 
                       0, 0, 360, (255, 255, 255), -1)
            
            # Blend with original image
            alpha = 0.3 + random.uniform(0, 0.4)  # 30-70% opacity
            image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return image
    
    def generate_grade_set(self, grades: List[int], 
                          cards_per_grade: int = 5) -> Dict[int, List[str]]:
        """
        Generate a set of cards for specified grades.
        
        Args:
            grades: List of target grades to generate
            cards_per_grade: Number of cards to generate per grade
            
        Returns:
            Dictionary mapping grade to list of generated file paths
        """
        generated = {}
        
        for grade in grades:
            generated[grade] = []
            
            for i in range(cards_per_grade):
                config = self._create_config_for_grade(grade)
                filename = f"grade_{grade:02d}_card_{i+1:03d}.jpg"
                save_path = str(self.output_dir / filename)
                
                self.generate_card(config, save_path)
                generated[grade].append(save_path)
        
        # Save metadata
        metadata_path = self.output_dir / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(generated, f, indent=2)
        
        return generated
    
    def _create_config_for_grade(self, target_grade: int) -> SyntheticCardConfig:
        """Create a realistic configuration for the target grade."""
        config = SyntheticCardConfig(target_grade=target_grade)
        
        if target_grade == 10:  # Gem Mint
            # Perfect card
            pass
        
        elif target_grade == 9:  # Mint
            # Very minor imperfections
            config.centering_error = random.uniform(0.02, 0.05)
            if random.random() < 0.3:
                config.edge_whitening = random.uniform(0.01, 0.03)
        
        elif target_grade == 8:  # NM-Mint
            config.centering_error = random.uniform(0.05, 0.12)
            config.edge_whitening = random.uniform(0.02, 0.08)
            if random.random() < 0.4:
                config.nick_count = random.randint(1, 2)
            # Minor corner wear
            corner = random.choice(list(config.corner_wear.keys()))
            config.corner_wear[corner] = random.uniform(0.1, 0.3)
        
        elif target_grade == 7:  # Near Mint
            config.centering_error = random.uniform(0.10, 0.20)
            config.edge_whitening = random.uniform(0.05, 0.15)
            config.nick_count = random.randint(1, 3)
            # Corner wear on 1-2 corners
            corners_to_wear = random.sample(list(config.corner_wear.keys()), 
                                          random.randint(1, 2))
            for corner in corners_to_wear:
                config.corner_wear[corner] = random.uniform(0.2, 0.5)
            config.scratch_count = random.randint(0, 1)
        
        elif target_grade == 6:  # Excellent
            config.centering_error = random.uniform(0.15, 0.25)
            config.edge_whitening = random.uniform(0.10, 0.25)
            config.nick_count = random.randint(2, 5)
            # More corner wear
            corners_to_wear = random.sample(list(config.corner_wear.keys()),
                                          random.randint(2, 3))
            for corner in corners_to_wear:
                config.corner_wear[corner] = random.uniform(0.3, 0.6)
            config.scratch_count = random.randint(1, 3)
            config.surface_noise = random.uniform(0.1, 0.2)
        
        elif target_grade <= 5:  # Lower grades
            config.centering_error = random.uniform(0.20, 0.35)
            config.edge_whitening = random.uniform(0.20, 0.40)
            config.nick_count = random.randint(3, 8)
            # Heavy corner wear
            for corner in config.corner_wear:
                config.corner_wear[corner] = random.uniform(0.4, 0.8)
            config.scratch_count = random.randint(2, 6)
            config.surface_noise = random.uniform(0.2, 0.4)
            if random.random() < 0.3:
                config.glare_regions = random.randint(1, 2)
        
        return config


def main():
    """Command line interface for generating synthetic cards."""
    parser = argparse.ArgumentParser(description="Generate synthetic trading cards")
    parser.add_argument("--output-dir", default="examples/synthetic_cards",
                       help="Output directory for generated cards")
    parser.add_argument("--grades", nargs="+", type=int, 
                       default=[10, 9, 8, 7, 6, 5],
                       help="Grades to generate")
    parser.add_argument("--count-per-grade", type=int, default=5,
                       help="Number of cards per grade")
    parser.add_argument("--single-grade", type=int,
                       help="Generate a single card of specified grade")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    generator = SyntheticCardGenerator(args.output_dir)
    
    if args.single_grade:
        config = generator._create_config_for_grade(args.single_grade)
        filename = f"grade_{args.single_grade:02d}_sample.jpg"
        save_path = str(Path(args.output_dir) / filename)
        
        generator.generate_card(config, save_path)
        print(f"Generated single card: {save_path}")
    else:
        generated = generator.generate_grade_set(args.grades, args.count_per_grade)
        
        total_cards = sum(len(files) for files in generated.values())
        print(f"Generated {total_cards} synthetic cards in {args.output_dir}")
        
        for grade, files in generated.items():
            print(f"  Grade {grade}: {len(files)} cards")


if __name__ == "__main__":
    main()