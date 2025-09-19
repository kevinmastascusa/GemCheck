"""
Specialized visualization for holographic card analysis with enhanced overlays.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .rarity_detector import RarityFeatures
from .visual_analyzer import PokemonDefectAnalysis, PokemonCardRegions

logger = logging.getLogger(__name__)


@dataclass
class HolographicOverlaySettings:
    """Settings for holographic visualization overlays."""
    show_holo_mask: bool = True
    show_scratch_lines: bool = True
    show_peeling_areas: bool = True
    show_pattern_analysis: bool = True
    show_rainbow_disruption: bool = True
    overlay_alpha: float = 0.6
    highlight_severity: bool = True


class HolographicVisualizer:
    """Specialized visualizer for holographic card analysis."""
    
    def __init__(self):
        # Color scheme for different holo elements
        self.colors = {
            'holo_area': (255, 215, 0),      # Gold for holo areas
            'scratch': (255, 0, 255),        # Magenta for scratches
            'peeling': (255, 0, 0),          # Red for peeling
            'pattern_good': (0, 255, 0),     # Green for good patterns
            'pattern_poor': (0, 0, 255),     # Blue for poor patterns
            'rainbow_disruption': (255, 165, 0),  # Orange for disruption
            'reverse_holo': (0, 255, 255),   # Cyan for reverse holo
            'gold_foil': (255, 215, 0),      # Gold
            'rainbow_foil': (128, 0, 128)    # Purple for rainbow
        }
    
    def create_holographic_overlay(self, image: np.ndarray, 
                                 rarity_features: RarityFeatures,
                                 defect_analysis: PokemonDefectAnalysis,
                                 regions: PokemonCardRegions,
                                 settings: Optional[HolographicOverlaySettings] = None) -> np.ndarray:
        """
        Create comprehensive holographic analysis overlay.
        
        Args:
            image: Original card image
            rarity_features: Detected rarity features
            defect_analysis: Defect analysis results
            regions: Detected card regions
            settings: Overlay settings
            
        Returns:
            Image with holographic overlay
        """
        if settings is None:
            settings = HolographicOverlaySettings()
        
        try:
            overlay = image.copy()
            h, w = overlay.shape[:2]
            
            # Create overlay layers
            holo_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 1. Show holographic areas
            if settings.show_holo_mask and rarity_features.has_holographic:
                self._draw_holographic_areas(image, holo_overlay, rarity_features)
            
            # 2. Show scratch analysis
            if settings.show_scratch_lines and defect_analysis.holo_scratches:
                self._draw_holo_scratches(holo_overlay, defect_analysis.holo_scratches, 
                                        settings.highlight_severity)
            
            # 3. Show foil peeling areas
            if settings.show_peeling_areas and defect_analysis.foil_peeling:
                self._draw_foil_peeling(holo_overlay, defect_analysis.foil_peeling)
            
            # 4. Show pattern analysis
            if settings.show_pattern_analysis:
                self._draw_pattern_analysis(holo_overlay, rarity_features, regions)
            
            # 5. Show rainbow disruption
            if settings.show_rainbow_disruption:
                self._draw_rainbow_analysis(image, holo_overlay, rarity_features)
            
            # 6. Add informational overlays
            self._add_holographic_info_overlay(overlay, rarity_features, defect_analysis)
            
            # Blend overlays
            alpha = settings.overlay_alpha
            result = cv2.addWeighted(overlay, 1 - alpha, holo_overlay, alpha, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Holographic overlay creation failed: {e}")
            return image

    def _draw_holographic_areas(self, image: np.ndarray, overlay: np.ndarray, 
                               features: RarityFeatures):
        """Draw detected holographic areas on overlay."""
        # Recreate holo detection for visualization
        from .rarity_detector import PokemonRarityDetector
        detector = PokemonRarityDetector()
        
        # Get holo mask using the same method as detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Basic holo detection for visualization
        holo_mask = cv2.bitwise_and(
            saturation > 80,
            value > 120
        )
        
        # Color code different holo types
        if features.has_reverse_holo:
            color = self.colors['reverse_holo']
        elif features.has_gold_foil:
            color = self.colors['gold_foil']
        elif features.has_rainbow_foil:
            color = self.colors['rainbow_foil']
        else:
            color = self.colors['holo_area']
        
        # Apply color to holo areas
        overlay[holo_mask > 0] = color
        
        # Add contours for better visibility
        contours, _ = cv2.findContours(holo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    def _draw_holo_scratches(self, overlay: np.ndarray, scratches: List[Dict[str, Any]], 
                           highlight_severity: bool):
        """Draw holographic scratches on overlay."""
        for scratch in scratches:
            start = scratch['start']
            end = scratch['end']
            severity = scratch.get('severity', 0.5)
            
            # Color intensity based on severity
            if highlight_severity:
                intensity = int(255 * severity)
                color = (intensity, 0, 255 - intensity)  # Red to blue gradient
            else:
                color = self.colors['scratch']
            
            # Draw scratch line with thickness based on severity
            thickness = max(1, int(severity * 4))
            cv2.line(overlay, start, end, color, thickness)
            
            # Add severity indicator
            if highlight_severity and severity > 0.5:
                center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                cv2.circle(overlay, center, 5, (255, 255, 255), -1)
                cv2.circle(overlay, center, 5, color, 2)

    def _draw_foil_peeling(self, overlay: np.ndarray, peeling_areas: List[Tuple[int, int, int, int]]):
        """Draw foil peeling areas on overlay."""
        for x, y, w, h in peeling_areas:
            # Draw peeling area with hatched pattern
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.colors['peeling'], 2)
            
            # Add diagonal lines to indicate peeling
            for i in range(0, max(w, h), 10):
                start_x = x + i
                start_y = y
                end_x = x
                end_y = y + i
                
                if start_x < x + w:
                    cv2.line(overlay, (start_x, start_y), (end_x, end_y), 
                           self.colors['peeling'], 1)

    def _draw_pattern_analysis(self, overlay: np.ndarray, features: RarityFeatures, 
                             regions: PokemonCardRegions):
        """Draw pattern analysis visualization."""
        if features.holo_pattern_type and regions.artwork:
            x, y, w, h = regions.artwork
            
            # Pattern type indicator
            pattern_colors = {
                'cosmos': (255, 255, 0),     # Yellow
                'linear': (0, 255, 255),     # Cyan
                'crosshatch': (255, 0, 255), # Magenta
                'rainbow': (128, 0, 128),    # Purple
                'lightning': (255, 165, 0),  # Orange
                'leaf': (0, 255, 0)          # Green
            }
            
            color = pattern_colors.get(features.holo_pattern_type, (255, 255, 255))
            
            # Draw pattern indicator in artwork region
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
            
            # Add pattern type label
            label = f"Pattern: {features.holo_pattern_type}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x + (w - text_w) // 2
            text_y = y - 10
            
            if text_y > 20:
                cv2.putText(overlay, label, (text_x, text_y), font, font_scale, 
                          color, thickness)

    def _draw_rainbow_analysis(self, image: np.ndarray, overlay: np.ndarray, 
                             features: RarityFeatures):
        """Draw rainbow effect analysis."""
        if features.has_rainbow_foil:
            # Analyze rainbow distribution
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            saturation = hsv[:, :, 1]
            
            # Create rainbow visualization
            rainbow_mask = saturation > 100
            unique_hues = np.unique(hue[rainbow_mask])
            
            # Create rainbow overlay
            for i, hue_val in enumerate(unique_hues[::10]):  # Sample every 10th hue
                hue_mask = np.abs(hue - hue_val) < 5
                combined_mask = cv2.bitwise_and(rainbow_mask, hue_mask)
                
                # Convert hue to RGB for visualization
                hue_color = cv2.cvtColor(np.array([[[hue_val, 255, 255]]], dtype=np.uint8), 
                                       cv2.COLOR_HSV2RGB)[0, 0]
                
                overlay[combined_mask] = hue_color

    def _add_holographic_info_overlay(self, overlay: np.ndarray, 
                                    features: RarityFeatures,
                                    defects: PokemonDefectAnalysis):
        """Add informational text overlay about holographic analysis."""
        h, w = overlay.shape[:2]
        
        # Background for text
        info_bg = np.zeros((150, 400, 3), dtype=np.uint8)
        info_bg[:] = (0, 0, 0, 128)  # Semi-transparent black
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        
        # Holographic information
        info_lines = []
        
        if features.has_holographic:
            info_lines.append(f"Holographic: YES ({features.holo_intensity:.1%} coverage)")
            info_lines.append(f"Pattern: {features.holo_pattern_type}")
            info_lines.append(f"Shine Intensity: {features.shine_intensity:.2f}")
            
            if features.has_reverse_holo:
                info_lines.append("Type: Reverse Holo")
            elif features.has_gold_foil:
                info_lines.append("Type: Gold Foil")
            elif features.has_rainbow_foil:
                info_lines.append("Type: Rainbow Foil")
            
            # Defect information
            if defects.holo_scratches:
                info_lines.append(f"Holo Scratches: {len(defects.holo_scratches)}")
            
            if defects.foil_peeling:
                info_lines.append(f"Foil Peeling Areas: {len(defects.foil_peeling)}")
            
            info_lines.append(f"Holo Wear: {defects.holo_wear:.1%}")
        else:
            info_lines.append("Holographic: NO")
        
        # Draw text lines
        for i, line in enumerate(info_lines):
            y_pos = 20 + i * 20
            if y_pos < info_bg.shape[0] - 10:
                cv2.putText(info_bg, line, (10, y_pos), font, font_scale, color, thickness)
        
        # Place info overlay in bottom-left corner
        info_h, info_w = info_bg.shape[:2]
        if h > info_h and w > info_w:
            # Blend info background
            roi = overlay[h - info_h:h, 0:info_w]
            blended = cv2.addWeighted(roi, 0.7, info_bg, 0.3, 0)
            overlay[h - info_h:h, 0:info_w] = blended

    def create_holo_comparison_overlay(self, original: np.ndarray, 
                                     processed: np.ndarray,
                                     features: RarityFeatures) -> np.ndarray:
        """Create side-by-side comparison showing original vs holographic analysis."""
        h, w = original.shape[:2]
        
        # Create side-by-side image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w:] = processed
        
        # Add dividing line
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Holo Analysis", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return comparison

    def create_holo_severity_heatmap(self, image: np.ndarray, 
                                   defects: PokemonDefectAnalysis) -> np.ndarray:
        """Create a heatmap showing holographic defect severity."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Add scratch severity to heatmap
        for scratch in defects.holo_scratches:
            start = scratch['start']
            end = scratch['end']
            severity = scratch.get('severity', 0.5)
            
            # Draw line on heatmap with severity intensity
            cv2.line(heatmap, start, end, severity, max(1, int(severity * 5)))
        
        # Add peeling areas
        for x, y, w_area, h_area in defects.foil_peeling:
            severity = 0.8  # High severity for peeling
            cv2.rectangle(heatmap, (x, y), (x + w_area, y + h_area), severity, -1)
        
        # Convert to color heatmap
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.7, heatmap_colored, 0.3, 0)
        
        return result