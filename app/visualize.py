"""
Visualization and overlay creation for card analysis results.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from app.schema import CenteringFindings, EdgeFindings, CornerFindings, SurfaceFindings, GlareFindings

logger = logging.getLogger(__name__)


def create_centering_overlay(image: np.ndarray, findings: CenteringFindings) -> np.ndarray:
    """
    Create visual overlay showing centering analysis.
    
    Args:
        image: Original rectified image
        findings: Centering analysis findings
        
    Returns:
        Image with centering overlay
    """
    try:
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Calculate inner frame from findings
        left = int(findings.left_margin_px)
        top = int(findings.top_margin_px)
        right = int(findings.right_margin_px)
        bottom = int(findings.bottom_margin_px)
        
        # Draw outer rectangle (card edge)
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), (255, 0, 0), 3)  # Red
        
        # Draw inner rectangle (detected frame)
        inner_x = left
        inner_y = top
        inner_w = w - left - right
        inner_h = h - top - bottom
        
        if inner_w > 0 and inner_h > 0:
            cv2.rectangle(overlay, (inner_x, inner_y), 
                         (inner_x + inner_w - 1, inner_y + inner_h - 1), (0, 255, 0), 2)  # Green
        
        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)  # White
        font_thickness = 2
        
        # Margin measurements
        cv2.putText(overlay, f"L: {left}px", (5, 30), 
                   font, font_scale, font_color, font_thickness)
        cv2.putText(overlay, f"R: {right}px", (w-100, 30), 
                   font, font_scale, font_color, font_thickness)
        cv2.putText(overlay, f"T: {top}px", (w//2-50, 25), 
                   font, font_scale, font_color, font_thickness)
        cv2.putText(overlay, f"B: {bottom}px", (w//2-50, h-10), 
                   font, font_scale, font_color, font_thickness)
        
        return overlay
        
    except Exception as e:
        logger.error(f"Centering overlay creation failed: {e}")
        return image


def create_edge_overlay(image: np.ndarray, findings: EdgeFindings) -> np.ndarray:
    """
    Create visual overlay showing edge condition analysis.
    
    Args:
        image: Original rectified image
        findings: Edge analysis findings
        
    Returns:
        Image with edge overlay
    """
    try:
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Create simple edge visualization
        border_width = 6
        
        # Draw edge border
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), (0, 255, 0), 2)
        cv2.rectangle(overlay, (border_width, border_width), 
                     (w-border_width-1, h-border_width-1), (0, 255, 0), 1)
        
        # Add statistics text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        stats_text = [
            f"Whitening: {findings.whitening_percentage:.1f}%",
            f"Clean: {findings.clean_edge_percentage:.1f}%",
            f"Nicks: {findings.nick_count}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (10, 30 + i*25), 
                       font, 0.6, text_color, 2)
        
        return overlay
        
    except Exception as e:
        logger.error(f"Edge overlay creation failed: {e}")
        return image


def create_corner_overlay(image: np.ndarray, findings: CornerFindings) -> np.ndarray:
    """
    Create visual overlay showing corner condition analysis.
    
    Args:
        image: Original rectified image
        findings: Corner analysis findings
        
    Returns:
        Image with corner overlay
    """
    try:
        overlay = image.copy()
        h, w = overlay.shape[:2]
        corner_size = 50
        
        # Define corner positions
        corners = {
            'top_left': (0, 0, corner_size, corner_size),
            'top_right': (w-corner_size, 0, corner_size, corner_size),
            'bottom_right': (w-corner_size, h-corner_size, corner_size, corner_size),
            'bottom_left': (0, h-corner_size, corner_size, corner_size)
        }
        
        # Color code corners by score
        for corner_name, (x, y, cw, ch) in corners.items():
            score = findings.corner_scores.get(corner_name, 0)
            
            # Determine color based on score
            if score >= 90:
                color = (0, 255, 0)  # Green - excellent
            elif score >= 70:
                color = (255, 255, 0)  # Yellow - good
            elif score >= 50:
                color = (255, 165, 0)  # Orange - fair
            else:
                color = (255, 0, 0)  # Red - poor
            
            # Draw corner rectangle
            cv2.rectangle(overlay, (x, y), (x+cw, y+ch), color, 3)
            
            # Add score text
            text_x = x + 5 if x == 0 else x - 30
            text_y = y + 20 if y == 0 else y - 5
            cv2.putText(overlay, f"{score:.0f}", (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return overlay
        
    except Exception as e:
        logger.error(f"Corner overlay creation failed: {e}")
        return image


def create_surface_overlay(image: np.ndarray, findings: SurfaceFindings) -> np.ndarray:
    """
    Create visual overlay showing surface condition analysis.
    
    Args:
        image: Original rectified image
        findings: Surface analysis findings
        
    Returns:
        Image with surface overlay
    """
    try:
        overlay = image.copy()
        
        # Draw defect regions
        for defect in findings.defect_regions:
            bbox = defect.bbox
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            
            # Color code by defect type
            if defect.defect_type == "scratch":
                color = (255, 0, 0)  # Red
            elif defect.defect_type == "print_line":
                color = (0, 0, 255)  # Blue
            else:
                color = (255, 165, 0)  # Orange
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
            
            # Add defect type label
            cv2.putText(overlay, defect.defect_type[:3].upper(), 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add surface statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        stats_text = [
            f"Defects: {findings.defect_percentage:.2f}%",
            f"Scratches: {findings.scratch_count}",
            f"Quality: {findings.surface_quality_score:.2f}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (10, image.shape[0] - 80 + i*20), 
                       font, 0.5, text_color, 1)
        
        return overlay
        
    except Exception as e:
        logger.error(f"Surface overlay creation failed: {e}")
        return image


def create_glare_overlay(image: np.ndarray, findings: GlareFindings) -> np.ndarray:
    """
    Create visual overlay showing glare analysis.
    
    Args:
        image: Original rectified image
        findings: Glare analysis findings
        
    Returns:
        Image with glare overlay
    """
    try:
        overlay = image.copy()
        
        if not findings.glare_detected:
            # Add "No Glare" text
            cv2.putText(overlay, "No Significant Glare Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return overlay
        
        # Draw bounding boxes for glare regions
        for i, region in enumerate(findings.affected_regions):
            x, y, w, h = region.x, region.y, region.width, region.height
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(overlay, f"GLARE {i+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add glare statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        stats_text = [
            f"Glare: {findings.glare_percentage:.2f}%",
            f"Penalty: -{findings.penalty_applied:.1f} pts"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (10, 30 + i*25),
                       font, 0.6, text_color, 2)
        
        return overlay
        
    except Exception as e:
        logger.error(f"Glare overlay creation failed: {e}")
        return image