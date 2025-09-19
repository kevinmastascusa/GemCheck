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
    Create advanced computational photography overlay showing centering analysis.
    
    Args:
        image: Original rectified image
        findings: Centering analysis findings
        
    Returns:
        Image with advanced centering overlay
    """
    try:
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Calculate inner frame from findings
        left = int(findings.left_margin_px)
        top = int(findings.top_margin_px)
        right = int(findings.right_margin_px)
        bottom = int(findings.bottom_margin_px)
        
        # Create semi-transparent overlay for better visualization
        overlay_alpha = overlay.copy()
        
        # Draw outer rectangle (card edge)
        cv2.rectangle(overlay_alpha, (0, 0), (w-1, h-1), (255, 0, 0), 3)  # Red
        
        # Draw inner rectangle (detected frame)
        inner_x = left
        inner_y = top
        inner_w = w - left - right
        inner_h = h - top - bottom
        
        if inner_w > 0 and inner_h > 0:
            cv2.rectangle(overlay_alpha, (inner_x, inner_y), 
                         (inner_x + inner_w - 1, inner_y + inner_h - 1), (0, 255, 0), 2)  # Green
        
        # Draw centering guide lines
        center_x = w // 2
        center_y = h // 2
        
        # Vertical center line
        cv2.line(overlay_alpha, (center_x, 0), (center_x, h), (0, 255, 255), 1)  # Cyan
        # Horizontal center line  
        cv2.line(overlay_alpha, (0, center_y), (w, center_y), (0, 255, 255), 1)  # Cyan
        
        # Draw margin measurement lines with arrows
        arrow_color = (255, 255, 0)  # Yellow
        arrow_thickness = 2
        
        # Left margin arrow
        cv2.arrowedLine(overlay_alpha, (0, center_y), (left, center_y), arrow_color, arrow_thickness)
        # Right margin arrow
        cv2.arrowedLine(overlay_alpha, (w-1, center_y), (w-right-1, center_y), arrow_color, arrow_thickness)
        # Top margin arrow
        cv2.arrowedLine(overlay_alpha, (center_x, 0), (center_x, top), arrow_color, arrow_thickness)
        # Bottom margin arrow
        cv2.arrowedLine(overlay_alpha, (center_x, h-1), (center_x, h-bottom-1), arrow_color, arrow_thickness)
        
        # Calculate centering error visualization
        h_error = findings.horizontal_error
        v_error = findings.vertical_error
        combined_error = findings.combined_error
        
        # Draw error indicator circles
        error_center_x = inner_x + inner_w // 2
        error_center_y = inner_y + inner_h // 2
        
        # Error magnitude circle (larger circle = worse centering)
        error_radius = int(min(50, combined_error * 100))
        if error_radius > 5:
            cv2.circle(overlay_alpha, (error_center_x, error_center_y), error_radius, (255, 0, 255), 2)  # Magenta
        
        # Perfect center indicator
        cv2.circle(overlay_alpha, (center_x, center_y), 5, (0, 255, 0), -1)  # Green dot
        
        # Actual center indicator  
        cv2.circle(overlay_alpha, (error_center_x, error_center_y), 5, (255, 0, 0), -1)  # Red dot
        
        # Draw error vector from perfect center to actual center
        if error_radius > 5:
            cv2.arrowedLine(overlay_alpha, (center_x, center_y), (error_center_x, error_center_y), (255, 0, 255), 2)
        
        # Add advanced text annotations with background boxes
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # White
        font_thickness = 2
        
        # Function to draw text with background
        def draw_text_with_bg(img, text, pos, font, scale, color, thickness, bg_color=(0, 0, 0)):
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = pos
            cv2.rectangle(img, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), bg_color, -1)
            cv2.putText(img, text, pos, font, scale, color, thickness)
        
        # Margin measurements with backgrounds
        draw_text_with_bg(overlay_alpha, f"L: {left}px", (5, 30), font, 0.6, font_color, 2)
        draw_text_with_bg(overlay_alpha, f"R: {right}px", (w-100, 30), font, 0.6, font_color, 2)
        draw_text_with_bg(overlay_alpha, f"T: {top}px", (w//2-40, 25), font, 0.6, font_color, 2)
        draw_text_with_bg(overlay_alpha, f"B: {bottom}px", (w//2-40, h-10), font, 0.6, font_color, 2)
        
        # Error measurements
        error_y_start = h - 120
        draw_text_with_bg(overlay_alpha, f"H Error: {h_error:.4f}", (10, error_y_start), font, 0.5, font_color, 1)
        draw_text_with_bg(overlay_alpha, f"V Error: {v_error:.4f}", (10, error_y_start + 25), font, 0.5, font_color, 1)
        draw_text_with_bg(overlay_alpha, f"Combined: {combined_error:.4f}", (10, error_y_start + 50), font, 0.5, font_color, 1)
        
        # Score with color coding
        score = findings.centering_score
        if score >= 90:
            score_color = (0, 255, 0)  # Green
        elif score >= 70:
            score_color = (255, 255, 0)  # Yellow
        elif score >= 50:
            score_color = (255, 165, 0)  # Orange
        else:
            score_color = (255, 0, 0)  # Red
            
        draw_text_with_bg(overlay_alpha, f"Score: {score:.1f}/100", (10, error_y_start + 75), font, 0.6, score_color, 2)
        
        # Blend overlay with original image for transparency effect
        alpha = 0.8
        overlay = cv2.addWeighted(overlay, 1-alpha, overlay_alpha, alpha, 0)
        
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