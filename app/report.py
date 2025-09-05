"""
Report generation and export functionality for card grading results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import base64
from io import BytesIO

import numpy as np
import cv2
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from app.schema import CardAnalysisResult, GradingResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive reports for card grading analysis.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='ScoreText',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
    
    def generate_pdf_report(self, result: CardAnalysisResult, 
                          output_path: str,
                          original_image: Optional[np.ndarray] = None,
                          overlay_images: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        Generate a comprehensive PDF report.
        
        Args:
            result: Analysis result to report on
            output_path: Path to save PDF file
            original_image: Original card image (optional)
            overlay_images: Dictionary of overlay images by type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            title = Paragraph("Card Grading Analysis Report", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Analysis date
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            date_p = Paragraph(f"Analysis Date: {date_str}", self.styles['Normal'])
            story.append(date_p)
            story.append(Spacer(1, 20))
            
            # Overall Grade
            grade_data = [
                ['Final Grade', f"{result.final_grade.numeric} - {result.final_grade.label}"],
                ['Overall Score', f"{result.final_grade.overall_score:.2f} / 100"],
                ['Confidence', f"{result.final_grade.confidence:.1%}"]
            ]
            
            grade_table = Table(grade_data, colWidths=[2.5*inch, 3*inch])
            grade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(grade_table)
            story.append(Spacer(1, 20))
            
            # Sub-scores breakdown
            story.append(Paragraph("Detailed Sub-Scores", self.styles['SectionHeader']))
            
            subscores_data = [['Metric', 'Score', 'Weight', 'Weighted Score']]
            
            if result.centering:
                subscores_data.append([
                    'Centering',
                    f"{result.centering.centering_score:.2f}",
                    f"{result.final_grade.sub_scores['centering']['weight']:.1%}",
                    f"{result.final_grade.sub_scores['centering']['weighted_score']:.2f}"
                ])
            
            if result.edges:
                subscores_data.append([
                    'Edges',
                    f"{result.edges.edge_score:.2f}",
                    f"{result.final_grade.sub_scores['edges']['weight']:.1%}",
                    f"{result.final_grade.sub_scores['edges']['weighted_score']:.2f}"
                ])
            
            if result.corners:
                subscores_data.append([
                    'Corners',
                    f"{result.corners.corner_score:.2f}",
                    f"{result.final_grade.sub_scores['corners']['weight']:.1%}",
                    f"{result.final_grade.sub_scores['corners']['weighted_score']:.2f}"
                ])
            
            if result.surface:
                subscores_data.append([
                    'Surface',
                    f"{result.surface.surface_quality_score:.2f}",
                    f"{result.final_grade.sub_scores['surface']['weight']:.1%}",
                    f"{result.final_grade.sub_scores['surface']['weighted_score']:.2f}"
                ])
            
            subscores_table = Table(subscores_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
            subscores_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(subscores_table)
            story.append(Spacer(1, 20))
            
            # Detailed findings
            self._add_detailed_findings(story, result)
            
            # Add images if available
            if original_image is not None:
                story.append(Paragraph("Original Image", self.styles['SectionHeader']))
                img_buffer = self._image_to_buffer(original_image)
                if img_buffer:
                    rl_image = RLImage(img_buffer, width=3*inch, height=4*inch)
                    story.append(rl_image)
                    story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF report generation failed: {e}")
            return False
    
    def _add_detailed_findings(self, story: List, result: CardAnalysisResult):
        """Add detailed findings sections to the report."""
        
        # Centering details
        if result.centering:
            story.append(Paragraph("Centering Analysis", self.styles['SectionHeader']))
            centering_data = [
                ['Left Margin', f"{result.centering.left_margin_px:.1f} px"],
                ['Right Margin', f"{result.centering.right_margin_px:.1f} px"],
                ['Top Margin', f"{result.centering.top_margin_px:.1f} px"],
                ['Bottom Margin', f"{result.centering.bottom_margin_px:.1f} px"],
                ['Horizontal Error', f"{result.centering.horizontal_error:.3f}"],
                ['Vertical Error', f"{result.centering.vertical_error:.3f}"],
                ['Combined Error', f"{result.centering.combined_error:.3f}"]
            ]
            
            centering_table = Table(centering_data, colWidths=[2*inch, 1.5*inch])
            centering_table.setStyle(self._get_detail_table_style())
            story.append(centering_table)
            story.append(Spacer(1, 12))
        
        # Edges details
        if result.edges:
            story.append(Paragraph("Edge Analysis", self.styles['SectionHeader']))
            edge_data = [
                ['Whitening Percentage', f"{result.edges.whitening_percentage:.2f}%"],
                ['Clean Edge Percentage', f"{result.edges.clean_edge_percentage:.2f}%"],
                ['Nick Count', str(result.edges.nick_count)]
            ]
            
            edge_table = Table(edge_data, colWidths=[2*inch, 1.5*inch])
            edge_table.setStyle(self._get_detail_table_style())
            story.append(edge_table)
            story.append(Spacer(1, 12))
        
        # Corners details
        if result.corners:
            story.append(Paragraph("Corner Analysis", self.styles['SectionHeader']))
            corner_data = [['Corner', 'Score']]
            
            for corner, score in result.corners.corner_scores.items():
                corner_data.append([corner.replace('_', ' ').title(), f"{score:.1f}"])
            
            corner_table = Table(corner_data, colWidths=[2*inch, 1.5*inch])
            corner_table.setStyle(self._get_detail_table_style())
            story.append(corner_table)
            story.append(Spacer(1, 12))
        
        # Surface details
        if result.surface:
            story.append(Paragraph("Surface Analysis", self.styles['SectionHeader']))
            surface_data = [
                ['Defect Percentage', f"{result.surface.defect_percentage:.3f}%"],
                ['Scratch Count', str(result.surface.scratch_count)],
                ['Print Line Count', str(result.surface.print_line_count)]
            ]
            
            surface_table = Table(surface_data, colWidths=[2*inch, 1.5*inch])
            surface_table.setStyle(self._get_detail_table_style())
            story.append(surface_table)
            story.append(Spacer(1, 12))
        
        # Glare details
        if result.glare and result.glare.glare_detected:
            story.append(Paragraph("Glare Analysis", self.styles['SectionHeader']))
            glare_data = [
                ['Glare Percentage', f"{result.glare.glare_percentage:.2f}%"],
                ['Penalty Applied', f"-{result.glare.penalty_applied:.1f} points"],
                ['Affected Regions', str(len(result.glare.affected_regions))]
            ]
            
            glare_table = Table(glare_data, colWidths=[2*inch, 1.5*inch])
            glare_table.setStyle(self._get_detail_table_style())
            story.append(glare_table)
            story.append(Spacer(1, 12))
    
    def _get_detail_table_style(self) -> TableStyle:
        """Get standard table style for detail tables."""
        return TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
    
    def _image_to_buffer(self, image: np.ndarray) -> Optional[BytesIO]:
        """Convert numpy image to BytesIO buffer for ReportLab."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Encode to PNG
            success, buffer = cv2.imencode('.png', image)
            if not success:
                return None
                
            return BytesIO(buffer.tobytes())
            
        except Exception as e:
            logger.error(f"Image buffer conversion failed: {e}")
            return None
    
    def export_to_csv(self, results: List[CardAnalysisResult], output_path: str) -> bool:
        """
        Export analysis results to CSV format.
        
        Args:
            results: List of analysis results
            output_path: Path to save CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'analysis_date', 'final_grade', 'overall_score', 'confidence',
                    'centering_score', 'edge_score', 'corner_score', 'surface_score',
                    'horizontal_error', 'vertical_error', 'whitening_percent',
                    'nick_count', 'defect_percent', 'scratch_count', 'glare_percent'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'analysis_date': result.analysis_date.isoformat(),
                        'final_grade': f"{result.final_grade.numeric} - {result.final_grade.label}",
                        'overall_score': result.final_grade.overall_score,
                        'confidence': result.final_grade.confidence,
                        'centering_score': result.centering.centering_score if result.centering else '',
                        'edge_score': result.edges.edge_score if result.edges else '',
                        'corner_score': result.corners.corner_score if result.corners else '',
                        'surface_score': result.surface.surface_quality_score if result.surface else '',
                        'horizontal_error': result.centering.horizontal_error if result.centering else '',
                        'vertical_error': result.centering.vertical_error if result.centering else '',
                        'whitening_percent': result.edges.whitening_percentage if result.edges else '',
                        'nick_count': result.edges.nick_count if result.edges else '',
                        'defect_percent': result.surface.defect_percentage if result.surface else '',
                        'scratch_count': result.surface.scratch_count if result.surface else '',
                        'glare_percent': result.glare.glare_percentage if result.glare else ''
                    }
                    writer.writerow(row)
            
            logger.info(f"CSV export completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    def export_to_json(self, results: List[CardAnalysisResult], output_path: str) -> bool:
        """
        Export analysis results to JSON format.
        
        Args:
            results: List of analysis results
            output_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert results to JSON-serializable format
            json_data = []
            for result in results:
                json_data.append(result.dict())
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(json_data, jsonfile, indent=2, default=str)
            
            logger.info(f"JSON export completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False


def generate_batch_report(results: List[CardAnalysisResult], 
                         output_dir: str,
                         format_types: List[str] = ['pdf', 'csv', 'json']) -> Dict[str, bool]:
    """
    Generate batch reports in multiple formats.
    
    Args:
        results: List of analysis results
        output_dir: Directory to save reports
        format_types: List of formats to generate
        
    Returns:
        Dictionary of format -> success status
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"card_analysis_batch_{timestamp}"
        
        reporter = ReportGenerator()
        status = {}
        
        if 'csv' in format_types:
            csv_path = output_path / f"{base_filename}.csv"
            status['csv'] = reporter.export_to_csv(results, str(csv_path))
        
        if 'json' in format_types:
            json_path = output_path / f"{base_filename}.json"
            status['json'] = reporter.export_to_json(results, str(json_path))
        
        if 'pdf' in format_types and results:
            # Generate individual PDFs for each result
            for i, result in enumerate(results):
                pdf_path = output_path / f"{base_filename}_card_{i+1:03d}.pdf"
                success = reporter.generate_pdf_report(result, str(pdf_path))
                if i == 0:  # Track status of first PDF as representative
                    status['pdf'] = success
        
        return status
        
    except Exception as e:
        logger.error(f"Batch report generation failed: {e}")
        return {fmt: False for fmt in format_types}