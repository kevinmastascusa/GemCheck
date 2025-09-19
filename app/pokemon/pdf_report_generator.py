"""
Comprehensive PDF Report Generator for Pokémon Card Grading Analysis.
Creates detailed reports with visual overlays and explanations for each card part.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import base64
from io import BytesIO

# PDF generation libraries
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, white, red, green, blue, orange
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

from .card_types import PokemonCardType, PokemonRarity, PokemonCardEra
from .template_processor import CardTemplate, CardPartOutline, CardPartType
from .visual_analyzer import PokemonDefectAnalysis
from .tcg_data_integration import TCGCardData

logger = logging.getLogger(__name__)


@dataclass
class GradingReportData:
    """Complete data for generating a grading report."""
    # Card identification
    card_name: str
    set_name: str
    card_number: str
    rarity: PokemonRarity
    era: PokemonCardEra
    
    # Analysis results
    template: CardTemplate
    defect_analysis: PokemonDefectAnalysis
    
    # Grading scores
    overall_grade: float
    centering_score: float
    surface_score: float
    edges_score: float
    corners_score: float
    
    # Grade explanation
    grade_reasoning: str
    improvement_suggestions: List[str]
    
    # Images (as numpy arrays)
    original_image: np.ndarray
    overlay_images: Dict[str, np.ndarray]
    
    # Metadata
    analysis_date: datetime
    processing_time: float
    
    # Optional fields with defaults
    tcg_data: Optional[TCGCardData] = None
    gemcheck_version: str = "1.0.0"


class PokemonCardPDFReportGenerator:
    """Generates comprehensive PDF reports for Pokémon card grading analysis."""
    
    def __init__(self):
        self.page_width, self.page_height = A4
        self.margin = 0.75 * inch
        self.content_width = self.page_width - 2 * self.margin
        self.content_height = self.page_height - 2 * self.margin
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
        # Grade color mapping
        self.grade_colors = {
            10: Color(0, 0.8, 0),      # Green
            9: Color(0.2, 0.8, 0.2),   # Light Green
            8: Color(0.5, 0.8, 0.2),   # Yellow-Green
            7: Color(0.8, 0.8, 0),     # Yellow
            6: Color(0.8, 0.6, 0),     # Orange
            5: Color(0.8, 0.4, 0),     # Dark Orange
            4: Color(0.8, 0.2, 0),     # Red-Orange
            3: Color(0.8, 0, 0),       # Red
            2: Color(0.6, 0, 0),       # Dark Red
            1: Color(0.4, 0, 0)        # Very Dark Red
        }

    def _create_custom_styles(self):
        """Create custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=Color(0.2, 0.2, 0.8)
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=Color(0.3, 0.3, 0.7)
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=Color(0.4, 0.4, 0.6)
        ))
        
        # Analysis text style
        self.styles.add(ParagraphStyle(
            name='AnalysisText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Grade text style
        self.styles.add(ParagraphStyle(
            name='GradeText',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=8,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

    def generate_report(self, report_data: GradingReportData, output_path: str) -> bool:
        """
        Generate a comprehensive PDF report for the card analysis.
        
        Args:
            report_data: Complete grading analysis data
            output_path: Path for the output PDF file
            
        Returns:
            True if report generation successful, False otherwise
        """
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )
            
            # Build report content
            story = []
            
            # Title page
            self._add_title_page(story, report_data)
            
            # Executive summary
            self._add_executive_summary(story, report_data)
            
            # Overall grading analysis
            self._add_overall_analysis(story, report_data)
            
            # Detailed part analysis
            self._add_part_by_part_analysis(story, report_data)
            
            # Visual overlays
            self._add_visual_overlays(story, report_data)
            
            # Defect analysis
            self._add_defect_analysis(story, report_data)
            
            # Market information (if available)
            if report_data.tcg_data:
                self._add_market_information(story, report_data)
            
            # Recommendations
            self._add_recommendations(story, report_data)
            
            # Technical appendix
            self._add_technical_appendix(story, report_data)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF report generation failed: {e}")
            return False

    def _add_title_page(self, story: List, report_data: GradingReportData):
        """Add title page with card information and overall grade."""
        # GemCheck logo and title
        story.append(Paragraph("GemCheck", self.styles['CustomTitle']))
        story.append(Paragraph("Professional Pokémon Card Grading Report", self.styles['Heading2']))
        story.append(Spacer(1, 0.5 * inch))
        
        # Card image (if available)
        if report_data.original_image is not None:
            card_image = self._numpy_to_reportlab_image(report_data.original_image, width=3*inch)
            story.append(card_image)
            story.append(Spacer(1, 0.3 * inch))
        
        # Card information table
        card_info = [
            ['Card Name:', report_data.card_name],
            ['Set:', report_data.set_name],
            ['Card Number:', report_data.card_number],
            ['Rarity:', report_data.rarity.value.replace('_', ' ').title()],
            ['Era:', report_data.era.value.replace('_', ' ').title()],
            ['Analysis Date:', report_data.analysis_date.strftime('%B %d, %Y')],
            ['Processing Time:', f"{report_data.processing_time:.2f} seconds"]
        ]
        
        info_table = Table(card_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 0.5 * inch))
        
        # Overall grade highlight
        grade_int = int(round(report_data.overall_grade))
        grade_color = self.grade_colors.get(grade_int, black)
        
        grade_text = f"Overall Grade: {report_data.overall_grade:.1f}/10"
        if grade_int == 10:
            grade_text += " (GEM MINT)"
        elif grade_int == 9:
            grade_text += " (MINT)"
        elif grade_int == 8:
            grade_text += " (NM-MT)"
        elif grade_int == 7:
            grade_text += " (NM)"
        elif grade_int >= 5:
            grade_text += " (EX)"
        else:
            grade_text += " (VG or below)"
        
        story.append(Paragraph(grade_text, self.styles['GradeText']))
        story.append(PageBreak())

    def _add_executive_summary(self, story: List, report_data: GradingReportData):
        """Add executive summary section."""
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Summary paragraph
        summary_text = f"""
        This report presents a comprehensive analysis of the {report_data.card_name} 
        from the {report_data.set_name} set. Using advanced computer vision and 
        machine learning techniques, GemCheck has evaluated this card across multiple 
        criteria including centering, surface condition, edge quality, and corner integrity.
        
        The card received an overall grade of {report_data.overall_grade:.1f}/10, 
        based on professional PSA grading standards. This analysis utilized 
        computational photography overlays and pixel-perfect measurements to 
        provide objective, reproducible results.
        """
        
        story.append(Paragraph(summary_text, self.styles['AnalysisText']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Score summary table
        scores = [
            ['Grading Category', 'Score', 'Weight', 'Weighted Score'],
            ['Centering', f"{report_data.centering_score:.1f}/100", '35%', f"{report_data.centering_score * 0.35:.1f}"],
            ['Surface', f"{report_data.surface_score:.1f}/100", '25%', f"{report_data.surface_score * 0.25:.1f}"],
            ['Edges', f"{report_data.edges_score:.1f}/100", '20%', f"{report_data.edges_score * 0.20:.1f}"],
            ['Corners', f"{report_data.corners_score:.1f}/100", '20%', f"{report_data.corners_score * 0.20:.1f}"],
        ]
        
        score_table = Table(scores, colWidths=[2*inch, 1*inch, 1*inch, 1.2*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, black),
        ]))
        
        story.append(score_table)
        story.append(Spacer(1, 0.3 * inch))

    def _add_overall_analysis(self, story: List, report_data: GradingReportData):
        """Add overall grading analysis section."""
        story.append(Paragraph("Overall Grading Analysis", self.styles['SectionHeader']))
        
        # Grade reasoning
        story.append(Paragraph("Grade Explanation", self.styles['SubsectionHeader']))
        story.append(Paragraph(report_data.grade_reasoning, self.styles['AnalysisText']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Key findings
        story.append(Paragraph("Key Findings", self.styles['SubsectionHeader']))
        
        findings = []
        
        # Centering analysis
        if report_data.centering_score >= 90:
            findings.append("✓ Excellent centering with minimal deviation from perfect alignment")
        elif report_data.centering_score >= 70:
            findings.append("• Good centering with minor alignment issues")
        else:
            findings.append("⚠ Centering issues detected - significant impact on grade")
        
        # Surface analysis
        if report_data.surface_score >= 90:
            findings.append("✓ Pristine surface condition with no visible defects")
        elif report_data.surface_score >= 70:
            findings.append("• Minor surface imperfections detected")
        else:
            findings.append("⚠ Significant surface defects affecting grade")
        
        # Edge analysis
        if report_data.edges_score >= 90:
            findings.append("✓ Sharp, clean edges with no whitening")
        elif report_data.edges_score >= 70:
            findings.append("• Minor edge wear or whitening present")
        else:
            findings.append("⚠ Noticeable edge wear impacting grade")
        
        # Corner analysis
        if report_data.corners_score >= 90:
            findings.append("✓ Sharp corners with no visible wear")
        elif report_data.corners_score >= 70:
            findings.append("• Minor corner softening detected")
        else:
            findings.append("⚠ Corner damage or rounding present")
        
        for finding in findings:
            story.append(Paragraph(finding, self.styles['AnalysisText']))
        
        story.append(Spacer(1, 0.3 * inch))

    def _add_part_by_part_analysis(self, story: List, report_data: GradingReportData):
        """Add detailed analysis for each card part."""
        story.append(Paragraph("Detailed Part-by-Part Analysis", self.styles['SectionHeader']))
        
        # Centering analysis
        self._add_centering_analysis(story, report_data)
        
        # Surface analysis
        self._add_surface_analysis(story, report_data)
        
        # Edge analysis
        self._add_edge_analysis(story, report_data)
        
        # Corner analysis
        self._add_corner_analysis(story, report_data)

    def _add_centering_analysis(self, story: List, report_data: GradingReportData):
        """Add detailed centering analysis."""
        story.append(Paragraph("Centering Analysis", self.styles['SubsectionHeader']))
        
        # Get centering data from template
        centering_part = report_data.template.parts.get(CardPartType.INNER_FRAME)
        
        if centering_part:
            # Calculate centering measurements
            x, y, w, h = centering_part.bounding_box
            card_w, card_h = report_data.template.width, report_data.template.height
            
            left_margin = x
            right_margin = card_w - (x + w)
            top_margin = y
            bottom_margin = card_h - (y + h)
            
            # Centering measurements table
            centering_data = [
                ['Measurement', 'Value (pixels)', 'Percentage'],
                ['Left Margin', f"{left_margin}", f"{left_margin/card_w*100:.1f}%"],
                ['Right Margin', f"{right_margin}", f"{right_margin/card_w*100:.1f}%"],
                ['Top Margin', f"{top_margin}", f"{top_margin/card_h*100:.1f}%"],
                ['Bottom Margin', f"{bottom_margin}", f"{bottom_margin/card_h*100:.1f}%"],
                ['Horizontal Error', '', f"{abs(left_margin-right_margin)/(left_margin+right_margin)*100:.2f}%"],
                ['Vertical Error', '', f"{abs(top_margin-bottom_margin)/(top_margin+bottom_margin)*100:.2f}%"]
            ]
            
            centering_table = Table(centering_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            centering_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(centering_table)
        
        # Centering analysis text
        analysis_text = f"""
        The centering analysis achieved a score of {report_data.centering_score:.1f}/100. 
        This evaluation uses computational photography to measure pixel-perfect margins 
        and calculate deviation from ideal centering. The analysis considers both 
        horizontal and vertical alignment according to PSA grading standards.
        """
        
        if report_data.centering_score >= 90:
            analysis_text += " The card demonstrates excellent centering with minimal deviation."
        elif report_data.centering_score >= 70:
            analysis_text += " The card shows good centering with minor alignment issues."
        else:
            analysis_text += " Significant centering issues were detected that impact the overall grade."
        
        story.append(Paragraph(analysis_text, self.styles['AnalysisText']))
        story.append(Spacer(1, 0.2 * inch))

    def _add_surface_analysis(self, story: List, report_data: GradingReportData):
        """Add detailed surface analysis."""
        story.append(Paragraph("Surface Condition Analysis", self.styles['SubsectionHeader']))
        
        # Surface defects summary
        defects = report_data.defect_analysis
        
        surface_data = [
            ['Defect Type', 'Count', 'Severity Impact'],
            ['Scratches', f"{len(defects.surface_scratches)}", self._get_severity_text(len(defects.surface_scratches), 'scratches')],
            ['Holo Scratches', f"{len(defects.holo_scratches)}", self._get_severity_text(len(defects.holo_scratches), 'holo_scratches')],
            ['Print Lines', f"{len(defects.print_lines)}", self._get_severity_text(len(defects.print_lines), 'print_lines')],
            ['Indentations', f"{len(defects.indentations)}", self._get_severity_text(len(defects.indentations), 'indentations')],
            ['Staining', f"{len(defects.staining)}", self._get_severity_text(len(defects.staining), 'staining')]
        ]
        
        surface_table = Table(surface_data, colWidths=[2*inch, 1*inch, 2*inch])
        surface_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(surface_table)
        
        # Surface analysis explanation
        analysis_text = f"""
        The surface condition analysis achieved a score of {report_data.surface_score:.1f}/100. 
        This evaluation examines the card surface for scratches, print defects, indentations, 
        and other imperfections that affect the card's appearance and value. Special attention 
        is paid to holographic areas where defects are more visible and impactful.
        """
        
        story.append(Paragraph(analysis_text, self.styles['AnalysisText']))
        story.append(Spacer(1, 0.2 * inch))

    def _add_edge_analysis(self, story: List, report_data: GradingReportData):
        """Add detailed edge analysis."""
        story.append(Paragraph("Edge Condition Analysis", self.styles['SubsectionHeader']))
        
        # Edge whitening data
        edge_data = report_data.defect_analysis.edge_whitening
        
        edge_info = [
            ['Edge', 'Whitening %', 'Condition'],
            ['Top', f"{edge_data.get('top', 0)*100:.1f}%", self._get_edge_condition(edge_data.get('top', 0))],
            ['Bottom', f"{edge_data.get('bottom', 0)*100:.1f}%", self._get_edge_condition(edge_data.get('bottom', 0))],
            ['Left', f"{edge_data.get('left', 0)*100:.1f}%", self._get_edge_condition(edge_data.get('left', 0))],
            ['Right', f"{edge_data.get('right', 0)*100:.1f}%", self._get_edge_condition(edge_data.get('right', 0))]
        ]
        
        edge_table = Table(edge_info, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        edge_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(edge_table)
        
        analysis_text = f"""
        The edge condition analysis achieved a score of {report_data.edges_score:.1f}/100. 
        This evaluation examines each edge for whitening, nicks, and wear patterns. 
        Edge whitening occurs when the card's white core becomes visible due to wear 
        or damage, significantly impacting the card's grade.
        """
        
        story.append(Paragraph(analysis_text, self.styles['AnalysisText']))
        story.append(Spacer(1, 0.2 * inch))

    def _add_corner_analysis(self, story: List, report_data: GradingReportData):
        """Add detailed corner analysis."""
        story.append(Paragraph("Corner Condition Analysis", self.styles['SubsectionHeader']))
        
        # Corner condition data
        corner_data = report_data.defect_analysis.corner_peeling
        
        corner_info = [
            ['Corner', 'Wear Level', 'Condition'],
            ['Top Left', f"{corner_data.get('top_left', 0)*100:.1f}%", self._get_corner_condition(corner_data.get('top_left', 0))],
            ['Top Right', f"{corner_data.get('top_right', 0)*100:.1f}%", self._get_corner_condition(corner_data.get('top_right', 0))],
            ['Bottom Left', f"{corner_data.get('bottom_left', 0)*100:.1f}%", self._get_corner_condition(corner_data.get('bottom_left', 0))],
            ['Bottom Right', f"{corner_data.get('bottom_right', 0)*100:.1f}%", self._get_corner_condition(corner_data.get('bottom_right', 0))]
        ]
        
        corner_table = Table(corner_info, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        corner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(corner_table)
        
        analysis_text = f"""
        The corner condition analysis achieved a score of {report_data.corners_score:.1f}/100. 
        This evaluation examines each corner for sharpness, wear, and potential damage. 
        Corner condition is critical for high-grade cards, as even minor rounding or 
        wear can significantly impact the overall grade.
        """
        
        story.append(Paragraph(analysis_text, self.styles['AnalysisText']))
        story.append(PageBreak())

    def _add_visual_overlays(self, story: List, report_data: GradingReportData):
        """Add visual overlay images."""
        story.append(Paragraph("Visual Analysis Overlays", self.styles['SectionHeader']))
        
        overlay_descriptions = {
            'centering': 'Centering analysis with margin measurements and alignment guides',
            'surface': 'Surface defect detection highlighting scratches and imperfections',
            'edges': 'Edge condition analysis showing whitening and wear patterns',
            'corners': 'Corner condition assessment with wear level indicators'
        }
        
        for overlay_type, description in overlay_descriptions.items():
            if overlay_type in report_data.overlay_images:
                story.append(Paragraph(f"{overlay_type.title()} Overlay", self.styles['SubsectionHeader']))
                story.append(Paragraph(description, self.styles['AnalysisText']))
                
                overlay_image = self._numpy_to_reportlab_image(
                    report_data.overlay_images[overlay_type], 
                    width=4*inch
                )
                story.append(overlay_image)
                story.append(Spacer(1, 0.2 * inch))

    def _add_defect_analysis(self, story: List, report_data: GradingReportData):
        """Add detailed defect analysis section."""
        story.append(Paragraph("Detailed Defect Analysis", self.styles['SectionHeader']))
        
        defects = report_data.defect_analysis
        
        # Holographic defects (if applicable)
        if defects.holo_scratches:
            story.append(Paragraph("Holographic Surface Defects", self.styles['SubsectionHeader']))
            
            holo_text = f"""
            {len(defects.holo_scratches)} holographic scratches were detected with 
            an overall holographic wear rating of {defects.holo_wear:.2f}. 
            Holographic surfaces are particularly susceptible to scratching and 
            require careful handling to maintain their pristine appearance.
            """
            
            story.append(Paragraph(holo_text, self.styles['AnalysisText']))
            story.append(Spacer(1, 0.1 * inch))
        
        # Print defects
        if defects.print_lines:
            story.append(Paragraph("Print Quality Issues", self.styles['SubsectionHeader']))
            
            print_text = f"""
            {len(defects.print_lines)} print lines were detected. Print lines are 
            horizontal defects that occur during the manufacturing process and are 
            not considered as severe as post-production damage. Color misalignment 
            factor: {defects.color_misalignment:.3f}.
            """
            
            story.append(Paragraph(print_text, self.styles['AnalysisText']))
            story.append(Spacer(1, 0.1 * inch))
        
        # Artwork and text quality
        story.append(Paragraph("Content Quality Assessment", self.styles['SubsectionHeader']))
        
        quality_text = f"""
        Artwork damage factor: {defects.artwork_damage:.2f}
        Text legibility score: {defects.text_legibility:.2f}
        Symbol clarity score: {defects.symbol_clarity:.2f}
        
        These metrics assess the clarity and integrity of the card's visual elements, 
        which are important for both aesthetic appeal and functional readability.
        """
        
        story.append(Paragraph(quality_text, self.styles['AnalysisText']))
        story.append(Spacer(1, 0.2 * inch))

    def _add_market_information(self, story: List, report_data: GradingReportData):
        """Add market information if TCG data is available."""
        story.append(Paragraph("Market Information", self.styles['SectionHeader']))
        
        tcg_data = report_data.tcg_data
        
        # Card details
        details_text = f"""
        Official Set: {tcg_data.set_name}
        Card Number: {tcg_data.number}/{tcg_data.set_total}
        Rarity: {tcg_data.rarity}
        Artist: {tcg_data.artist or 'Unknown'}
        Release Date: {tcg_data.release_date or 'Unknown'}
        """
        
        story.append(Paragraph("Card Details", self.styles['SubsectionHeader']))
        story.append(Paragraph(details_text, self.styles['AnalysisText']))
        
        # Market context
        market_text = """
        Market values can vary significantly based on card condition, rarity, and demand. 
        The grade assigned by this analysis provides an objective assessment that can 
        help determine the card's position within the market range for this specific card.
        """
        
        story.append(Paragraph("Market Context", self.styles['SubsectionHeader']))
        story.append(Paragraph(market_text, self.styles['AnalysisText']))
        story.append(Spacer(1, 0.2 * inch))

    def _add_recommendations(self, story: List, report_data: GradingReportData):
        """Add recommendations section."""
        story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        
        # Grade-specific recommendations
        grade = report_data.overall_grade
        
        if grade >= 9.5:
            rec_text = """
            Exceptional card quality! This card demonstrates museum-quality condition 
            and would be an excellent candidate for professional grading. Consider 
            submitting to PSA or BGS for authentication and encapsulation.
            """
        elif grade >= 8.5:
            rec_text = """
            Excellent card quality with minor imperfections. This card would likely 
            grade well with professional services and represents strong collectible value.
            """
        elif grade >= 7.0:
            rec_text = """
            Good card quality suitable for most collections. While not gem mint, 
            this card displays well and maintains solid collectible appeal.
            """
        else:
            rec_text = """
            The card shows significant wear or defects. Consider this for played 
            condition collections or as a placeholder until a higher grade example 
            can be obtained.
            """
        
        story.append(Paragraph(rec_text, self.styles['AnalysisText']))
        
        # Specific improvement suggestions
        if report_data.improvement_suggestions:
            story.append(Paragraph("Specific Observations", self.styles['SubsectionHeader']))
            
            for suggestion in report_data.improvement_suggestions:
                story.append(Paragraph(f"• {suggestion}", self.styles['AnalysisText']))
        
        story.append(Spacer(1, 0.2 * inch))

    def _add_technical_appendix(self, story: List, report_data: GradingReportData):
        """Add technical appendix with methodology."""
        story.append(PageBreak())
        story.append(Paragraph("Technical Appendix", self.styles['SectionHeader']))
        
        methodology_text = f"""
        This analysis was performed using GemCheck v{report_data.gemcheck_version}, 
        an advanced computer vision system designed specifically for Pokémon card 
        grading. The analysis process includes:
        
        1. Card Detection and Rectification: Automatic detection and perspective 
           correction of the card within the image.
        
        2. Template-Based Part Isolation: Each card component is identified and 
           isolated using machine learning-trained templates.
        
        3. Computational Photography Analysis: Pixel-level analysis of centering, 
           surface defects, edge condition, and corner integrity.
        
        4. PSA-Standard Scoring: Grades are calculated using established PSA 
           grading criteria with mathematical precision.
        
        Processing Statistics:
        • Template Quality: {report_data.template.template_quality:.2f}
        • Parts Detected: {len(report_data.template.parts)}
        • Analysis Confidence: {report_data.template.detection_confidence:.2f}
        • Processing Time: {report_data.processing_time:.2f} seconds
        
        This objective, reproducible analysis provides consistent grading results 
        that align with professional grading standards while offering detailed 
        insights into specific card condition factors.
        """
        
        story.append(Paragraph(methodology_text, self.styles['AnalysisText']))
        
        # Disclaimer
        disclaimer_text = """
        DISCLAIMER: This analysis is provided for informational purposes only. 
        While GemCheck uses professional-grade analysis techniques, official 
        grading should be obtained from recognized grading services such as PSA, 
        BGS, or CGC for authentication and market purposes. Results may vary 
        based on image quality, lighting conditions, and other factors.
        """
        
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("Disclaimer", self.styles['SubsectionHeader']))
        story.append(Paragraph(disclaimer_text, self.styles['AnalysisText']))

    def _numpy_to_reportlab_image(self, numpy_image: np.ndarray, width: float) -> Image:
        """Convert numpy image to ReportLab Image object."""
        try:
            # Convert numpy array to PIL Image
            if numpy_image.dtype != np.uint8:
                numpy_image = (numpy_image * 255).astype(np.uint8)
            
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(numpy_image)
            
            # Calculate height maintaining aspect ratio
            aspect_ratio = pil_image.height / pil_image.width
            height = width * aspect_ratio
            
            # Save to BytesIO
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Create ReportLab Image
            return Image(img_buffer, width=width, height=height)
            
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            # Return placeholder
            return Paragraph("Image could not be loaded", self.styles['Normal'])

    def _get_severity_text(self, count: int, defect_type: str) -> str:
        """Get severity description based on defect count and type."""
        if count == 0:
            return "None detected"
        elif count <= 2:
            return "Minor impact"
        elif count <= 5:
            return "Moderate impact"
        else:
            return "Significant impact"

    def _get_edge_condition(self, whitening_level: float) -> str:
        """Get edge condition description."""
        if whitening_level < 0.05:
            return "Excellent"
        elif whitening_level < 0.15:
            return "Good"
        elif whitening_level < 0.30:
            return "Fair"
        else:
            return "Poor"

    def _get_corner_condition(self, wear_level: float) -> str:
        """Get corner condition description."""
        if wear_level < 0.10:
            return "Sharp"
        elif wear_level < 0.25:
            return "Minor wear"
        elif wear_level < 0.50:
            return "Moderate wear"
        else:
            return "Significant wear"


def generate_pokemon_card_report(report_data: GradingReportData, output_path: str) -> bool:
    """
    Convenience function to generate a Pokémon card grading report.
    
    Args:
        report_data: Complete grading analysis data
        output_path: Path for the output PDF file
        
    Returns:
        True if successful, False otherwise
    """
    generator = PokemonCardPDFReportGenerator()
    return generator.generate_report(report_data, output_path)