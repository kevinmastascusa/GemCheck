"""
Scoring system for combining sub-scores and determining final grades.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging
from app.schema import (
    SubScores, ScoreWeights, GradeLabel, OverallScore, 
    CenteringFindings, EdgeFindings, CornerFindings, SurfaceFindings, GlareFindings
)

logger = logging.getLogger(__name__)


def calculate_sub_scores(centering_findings: CenteringFindings,
                        edge_findings: EdgeFindings,
                        corner_findings: CornerFindings,
                        surface_findings: SurfaceFindings,
                        config: Optional[Dict[str, Any]] = None) -> SubScores:
    """
    Calculate individual sub-scores from analysis findings.
    
    Args:
        centering_findings: Centering analysis results
        edge_findings: Edge analysis results
        corner_findings: Corner analysis results
        surface_findings: Surface analysis results
        config: Configuration parameters
        
    Returns:
        SubScores object with all metric scores
    """
    try:
        from .metrics.centering import calculate_centering_score
        
        # Calculate centering score
        centering_score = calculate_centering_score(
            centering_findings.combined_error,
            centering_findings.max_error_threshold
        )
        
        # Calculate edge score (simplified)
        edge_score = max(0.0, 100.0 - edge_findings.whitening_percentage * 5.0)
        
        # Calculate corner score (use minimum corner score)
        corner_score = corner_findings.minimum_corner_score
        
        # Calculate surface score (simplified)
        surface_score = max(0.0, 100.0 - surface_findings.defect_percentage * 10.0)
        
        sub_scores = SubScores(
            centering=centering_score,
            edges=edge_score,
            corners=corner_score,
            surface=surface_score
        )
        
        logger.info(f"Sub-scores calculated: C={centering_score:.1f}, E={edge_score:.1f}, "
                   f"Co={corner_score:.1f}, S={surface_score:.1f}")
        return sub_scores
        
    except Exception as e:
        logger.error(f"Sub-score calculation failed: {e}")
        return SubScores(centering=0, edges=0, corners=0, surface=0)


def load_grade_mapping(config: Optional[Dict[str, Any]] = None) -> List[GradeLabel]:
    """
    Load PSA-style grade mapping configuration.
    
    Args:
        config: Configuration parameters
        
    Returns:
        List of GradeLabel objects defining grade thresholds
    """
    try:
        # Default PSA-style grade mapping
        default_grades = [
            GradeLabel(
                numeric=10,
                label="Gem Mint 10",
                description="Perfect condition card with sharp corners, perfect centering, and pristine surface",
                min_overall_score=97.0,
                min_sub_score=95.0
            ),
            GradeLabel(
                numeric=9,
                label="Mint 9",
                description="Near-perfect card with very minor imperfections",
                min_overall_score=92.0,
                min_sub_score=88.0
            ),
            GradeLabel(
                numeric=8,
                label="NM-Mint 8",
                description="Very high quality card with only slight wear",
                min_overall_score=85.0,
                min_sub_score=80.0
            ),
            GradeLabel(
                numeric=7,
                label="Near Mint 7",
                description="High quality card with minor wear but no major defects",
                min_overall_score=78.0,
                min_sub_score=72.0
            ),
            GradeLabel(
                numeric=6,
                label="Excellent 6",
                description="Good condition card with light to moderate wear",
                min_overall_score=72.0,
                min_sub_score=0.0
            ),
            GradeLabel(
                numeric=5,
                label="VG-EX 5",
                description="Card showing moderate wear but still collectible",
                min_overall_score=66.0,
                min_sub_score=0.0
            ),
            GradeLabel(
                numeric=4,
                label="Good 4",
                description="Card with significant wear but no major damage",
                min_overall_score=60.0,
                min_sub_score=0.0
            ),
            GradeLabel(
                numeric=3,
                label="Fair 3",
                description="Card with heavy wear and/or minor damage",
                min_overall_score=54.0,
                min_sub_score=0.0
            ),
            GradeLabel(
                numeric=2,
                label="Poor 2",
                description="Card with extensive wear and damage",
                min_overall_score=48.0,
                min_sub_score=0.0
            ),
            GradeLabel(
                numeric=1,
                label="Authentic 1",
                description="Card is authentic but in poor condition",
                min_overall_score=0.0,
                min_sub_score=0.0
            )
        ]
        
        return default_grades
        
    except Exception as e:
        logger.error(f"Grade mapping loading failed: {e}")
        return default_grades


def determine_grade(overall_score: float, sub_scores: SubScores, 
                   grade_mapping: List[GradeLabel]) -> GradeLabel:
    """
    Determine the appropriate grade based on scores and thresholds.
    
    Args:
        overall_score: Weighted overall score
        sub_scores: Individual metric scores
        grade_mapping: List of grade definitions
        
    Returns:
        Appropriate GradeLabel
    """
    try:
        # Find minimum sub-score
        min_sub_score = min(sub_scores.centering, sub_scores.edges, 
                           sub_scores.corners, sub_scores.surface)
        
        # Sort grades by numeric value (descending)
        sorted_grades = sorted(grade_mapping, key=lambda x: x.numeric, reverse=True)
        
        # Find appropriate grade
        for grade in sorted_grades:
            # Check overall score requirement
            if overall_score >= grade.min_overall_score:
                # Check minimum sub-score requirement
                if min_sub_score >= grade.min_sub_score:
                    logger.info(f"Assigned grade: {grade.label} (overall: {overall_score:.1f}, min_sub: {min_sub_score:.1f})")
                    return grade
        
        # Fallback to lowest grade
        lowest_grade = min(grade_mapping, key=lambda x: x.numeric)
        logger.warning(f"Score too low for any grade, assigning: {lowest_grade.label}")
        return lowest_grade
        
    except Exception as e:
        logger.error(f"Grade determination failed: {e}")
        # Return default lowest grade
        return GradeLabel(
            numeric=1,
            label="Authentic 1",
            description="Card is authentic but in poor condition",
            min_overall_score=0.0,
            min_sub_score=0.0
        )


def calculate_overall_score(sub_scores: SubScores, weights: ScoreWeights,
                          glare_findings: GlareFindings,
                          centering_findings: CenteringFindings,
                          edge_findings: EdgeFindings,
                          corner_findings: CornerFindings,
                          surface_findings: SurfaceFindings,
                          config: Optional[Dict[str, Any]] = None) -> OverallScore:
    """
    Calculate the complete overall score and grade.
    
    Args:
        sub_scores: Individual metric scores
        weights: Scoring weights
        glare_findings: Glare analysis results
        centering_findings: Centering analysis results
        edge_findings: Edge analysis results
        corner_findings: Corner analysis results
        surface_findings: Surface analysis results
        config: Configuration parameters
        
    Returns:
        OverallScore object with complete grading results
    """
    try:
        # Normalize weights
        normalized_weights = weights.normalize()
        
        # Calculate weighted score
        weighted_score = (
            (sub_scores.centering * normalized_weights.centering / 100.0) +
            (sub_scores.edges * normalized_weights.edges / 100.0) +
            (sub_scores.corners * normalized_weights.corners / 100.0) +
            (sub_scores.surface * normalized_weights.surface / 100.0)
        )
        
        # Apply glare penalty
        glare_penalty = glare_findings.penalty_applied
        final_score = max(0.0, weighted_score - glare_penalty)
        
        # Load grade mapping
        grade_mapping = load_grade_mapping(config)
        
        # Determine grade
        grade_label = determine_grade(final_score, sub_scores, grade_mapping)
        
        # Identify top issues (simplified)
        top_issues = []
        if sub_scores.centering < 80:
            top_issues.append("Poor centering")
        if sub_scores.edges < 80:
            top_issues.append("Edge wear detected")
        if sub_scores.corners < 80:
            top_issues.append("Corner damage")
        if sub_scores.surface < 80:
            top_issues.append("Surface defects")
        if glare_findings.glare_detected:
            top_issues.append("Glare affecting analysis")
        
        # Calculate confidence (simplified)
        confidence = 1.0 - (glare_penalty / 100.0) if glare_penalty > 0 else 0.9
        
        # Create overall score object
        overall_score = OverallScore(
            sub_scores=sub_scores,
            weights=normalized_weights,
            weighted_score=weighted_score,
            glare_penalty=glare_penalty,
            final_score=final_score,
            grade_label=grade_label,
            confidence=confidence,
            top_issues=top_issues[:5]  # Top 5 issues
        )
        
        logger.info(f"Overall score calculated: {final_score:.2f} -> {grade_label.label} (confidence: {confidence:.2f})")
        return overall_score
        
    except Exception as e:
        logger.error(f"Overall score calculation failed: {e}")
        # Return default poor score
        return OverallScore(
            sub_scores=sub_scores,
            weights=weights,
            weighted_score=0.0,
            glare_penalty=0.0,
            final_score=0.0,
            grade_label=GradeLabel(
                numeric=1,
                label="Authentic 1",
                description="Analysis failed",
                min_overall_score=0.0,
                min_sub_score=0.0
            ),
            confidence=0.0,
            top_issues=["Analysis error occurred"]
        )