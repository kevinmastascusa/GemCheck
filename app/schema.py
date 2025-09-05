"""
Data models for the PSA-style card grading application.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BoundingBox(BaseModel):
    """Represents a bounding box with coordinates."""
    x: int
    y: int
    width: int
    height: int


class DefectRegion(BaseModel):
    """Represents a detected defect region."""
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    defect_type: str
    severity: float = Field(ge=0.0, le=1.0)
    area_pixels: int


class CenteringFindings(BaseModel):
    """Detailed findings for centering analysis."""
    left_margin_px: float
    right_margin_px: float
    top_margin_px: float
    bottom_margin_px: float
    horizontal_error: float = Field(ge=0.0, le=1.0)
    vertical_error: float = Field(ge=0.0, le=1.0)
    combined_error: float = Field(ge=0.0, le=1.0)
    max_error_threshold: float
    inner_frame_detected: bool
    detection_method: str  # "edge_based" or "color_based"
    centering_score: float = Field(ge=0.0, le=100.0)


class EdgeFindings(BaseModel):
    """Detailed findings for edge condition analysis."""
    total_perimeter_px: int
    whitened_perimeter_px: int
    whitening_percentage: float = Field(ge=0.0, le=100.0)
    nick_count: int
    largest_nick_area_px: int
    clean_edge_percentage: float = Field(ge=0.0, le=100.0)
    whitening_threshold: float
    edge_score: float = Field(ge=0.0, le=100.0)


class CornerFindings(BaseModel):
    """Detailed findings for corner condition analysis."""
    corner_scores: Dict[str, float] = Field(default_factory=dict)  # "top_left", "top_right", etc.
    corner_sharpness: Dict[str, float] = Field(default_factory=dict)
    corner_whitening: Dict[str, float] = Field(default_factory=dict)
    corner_damage_area: Dict[str, int] = Field(default_factory=dict)
    minimum_corner_score: float = Field(ge=0.0, le=100.0)
    sharpness_threshold: float
    corner_score: float = Field(ge=0.0, le=100.0)


class SurfaceFindings(BaseModel):
    """Detailed findings for surface condition analysis."""
    total_area_px: int
    defect_area_px: int
    defect_percentage: float = Field(ge=0.0, le=100.0)
    scratch_count: int
    print_line_count: int
    stain_count: int
    defect_regions: List[DefectRegion] = Field(default_factory=list)
    ml_assist_used: bool = False
    ml_confidence: Optional[float] = None
    surface_quality_score: float = Field(ge=0.0, le=1.0)


class GlareFindings(BaseModel):
    """Detailed findings for glare/reflection analysis."""
    glare_detected: bool
    glare_area_px: int
    glare_percentage: float = Field(ge=0.0, le=100.0)
    penalty_applied: float = Field(ge=0.0, le=100.0)
    affected_regions: List[BoundingBox] = Field(default_factory=list)
    glare_threshold: float


class SubScores(BaseModel):
    """Individual metric scores."""
    centering: float = Field(ge=0.0, le=100.0)
    edges: float = Field(ge=0.0, le=100.0)
    corners: float = Field(ge=0.0, le=100.0)
    surface: float = Field(ge=0.0, le=100.0)


class ScoreWeights(BaseModel):
    """Weights for combining sub-scores."""
    centering: float = Field(default=35.0, ge=0.0, le=100.0)
    edges: float = Field(default=20.0, ge=0.0, le=100.0)
    corners: float = Field(default=20.0, ge=0.0, le=100.0)
    surface: float = Field(default=25.0, ge=0.0, le=100.0)
    
    def normalize(self) -> 'ScoreWeights':
        """Normalize weights to sum to 100."""
        total = self.centering + self.edges + self.corners + self.surface
        if total == 0:
            return ScoreWeights()
        factor = 100.0 / total
        return ScoreWeights(
            centering=self.centering * factor,
            edges=self.edges * factor,
            corners=self.corners * factor,
            surface=self.surface * factor
        )


class GradeLabel(BaseModel):
    """PSA-style grade label."""
    numeric: int = Field(ge=1, le=10)
    label: str
    description: str
    min_overall_score: float
    min_sub_score: float


class OverallScore(BaseModel):
    """Final grading result."""
    sub_scores: SubScores
    weights: ScoreWeights
    weighted_score: float = Field(ge=0.0, le=100.0)
    glare_penalty: float = Field(ge=0.0, le=100.0)
    final_score: float = Field(ge=0.0, le=100.0)
    grade_label: GradeLabel
    confidence: float = Field(ge=0.0, le=1.0)
    top_issues: List[str] = Field(default_factory=list)


class CardAnalysis(BaseModel):
    """Complete analysis results for a card."""
    image_path: str
    timestamp: str
    preprocessing_info: Dict[str, Any] = Field(default_factory=dict)
    centering_findings: CenteringFindings
    edge_findings: EdgeFindings
    corner_findings: CornerFindings
    surface_findings: SurfaceFindings
    glare_findings: GlareFindings
    overall_score: OverallScore
    processing_time_seconds: float
    config_version: str = "1.0"


# Aliases for compatibility
CardAnalysisResult = CardAnalysis

class GradingResult(BaseModel):
    """Final grading result with grade and confidence."""
    numeric: int = Field(ge=1, le=10)
    label: str
    overall_score: float
    confidence: float
    sub_scores: Dict[str, Dict[str, float]]
    description: Optional[str] = None

class CalibrationReference(BaseModel):
    """Reference card for calibration."""
    image_path: str
    labels: Dict[str, str] = Field(default_factory=dict)  # e.g., {"centering": "good", "surface": "poor"}
    known_grade: Optional[int] = None


class AppConfig(BaseModel):
    """Application configuration."""
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    centering_max_error: float = Field(default=0.25)
    edge_whitening_threshold: float = Field(default=0.15)
    corner_sharpness_threshold: float = Field(default=0.3)
    surface_scratch_threshold: float = Field(default=0.02)
    glare_threshold: float = Field(default=0.8)
    ml_enabled: bool = Field(default=False)
    batch_mode: bool = Field(default=False)
    export_overlays: bool = Field(default=True)
    canonical_width: int = Field(default=750)
    canonical_height: int = Field(default=1050)
    max_image_size: int = Field(default=2000)