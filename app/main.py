"""
Streamlit UI for PSA-style card grading application.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
from datetime import datetime
import logging
import time
from typing import Dict, Any, Optional, List

from app.disclaimer import show_disclaimer, show_main_disclaimer
from app.io_utils import safe_read_image, find_images_in_folder, validate_image_file
from app.preprocess import preprocess_card_image
from app.detect_card import detect_and_rectify_card
from app.metrics.centering import analyze_centering
from app.metrics.edges_corners import analyze_edges, analyze_corners
from app.metrics.surface import analyze_surface
from app.metrics.glare import analyze_glare
from app.scoring import calculate_sub_scores, calculate_overall_score, load_grade_mapping
from app.visualize import (
    create_centering_overlay, create_edge_overlay, create_corner_overlay,
    create_surface_overlay, create_glare_overlay
)
from app.report import ReportGenerator, generate_batch_report
from app.calibration import ThresholdCalibrator, run_calibration_workflow
from app.ml.optional_defect_model import create_defect_model
from app.schema import AppConfig, CardAnalysis, ScoreWeights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="PSA-Style Card Grader",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.stMetric .metric-container {
    background-color: #f0f2f6;
    border: 1px solid #d4e6f1;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.grade-excellent { background-color: #d4edda !important; }
.grade-good { background-color: #fff3cd !important; }
.grade-fair { background-color: #f8d7da !important; }
.grade-poor { background-color: #f8d7da !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config() -> AppConfig:
    """Load application configuration from files."""
    try:
        # Load default config
        config_data = {
            'weights': {'centering': 35.0, 'edges': 20.0, 'corners': 20.0, 'surface': 25.0},
            'centering_max_error': 0.25,
            'edge_whitening_threshold': 0.15,
            'corner_sharpness_threshold': 0.3,
            'surface_scratch_threshold': 0.02,
            'glare_threshold': 0.8,
            'ml_enabled': False
        }
        
        # Try to load from files
        config_files = {
            'weights': Path('config/weights.yaml'),
            'thresholds': Path('config/thresholds.yaml'),
            'grade_map': Path('config/grade_map.yaml')
        }
        
        for key, file_path in config_files.items():
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        data = yaml.safe_load(f)
                        config_data.update(data)
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        return AppConfig(**config_data)
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        return AppConfig()


@st.cache_resource
def load_ml_model(enabled: bool, model_path: Optional[str] = None):
    """Load ML model if enabled."""
    if not enabled:
        return None
    try:
        return create_defect_model("placeholder", model_path)
    except Exception as e:
        logger.error(f"ML model loading failed: {e}")
        return None


def analyze_single_card(image: np.ndarray, config: AppConfig, ml_model=None) -> CardAnalysis:
    """Analyze a single card image."""
    try:
        start_time = time.time()
        
        # Preprocessing
        st.text("🔄 Preprocessing image...")
        progress_bar = st.progress(0)
        
        preprocessing_config = {
            'max_size': config.max_image_size,
            'target_width': config.canonical_width,
            'target_height': config.canonical_height,
            'white_balance': True,
            'enhance_contrast': True,
            'remove_background': True
        }
        
        preprocess_results = preprocess_card_image(image, preprocessing_config)
        progress_bar.progress(20)
        
        rectified_image = preprocess_results.get('rectified_image')
        if rectified_image is None:
            st.error("Failed to rectify card image")
            return None
        
        # Analysis stages
        st.text("🔍 Analyzing centering...")
        centering_config = {
            'max_error_threshold': config.centering_max_error
        }
        centering_findings = analyze_centering(rectified_image, centering_config)
        progress_bar.progress(40)
        
        st.text("📏 Analyzing edges...")
        edge_config = {
            'whitening_threshold': config.edge_whitening_threshold
        }
        edge_findings = analyze_edges(rectified_image, edge_config)
        progress_bar.progress(55)
        
        st.text("📐 Analyzing corners...")
        corner_config = {
            'sharpness_threshold': config.corner_sharpness_threshold
        }
        corner_findings = analyze_corners(rectified_image, corner_config)
        progress_bar.progress(70)
        
        st.text("🔍 Analyzing surface...")
        surface_config = {
            'scratch_threshold': config.surface_scratch_threshold,
            'use_ml_assist': config.ml_enabled and ml_model is not None
        }
        surface_findings = analyze_surface(rectified_image, surface_config, ml_model)
        progress_bar.progress(85)
        
        st.text("✨ Analyzing glare...")
        glare_config = {
            'glare_threshold': config.glare_threshold
        }
        glare_findings = analyze_glare(rectified_image, glare_config)
        progress_bar.progress(95)
        
        # Calculate scores
        st.text("📊 Calculating scores...")
        sub_scores = calculate_sub_scores(
            centering_findings, edge_findings, corner_findings, surface_findings
        )
        
        weights = ScoreWeights(**config.weights.dict() if hasattr(config.weights, 'dict') else config.weights.__dict__)
        overall_score = calculate_overall_score(
            sub_scores, weights, glare_findings,
            centering_findings, edge_findings, corner_findings, surface_findings
        )
        
        # Create analysis object
        analysis = CardAnalysis(
            image_path="uploaded_image",
            timestamp=datetime.now().isoformat(),
            preprocessing_info=preprocess_results.get('processing_steps', []),
            centering_findings=centering_findings,
            edge_findings=edge_findings,
            corner_findings=corner_findings,
            surface_findings=surface_findings,
            glare_findings=glare_findings,
            overall_score=overall_score,
            processing_time_seconds=time.time() - start_time
        )
        
        progress_bar.progress(100)
        st.text("✅ Analysis complete!")
        time.sleep(0.5)  # Brief pause for user feedback
        progress_bar.empty()
        
        return analysis, rectified_image
        
    except Exception as e:
        logger.error(f"Card analysis failed: {e}")
        st.error(f"Analysis failed: {str(e)}")
        return None, None


def main():
    """Main Streamlit application."""
    
    # Load configuration
    config = load_config()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'rectified_image' not in st.session_state:
        st.session_state.rectified_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    
    # Title and main disclaimer
    st.title("🃏 PSA-Style Card Pre-Grader")
    show_main_disclaimer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    show_disclaimer()
    
    # File upload
    st.sidebar.subheader("📤 Upload Card")
    upload_mode = st.sidebar.radio("Upload Mode", ["Single Image", "Batch Folder"])
    
    uploaded_file = None
    batch_folder = None
    
    if upload_mode == "Single Image":
        uploaded_file = st.sidebar.file_uploader(
            "Choose card image", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        )
    else:
        batch_folder = st.sidebar.text_input("Folder Path (for batch processing)")
    
    # ML Model Toggle
    st.sidebar.subheader("🤖 ML Assistance")
    ml_enabled = st.sidebar.checkbox(
        "Enable ML surface defect detection (experimental)",
        value=config.ml_enabled
    )
    config.ml_enabled = ml_enabled
    
    # Load ML model if enabled
    ml_model = load_ml_model(ml_enabled)
    if ml_enabled and ml_model is None:
        st.sidebar.warning("ML model not available, falling back to classical methods")
    
    # Weight configuration
    st.sidebar.subheader("⚖️ Scoring Weights")
    with st.sidebar.expander("Adjust Weights"):
        centering_weight = st.slider("Centering", 0, 100, int(config.weights.centering))
        edges_weight = st.slider("Edges", 0, 100, int(config.weights.edges))
        corners_weight = st.slider("Corners", 0, 100, int(config.weights.corners))
        surface_weight = st.slider("Surface", 0, 100, int(config.weights.surface))
        
        total_weight = centering_weight + edges_weight + corners_weight + surface_weight
        if total_weight != 100:
            st.warning(f"Weights sum to {total_weight}%, they will be normalized to 100%")
        
        config.weights = ScoreWeights(
            centering=centering_weight,
            edges=edges_weight,
            corners=corners_weight,
            surface=surface_weight
        )
    
    # Threshold configuration
    st.sidebar.subheader("🎚️ Detection Thresholds")
    with st.sidebar.expander("Adjust Thresholds"):
        config.centering_max_error = st.slider(
            "Max Centering Error", 0.1, 1.0, config.centering_max_error, 0.05
        )
        config.edge_whitening_threshold = st.slider(
            "Edge Whitening Threshold", 0.05, 0.5, config.edge_whitening_threshold, 0.01
        )
        config.corner_sharpness_threshold = st.slider(
            "Corner Sharpness Threshold", 0.1, 1.0, config.corner_sharpness_threshold, 0.05
        )
        config.surface_scratch_threshold = st.slider(
            "Surface Scratch Threshold", 0.01, 0.1, config.surface_scratch_threshold, 0.005
        )
        config.glare_threshold = st.slider(
            "Glare Detection Threshold", 0.5, 1.0, config.glare_threshold, 0.05
        )
    
    # Main content area
    if uploaded_file is not None:
        # Process single image
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.session_state.original_image = image
        
        # Analyze button
        if st.button("🔍 Analyze Card", type="primary"):
            with st.spinner("Analyzing card..."):
                results = analyze_single_card(image, config, ml_model)
                if results[0] is not None:
                    st.session_state.analysis_results = results[0]
                    st.session_state.rectified_image = results[1]
                    st.success("Analysis completed!")
                else:
                    st.error("Analysis failed!")
    
    elif batch_folder and Path(batch_folder).exists():
        st.info(f"Batch mode: Found folder {batch_folder}")
        # Batch processing would be implemented here
        if st.button("🔄 Process Batch"):
            st.info("Batch processing not fully implemented in this demo")
    
    # Display results if available
    if st.session_state.analysis_results is not None:
        display_analysis_results()


def display_analysis_results():
    """Display the analysis results in tabs."""
    analysis = st.session_state.analysis_results
    rectified_image = st.session_state.rectified_image
    original_image = st.session_state.original_image
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview", "🎯 Centering", "📏 Edges", "📐 Corners", 
        "🔍 Surface", "✨ Glare", "⚙️ Config", "📁 Export"
    ])
    
    with tabs[0]:  # Overview
        display_overview_tab(analysis, original_image, rectified_image)
    
    with tabs[1]:  # Centering
        display_centering_tab(analysis, rectified_image)
    
    with tabs[2]:  # Edges
        display_edges_tab(analysis, rectified_image)
    
    with tabs[3]:  # Corners
        display_corners_tab(analysis, rectified_image)
    
    with tabs[4]:  # Surface
        display_surface_tab(analysis, rectified_image)
    
    with tabs[5]:  # Glare
        display_glare_tab(analysis, rectified_image)
    
    with tabs[6]:  # Config
        display_config_tab(analysis)
    
    with tabs[7]:  # Export
        display_export_tab(analysis)


def display_overview_tab(analysis: CardAnalysis, original_image: np.ndarray, rectified_image: np.ndarray):
    """Display overview tab with summary."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Images")
        if original_image is not None:
            st.image(original_image, caption="Original Image", use_column_width=True)
        if rectified_image is not None:
            st.image(rectified_image, caption="Rectified Image", use_column_width=True)
    
    with col2:
        st.subheader("🏆 Overall Grade")
        grade = analysis.overall_score.grade_label
        
        # Display grade with color coding
        grade_class = "grade-excellent"
        if grade.numeric >= 9:
            grade_class = "grade-excellent"
        elif grade.numeric >= 7:
            grade_class = "grade-good"
        elif grade.numeric >= 5:
            grade_class = "grade-fair"
        else:
            grade_class = "grade-poor"
        
        st.markdown(f"""
        <div class="{grade_class}" style="padding: 20px; border-radius: 10px; text-align: center;">
            <h1>{grade.numeric}</h1>
            <h3>{grade.label}</h3>
            <p>Final Score: {analysis.overall_score.final_score:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Description:** {grade.description}")
        
        # Sub-scores
        st.subheader("📊 Sub-Scores")
        
        scores = analysis.overall_score.sub_scores
        weights = analysis.overall_score.weights
        
        metrics_data = [
            ["Centering", scores.centering, weights.centering],
            ["Edges", scores.edges, weights.edges],
            ["Corners", scores.corners, weights.corners],
            ["Surface", scores.surface, weights.surface]
        ]
        
        for metric, score, weight in metrics_data:
            col_metric, col_score, col_weight = st.columns([2, 1, 1])
            with col_metric:
                st.write(f"**{metric}**")
            with col_score:
                st.metric("Score", f"{score:.1f}", delta=None)
            with col_weight:
                st.write(f"Weight: {weight:.0f}%")
        
        # Top issues
        if analysis.overall_score.top_issues:
            st.subheader("⚠️ Top Issues")
            for i, issue in enumerate(analysis.overall_score.top_issues, 1):
                st.write(f"{i}. {issue}")
        
        # Processing info
        st.subheader("ℹ️ Processing Info")
        st.write(f"**Processing time:** {analysis.processing_time_seconds:.2f} seconds")
        st.write(f"**Confidence:** {analysis.overall_score.confidence:.2f}")
        if analysis.overall_score.glare_penalty > 0:
            st.write(f"**Glare penalty:** -{analysis.overall_score.glare_penalty:.1f} points")


def display_centering_tab(analysis: CardAnalysis, rectified_image: np.ndarray):
    """Display centering analysis tab."""
    st.subheader("🎯 Centering Analysis")
    
    findings = analysis.centering_findings
    score = analysis.overall_score.sub_scores.centering
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Centering Score", f"{score:.1f}/100")
        
        st.subheader("📏 Margin Measurements")
        st.write(f"**Left margin:** {findings.left_margin_px:.0f} pixels")
        st.write(f"**Right margin:** {findings.right_margin_px:.0f} pixels")
        st.write(f"**Top margin:** {findings.top_margin_px:.0f} pixels")
        st.write(f"**Bottom margin:** {findings.bottom_margin_px:.0f} pixels")
        
        st.subheader("📐 Error Calculations")
        st.write(f"**Horizontal error:** {findings.horizontal_error:.4f}")
        st.write(f"**Vertical error:** {findings.vertical_error:.4f}")
        st.write(f"**Combined error:** {findings.combined_error:.4f}")
        st.write(f"**Max threshold:** {findings.max_error_threshold:.4f}")
        
        st.subheader("🔧 Detection Details")
        st.write(f"**Method used:** {findings.detection_method}")
        st.write(f"**Frame detected:** {'Yes' if findings.inner_frame_detected else 'No'}")
    
    with col2:
        from .visualize import create_centering_overlay
        overlay = create_centering_overlay(rectified_image, findings)
        st.image(overlay, caption="Centering Analysis Overlay", use_column_width=True)
        
        # Formula explanation
        st.subheader("🧮 Scoring Formula")
        st.code(f"""
        Combined Error = √(H²+ V²) / √2
        where:
        H = |left - right| / (left + right)
        V = |top - bottom| / (top + bottom)
        
        Score = max(0, 100 × (1 - Combined_Error / {findings.max_error_threshold}))
        
        Current calculation:
        H = {findings.horizontal_error:.4f}
        V = {findings.vertical_error:.4f}
        Combined = {findings.combined_error:.4f}
        Score = {score:.1f}
        """)


# Additional display functions would continue here...
# For brevity, I'm including the essential structure

if __name__ == "__main__":
    main()