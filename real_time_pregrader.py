"""
GemCheck Real-time PSA Card Pre-grader with Live Camera Feed and Computational Photography Overlays
Revolutionary card authentication with professional-grade analysis
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, Any
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
from datetime import datetime

from app.detect_card import CardDetector, detect_and_rectify_card
from app.metrics.centering import analyze_centering
from app.visualize import create_centering_overlay
from app.scoring import calculate_sub_scores, calculate_overall_score, load_grade_mapping
from app.schema import ScoreWeights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimePSAPregrader:
    """GemCheck Real-time PSA card pre-grader with live camera feed and computational photography overlays."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GemCheck - Real-Time PSA Card Pre-Grader")
        self.root.geometry("1400x900")
        
        # Camera setup
        self.camera = None
        self.camera_active = False
        self.current_frame = None
        self.analysis_results = None
        
        # Analysis configuration
        self.config = {
            'centering_max_error': 0.25,
            'target_width': 750,
            'target_height': 1050,
            'analysis_interval': 0.5  # Analyze every 0.5 seconds
        }
        
        self.weights = ScoreWeights(
            centering=35.0,
            edges=20.0,
            corners=20.0,
            surface=25.0
        )
        
        # Threading
        self.camera_thread = None
        self.analysis_thread = None
        self.running = False
        
        # Last analysis time
        self.last_analysis_time = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title with branding
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="GemCheck", 
                               font=('Arial', 20, 'bold'), foreground='#2E8B57')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Real-Time PSA Card Pre-Grader", 
                                  font=('Arial', 14))
        subtitle_label.pack()
        
        # Camera control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Camera", 
                                      command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Camera", 
                                     command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.capture_button = ttk.Button(control_frame, text="Capture & Analyze", 
                                        command=self.capture_and_analyze, state=tk.DISABLED)
        self.capture_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Camera selection
        ttk.Label(control_frame, text="Camera:").pack(side=tk.LEFT, padx=(20, 5))
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var, 
                                   values=["0", "1", "2"], width=5)
        camera_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Analysis Configuration")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Centering threshold
        ttk.Label(config_frame, text="Centering Error Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=self.config['centering_max_error'])
        threshold_scale = ttk.Scale(config_frame, from_=0.1, to=0.5, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.threshold_label = ttk.Label(config_frame, text=f"{self.config['centering_max_error']:.2f}")
        self.threshold_label.grid(row=0, column=2, padx=5, pady=5)
        threshold_scale.configure(command=self.update_threshold)
        
        config_frame.columnconfigure(1, weight=1)
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame
        video_frame = ttk.LabelFrame(content_frame, text="Live Camera Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Camera not started", 
                                    background="black", foreground="white")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Analysis frame
        analysis_frame = ttk.LabelFrame(content_frame, text="Analysis Results")
        analysis_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        analysis_frame.configure(width=350)
        
        # Grade display
        self.grade_frame = ttk.Frame(analysis_frame)
        self.grade_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.grade_label = ttk.Label(self.grade_frame, text="Grade: --", 
                                    font=('Arial', 14, 'bold'))
        self.grade_label.pack()
        
        self.score_label = ttk.Label(self.grade_frame, text="Score: --/100")
        self.score_label.pack()
        
        # Centering details
        centering_frame = ttk.LabelFrame(analysis_frame, text="Centering Analysis")
        centering_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.centering_score_label = ttk.Label(centering_frame, text="Score: --/100")
        self.centering_score_label.pack(anchor=tk.W)
        
        self.h_error_label = ttk.Label(centering_frame, text="H Error: --")
        self.h_error_label.pack(anchor=tk.W)
        
        self.v_error_label = ttk.Label(centering_frame, text="V Error: --")
        self.v_error_label.pack(anchor=tk.W)
        
        self.combined_error_label = ttk.Label(centering_frame, text="Combined: --")
        self.combined_error_label.pack(anchor=tk.W)
        
        # Margin measurements
        margin_frame = ttk.LabelFrame(analysis_frame, text="Margin Measurements")
        margin_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.left_margin_label = ttk.Label(margin_frame, text="Left: -- px")
        self.left_margin_label.pack(anchor=tk.W)
        
        self.right_margin_label = ttk.Label(margin_frame, text="Right: -- px")
        self.right_margin_label.pack(anchor=tk.W)
        
        self.top_margin_label = ttk.Label(margin_frame, text="Top: -- px")
        self.top_margin_label.pack(anchor=tk.W)
        
        self.bottom_margin_label = ttk.Label(margin_frame, text="Bottom: -- px")
        self.bottom_margin_label.pack(anchor=tk.W)
        
        # Status frame
        status_frame = ttk.Frame(analysis_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready")
        self.status_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: --")
        self.fps_label.pack(anchor=tk.W)
        
    def update_threshold(self, value):
        """Update centering threshold."""
        self.config['centering_max_error'] = float(value)
        self.threshold_label.configure(text=f"{float(value):.2f}")
        
    def start_camera(self):
        """Start the camera feed."""
        try:
            camera_index = int(self.camera_var.get())
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_index}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_active = True
            self.running = True
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=self.analysis_loop)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
            # Update UI
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.capture_button.configure(state=tk.NORMAL)
            self.status_label.configure(text="Status: Camera Active")
            
            logger.info(f"Camera {camera_index} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            self.status_label.configure(text=f"Error: {e}")
            
    def stop_camera(self):
        """Stop the camera feed."""
        self.running = False
        self.camera_active = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
            
        # Update UI
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.capture_button.configure(state=tk.DISABLED)
        self.video_label.configure(image="", text="Camera stopped")
        self.status_label.configure(text="Status: Stopped")
        
        logger.info("Camera stopped")
        
    def camera_loop(self):
        """Main camera loop."""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running and self.camera_active:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                    
                # Store current frame for analysis
                self.current_frame = frame.copy()
                
                # Convert frame for display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                display_height = 480
                display_width = int(display_height * frame.shape[1] / frame.shape[0])
                display_frame = cv2.resize(display_frame, (display_width, display_height))
                
                # Convert to PhotoImage
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image)
                
                # Update video label
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo  # Keep a reference
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_start_time)
                    self.fps_label.configure(text=f"FPS: {fps:.1f}")
                    fps_start_time = current_time
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Camera loop error: {e}")
                break
                
    def analysis_loop(self):
        """Continuous analysis loop."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for analysis
                if (current_time - self.last_analysis_time > self.config['analysis_interval'] and 
                    self.current_frame is not None):
                    
                    self.analyze_current_frame()
                    self.last_analysis_time = current_time
                    
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                time.sleep(1)  # Wait before retrying
                
    def analyze_current_frame(self):
        """Analyze the current camera frame."""
        if self.current_frame is None:
            return
            
        try:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            # Detect and rectify card
            detector = CardDetector(
                target_width=self.config['target_width'],
                target_height=self.config['target_height']
            )
            
            card_bounds = detector.detect_card(frame_rgb)
            if card_bounds is None:
                self.update_analysis_ui(None)
                return
                
            rectified = detector.rectify_card(frame_rgb, card_bounds)
            if rectified is None:
                self.update_analysis_ui(None)
                return
                
            # Analyze centering
            centering_config = {
                'max_error_threshold': self.config['centering_max_error']
            }
            centering_findings = analyze_centering(rectified, centering_config)
            
            # Create overlay
            overlay = create_centering_overlay(rectified, centering_findings)
            
            # Display overlay in video feed
            self.display_overlay(overlay)
            
            # Update analysis results
            self.analysis_results = {
                'centering': centering_findings,
                'rectified': rectified,
                'overlay': overlay
            }
            
            self.update_analysis_ui(self.analysis_results)
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            self.update_analysis_ui(None)
            
    def display_overlay(self, overlay):
        """Display the analysis overlay in the video feed."""
        try:
            # Resize overlay for display
            display_height = 480
            display_width = int(display_height * overlay.shape[1] / overlay.shape[0])
            display_overlay = cv2.resize(overlay, (display_width, display_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(display_overlay)
            photo = ImageTk.PhotoImage(image)
            
            # Update video label
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            logger.error(f"Overlay display failed: {e}")
            
    def update_analysis_ui(self, results):
        """Update the analysis UI with results."""
        if results is None:
            # Clear results
            self.grade_label.configure(text="Grade: No Card Detected")
            self.score_label.configure(text="Score: --/100")
            self.centering_score_label.configure(text="Score: --/100")
            self.h_error_label.configure(text="H Error: --")
            self.v_error_label.configure(text="V Error: --")
            self.combined_error_label.configure(text="Combined: --")
            self.left_margin_label.configure(text="Left: -- px")
            self.right_margin_label.configure(text="Right: -- px")
            self.top_margin_label.configure(text="Top: -- px")
            self.bottom_margin_label.configure(text="Bottom: -- px")
            return
            
        try:
            centering = results['centering']
            
            # Calculate overall score (simplified for real-time)
            centering_score = centering.centering_score
            overall_score = centering_score  # Simplified - only centering for real-time
            
            # Determine grade
            if overall_score >= 90:
                grade = "10 (GEM MINT)"
            elif overall_score >= 85:
                grade = "9 (MINT)"
            elif overall_score >= 80:
                grade = "8 (NM-MT)"
            elif overall_score >= 70:
                grade = "7 (NM)"
            elif overall_score >= 60:
                grade = "6 (EX-MT)"
            elif overall_score >= 50:
                grade = "5 (EX)"
            else:
                grade = "4 (VG-EX) or lower"
                
            # Update UI
            self.grade_label.configure(text=f"Grade: {grade}")
            self.score_label.configure(text=f"Score: {overall_score:.1f}/100")
            
            self.centering_score_label.configure(text=f"Score: {centering.centering_score:.1f}/100")
            self.h_error_label.configure(text=f"H Error: {centering.horizontal_error:.4f}")
            self.v_error_label.configure(text=f"V Error: {centering.vertical_error:.4f}")
            self.combined_error_label.configure(text=f"Combined: {centering.combined_error:.4f}")
            
            self.left_margin_label.configure(text=f"Left: {centering.left_margin_px:.0f} px")
            self.right_margin_label.configure(text=f"Right: {centering.right_margin_px:.0f} px")
            self.top_margin_label.configure(text=f"Top: {centering.top_margin_px:.0f} px")
            self.bottom_margin_label.configure(text=f"Bottom: {centering.bottom_margin_px:.0f} px")
            
        except Exception as e:
            logger.error(f"UI update failed: {e}")
            
    def capture_and_analyze(self):
        """Capture current frame and perform detailed analysis."""
        if self.current_frame is None:
            return
            
        try:
            # Save captured frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_card_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            
            # Save analysis results if available
            if self.analysis_results:
                # Save overlay
                overlay_filename = f"analysis_overlay_{timestamp}.jpg"
                overlay_bgr = cv2.cvtColor(self.analysis_results['overlay'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(overlay_filename, overlay_bgr)
                
                # Save analysis data
                data_filename = f"analysis_data_{timestamp}.json"
                centering = self.analysis_results['centering']
                data = {
                    'timestamp': timestamp,
                    'centering_score': centering.centering_score,
                    'horizontal_error': centering.horizontal_error,
                    'vertical_error': centering.vertical_error,
                    'combined_error': centering.combined_error,
                    'margins': {
                        'left': centering.left_margin_px,
                        'right': centering.right_margin_px,
                        'top': centering.top_margin_px,
                        'bottom': centering.bottom_margin_px
                    }
                }
                
                with open(data_filename, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                self.status_label.configure(text=f"Captured: {filename}")
                logger.info(f"Captured and analyzed: {filename}")
            else:
                self.status_label.configure(text=f"Captured: {filename} (no analysis)")
                
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            self.status_label.configure(text=f"Capture error: {e}")
            
    def run(self):
        """Run the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle application closing."""
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    app = RealTimePSAPregrader()
    app.run()