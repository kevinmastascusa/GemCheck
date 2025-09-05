"""
Calibration system for auto-tuning detection thresholds and parameters.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import yaml
from dataclasses import dataclass
from scipy import optimize

from app.schema import CardAnalysisResult
from app.metrics.centering import analyze_centering
from app.metrics.edges_corners import analyze_edges, analyze_corners  
from app.metrics.surface import analyze_surface
from app.metrics.glare import analyze_glare

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTarget:
    """
    Represents a calibration target with known ground truth.
    """
    image_path: str
    expected_grade: int
    expected_centering_score: Optional[float] = None
    expected_edge_score: Optional[float] = None
    expected_corner_score: Optional[float] = None
    expected_surface_score: Optional[float] = None
    description: Optional[str] = None


@dataclass
class CalibrationResult:
    """
    Results of calibration optimization.
    """
    optimized_params: Dict[str, Any]
    original_params: Dict[str, Any]
    improvement_score: float
    target_accuracy: float
    iterations_used: int


class ThresholdCalibrator:
    """
    Automatically calibrates detection thresholds using known reference cards.
    """
    
    def __init__(self, config_path: str = "config/thresholds.yaml"):
        self.config_path = config_path
        self.current_params = self._load_config()
        self.original_params = self.current_params.copy()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load current threshold configuration."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters if config loading fails."""
        return {
            'centering': {
                'max_error_threshold': 0.25
            },
            'edges': {
                'border_width': 6,
                'whitening_threshold': 0.15,
                'acceptable_whitening_percent': 15.0
            },
            'corners': {
                'corner_size': 50,
                'sharpness_threshold': 0.3,
                'whitening_threshold': 0.15
            },
            'surface': {
                'scratch_threshold': 0.02,
                'min_defect_area': 5,
                'print_line_min_length': 30
            },
            'glare': {
                'highlight_threshold': 0.8,
                'min_region_area': 100,
                'max_penalty': 10.0
            }
        }
    
    def calibrate_from_targets(self, targets: List[CalibrationTarget]) -> CalibrationResult:
        """
        Calibrate thresholds using reference cards with known grades.
        
        Args:
            targets: List of calibration targets with ground truth
            
        Returns:
            Calibration results with optimized parameters
        """
        try:
            logger.info(f"Starting calibration with {len(targets)} targets")
            
            # Validate targets
            valid_targets = self._validate_targets(targets)
            if not valid_targets:
                raise ValueError("No valid calibration targets provided")
            
            # Define parameter bounds for optimization
            param_bounds = self._get_parameter_bounds()
            
            # Initial parameter vector
            x0 = self._params_to_vector(self.current_params)
            
            # Define objective function
            def objective(x):
                params = self._vector_to_params(x)
                return self._calculate_target_error(params, valid_targets)
            
            # Run optimization
            result = optimize.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 100, 'disp': True}
            )
            
            # Extract optimized parameters
            optimized_params = self._vector_to_params(result.x)
            
            # Calculate improvement
            original_error = objective(x0)
            optimized_error = result.fun
            improvement = max(0, (original_error - optimized_error) / original_error)
            
            # Calculate target accuracy
            accuracy = self._calculate_accuracy(optimized_params, valid_targets)
            
            calibration_result = CalibrationResult(
                optimized_params=optimized_params,
                original_params=self.original_params,
                improvement_score=improvement,
                target_accuracy=accuracy,
                iterations_used=result.nit
            )
            
            logger.info(f"Calibration completed: {improvement:.2%} improvement, "
                       f"{accuracy:.2%} target accuracy")
            
            return calibration_result
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise
    
    def _validate_targets(self, targets: List[CalibrationTarget]) -> List[CalibrationTarget]:
        """Validate calibration targets and filter invalid ones."""
        valid_targets = []
        
        for target in targets:
            try:
                # Check if image exists
                if not Path(target.image_path).exists():
                    logger.warning(f"Image not found: {target.image_path}")
                    continue
                
                # Check if image can be loaded
                image = cv2.imread(target.image_path)
                if image is None:
                    logger.warning(f"Cannot load image: {target.image_path}")
                    continue
                
                # Check if expected grade is valid
                if not (1 <= target.expected_grade <= 10):
                    logger.warning(f"Invalid grade {target.expected_grade}: {target.image_path}")
                    continue
                
                valid_targets.append(target)
                
            except Exception as e:
                logger.warning(f"Target validation failed for {target.image_path}: {e}")
                continue
        
        logger.info(f"Validated {len(valid_targets)}/{len(targets)} targets")
        return valid_targets
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Define bounds for optimization parameters."""
        return [
            (0.1, 0.5),    # centering.max_error_threshold
            (0.05, 0.3),   # edges.whitening_threshold
            (5.0, 30.0),   # edges.acceptable_whitening_percent
            (0.1, 0.5),    # corners.sharpness_threshold
            (0.05, 0.3),   # corners.whitening_threshold
            (0.005, 0.05), # surface.scratch_threshold
            (0.6, 0.95),   # glare.highlight_threshold
            (5.0, 20.0)    # glare.max_penalty
        ]
    
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to optimization vector."""
        return np.array([
            params['centering']['max_error_threshold'],
            params['edges']['whitening_threshold'],
            params['edges']['acceptable_whitening_percent'],
            params['corners']['sharpness_threshold'],
            params['corners']['whitening_threshold'],
            params['surface']['scratch_threshold'],
            params['glare']['highlight_threshold'],
            params['glare']['max_penalty']
        ])
    
    def _vector_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert optimization vector back to parameter dictionary."""
        params = self.current_params.copy()
        
        params['centering']['max_error_threshold'] = float(x[0])
        params['edges']['whitening_threshold'] = float(x[1])
        params['edges']['acceptable_whitening_percent'] = float(x[2])
        params['corners']['sharpness_threshold'] = float(x[3])
        params['corners']['whitening_threshold'] = float(x[4])
        params['surface']['scratch_threshold'] = float(x[5])
        params['glare']['highlight_threshold'] = float(x[6])
        params['glare']['max_penalty'] = float(x[7])
        
        return params
    
    def _calculate_target_error(self, params: Dict[str, Any], 
                              targets: List[CalibrationTarget]) -> float:
        """
        Calculate error between predicted and expected grades for targets.
        
        Args:
            params: Current parameter set
            targets: List of calibration targets
            
        Returns:
            Mean squared error across all targets
        """
        try:
            errors = []
            
            for target in targets:
                # Load and analyze image with current parameters
                image = cv2.imread(target.image_path)
                if image is None:
                    continue
                
                # Run analysis with current parameters
                predicted_scores = self._analyze_with_params(image, params)
                
                # Calculate overall predicted grade (simplified)
                predicted_grade = self._scores_to_grade(predicted_scores)
                
                # Calculate grade error
                grade_error = (predicted_grade - target.expected_grade) ** 2
                errors.append(grade_error)
                
                # Add sub-score errors if available
                if target.expected_centering_score is not None:
                    centering_error = (predicted_scores['centering'] - 
                                     target.expected_centering_score) ** 2
                    errors.append(centering_error * 0.1)  # Weight sub-score errors less
                
                if target.expected_edge_score is not None:
                    edge_error = (predicted_scores['edges'] - 
                                target.expected_edge_score) ** 2
                    errors.append(edge_error * 0.1)
            
            return np.mean(errors) if errors else float('inf')
            
        except Exception as e:
            logger.error(f"Error calculation failed: {e}")
            return float('inf')
    
    def _analyze_with_params(self, image: np.ndarray, 
                           params: Dict[str, Any]) -> Dict[str, float]:
        """Run analysis with specified parameters."""
        try:
            scores = {}
            
            # Centering analysis
            centering_result = analyze_centering(
                image, 
                max_error_threshold=params['centering']['max_error_threshold']
            )
            scores['centering'] = centering_result.centering_score
            
            # Edges analysis
            edge_result = analyze_edges(image, {
                'border_width': params['edges']['border_width'],
                'whitening_threshold': params['edges']['whitening_threshold']
            })
            
            # Corners analysis
            corner_result = analyze_corners(image, {
                'corner_size': params['corners']['corner_size'],
                'sharpness_threshold': params['corners']['sharpness_threshold']
            })
            scores['edges'] = edge_result.edge_score
            scores['corners'] = corner_result.corner_score
            
            # Surface analysis
            surface_result = analyze_surface(
                image,
                scratch_threshold=params['surface']['scratch_threshold'],
                min_defect_area=params['surface']['min_defect_area']
            )
            scores['surface'] = surface_result.surface_quality_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Analysis with parameters failed: {e}")
            return {'centering': 50.0, 'edges': 50.0, 'corners': 50.0, 'surface': 50.0}
    
    def _scores_to_grade(self, scores: Dict[str, float]) -> float:
        """Convert sub-scores to overall grade estimate."""
        # Simplified grade calculation (using default weights)
        weights = {'centering': 0.35, 'edges': 0.20, 'corners': 0.20, 'surface': 0.25}
        
        overall_score = sum(scores[metric] * weight 
                          for metric, weight in weights.items() 
                          if metric in scores)
        
        # Convert to grade scale (simplified mapping)
        if overall_score >= 97:
            return 10.0
        elif overall_score >= 92:
            return 9.0
        elif overall_score >= 85:
            return 8.0
        elif overall_score >= 78:
            return 7.0
        elif overall_score >= 72:
            return 6.0
        elif overall_score >= 66:
            return 5.0
        elif overall_score >= 60:
            return 4.0
        elif overall_score >= 54:
            return 3.0
        elif overall_score >= 48:
            return 2.0
        else:
            return 1.0
    
    def _calculate_accuracy(self, params: Dict[str, Any], 
                          targets: List[CalibrationTarget]) -> float:
        """Calculate accuracy of grade predictions with given parameters."""
        try:
            correct_predictions = 0
            total_predictions = 0
            
            for target in targets:
                image = cv2.imread(target.image_path)
                if image is None:
                    continue
                
                predicted_scores = self._analyze_with_params(image, params)
                predicted_grade = self._scores_to_grade(predicted_scores)
                
                # Allow ±0.5 grade tolerance
                if abs(predicted_grade - target.expected_grade) <= 0.5:
                    correct_predictions += 1
                
                total_predictions += 1
            
            return correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 0.0
    
    def save_optimized_config(self, result: CalibrationResult, 
                            output_path: Optional[str] = None) -> bool:
        """
        Save optimized parameters to configuration file.
        
        Args:
            result: Calibration results
            output_path: Optional custom output path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = output_path or self.config_path
            
            # Create backup of original config
            backup_path = str(output_file).replace('.yaml', '_backup.yaml')
            with open(self.config_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            
            # Save optimized config
            with open(output_file, 'w') as file:
                yaml.dump(result.optimized_params, file, default_flow_style=False, indent=2)
            
            logger.info(f"Optimized config saved to {output_file}")
            logger.info(f"Original config backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save optimized config: {e}")
            return False


def create_calibration_targets_from_known_cards() -> List[CalibrationTarget]:
    """
    Create calibration targets for common reference cards.
    This is a placeholder - in practice, you'd have actual reference images.
    
    Returns:
        List of calibration targets
    """
    targets = [
        CalibrationTarget(
            image_path="examples/reference_cards/gem_mint_10.jpg",
            expected_grade=10,
            expected_centering_score=98.0,
            expected_edge_score=98.0,
            expected_corner_score=98.0,
            expected_surface_score=98.0,
            description="Perfect condition reference card"
        ),
        CalibrationTarget(
            image_path="examples/reference_cards/mint_9.jpg",
            expected_grade=9,
            expected_centering_score=94.0,
            expected_edge_score=92.0,
            expected_corner_score=94.0,
            expected_surface_score=95.0,
            description="Mint condition with minor centering issue"
        ),
        CalibrationTarget(
            image_path="examples/reference_cards/nm_mint_8.jpg",
            expected_grade=8,
            expected_centering_score=88.0,
            expected_edge_score=85.0,
            expected_corner_score=87.0,
            expected_surface_score=90.0,
            description="Near mint with light wear"
        ),
        CalibrationTarget(
            image_path="examples/reference_cards/near_mint_7.jpg",
            expected_grade=7,
            expected_centering_score=82.0,
            expected_edge_score=78.0,
            expected_corner_score=80.0,
            expected_surface_score=85.0,
            description="Near mint with visible wear"
        )
    ]
    
    return targets


def run_calibration_workflow(targets: Optional[List[CalibrationTarget]] = None,
                           config_path: str = "config/thresholds.yaml") -> CalibrationResult:
    """
    Run complete calibration workflow.
    
    Args:
        targets: Optional list of calibration targets (uses defaults if None)
        config_path: Path to threshold configuration file
        
    Returns:
        Calibration results
    """
    try:
        # Use default targets if none provided
        if targets is None:
            targets = create_calibration_targets_from_known_cards()
        
        # Initialize calibrator
        calibrator = ThresholdCalibrator(config_path)
        
        # Run calibration
        result = calibrator.calibrate_from_targets(targets)
        
        # Save optimized configuration
        calibrator.save_optimized_config(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Calibration workflow failed: {e}")
        raise