"""
Uncertainty Calibration and Conformal Prediction module for ET-Pfam.
Re-exports the core implementations from src.uncertainty_calibration.
"""
from src.uncertainty_calibration import (
    ConformalPredictionWrapper,
    compute_random_neighbor_score,
    OODDetector,
)

__all__ = [
    "ConformalPredictionWrapper",
    "compute_random_neighbor_score",
    "OODDetector",
]
