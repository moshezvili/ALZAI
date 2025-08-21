"""
Clinical ML Pipeline - Senior ML Engineer Assignment

A comprehensive end-to-end machine learning pipeline for binary classification
on synthetic clinical data, addressing diagnosis year uncertainty and handling
large-scale patient-year data efficiently.
"""

__version__ = "1.0.0"
__author__ = "Clinical ML Team"
__email__ = "ml-team@clinical.ai"

from .data_generation import ClinicalDataGenerator
from .pipeline import (
    TemporalFeatureEngineer,
    CategoricalEncoder,
    MissingValueHandler,
    DataValidator
)
from .utils import (
    ExperimentTracker,
    ThresholdOptimizer,
    ModelEvaluator
)

__all__ = [
    'ClinicalDataGenerator',
    'TemporalFeatureEngineer', 
    'CategoricalEncoder',
    'MissingValueHandler',
    'DataValidator',
    'ExperimentTracker',
    'ThresholdOptimizer',
    'ModelEvaluator'
]
