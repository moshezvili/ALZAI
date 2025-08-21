"""
Model utilities for threshold optimization and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """Optimize classification threshold based on different strategies."""
    
    def __init__(self, method: str = 'f1_optimal'):
        """
        Initialize threshold optimizer.
        
        Args:
            method: Optimization method ('f1_optimal', 'precision_recall_curve', 'youden_j')
        """
        self.method = method
    
    def optimize(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Find optimal threshold.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            
        Returns:
            Optimal threshold value
        """
        if self.method == 'f1_optimal':
            return self._optimize_f1(y_true, y_proba)
        elif self.method == 'precision_recall_curve':
            return self._optimize_precision_recall(y_true, y_proba)
        elif self.method == 'youden_j':
            return self._optimize_youden_j(y_true, y_proba)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def _optimize_f1(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find threshold that maximizes F1 score."""
        thresholds = np.linspace(0.1, 0.9, 100)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal threshold for F1: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def _optimize_precision_recall(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find threshold using precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold that maximizes F1 score from PR curve
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        logger.info(f"Optimal threshold from PR curve: {best_threshold:.3f}")
        return best_threshold
    
    def _optimize_youden_j(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find threshold using Youden's J statistic (sensitivity + specificity - 1)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Youden's J = Sensitivity + Specificity - 1 = TPR + (1-FPR) - 1 = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        best_threshold = thresholds[best_idx]
        logger.info(f"Optimal threshold from Youden's J: {best_threshold:.3f}")
        return best_threshold

class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        
        # Clinical metrics
        metrics['number_needed_to_screen'] = 1 / metrics['ppv'] if metrics['ppv'] > 0 else float('inf')
        
        return metrics
    
    def slice_analysis(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      y_proba: np.ndarray,
                      slice_feature: np.ndarray,
                      slice_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Perform slice analysis across different subgroups.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels  
            y_proba: Predicted probabilities
            slice_feature: Feature values for slicing
            slice_names: Names for slice values
            
        Returns:
            Dictionary with metrics for each slice
        """
        slice_results = {}
        unique_values = np.unique(slice_feature)
        
        for i, value in enumerate(unique_values):
            slice_name = slice_names[i] if slice_names else str(value)
            mask = slice_feature == value
            
            if np.sum(mask) < 10:  # Skip slices with too few samples
                continue
            
            slice_metrics = self.calculate_metrics(
                y_true[mask], y_pred[mask], y_proba[mask]
            )
            slice_metrics['sample_size'] = int(np.sum(mask))
            slice_results[slice_name] = slice_metrics
        
        return slice_results
    
    def generate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> str:
        """Generate detailed classification report."""
        return classification_report(y_true, y_pred)
    
    def calculate_calibration_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_proba: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate calibration metrics.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration metrics
        """
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )
        
        # Brier score (lower is better)
        brier_score = np.mean((y_proba - y_true) ** 2)
        
        # Expected Calibration Error
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'brier_score': brier_score,
            'expected_calibration_error': ece
        }

class ModelComparator:
    """Compare multiple models."""
    
    def __init__(self):
        """Initialize comparator."""
        self.results = {}
    
    def add_model(self, 
                  name: str, 
                  y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  y_proba: np.ndarray):
        """Add model results for comparison."""
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
        self.results[name] = metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Create comparison table."""
        if not self.results:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.results).T
        
        # Sort by ROC-AUC (or another primary metric)
        if 'roc_auc' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'roc_auc') -> str:
        """Get name of best performing model."""
        if not self.results:
            return None
        
        best_score = -1
        best_model = None
        
        for model_name, metrics in self.results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        return best_model

def calculate_statistical_significance(results1: Dict[str, float], 
                                     results2: Dict[str, float],
                                     metric: str = 'roc_auc') -> Dict[str, float]:
    """
    Calculate statistical significance between two model results.
    
    This is a simplified version - in practice, you'd want to use
    proper statistical tests with multiple CV folds.
    """
    if metric not in results1 or metric not in results2:
        return {'p_value': None, 'significant': False}
    
    # Simplified difference test
    diff = abs(results1[metric] - results2[metric])
    
    # Rule of thumb: difference > 0.01 for AUC is often meaningful
    significant = diff > 0.01 if 'auc' in metric else diff > 0.05
    
    return {
        'difference': diff,
        'significant': significant,
        'p_value': None  # Would require proper statistical test
    }
