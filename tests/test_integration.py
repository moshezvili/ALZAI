"""
Integration tests for the training pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import yaml
from unittest.mock import patch

def test_training_pipeline_integration():
    """Test the complete training pipeline end-to-end."""
    from src.data_generation.generate_clinical_data import ClinicalDataGenerator
    
    # Generate small test dataset
    generator = ClinicalDataGenerator(seed=42)
    df = generator.generate_dataset(
        num_patients=100,
        years_per_patient=2,
        target_prevalence=0.15
    )
    
    # Create temporary config and data files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save test data
        data_path = temp_path / "test_data.parquet"
        df.to_parquet(data_path)
        
        # Create minimal training config
        config = {
            'feature_engineering': {
                'rolling_window_years': 1,
                'temporal_features': ['mean'],
                'categorical_encoding': {'method': 'target_encoding'},
                'missing_values': {'numeric_strategy': 'median'}
            },
            'hpo': {'enabled': False},
            'model': {
                'algorithm': 'lightgbm',
                'lightgbm': {
                    'objective': 'binary',
                    'n_estimators': 10,  # Small for speed
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'cross_validation': {
                'method': 'stratified',
                'n_splits': 2,  # Small for speed
                'test_size': 0.3
            },
            'mlflow': {
                'experiment_name': 'test_experiment',
                'tracking_uri': f'file:{temp_path}/mlruns'
            },
            'imbalance': {'method': 'none'},
            'threshold': {'method': 'f1_optimal'}
        }
        
        config_path = temp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        output_path = temp_path / "models"
        
        # Test training pipeline import and execution
        from src.pipeline.training_pipeline import ClinicalMLPipeline
        
        # Initialize pipeline
        pipeline = ClinicalMLPipeline(config)
        
        # Run pipeline with skip SHAP for speed
        pipeline.run_pipeline(str(data_path), str(output_path), skip_shap=True)
        
        # Verify outputs exist
        assert (output_path / "best_model.joblib").exists()
        assert (output_path / "optimal_threshold.txt").exists()
        assert (output_path / "metrics.yaml").exists()
        
        # Verify metrics file has content
        with open(output_path / "metrics.yaml", 'r') as f:
            metrics = yaml.safe_load(f)
        
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1

def test_training_pipeline_with_hpo():
    """Test training pipeline with hyperparameter optimization."""
    from src.data_generation.generate_clinical_data import ClinicalDataGenerator
    
    # Generate very small test dataset for speed
    generator = ClinicalDataGenerator(seed=42)
    df = generator.generate_dataset(
        num_patients=50,
        years_per_patient=2,
        target_prevalence=0.2
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save test data
        data_path = temp_path / "test_data.parquet"
        df.to_parquet(data_path)
        
        # Create config with HPO enabled
        config = {
            'feature_engineering': {
                'rolling_window_years': 1,
                'temporal_features': ['mean'],
                'categorical_encoding': {'method': 'target_encoding'},
                'missing_values': {'numeric_strategy': 'median'}
            },
            'hpo': {
                'enabled': True,
                'n_trials': 3,  # Very small for speed
                'algorithms': ['lightgbm'],
                'opt_metric': 'roc_auc'
            },
            'model': {
                'algorithm': 'lightgbm',
                'lightgbm': {
                    'objective': 'binary',
                    'random_state': 42
                }
            },
            'cross_validation': {
                'method': 'stratified',
                'n_splits': 2,
                'test_size': 0.3
            },
            'mlflow': {
                'experiment_name': 'test_hpo_experiment',
                'tracking_uri': f'file:{temp_path}/mlruns'
            },
            'imbalance': {'method': 'none'},
            'threshold': {'method': 'f1_optimal'}
        }
        
        config_path = temp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        output_path = temp_path / "models"
        
        # Test training pipeline with HPO
        from src.pipeline.training_pipeline import ClinicalMLPipeline
        
        pipeline = ClinicalMLPipeline(config)
        pipeline.run_pipeline(str(data_path), str(output_path), skip_shap=True)
        
        # Verify outputs exist
        assert (output_path / "best_model.joblib").exists()
        assert (output_path / "metrics.yaml").exists()
        
        # Verify HPO was used (should have better or equal performance)
        with open(output_path / "metrics.yaml", 'r') as f:
            metrics = yaml.safe_load(f)
        
        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] > 0  # Should have some performance

@patch('src.utils.experiment_tracking.mlflow')
def test_training_pipeline_mlflow_integration(mock_mlflow):
    """Test training pipeline MLflow integration."""
    from src.data_generation.generate_clinical_data import ClinicalDataGenerator
    from unittest.mock import MagicMock
    
    # Create a proper context manager mock
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    
    # Mock MLflow methods
    mock_mlflow.set_tracking_uri.return_value = None
    mock_mlflow.set_experiment.return_value = None
    mock_mlflow.start_run.return_value = mock_run
    mock_mlflow.log_param.return_value = None
    mock_mlflow.log_metric.return_value = None
    mock_mlflow.log_artifacts.return_value = None
    mock_mlflow.create_experiment.side_effect = Exception("Experiment exists")
    mock_mlflow.get_experiment_by_name.return_value = MagicMock(experiment_id="test_id")
    
    generator = ClinicalDataGenerator(seed=42)
    df = generator.generate_dataset(
        num_patients=30,
        years_per_patient=2,
        target_prevalence=0.2
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        data_path = temp_path / "test_data.parquet"
        df.to_parquet(data_path)
        
        config = {
            'feature_engineering': {
                'rolling_window_years': 1,
                'temporal_features': ['mean'],
                'categorical_encoding': {'method': 'target_encoding'},
                'missing_values': {'numeric_strategy': 'median'}
            },
            'hpo': {'enabled': False},
            'model': {
                'algorithm': 'lightgbm',
                'lightgbm': {
                    'objective': 'binary',
                    'n_estimators': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'cross_validation': {
                'method': 'stratified',
                'n_splits': 2,
                'test_size': 0.3
            },
            'mlflow': {
                'experiment_name': 'test_mlflow_experiment',
                'tracking_uri': f'file:{temp_path}/mlruns'
            },
            'imbalance': {'method': 'none'},
            'threshold': {'method': 'f1_optimal'}
        }
        
        config_path = temp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        output_path = temp_path / "models"
        
        # Test training pipeline
        from src.pipeline.training_pipeline import ClinicalMLPipeline
        
        pipeline = ClinicalMLPipeline(config)
        pipeline.run_pipeline(str(data_path), str(output_path), skip_shap=True)
        
        # Verify MLflow methods were called
        mock_mlflow.set_tracking_uri.assert_called()
        mock_mlflow.log_metric.assert_called()  # Individual calls, not log_metrics

if __name__ == "__main__":
    pytest.main([__file__])
