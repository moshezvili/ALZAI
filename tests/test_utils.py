"""
Test suite for utilities and experiment tracking.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.utils.experiment_tracking import ExperimentTracker, setup_experiment_tracking


class TestExperimentTracker:
    """Test experiment tracking functionality."""
    
    @patch('mlflow.create_experiment')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.set_experiment')
    @patch('mlflow.set_tracking_uri')
    def test_init(self, mock_set_uri, mock_set_exp, mock_get_exp, mock_create_exp):
        """Test ExperimentTracker initialization."""
        # Mock successful experiment creation
        mock_create_exp.return_value = "test_exp_id"
        
        config = {
            'mlflow': {
                'tracking_uri': 'file:./test_mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        tracker = ExperimentTracker(config)
        
        assert tracker.tracking_uri == 'file:./test_mlruns'
        # Allow for timestamp suffix in experiment name
        assert tracker.experiment_name.startswith('test_experiment')
        mock_set_uri.assert_called_once_with('file:./test_mlruns')
    
    @patch('mlflow.start_run')
    def test_start_run(self, mock_start_run):
        """Test starting MLflow run."""
        config = {
            'mlflow': {
                'tracking_uri': 'file:./test_mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        tracker = ExperimentTracker(config)
        tracker.start_run("test_run")
        
        mock_start_run.assert_called_once_with(run_name="test_run")
    
    @patch('mlflow.log_param')
    def test_log_params(self, mock_log_param):
        """Test logging parameters."""
        config = {
            'mlflow': {
                'tracking_uri': 'file:./test_mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        tracker = ExperimentTracker(config)
        params = {'learning_rate': 0.01, 'n_estimators': 100}
        tracker.log_params(params)
        
        # Check that log_param was called for each parameter (values are converted to strings)
        assert mock_log_param.call_count == 2
        mock_log_param.assert_any_call('learning_rate', '0.01')
        mock_log_param.assert_any_call('n_estimators', '100')
    
    @patch('mlflow.log_metric')
    def test_log_metrics(self, mock_log_metric):
        """Test logging metrics."""
        config = {
            'mlflow': {
                'tracking_uri': 'file:./test_mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        tracker = ExperimentTracker(config)
        metrics = {'accuracy': 0.95, 'f1_score': 0.92}
        tracker.log_metrics(metrics)
        
        # Check that log_metric was called for each metric
        assert mock_log_metric.call_count == 2
        mock_log_metric.assert_any_call('accuracy', 0.95, step=None)
        mock_log_metric.assert_any_call('f1_score', 0.92, step=None)
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        config = {
            'mlflow': {
                'tracking_uri': 'file:./test_mlruns',
                'experiment_name': 'test_experiment'
            }
        }
        
        tracker = ExperimentTracker(config)
        nested_dict = {
            'model': {
                'learning_rate': 0.01,
                'params': {
                    'n_estimators': 100
                }
            }
        }
        
        flattened = tracker._flatten_dict(nested_dict)
        
        assert 'model.learning_rate' in flattened
        assert 'model.params.n_estimators' in flattened
        assert flattened['model.learning_rate'] == '0.01'
        assert flattened['model.params.n_estimators'] == '100'


def test_setup_experiment_tracking():
    """Test experiment tracking setup function."""
    # Test MLflow setup
    mlflow_config = {
        'experiment_tracking': {
            'backend': 'mlflow',
            'mlflow': {
                'tracking_uri': 'file:./test_mlruns',
                'experiment_name': 'test_experiment'
            }
        }
    }
    
    with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
        tracker = setup_experiment_tracking(mlflow_config)
        assert isinstance(tracker, ExperimentTracker)
    
    # Test config without experiment_tracking key (should default to mlflow with empty config)
    empty_config = {}
    with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
        tracker = setup_experiment_tracking(empty_config)
        assert isinstance(tracker, ExperimentTracker)  # Defaults to mlflow
    
    # Test config with explicit None backend
    none_config = {
        'experiment_tracking': {
            'backend': 'none'
        }
    }
    tracker = setup_experiment_tracking(none_config)
    assert tracker is None


# Test integration scenarios
class TestIntegration:
    """Test integration scenarios."""
    
    def test_data_generation_and_preprocessing(self):
        """Test data generation followed by preprocessing."""
        from src.data_generation.generate_clinical_data import ClinicalDataGenerator
        from src.pipeline.preprocessing import MissingValueHandler
        
        # Generate small dataset
        generator = ClinicalDataGenerator(seed=42)
        df = generator.generate_dataset(
            num_patients=50,
            years_per_patient=2,
            target_prevalence=0.1
        )
        
        # Preprocess data
        handler = MissingValueHandler()
        handler.fit(df)
        df_processed = handler.transform(df)
        
        # Verify pipeline works
        assert len(df_processed) > 0
        assert 'target' in df_processed.columns
        assert df_processed.isnull().sum().sum() == 0  # No missing values
    
    def test_feature_engineering_pipeline(self):
        """Test feature engineering components together."""
        from src.pipeline.feature_engineering import TemporalFeatureEngineer, CategoricalEncoder
        
        # Create test data
        df = pd.DataFrame({
            'patient_id': ['P1', 'P1', 'P2', 'P2'] * 3,
            'year': [2018, 2019, 2018, 2019] * 3,
            'category': ['A', 'B', 'A', 'B'] * 3,
            'value': [10, 15, 8, 12] * 3,
            'target': [0, 1, 0, 1] * 3
        })
        
        # Apply temporal feature engineering
        temp_engineer = TemporalFeatureEngineer(
            window_years=2,
            features=['value'],
            aggregations=['mean']
        )
        temp_engineer.fit(df)
        df_temporal = temp_engineer.transform(df)
        
        # Apply categorical encoding
        cat_encoder = CategoricalEncoder(method='target_encoding')
        cat_encoder.fit(df_temporal, df_temporal['target'])
        df_encoded = cat_encoder.transform(df_temporal)
        
        # Verify pipeline works
        assert len(df_encoded) > 0
        assert 'value_rolling_2y_mean' in df_encoded.columns or any('rolling' in col for col in df_encoded.columns)


if __name__ == "__main__":
    pytest.main([__file__])
