"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

@pytest.fixture
def sample_clinical_data():
    """Create sample clinical data for testing."""
    np.random.seed(42)
    
    n_patients = 100
    years = [2018, 2019, 2020]
    
    data = []
    for patient_id in range(n_patients):
        for year in years:
            data.append({
                'patient_id': f'P{patient_id:04d}',
                'year': year,
                'age': np.random.randint(25, 85),
                'gender': np.random.choice(['M', 'F']),
                'bmi': np.random.normal(27, 5),
                'systolic_bp': np.random.normal(130, 20),
                'diastolic_bp': np.random.normal(80, 15),
                'cholesterol': np.random.normal(200, 40),
                'glucose': np.random.normal(100, 20),
                'smoking_status': np.random.choice(['Never', 'Former', 'Current']),
                'num_visits': np.random.poisson(3),
                'medications_count': np.random.poisson(2),
                'lab_abnormal_flag': np.random.choice([True, False]),
                'primary_diagnosis': np.random.choice(['I10', 'E11', 'I25', 'J44']),
                'target': np.random.choice([0, 1], p=[0.9, 0.1])
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data_generation': {
            'num_patients': 1000,
            'years_per_patient': 3,
            'target_prevalence': 0.07
        },
        'feature_engineering': {
            'rolling_window_years': 2,
            'temporal_features': ['mean', 'std'],
            'categorical_encoding': {
                'method': 'target_encoding',
                'handle_unknown': 'ignore'
            },
            'missing_values': {
                'numeric_strategy': 'median',
                'categorical_strategy': 'most_frequent'
            }
        },
        'model': {
            'algorithm': 'lightgbm',
            'lightgbm': {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'random_state': 42
            }
        },
        'imbalance': {
            'method': 'smote',
            'smote': {
                'sampling_strategy': 0.1,
                'random_state': 42
            }
        },
        'cross_validation': {
            'method': 'time_series_split',
            'n_splits': 3,
            'test_size': 0.2
        },
        'threshold': {
            'method': 'f1_optimal'
        },
        'mlflow': {
            'experiment_name': 'test_experiment',
            'tracking_uri': 'file:./test_mlruns'
        }
    }
