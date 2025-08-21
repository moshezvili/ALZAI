"""
Unit tests for the clinical ML pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# Test data generation
from src.data_generation.generate_clinical_data import ClinicalDataGenerator

class TestClinicalDataGenerator:
    """Test clinical data generation."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = ClinicalDataGenerator(seed=42)
        assert generator.seed == 42
        assert len(generator.icd_codes) > 0
        assert len(generator.medications) > 0
    
    def test_generate_patient_demographics(self):
        """Test patient demographics generation."""
        generator = ClinicalDataGenerator(seed=42)
        demographics = generator.generate_patient_demographics("P001")
        
        assert 'gender' in demographics
        assert demographics['gender'] in ['M', 'F']
        assert 'birth_year' in demographics
        assert 1940 <= demographics['birth_year'] <= 2000
        assert 'baseline_bmi' in demographics
        assert 16 <= demographics['baseline_bmi'] <= 50
        assert 'smoking_status' in demographics
        assert demographics['smoking_status'] in ['Never', 'Former', 'Current']
    
    def test_generate_year_data(self):
        """Test year data generation."""
        generator = ClinicalDataGenerator(seed=42)
        demographics = generator.generate_patient_demographics("P001")
        year_data = generator.generate_year_data("P001", 2020, demographics)
        
        required_fields = [
            'patient_id', 'year', 'age', 'gender', 'bmi', 'systolic_bp',
            'diastolic_bp', 'cholesterol', 'glucose', 'smoking_status',
            'num_visits', 'medications_count', 'lab_abnormal_flag',
            'primary_diagnosis'
        ]
        
        for field in required_fields:
            assert field in year_data
        
        assert year_data['patient_id'] == "P001"
        assert year_data['year'] == 2020
        assert year_data['age'] == 2020 - demographics['birth_year']
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        generator = ClinicalDataGenerator(seed=42)
        df = generator.generate_dataset(
            num_patients=100,
            years_per_patient=3,
            target_prevalence=0.1
        )
        
        assert len(df) > 0
        assert 'target' in df.columns
        assert 'patient_id' in df.columns
        assert df['patient_id'].nunique() <= 100
        assert 0.05 <= df['target'].mean() <= 0.15  # Roughly 10% prevalence

# Test preprocessing
from src.pipeline.preprocessing import MissingValueHandler, DataValidator

class TestMissingValueHandler:
    """Test missing value handling."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        # Create test data with missing values
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0],
            'categorical_col': ['A', 'B', None, 'A'],
            'target': [0, 1, 0, 1]
        })
        
        handler = MissingValueHandler()
        handler.fit(df)
        
        df_transformed = handler.transform(df)
        
        # Check no missing values remain
        assert not df_transformed.isnull().any().any()
        
        # Check indicator columns were added
        missing_indicators = [col for col in df_transformed.columns if '_was_missing' in col]
        assert len(missing_indicators) > 0

class TestDataValidator:
    """Test data validation."""
    
    def test_clinical_rules(self):
        """Test clinical validation rules."""
        validator = DataValidator()
        validator.setup_clinical_rules()
        
        # Create test data with violations
        df = pd.DataFrame({
            'age': [25, 150, 45],  # One violation (150 > 120)
            'bmi': [22, 5, 35],   # One violation (5 < 10) 
            'gender': ['M', 'F', 'X'],  # One violation ('X' not in ['M', 'F'])
            'systolic_bp': [120, 300, 140]  # One violation (300 > 250)
        })
        
        violations = validator.validate(df)
        
        assert len(violations) > 0
        assert 'age' in violations
        assert 'bmi' in violations
        assert 'gender' in violations
        assert 'systolic_bp' in violations

# Test feature engineering
from src.pipeline.feature_engineering import TemporalFeatureEngineer, CategoricalEncoder

class TestTemporalFeatureEngineer:
    """Test temporal feature engineering."""
    
    def test_transform(self):
        """Test temporal feature creation."""
        # Create test data
        df = pd.DataFrame({
            'patient_id': ['P1', 'P1', 'P1', 'P2', 'P2'],
            'year': [2018, 2019, 2020, 2019, 2020],
            'value': [10, 15, 20, 5, 8]
        })
        
        engineer = TemporalFeatureEngineer(
            window_years=2,
            features=['value'],
            aggregations=['mean', 'std']
        )
        
        engineer.fit(df)
        df_transformed = engineer.transform(df)
        
        # Check new features were created
        rolling_features = [col for col in df_transformed.columns if 'rolling' in col]
        assert len(rolling_features) > 0
        
        lag_features = [col for col in df_transformed.columns if 'lag' in col]
        assert len(lag_features) > 0

class TestCategoricalEncoder:
    """Test categorical encoding."""
    
    def test_target_encoding(self):
        """Test target encoding."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'C'],
            'target': [1, 0, 1, 0, 1]
        })
        
        encoder = CategoricalEncoder(method='target_encoding')
        encoder.fit(df, df['target'])
        
        df_transformed = encoder.transform(df)
        
        # Check categorical column was transformed to numeric
        assert pd.api.types.is_numeric_dtype(df_transformed['category'])

# Test model utilities
from src.utils.model_utils import ThresholdOptimizer, ModelEvaluator

class TestThresholdOptimizer:
    """Test threshold optimization."""
    
    def test_f1_optimization(self):
        """Test F1 score optimization."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_proba = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.6, 0.4])
        
        optimizer = ThresholdOptimizer(method='f1_optimal')
        threshold = optimizer.optimize(y_true, y_proba)
        
        assert 0.0 <= threshold <= 1.0

class TestModelEvaluator:
    """Test model evaluation."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.4, 0.8, 0.3])
        
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
        
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'pr_auc', 'specificity', 'sensitivity'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

# Test API
def test_api_import():
    """Test API can be imported."""
    from src.serving.api import app, PatientData
    assert app is not None
    assert PatientData is not None

def test_patient_data_validation():
    """Test patient data validation."""
    from src.serving.api import PatientData
    
    # Valid data
    valid_data = {
        "patient_id": "P001",
        "year": 2023,
        "age": 65.0,
        "gender": "M",
        "bmi": 28.3,
        "systolic_bp": 140.0,
        "diastolic_bp": 90.0,
        "cholesterol": 220.0,
        "glucose": 110.0,
        "smoking_status": "Former",
        "num_visits": 3,
        "medications_count": 2,
        "lab_abnormal_flag": True,
        "primary_diagnosis": "I10"
    }
    
    patient = PatientData(**valid_data)
    assert patient.patient_id == "P001"
    assert patient.gender == "M"
    
    # Invalid data should raise validation error
    invalid_data = valid_data.copy()
    invalid_data['age'] = 150  # Invalid age
    
    with pytest.raises(Exception):  # Pydantic validation error
        PatientData(**invalid_data)

if __name__ == "__main__":
    pytest.main([__file__])
