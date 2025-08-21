#!/usr/bin/env python3
"""
Test script for feature engineering improvements
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.feature_engineering import (
    TemporalFeatureEngineer, 
    CategoricalEncoder, 
    FeatureSelector,
    create_preprocessing_pipeline
)
from src.pipeline.preprocessing import MissingValueHandler, DataScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test clinical data with temporal patterns."""
    np.random.seed(42)
    
    n_patients = 100
    years = [2015, 2016, 2017, 2018, 2019]
    
    data = []
    for patient_id in range(1, n_patients + 1):
        for year in years:
            # Create some temporal patterns
            base_age = 50 + np.random.normal(0, 10)
            age = base_age + (year - 2015)
            
            # BMI with slight temporal trend
            base_bmi = 25 + np.random.normal(0, 4)
            bmi = base_bmi + 0.2 * (year - 2015) + np.random.normal(0, 1)
            
            # Glucose with some missing values
            glucose = 100 + np.random.normal(0, 20)
            if np.random.random() < 0.05:  # 5% missing
                glucose = np.nan
            
            # Categorical features
            gender = np.random.choice(['M', 'F'])
            smoking = np.random.choice(['Never', 'Former', 'Current'])
            
            # Target with some realistic patterns
            risk_score = (age - 50) * 0.02 + (bmi - 25) * 0.03 + (glucose - 100) * 0.001
            if smoking == 'Current':
                risk_score += 0.5
            elif smoking == 'Former':
                risk_score += 0.2
            
            target = 1 if (risk_score + np.random.normal(0, 0.5)) > 0.5 else 0
            
            data.append({
                'patient_id': patient_id,
                'year': year,
                'age': age,
                'bmi': bmi,
                'glucose': glucose,
                'gender': gender,
                'smoking_status': smoking,
                'target': target
            })
    
    return pd.DataFrame(data)

def test_temporal_features():
    """Test temporal feature engineering with improved trend calculation."""
    logger.info("Testing temporal feature engineering...")
    
    df = create_test_data()
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Test temporal feature engineer
    temporal = TemporalFeatureEngineer(
        window_years=3,
        features=['age', 'bmi', 'glucose'],
        aggregations=['mean', 'std', 'trend', 'min', 'max']
    )
    
    temporal.fit(X)
    X_temporal = temporal.transform(X)
    
    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"After temporal features: {X_temporal.shape[1]}")
    logger.info(f"Added features: {X_temporal.shape[1] - X.shape[1] + 2}")  # +2 for patient_id, year removed
    
    # Check that trend features exist and have reasonable values
    trend_cols = [col for col in X_temporal.columns if 'trend' in col]
    logger.info(f"Trend features created: {len(trend_cols)}")
    for col in trend_cols[:2]:  # Show first 2
        logger.info(f"{col}: mean={X_temporal[col].mean():.4f}, std={X_temporal[col].std():.4f}")
    
    return X_temporal, y

def test_categorical_encoding():
    """Test K-Fold target encoding."""
    logger.info("Testing K-Fold categorical encoding...")
    
    df = create_test_data()
    X = df[['patient_id', 'year', 'gender', 'smoking_status', 'age']]
    y = df['target']
    
    # Test K-Fold target encoding
    encoder = CategoricalEncoder(
        method='target_encoding',
        cv_folds=5,
        random_state=42
    )
    
    encoder.fit(X, y)
    X_encoded = encoder.transform(X)
    
    logger.info(f"Categorical features encoded: {len(encoder.categorical_features_)}")
    logger.info(f"Global means stored: {len(encoder.global_means_)}")
    
    # Check encoding values are reasonable
    for feature in encoder.categorical_features_:
        if feature in X_encoded.columns:
            logger.info(f"{feature}: min={X_encoded[feature].min():.4f}, max={X_encoded[feature].max():.4f}")
    
    return X_encoded, y

def test_scaling():
    """Test data scaling after encoding."""
    logger.info("Testing data scaling...")
    
    df = create_test_data()
    X = df.drop(columns=['target'])
    y = df['target']
    
    # First apply missing value handler
    missing_handler = MissingValueHandler(
        numeric_strategy='median',
        categorical_strategy='most_frequent'
    )
    missing_handler.fit(X)
    X_clean = missing_handler.transform(X)
    
    # Then categorical encoding
    encoder = CategoricalEncoder(method='target_encoding', cv_folds=3)
    encoder.fit(X_clean, y)
    X_encoded = encoder.transform(X_clean)
    
    # Finally scaling
    scaler = DataScaler(method='standard')
    scaler.fit(X_encoded)
    X_scaled = scaler.transform(X_encoded)
    
    # Check scaling worked
    numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
    logger.info(f"Numeric features scaled: {len(numeric_cols)}")
    
    for col in numeric_cols[:3]:  # Show first 3
        mean_val = X_scaled[col].mean()
        std_val = X_scaled[col].std()
        logger.info(f"{col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    return X_scaled

def test_full_pipeline():
    """Test the complete preprocessing pipeline."""
    logger.info("Testing complete preprocessing pipeline...")
    
    # Load configuration
    config_path = project_root / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    df = create_test_data()
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Create pipeline
    pipeline_steps = create_preprocessing_pipeline(config)
    
    logger.info(f"Pipeline has {len(pipeline_steps)} steps:")
    for i, step in enumerate(pipeline_steps):
        logger.info(f"  {i+1}. {step.__class__.__name__}")
    
    # Apply pipeline step by step
    X_current = X.copy()
    for i, step in enumerate(pipeline_steps):
        logger.info(f"\nApplying step {i+1}: {step.__class__.__name__}")
        logger.info(f"Input shape: {X_current.shape}")
        
        if hasattr(step, 'fit'):
            if step.__class__.__name__ in ['CategoricalEncoder', 'FeatureSelector']:
                step.fit(X_current, y)
            else:
                step.fit(X_current)
        
        X_current = step.transform(X_current)
        logger.info(f"Output shape: {X_current.shape}")
    
    logger.info(f"\nFinal pipeline output:")
    logger.info(f"Shape: {X_current.shape}")
    logger.info(f"Columns: {list(X_current.columns)}")
    logger.info(f"Data types: {X_current.dtypes.value_counts().to_dict()}")
    
    return X_current

def main():
    """Run all tests."""
    logger.info("Starting feature engineering improvement tests...")
    
    try:
        # Test individual components
        logger.info("\n" + "="*50)
        test_temporal_features()
        
        logger.info("\n" + "="*50)
        test_categorical_encoding()
        
        logger.info("\n" + "="*50)
        test_scaling()
        
        logger.info("\n" + "="*50)
        test_full_pipeline()
        
        logger.info("\n" + "="*50)
        logger.info("All tests completed successfully! ✅")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
