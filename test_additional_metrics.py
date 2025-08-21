#!/usr/bin/env python3
"""
Test script for additional metrics in cross-validation
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

from src.pipeline.training_pipeline import ClinicalMLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_test_data():
    """Create very simple test data for quick testing."""
    np.random.seed(42)
    
    n_patients = 50
    years = [2018, 2019]
    
    data = []
    for patient_id in range(1, n_patients + 1):
        for year in years:
            age = 50 + np.random.normal(0, 10)
            bmi = 25 + np.random.normal(0, 4)
            glucose = 100 + np.random.normal(0, 20)
            gender = np.random.choice(['M', 'F'])
            smoking = np.random.choice(['Never', 'Current'])
            
            # Simple target
            risk_score = (age - 50) * 0.02 + (bmi - 25) * 0.03
            if smoking == 'Current':
                risk_score += 0.5
            
            target = 1 if (risk_score + np.random.normal(0, 0.5)) > 0.3 else 0
            
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

def test_cv_metrics():
    """Test that all metrics are calculated in cross-validation."""
    logger.info("Testing cross-validation metrics...")
    
    # Create simple config
    config = {
        'random_seed': 42,
        'feature_engineering': {
            'rolling_window_years': 2,
            'temporal_features': ['mean', 'std'],
            'categorical_encoding': {
                'method': 'target_encoding',
                'cv_folds': 2
            },
            'scaling': {
                'enabled': True,
                'method': 'standard'
            }
        },
        'model': {
            'algorithm': 'lightgbm',
            'lightgbm': {
                'n_estimators': 10,
                'num_leaves': 10,
                'verbose': -1
            }
        },
        'cross_validation': {
            'n_splits': 2,
            'test_size': 0.3,
            'gap': 0
        },
        'hpo': {
            'enabled': False
        },
        'mlflow': {
            'experiment_name': 'test_metrics',
            'tracking_uri': 'file:./test_mlruns'
        }
    }
    
    # Create test data
    df = create_simple_test_data()
    logger.info(f"Created test data with shape: {df.shape}")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Initialize pipeline
    pipeline = ClinicalMLPipeline(config)
    
    # Prepare data
    X, y = pipeline.prepare_features(df)
    
    # Test CV metrics (this should now include precision, recall, accuracy)
    logger.info("Testing cross-validation training...")
    cv_metrics = pipeline.train_model(X, y)
    
    logger.info("Cross-validation metrics calculated:")
    for metric, value in cv_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    # Check that all expected metrics are present
    expected_metrics = ['cv_roc_auc', 'cv_pr_auc', 'cv_f1', 'cv_precision', 'cv_recall', 'cv_accuracy']
    missing_metrics = [m for m in expected_metrics if m not in cv_metrics]
    
    if missing_metrics:
        logger.error(f"Missing metrics: {missing_metrics}")
        return False
    else:
        logger.info("✅ All expected cross-validation metrics are present!")
        return True

def main():
    """Run the test."""
    logger.info("Starting metrics test...")
    
    try:
        success = test_cv_metrics()
        
        if success:
            logger.info("✅ Test completed successfully!")
            return 0
        else:
            logger.error("❌ Test failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
