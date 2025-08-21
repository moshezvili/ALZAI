#!/usr/bin/env python3
"""
Test integration fix for string dtypes issue.
"""

import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_integration_with_string_fix():
    """Test that string dtypes are properly converted before LightGBM."""
    
    from src.data_generation.generate_clinical_data import ClinicalDataGenerator
    from src.pipeline.training_pipeline import ClinicalMLPipeline
    
    # Generate very small test dataset
    logger.info("Generating test dataset...")
    generator = ClinicalDataGenerator(seed=42)
    df = generator.generate_dataset(
        num_patients=50,
        years_per_patient=2,
        target_prevalence=0.15
    )
    
    logger.info(f"Generated dataset shape: {df.shape}")
    logger.info(f"Data types before processing:\n{df.dtypes}")
    
    # Check for string columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    logger.info(f"String columns: {string_cols}")
    
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
                'categorical_encoding': {
                    'method': 'target_encoding',
                    'cv_folds': 2  # Small for speed
                },
                'scaling': {
                    'enabled': True,
                    'method': 'standard'
                },
                'missing_values': {'numeric_strategy': 'median'}
            },
            'hpo': {'enabled': False},
            'model': {
                'algorithm': 'lightgbm',
                'lightgbm': {
                    'objective': 'binary',
                    'n_estimators': 5,  # Very small for speed
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            'cross_validation': {
                'method': 'stratified',
                'n_splits': 2,  # Small for speed
                'test_size': 0.3
            },
            'mlflow': {
                'experiment_name': 'test_integration_fix',
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
        logger.info("Starting training pipeline...")
        pipeline = ClinicalMLPipeline(config)
        
        try:
            pipeline.run_pipeline(str(data_path), str(output_path), skip_shap=True)
            logger.info("✅ Training pipeline completed successfully!")
            
            # Verify outputs exist
            assert (output_path / "best_model.joblib").exists(), "Model file not found"
            assert (output_path / "optimal_threshold.txt").exists(), "Threshold file not found"
            assert (output_path / "metrics.yaml").exists(), "Metrics file not found"
            
            # Verify metrics file has content
            with open(output_path / "metrics.yaml", 'r') as f:
                metrics = yaml.safe_load(f)
            
            logger.info(f"Final metrics: {metrics}")
            assert 'roc_auc' in metrics, "ROC AUC not found in metrics"
            assert 'pr_auc' in metrics, "PR AUC not found in metrics"
            assert 0 <= metrics['roc_auc'] <= 1, f"Invalid ROC AUC: {metrics['roc_auc']}"
            assert 0 <= metrics['pr_auc'] <= 1, f"Invalid PR AUC: {metrics['pr_auc']}"
            
            logger.info("✅ All validations passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_integration_with_string_fix()
    if success:
        print("🎉 Integration test passed! String dtypes issue is fixed.")
    else:
        print("❌ Integration test failed.")
        exit(1)
