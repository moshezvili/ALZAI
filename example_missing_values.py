#!/usr/bin/env python3
"""
Example: Missing Value Simulation in Clinical Data

This script demonstrates how the improved ClinicalDataGenerator
simulates realistic missing values in clinical datasets.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_generation.generate_clinical_data import ClinicalDataGenerator
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_missing_values():
    """Demonstrate missing value simulation functionality."""
    logger.info("=== Missing Value Simulation Example ===")
    
    # 1. Default missing value rates
    logger.info("\n1. Testing with default missing value rates")
    generator = ClinicalDataGenerator(seed=42)
    logger.info(f"Default rates: {generator.missing_value_rates}")
    
    # Generate data
    dask_df = generator.generate_dask_dataset(
        num_patients=2000,
        years_per_patient=3,
        batch_size=400
    )
    df = generator.get_computed_dataframe(dask_df)
    
    # Analyze missing values
    logger.info(f"Dataset shape: {df.shape}")
    logger.info("\nMissing value analysis:")
    
    missing_analysis = {}
    total_records = len(df)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            percentage = (missing_count / total_records) * 100
            missing_analysis[col] = {
                'count': missing_count,
                'percentage': percentage
            }
            logger.info(f"  {col}: {missing_count:,} missing ({percentage:.1f}%)")
    
    # 2. Custom missing value rates for different scenarios
    logger.info("\n\n2. Testing different missing value scenarios")
    
    scenarios = {
        "High Quality Lab": {
            'cholesterol': 0.01,  # Very low missing rate
            'bmi': 0.02,
            'glucose': 0.005,
            'blood_pressure': 0.01
        },
        "Standard Clinical Practice": {
            'cholesterol': 0.05,
            'bmi': 0.10,
            'glucose': 0.03,
            'blood_pressure': 0.02
        },
        "Resource-Limited Setting": {
            'cholesterol': 0.25,  # High missing rate
            'bmi': 0.15,
            'glucose': 0.20,
            'blood_pressure': 0.08
        }
    }
    
    for scenario_name, rates in scenarios.items():
        logger.info(f"\n--- {scenario_name} ---")
        
        generator_scenario = ClinicalDataGenerator(seed=42, missing_value_rates=rates)
        dask_df_scenario = generator_scenario.generate_dask_dataset(
            num_patients=1000,
            years_per_patient=2,
            batch_size=200
        )
        df_scenario = generator_scenario.get_computed_dataframe(dask_df_scenario)
        
        logger.info(f"Expected rates: {rates}")
        logger.info("Actual missing rates:")
        
        for col in ['bmi', 'cholesterol', 'glucose', 'systolic_bp']:
            if col in df_scenario.columns:
                missing_count = df_scenario[col].isnull().sum()
                actual_rate = (missing_count / len(df_scenario)) * 100
                
                # Map column to expected rate
                expected_key = {
                    'cholesterol': 'cholesterol',
                    'bmi': 'bmi', 
                    'glucose': 'glucose',
                    'systolic_bp': 'blood_pressure'
                }.get(col)
                
                if expected_key:
                    expected_rate = rates[expected_key] * 100
                    logger.info(f"  {col}: {actual_rate:.1f}% (expected: {expected_rate:.1f}%)")
    
    # 3. Show impact on data quality
    logger.info("\n\n3. Impact on downstream ML pipeline")
    
    # Generate data with missing values
    generator_with_missing = ClinicalDataGenerator(
        seed=42, 
        missing_value_rates={
            'cholesterol': 0.10,
            'bmi': 0.15,
            'glucose': 0.05,
            'blood_pressure': 0.03
        }
    )
    
    dask_df_missing = generator_with_missing.generate_dask_dataset(
        num_patients=5000,
        years_per_patient=3,
        batch_size=500
    )
    dask_df_missing = generator_with_missing.add_target_variable_dask(dask_df_missing)
    df_missing = generator_with_missing.get_computed_dataframe(dask_df_missing)
    
    logger.info(f"Dataset with missing values: {df_missing.shape}")
    
    # Show complete case analysis impact
    complete_cases = df_missing.dropna()
    logger.info(f"Complete cases (no missing values): {complete_cases.shape}")
    logger.info(f"Data loss from missing values: {(1 - len(complete_cases)/len(df_missing))*100:.1f}%")
    
    # Show which features are most affected
    logger.info("\nFeatures most affected by missing values:")
    missing_summary = df_missing.isnull().sum().sort_values(ascending=False)
    for col, count in missing_summary.head(10).items():
        if count > 0:
            percentage = (count / len(df_missing)) * 100
            logger.info(f"  {col}: {percentage:.1f}% missing")
    
    logger.info("\n✅ Missing value simulation provides realistic data quality challenges")
    logger.info("   that will help test the robustness of ML pipelines!")

if __name__ == "__main__":
    example_missing_values()
