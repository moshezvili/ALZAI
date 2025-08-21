#!/usr/bin/env python3
"""
Test the updated configuration file
"""

import yaml
from pathlib import Path

def test_config():
    print('=== Testing Updated Configuration ===')
    
    config_path = Path('./config/training_config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print('\n1. Data Generation Config:')
        data_gen = config.get('data_generation', {})
        print(f'   Patients: {data_gen.get("num_patients")}')
        print(f'   Years per patient: {data_gen.get("years_per_patient")}')
        print(f'   Use Dask: {data_gen.get("use_dask")}')
        print(f'   Batch size: {data_gen.get("batch_size")}')
        
        print('\n2. Missing Value Config:')
        missing = data_gen.get('missing_values', {})
        print(f'   Enabled: {missing.get("enabled")}')
        rates = missing.get('rates')
        print(f'   Rates: {rates}')
        
        print('\n3. Available Scenarios:')
        scenarios = missing.get('scenarios', {})
        for name, scenario_rates in scenarios.items():
            print(f'   {name}: {scenario_rates}')
        
        print('\n4. Feature Engineering (Missing Values):')
        feat_eng = config.get('feature_engineering', {})
        mv_config = feat_eng.get('missing_values', {})
        print(f'   Numeric strategy: {mv_config.get("numeric_strategy")}')
        missing_indicators = mv_config.get('missing_indicators', {})
        print(f'   Missing indicators enabled: {missing_indicators.get("enabled")}')
        print(f'   Indicator features: {missing_indicators.get("features")}')
        
        print('\n5. Memory Management:')
        memory = config.get('memory', {})
        print(f'   Use Dask: {memory.get("use_dask")}')
        print(f'   Workers: {memory.get("n_workers")}')
        print(f'   Memory limit: {memory.get("memory_limit")}')
        
        print('\n6. Evaluation (Data Quality):')
        evaluation = config.get('evaluation', {})
        data_quality = evaluation.get('data_quality', {})
        print(f'   Missing value analysis: {data_quality.get("missing_value_analysis")}')
        missing_impact = data_quality.get('missing_impact_analysis', {})
        print(f'   Missing impact analysis: {missing_impact.get("enabled")}')
        
        print('\n✅ Configuration loaded and validated successfully!')
        return True
    else:
        print('❌ Configuration file not found')
        return False

if __name__ == "__main__":
    test_config()
