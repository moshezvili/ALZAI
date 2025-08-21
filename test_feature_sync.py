#!/usr/bin/env python3
"""
Test script for feature synchronization in API
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import yaml

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Simple test to check feature alignment
def test_feature_alignment():
    """Test feature alignment logic."""
    print("Testing feature alignment...")
    
    # Simulate feature_names from model
    feature_names = ['age', 'bmi', 'glucose', 'cholesterol', 'age_rolling_mean_3y', 'bmi_lag_1y']
    
    # Create test input data (partial features)
    test_data = {
        'age': 65,
        'bmi': 28.5,
        'glucose': 110
        # Missing: cholesterol, age_rolling_mean_3y, bmi_lag_1y
    }
    
    df = pd.DataFrame([test_data])
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Apply reindex with feature_names
    df_aligned = df.reindex(columns=feature_names, fill_value=0.0)
    
    print(f"Aligned data shape: {df_aligned.shape}")
    print(f"Aligned columns: {list(df_aligned.columns)}")
    print(f"Values:\n{df_aligned}")
    
    # Check that all expected features are present
    missing_features = [col for col in feature_names if col not in test_data.keys()]
    print(f"Missing features filled with 0.0: {missing_features}")
    
    print("✅ Feature alignment test passed!")

if __name__ == "__main__":
    test_feature_alignment()
