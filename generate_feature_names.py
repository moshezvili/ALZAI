#!/usr/bin/env python3
"""
Generate feature_names.txt from trained model for API serving.
"""

import joblib
import pandas as pd
from pathlib import Path

def generate_feature_names():
    """Generate feature names file from trained model."""
    
    # Load the trained pipeline
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        print("❌ No trained model found. Please train a model first.")
        return False
    
    try:
        pipeline = joblib.load(model_path)
        
        # Try to get feature names from the pipeline
        # Check if pipeline has feature_names_in_ attribute
        if hasattr(pipeline, 'feature_names_in_'):
            feature_names = list(pipeline.feature_names_in_)
        elif hasattr(pipeline, 'named_steps'):
            # For sklearn pipelines, get from last step
            last_step = list(pipeline.named_steps.values())[-1]
            if hasattr(last_step, 'feature_names_in_'):
                feature_names = list(last_step.feature_names_in_)
            else:
                print("❌ Cannot extract feature names from pipeline")
                return False
        else:
            print("❌ Cannot extract feature names from pipeline")
            return False
            
        # Save feature names
        feature_names_path = Path("models/feature_names.txt")
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        print(f"✅ Generated feature_names.txt with {len(feature_names)} features")
        print(f"📁 Saved to: {feature_names_path}")
        
        # Show first few feature names
        print(f"📋 First 10 features: {feature_names[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating feature names: {e}")
        return False

if __name__ == "__main__":
    generate_feature_names()
