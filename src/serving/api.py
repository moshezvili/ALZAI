"""
FastAPI serving endpoint for clinical ML model.

This module provides a REST API for model inference with proper
validation, error handling, and monitoring capabilities.

Key Features:
- Automatic feature alignment with training data
- Comprehensive input validation  
- Batch prediction support
- Health monitoring endpoints
- Debug endpoints for troubleshooting

Feature Synchronization:
The API automatically aligns input features with the exact feature set
used during model training by using df.reindex(columns=feature_names, fill_value=0.0)
to ensure prediction compatibility.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from contextlib import asynccontextmanager
import re
import pandas as pd
import numpy as np
import joblib
import yaml
import logging
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and config
model = None
threshold = 0.5
feature_names = []
config = {}

class PatientData(BaseModel):
    """Input schema for patient data."""
    
    patient_id: str = Field(..., description="Unique patient identifier")
    year: int = Field(..., ge=2000, le=2030, description="Year of record")
    age: float = Field(..., ge=0, le=120, description="Patient age")
    gender: str = Field(..., description="Patient gender (M/F)")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    systolic_bp: float = Field(..., ge=70, le=250, description="Systolic blood pressure")
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic blood pressure")
    cholesterol: float = Field(..., ge=100, le=500, description="Cholesterol level")
    glucose: float = Field(..., ge=50, le=500, description="Glucose level")
    smoking_status: str = Field(..., description="Smoking status")
    num_visits: int = Field(..., ge=0, le=50, description="Number of healthcare visits")
    medications_count: int = Field(..., ge=0, le=20, description="Number of medications")
    lab_abnormal_flag: bool = Field(..., description="Lab abnormality flag")
    primary_diagnosis: str = Field(..., description="Primary diagnosis code")
    additional_diagnoses: Optional[str] = Field("", description="Additional diagnosis codes")
    
    # Optional derived fields that might be in training data
    age_group: Optional[str] = Field(None, description="Age group category")
    bmi_category: Optional[str] = Field(None, description="BMI category")
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        """Validate gender field."""
        if v not in ['M', 'F']:
            raise ValueError("Gender must be M or F")
        return v
    
    @field_validator('smoking_status')
    @classmethod
    def validate_smoking_status(cls, v):
        """Validate smoking status field."""
        if v not in ['Never', 'Former', 'Current']:
            raise ValueError("Smoking status must be Never, Former, or Current")
        return v
    
    @field_validator('diastolic_bp')
    @classmethod
    def validate_blood_pressure(cls, v, info):
        """Validate blood pressure consistency."""
        # Get systolic BP from the data being validated
        if hasattr(info, 'data') and 'systolic_bp' in info.data:
            systolic_bp = info.data['systolic_bp']
            if v >= systolic_bp:  # diastolic >= systolic
                raise ValueError("Diastolic BP must be less than systolic BP")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "P000001",
                "year": 2023,
                "age": 65.5,
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
                "primary_diagnosis": "I10",
                "additional_diagnoses": "E78"
            }
        }
    }

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    
    patient_id: str = Field(..., description="Patient identifier")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0/1)")
    confidence: str = Field(..., description="Confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version")
    
class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    
    patients: List[PatientData] = Field(..., description="List of patient data")
    
class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of patients processed")
    processing_time_seconds: float = Field(..., description="Total processing time")

# Global startup time for uptime calculation
startup_time = time.time()

def load_model_artifacts():
    """Load model and related artifacts."""
    global model, threshold, feature_names, config
    
    try:
        model_dir = Path(os.getenv('MODEL_DIR', './models'))
        
        # Load model
        model_path = model_dir / 'best_model.joblib'
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load threshold
        threshold_path = model_dir / 'optimal_threshold.txt'
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
            logger.info(f"Threshold loaded: {threshold}")
        
        # Load feature names
        features_path = model_dir / 'feature_names.txt'
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"Feature names loaded: {len(feature_names)} features")
        else:
            logger.warning("feature_names.txt not found - feature alignment will be disabled")
            feature_names = []
        
        # Load config
        config_path = model_dir / 'training_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Training config loaded")
    
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting Clinical ML Prediction API...")
    load_model_artifacts()
    logger.info("API startup completed")
    yield
    # Shutdown (if needed)
    logger.info("API shutdown completed")

def create_app() -> FastAPI:
    """Create FastAPI application."""
    
    # Load serving configuration
    serving_config_path = Path('./config/serving_config.yaml')
    if serving_config_path.exists():
        with open(serving_config_path, 'r') as f:
            serving_config = yaml.safe_load(f)
    else:
        serving_config = {}
    
    api_config = serving_config.get('api', {})
    
    app = FastAPI(
        title=api_config.get('title', 'Clinical ML Prediction API'),
        description=api_config.get('description', 'Binary classification for clinical diagnosis prediction'),
        version=api_config.get('version', '1.0.0'),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"], 
        allow_headers=["*"],
    )
    
    return app

# Create app instance
app = create_app()

def preprocess_input(patient_data: PatientData) -> pd.DataFrame:
    """Preprocess input data to match training format."""
    
    # Convert to dictionary
    data_dict = patient_data.model_dump()
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Add birth_year feature (critical for model compatibility)
    df['birth_year'] = df['year'] - df['age']
    
    # Add derived features if missing
    if 'age_group' not in df.columns or df['age_group'].isna().any():
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100],
                               labels=['18-30', '31-45', '46-60', '61-75', '75+'])
        df['age_group'] = df['age_group'].astype(str)
    
    if 'bmi_category' not in df.columns or df['bmi_category'].isna().any():
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['bmi_category'] = df['bmi_category'].astype(str)
    
    # Handle missing features that might exist in training data
    # Add noise features if they were in training (they should be filtered out by preprocessing)
    for i in range(10):
        noise_col = f'noise_feature_{i}'
        if noise_col not in df.columns:
            df[noise_col] = 0.0  # Will be filtered out anyway
    
    # CRITICAL: Ensure exact feature alignment with training data
    if feature_names:
        logger.debug(f"Aligning features to training format. Expected {len(feature_names)} features.")
        
        # Store original columns for logging
        original_columns = set(df.columns)
        
        # Reindex to match exact training features, filling missing with appropriate defaults
        df = df.reindex(columns=feature_names, fill_value=0.0)
        
        # Log alignment results
        missing_features = [col for col in feature_names if col not in original_columns]
        extra_features = [col for col in original_columns if col not in feature_names]
        
        if missing_features:
            logger.info(f"Filled {len(missing_features)} missing features with defaults: {missing_features[:5]}...")
        
        if extra_features:
            logger.info(f"Dropped {len(extra_features)} extra features: {extra_features[:5]}...")
            
        logger.debug(f"Feature alignment completed. Final shape: {df.shape}")
    else:
        logger.warning("No feature_names available - prediction may fail due to feature mismatch")
    
    return df

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability < 0.2 or probability > 0.8:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """Make prediction for a single patient."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        df = preprocess_input(patient_data)
        
        # Validate feature alignment before prediction
        if feature_names and len(df.columns) != len(feature_names):
            raise ValueError(f"Feature count mismatch: got {len(df.columns)}, expected {len(feature_names)}")
        
        # Make prediction
        start_time = time.time()
        
        # Get probability
        probability = model.predict_proba(df)[0, 1]
        
        # Apply threshold
        prediction = 1 if probability >= threshold else 0
        
        processing_time = time.time() - start_time
        
        # Get confidence level
        confidence = get_confidence_level(probability)
        
        logger.info(f"Prediction for {patient_data.patient_id}: "
                   f"prob={probability:.3f}, pred={prediction}, "
                   f"time={processing_time:.3f}s")
        
        return PredictionResponse(
            patient_id=patient_data.patient_id,
            probability=float(probability),
            prediction=prediction,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version=config.get('model', {}).get('algorithm', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Prediction error for {patient_data.patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple patients."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.patients) > 1000:
        raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")
    
    try:
        start_time = time.time()
        predictions = []
        
        for patient_data in request.patients:
            try:
                # Preprocess input
                df = preprocess_input(patient_data)
                
                # Validate feature alignment before prediction
                if feature_names and len(df.columns) != len(feature_names):
                    raise ValueError(f"Feature count mismatch: got {len(df.columns)}, expected {len(feature_names)}")
                
                # Make prediction
                probability = model.predict_proba(df)[0, 1]
                prediction = 1 if probability >= threshold else 0
                confidence = get_confidence_level(probability)
                
                predictions.append(PredictionResponse(
                    patient_id=patient_data.patient_id,
                    probability=float(probability),
                    prediction=prediction,
                    confidence=confidence,
                    timestamp=datetime.now().isoformat(),
                    model_version=config.get('model', {}).get('algorithm', 'unknown')
                ))
                
            except Exception as e:
                logger.error(f"Error processing patient {patient_data.patient_id}: {e}")
                # Still add a response with error indication
                predictions.append(PredictionResponse(
                    patient_id=patient_data.patient_id,
                    probability=0.0,
                    prediction=0,
                    confidence="Error",
                    timestamp=datetime.now().isoformat(),
                    model_version="error"
                ))
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch prediction completed: {len(predictions)} patients "
                   f"in {processing_time:.3f}s")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "threshold": threshold,
        "num_features": len(feature_names) if feature_names else "unknown",
        "feature_names": feature_names[:20] if feature_names else [],  # Show first 20 features
        "total_features": len(feature_names) if feature_names else 0,
        "algorithm": config.get('model', {}).get('algorithm', 'unknown'),
        "training_config": config
    }
    
    return info

@app.get("/model/features")
async def get_model_features():
    """Get complete list of model features."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not feature_names:
        raise HTTPException(status_code=404, detail="Feature names not available")
    
    return {
        "total_features": len(feature_names),
        "feature_names": feature_names,
        "sample_input_format": {
            "patient_id": "P000001",
            "year": 2023,
            "age": 65.5,
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
            "primary_diagnosis": "I10",
            "additional_diagnoses": "E78"
        }
    }

@app.post("/debug/preprocess")
async def debug_preprocess(patient_data: PatientData):
    """Debug endpoint to check feature preprocessing without making prediction."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        df = preprocess_input(patient_data)
        
        return {
            "patient_id": patient_data.patient_id,
            "preprocessed_shape": df.shape,
            "preprocessed_columns": list(df.columns),
            "expected_features": len(feature_names) if feature_names else "unknown",
            "feature_alignment": "✅ Aligned" if feature_names and len(df.columns) == len(feature_names) else "❌ Misaligned",
            "sample_values": df.iloc[0].to_dict() if not df.empty else {}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": f"Validation error: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def main():
    """Main function to run the API server."""
    
    # Load serving configuration
    serving_config_path = Path('./config/serving_config.yaml')
    if serving_config_path.exists():
        with open(serving_config_path, 'r') as f:
            serving_config = yaml.safe_load(f)
    else:
        serving_config = {}
    
    server_config = serving_config.get('serving', {})
    
    uvicorn.run(
        "src.serving.api:app",
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8000),
        reload=server_config.get('reload', False),
        workers=server_config.get('workers', 1)
    )

if __name__ == "__main__":
    main()
