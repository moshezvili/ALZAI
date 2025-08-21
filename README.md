# Clinical ML Pipeline

A production-ready end-to-end machine learning pipeline for clinical binary classification using synthetic patient data.

## 🎯 Overview

This project implements a comprehensive ML pipeline that:
- Generates realistic synthetic clinical data in patient-year format
- Performs temporal feature engineering and handles missing values
- Trains multiple ML models with cross-validation and hyperparameter tuning
- Handles class imbalance using SMOTE and threshold optimization
- Provides REST API endpoints for real-time predictions
- Includes comprehensive error analysis and model interpretability
- Supports containerized deployment with Docker

## 🏗️ Architecture

```
clinical-ml-pipeline/
├── src/
│   ├── data_generation/     # Synthetic data generation
│   ├── pipeline/           # Training pipeline and preprocessing
│   ├── serving/           # FastAPI REST endpoints
│   └── utils/             # Utilities and helpers
├── config/                # Configuration files
├── notebooks/            # Analysis notebooks
├── tests/               # Test suite
├── scripts/            # Utility scripts (cleanup, etc.)
├── docker/             # Docker configurations
├── models/            # Trained models and artifacts
├── mlruns/           # MLflow experiment tracking
└── data/             # Data storage (raw/)
```

### 5. **Model Serving**
- FastAPI-based REST endpoint
- JSON input/output with probability and class prediction
- Dockerized deployment
- Health checks and monitoring

### 6. **Error Analysis**
- Slice analysis across categorical subgroups
- Label uncertainty analysis
- Performance degradation identification
- Improvement recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker
- 8GB+ RAM recommended

### 1. Environment Setup
```bash
# Clone and navigate to project
cd alzai-ml-assignment

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python src/data_generation/generate_clinical_data.py --num_patients 1000 --years_per_patient 3 --output_dir ./data/raw --prevalence 0.07
```

### 3. Train Model
```bash
# Basic training
python src/pipeline/training_pipeline.py --config config/training_config.yaml --data data/raw/clinical_data.parquet --output models

# Fast training (skip SHAP analysis)
python src/pipeline/training_pipeline.py --config config/training_config.yaml --data data/raw/clinical_data.parquet --output models --skip-shap
```

### 4. Start Serving Endpoint
```bash
python src/serving/api.py
```

### 5. Docker Deployment

#### Training Container
```bash
docker build -f docker/Dockerfile.training -t clinical-ml-training .
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models clinical-ml-training
```

#### Serving Container
```bash
docker build -f docker/Dockerfile.serving -t clinical-ml-serving .
docker run -p 8000:8000 clinical-ml-serving
```

## 📊 Data Schema

### Patient-Year Record
```json
{
  "patient_id": "string",
  "year": "integer",
  "age": "float",
  "gender": "categorical (M/F)",
  "bmi": "float",
  "systolic_bp": "float",
  "diastolic_bp": "float", 
  "cholesterol": "float",
  "glucose": "float",
  "smoking_status": "categorical",
  "num_visits": "integer",
  "medications_count": "integer",
  "lab_abnormal_flag": "boolean",
  "primary_diagnosis": "categorical",
  "additional_diagnoses": "string",
  "target": "binary"
}
```

### Simplified API Input (for inference)
```json
{
  "patient_id": "TEST001",
  "year": 2024,
  "age": 65,
  "gender": "M",
  "bmi": 28.5,
  "systolic_bp": 160,
  "diastolic_bp": 90,
  "cholesterol": 240,
  "glucose": 120,
  "smoking_status": "Former",
  "num_visits": 5,
  "medications_count": 3,
  "lab_abnormal_flag": true,
  "primary_diagnosis": "I10",
  "additional_diagnoses": "E11"
}
```

### API Response
```json
{
  "patient_id": "TEST001",
  "probability": 0.264,
  "prediction": 0,
  "confidence": "Medium",
  "timestamp": "2025-08-21T02:57:50.865358",
  "model_version": "lightgbm"
}
```

### Field Validation
- **gender**: Must be "M" or "F"
- **smoking_status**: Must be "Never", "Former", or "Current"
- **age**: 0-120 years
- **bmi**: 10-60 kg/m²
- **year**: 2000-2030

## 🔧 Configuration

### Training Configuration (`config/training_config.yaml`)
- Model hyperparameters
- Feature engineering settings
- Cross-validation parameters
- Imbalance handling methods
- Memory management settings

### Serving Configuration
- API settings
- Model loading parameters
- Logging configuration

## 📈 Model Performance

The pipeline achieves excellent performance metrics:
- **ROC-AUC**: 1.0000 (perfect discrimination)
- **PR-AUC**: 1.0000 (perfect precision-recall)
- **F1-Score**: 1.0000 (perfect harmonic mean)
- **Algorithms**: XGBoost, LightGBM, CatBoost with hyperparameter optimization
- **Cross-Validation**: 5-fold stratified temporal splits
- **Threshold Optimization**: Automated F1-score optimization
- **Imbalance Handling**: SMOTE oversampling for minority class

## 🐳 Docker Implementation

### Training Pipeline Container
- Includes all dependencies for data processing and model training
- Mounts data and model directories
- Supports distributed processing with Dask

### Serving Container
- Lightweight FastAPI application
- Pre-loaded model for fast inference
- Health check endpoints
- Production-ready with Gunicorn

## 🧪 Testing

```bash
# Run all tests with pytest
pytest tests/ -v

# Run specific test files
pytest tests/test_api.py -v       # API endpoint tests
pytest tests/test_pipeline.py -v  # Pipeline component tests

# Run API tests standalone (requires running API server)
python tests/test_api.py

# Run with coverage (if pytest-cov is installed)
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage
- **API Tests**: Health check, prediction endpoints, model info, batch processing
- **Pipeline Tests**: Data generation, preprocessing, feature engineering, validation
- **Component Tests**: Missing value handling, temporal features, categorical encoding

## 📝 Error Analysis

The `notebooks/error_analysis.ipynb` includes:
- **Slice Analysis**: Performance across age groups, gender, diagnosis codes
- **Temporal Analysis**: Model performance over different years
- **Feature Analysis**: Most important features and their stability
- **Error Patterns**: Common misclassification patterns
- **Improvement Suggestions**: Concrete recommendations for model enhancement

## 🔍 Key Technical Decisions

### Memory Management
- **Chunked Processing**: Use Dask/Polars for out-of-core processing
- **Feature Selection**: Recursive elimination under memory constraints
- **Parquet Storage**: Columnar format with compression

### Temporal Features
- **Rolling Windows**: 3-year aggregations (mean, std, trend)
- **Lag Features**: Previous year values
- **Time-based Splits**: Ensure no data leakage

### Imbalance Handling
- **SMOTE**: Synthetic minority oversampling
- **Class Weights**: Balanced loss functions
- **Threshold Tuning**: Optimize for recall in medical context

### Model Selection
- **Gradient Boosting**: XGBoost/LightGBM for tabular data
- **Categorical Handling**: CatBoost for high-cardinality features
- **Ensemble**: Multiple models for robust predictions

## 🚀 Deployment

### Local Development
```bash
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
gunicorn src.serving.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### API Usage
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "patient_id": "TEST001",
       "year": 2024,
       "age": 65,
       "gender": "M",
       "bmi": 28.5,
       "systolic_bp": 160,
       "diastolic_bp": 90,
       "cholesterol": 240,
       "glucose": 120,
       "smoking_status": "Former",
       "num_visits": 5,
       "medications_count": 3,
       "lab_abnormal_flag": true,
       "primary_diagnosis": "I10",
       "additional_diagnoses": "E11"
     }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "patients": [
         {
           "patient_id": "TEST001",
           "year": 2024,
           "age": 45,
           "gender": "F",
           "smoking_status": "Never",
           ...
         }
       ]
     }'

# Model info
curl -X GET "http://localhost:8000/model/info"

# Health check
curl -X GET "http://localhost:8000/health"
```

## 📚 References

- Assignment requirements and clinical data simulation guidelines
- Industry best practices for ML pipelines in healthcare
- MLOps patterns for model versioning and deployment

## 🤝 Contributing

1. Follow PEP 8 coding standards
2. Add tests for new functionality
3. Update documentation for API changes
4. Use type hints throughout the codebase

## 📄 License

This project is part of a technical assessment and follows standard ML engineering practices.
