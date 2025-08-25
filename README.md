# Clinical ML Pipeline

A production-ready end-to-end machine learning pipeline for clinical binary classification using synthetic patient data.

## 🎯 Overview

This project implements a comprehensive ML pipeline that:
- Generates realistic synthetic clinical data in patient-year format
- Performs temporal feature engineering with Dask-based distributed processing
- Trains multiple ML models with cross-validation and hyperparameter tuning
- Handles class imbalance using configurable SMOTE and threshold optimization
- Provides REST API endpoints for real-time predictions
- Includes optional SHAP-based model interpretability (disabled by default for performance)
- Supports containerized deployment with Docker and MLflow experiment tracking

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

## ✨ Key Features

- **🚀 Distributed Processing**: Dask-powered data loading and preprocessing
- **⚖️ Class Imbalance Handling**: Configurable SMOTE with intelligent sampling
- **🎯 Smart Feature Selection**: Recursive feature elimination and importance-based selection  
- **🔄 Temporal Cross-Validation**: Time-aware splits with configurable gaps
- **🎛️ Hyperparameter Optimization**: Optuna-based HPO with pruning
- **📊 Model Interpretability**: Optional SHAP explanations (disabled by default for performance)
- **🐳 Containerized Deployment**: Docker containers for training and serving
- **📈 Experiment Tracking**: MLflow integration for model versioning
- **💾 Memory Optimization**: Support for resource-constrained environments
- **🌐 REST API**: FastAPI-based serving with comprehensive validation

### 5. **Model Serving**
- FastAPI-based REST endpoint with comprehensive input validation
- JSON input/output with probability scores and confidence levels
- Dockerized deployment with MLflow model loading
- Health checks, model info endpoints, and batch processing support

### 6. **Error Analysis**
- Slice analysis across categorical subgroups
- Label uncertainty analysis
- Performance degradation identification
- Improvement recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)
- 4GB+ RAM (8GB+ recommended for full dataset)
- MLflow for experiment tracking

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
# Basic training (SHAP disabled by default for faster training)
python src/pipeline/training_pipeline.py --config config/training_config.yaml --data data/raw/clinical_data.parquet --output models

# Training with SHAP explanations (slower)
python src/pipeline/training_pipeline.py --config config/training_config.yaml --data data/raw/clinical_data.parquet --output models --enable-shap

# Using test data for quick experiments
python src/pipeline/training_pipeline.py --config config/training_config.yaml --data data/test/small_clinical_data.parquet --output models/test_run
```

**Available CLI Options:**
- `--config`: Path to training configuration YAML file (required)
- `--data`: Path to training data (parquet file or directory) (required)  
- `--output`: Output directory for trained models and artifacts (required)
- `--enable-shap`: Enable SHAP explanations (optional, disabled by default for faster training)

### 4. Start Serving Endpoint
```bash
# Set model directory environment variable
set MODEL_DIR=models/test_run_mlflow  # Windows
# export MODEL_DIR=models/test_run_mlflow  # Linux/Mac

# Start API server
python -m src.serving.api
```

### 5. Docker Deployment

#### Full Pipeline Testing (Recommended)
```bash
# Test complete pipeline with small dataset
cd docker
docker compose up --build

# This will:
# 1. Generate small test dataset (100 patients, 2 years)
# 2. Train model with optimized configuration
# 3. Start serving API on http://localhost:8000
# 4. Start MLflow server on http://localhost:5000
```

#### Using Pre-built Images from GitHub Container Registry
```bash
# Pull and run pre-built images (faster startup)
cd docker
docker compose -f docker-compose.ghcr.yml up

# Or pull images individually
docker pull ghcr.io/moshezvili/alzai-training:latest
docker pull ghcr.io/moshezvili/alzai-serving:latest
```

#### Individual Containers
```bash
# Training container only
docker compose up clinical-ml-training --build

# Serving container only (requires pre-trained model)
docker compose up clinical-ml-serving --build

# Stop all containers
docker compose down
```

#### Manual Docker Commands
```bash
# Build training container
docker build -f docker/Dockerfile.training -t clinical-ml-training .

# Run training with data mounting
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models clinical-ml-training

# Build and run serving container
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
```yaml
# Distributed processing settings
processing:
  sample_fraction: 0.2  # Use subset for faster training
  dask_config:
    n_workers: 1
    threads_per_worker: 2
    memory_limit: "512MB"

# Class imbalance handling
imbalance:
  method: "smote"  # or "none"
  smote:
    sampling_strategy: 0.1
    k_neighbors: 5
    random_state: 42

# Feature selection
feature_selection:
  method: "recursive_elimination"
  max_features: 50

# Hyperparameter optimization
hpo:
  enabled: true
  n_trials: 30
  algorithms: ["lightgbm", "xgboost", "catboost"]

# Cross validation
cross_validation:
  n_splits: 5
  test_size: 0.2
  gap: 1  # temporal gap between train/validation
```

### Free Server Configuration (`config/training_config_free_server.yaml`)
- Ultra-conservative memory settings (128MB)
- Single-threaded processing
- Optimized for resource-constrained environments

### Environment Variables
- `MODEL_DIR`: Directory containing trained model artifacts (required for serving)
- `MLFLOW_TRACKING_URI`: MLflow server URI (optional, defaults to local mlruns/)

### Serving Configuration
- API settings for FastAPI server
- Model loading and validation parameters
- Logging and monitoring configuration

## 🐳 Docker Implementation

### Training Pipeline Container
- Includes all dependencies for data processing and model training
- Mounts data and model directories for persistence
- Supports distributed processing with Dask
- Uses optimized configuration for containerized environments
- Automatically generates small test dataset (100 patients, 2 years)

### Serving Container  
- Lightweight FastAPI application with full dependency management
- Pre-loaded model artifacts for fast inference
- Health check and batch prediction endpoints
- Production-ready with comprehensive input validation
- Supports both single and batch prediction workflows

### Docker Compose Setup
- **clinical-ml-training**: Data generation and model training
- **clinical-ml-serving**: REST API serving with model inference
- **mlflow-server**: Experiment tracking and model registry
- **Shared network**: Container communication and data persistence

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

### Docker Pipeline Testing
```bash
# Test complete containerized pipeline
cd docker
docker compose up --build

# Validate endpoints while containers are running
curl -X GET "http://localhost:8000/health"
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @test_payload.json
```

### Test Coverage
- **API Tests**: Health check, prediction endpoints, model info, batch processing
- **Pipeline Tests**: Data generation, preprocessing, feature engineering, validation
- **Component Tests**: Missing value handling, temporal features, categorical encoding
- **Docker Tests**: End-to-end containerized pipeline validation

## 📝 Error Analysis

The comprehensive error analysis addresses assignment requirements with detailed slice analysis and improvement recommendations, available in two formats:

### Interactive Notebook: `notebooks/error_analysis.ipynb`
- **Slice Analysis**: Performance across age groups, gender, diagnosis codes, smoking status
- **Temporal Analysis**: Model performance over different years  
- **Feature Analysis**: Most important features and their stability
- **Error Patterns**: Common misclassification patterns and probability distributions
- **Calibration Assessment**: Model probability reliability analysis
- **Visualization**: ROC curves, confusion matrices, and performance plots

### Summary Report: `ERROR_ANALYSIS_REPORT.md`
- **Executive Summary**: Key findings and recommendations
- **Improvement Roadmap**: Short, medium, and long-term action items
- **Success Metrics**: Performance targets and monitoring guidelines
- **Technical Implementation**: Code integration and deployment considerations

#### Key Findings from Analysis
- **Perfect Performance**: Model achieves near-perfect metrics (ROC-AUC: 0.9994) indicating potential overfitting
- **Temporal Features Dominance**: Rolling mean features show highest importance 
- **Data Leakage Concerns**: Noise features ranking high suggest possible data leakage
- **Uniform Slice Performance**: Consistent high performance across all demographic subgroups

#### Running Error Analysis
```bash
# Open the interactive notebook
cd notebooks/
jupyter notebook error_analysis.ipynb

# Or review the markdown report
cat ERROR_ANALYSIS_REPORT.md
```

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
# Set model directory
set MODEL_DIR=models/test_run_mlflow  # Windows
# export MODEL_DIR=models/test_run_mlflow  # Linux/Mac

# Start development server
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

## 📄 License

This project is part of a technical assessment and follows standard ML engineering practices.
