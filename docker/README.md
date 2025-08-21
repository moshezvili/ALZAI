# Docker Deployment Guide

This directory contains Docker configurations for deploying the Clinical ML Pipeline.

## 🐳 Services

### 1. Training Service (`clinical-ml-training`)
- **Purpose**: Generate synthetic data and train ML models
- **Dockerfile**: `Dockerfile.training`
- **Process**: 
  1. Generates synthetic clinical data
  2. Trains the ML model using the pipeline
  3. Saves model artifacts to the `models/` directory

### 2. Serving Service (`clinical-ml-serving`)
- **Purpose**: Serve trained models via REST API
- **Dockerfile**: `Dockerfile.serving`  
- **Endpoints**: 
  - `GET /health` - Health check
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions
  - `GET /model/info` - Model metadata

### 3. MLflow Server (`mlflow-server`) [Optional]
- **Purpose**: Experiment tracking and model registry
- **Access**: http://localhost:5000
- **Data**: Stored in `mlruns/` directory

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB available RAM
- 2GB free disk space

### 1. Build and Run All Services
```bash
cd docker
docker-compose up --build
```

### 2. Wait for Training to Complete
The training service will:
- Generate 50,000 synthetic patient records
- Train and evaluate the ML model
- Save artifacts to `../models/`

### 3. Access the API
Once training completes, the serving API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Test Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "lab_abnormal_flag": true,
    "primary_diagnosis": "I10",
    "additional_diagnoses": "E78"
  }'
```

## 🔧 Configuration

### Environment Variables
- `MODEL_DIR`: Directory containing trained models (default: `/app/models`)
- `PYTHONPATH`: Python path for imports (set to `/app`)
- `MLFLOW_TRACKING_URI`: MLflow tracking URI

### Volume Mounts
- `../data:/app/data` - Data directory
- `../models:/app/models` - Model artifacts
- `../mlruns:/app/mlruns` - MLflow experiments
- `../config:/app/config` - Configuration files

## 🛠 Development

### Run Individual Services

#### Training Only
```bash
docker-compose up clinical-ml-training
```

#### Serving Only (requires pre-trained model)
```bash
docker-compose up clinical-ml-serving
```

#### MLflow Only
```bash
docker-compose up mlflow-server
```

### Custom Configuration
1. Edit `../config/training_config.yaml` for training parameters
2. Edit `../config/serving_config.yaml` for API configuration
3. Restart services: `docker-compose restart`

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f clinical-ml-serving
```

## 📊 Resource Usage

### Typical Resource Requirements:
- **Training**: 2GB RAM, 10-15 minutes for 50k patients
- **Serving**: 512MB RAM, <100ms response time
- **MLflow**: 256MB RAM
- **Disk**: ~500MB for models + data

### Performance Tuning:
- Reduce `num_patients` in config for faster training
- Disable HPO with `hpo.enabled: false` for speed
- Use `skip_shap: true` to skip SHAP explanations

## 🔍 Troubleshooting

### Common Issues:

#### Training Fails
```bash
# Check logs
docker-compose logs clinical-ml-training

# Common fixes:
# 1. Insufficient memory - reduce num_patients
# 2. Missing config - check config/training_config.yaml exists
```

#### Serving API Not Responding
```bash
# Check if training completed first
docker-compose logs clinical-ml-training | grep "Training completed"

# Check serving logs
docker-compose logs clinical-ml-serving

# Check model artifacts exist
ls -la ../models/
```

#### Port Conflicts
If ports 8000 or 5000 are in use:
```yaml
# Edit docker-compose.yml
ports:
  - "8001:8000"  # Change external port
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Check if model loaded
curl http://localhost:8000/model/info
```

## 🔐 Security Notes

- API runs as non-root user (`appuser`)
- No sensitive data in containers (synthetic data only)
- CORS enabled for development (disable in production)
- Health checks implemented for service monitoring

## 📝 Next Steps

1. **Production Deployment**: 
   - Use production WSGI server (Gunicorn)
   - Add authentication/authorization
   - Set up proper logging and monitoring

2. **CI/CD Integration**:
   - Add Docker image building to CI pipeline
   - Implement automated testing
   - Set up deployment automation

3. **Monitoring**:
   - Add Prometheus metrics
   - Set up log aggregation
   - Implement alerting
