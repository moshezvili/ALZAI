#!/bin/bash
# Docker build and test script

set -e  # Exit on any error

echo "🐳 Docker Build and Test Script"
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to docker directory
cd "$(dirname "$0")"

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Build images
echo "🔨 Building Docker images..."
docker-compose build --no-cache

# Test training container syntax
echo "🧪 Testing training container..."
docker-compose run --rm clinical-ml-training python --version

# Test serving container syntax  
echo "🧪 Testing serving container..."
docker-compose run --rm clinical-ml-serving python --version

# Check if config files are accessible
echo "📋 Checking configuration files..."
docker-compose run --rm clinical-ml-training ls -la config/

# Quick training test (small dataset)
echo "🎯 Quick training test..."
docker-compose run --rm clinical-ml-training python -c "
import sys
sys.path.append('/app')
from src.data_generation.generate_clinical_data import ClinicalDataGenerator
from src.pipeline.training_pipeline import ClinicalMLPipeline
import yaml
import tempfile
from pathlib import Path

print('✅ Imports successful')

# Generate tiny dataset
generator = ClinicalDataGenerator(seed=42)
df = generator.generate_dataset(num_patients=10, years_per_patient=1)
print(f'✅ Generated {len(df)} records')

# Test config loading
with open('/app/config/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Config loaded successfully')

print('🎉 Training container test passed!')
"

# Test serving container imports
echo "🎯 Quick serving test..."
docker-compose run --rm clinical-ml-serving python -c "
import sys
sys.path.append('/app')
from src.serving.api import create_app
import yaml

print('✅ API imports successful')

# Test config loading
try:
    with open('/app/config/serving_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('✅ Serving config loaded')
except FileNotFoundError:
    print('⚠️ Serving config not found, using defaults')

# Test app creation
app = create_app()
print('✅ FastAPI app created successfully')

print('🎉 Serving container test passed!')
"

echo ""
echo "✅ All Docker tests passed!"
echo ""
echo "🚀 To run the full pipeline:"
echo "   docker-compose up"
echo ""
echo "🔍 To run in background:"
echo "   docker-compose up -d"
echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f"
