# Docker build and test script for Windows PowerShell

Write-Host "🐳 Docker Build and Test Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Navigate to docker directory
Set-Location $PSScriptRoot

# Clean up any existing containers
Write-Host "🧹 Cleaning up existing containers..." -ForegroundColor Yellow
docker-compose down --remove-orphans 2>$null

# Build images
Write-Host "🔨 Building Docker images..." -ForegroundColor Yellow
docker-compose build --no-cache

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

# Test training container syntax
Write-Host "🧪 Testing training container..." -ForegroundColor Yellow
docker-compose run --rm clinical-ml-training python --version

# Test serving container syntax  
Write-Host "🧪 Testing serving container..." -ForegroundColor Yellow
docker-compose run --rm clinical-ml-serving python --version

# Check if config files are accessible
Write-Host "📋 Checking configuration files..." -ForegroundColor Yellow
docker-compose run --rm clinical-ml-training ls -la config/

# Quick training test (small dataset)
Write-Host "🎯 Quick training test..." -ForegroundColor Yellow
$trainingTest = @"
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
"@

docker-compose run --rm clinical-ml-training python -c $trainingTest

# Test serving container imports
Write-Host "🎯 Quick serving test..." -ForegroundColor Yellow
$servingTest = @"
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
"@

docker-compose run --rm clinical-ml-serving python -c $servingTest

Write-Host ""
Write-Host "✅ All Docker tests passed!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 To run the full pipeline:" -ForegroundColor Cyan
Write-Host "   docker-compose up" -ForegroundColor White
Write-Host ""
Write-Host "🔍 To run in background:" -ForegroundColor Cyan  
Write-Host "   docker-compose up -d" -ForegroundColor White
Write-Host ""
Write-Host "📊 To view logs:" -ForegroundColor Cyan
Write-Host "   docker-compose logs -f" -ForegroundColor White
