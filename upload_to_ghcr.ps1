# GitHub Container Registry Upload Script for ALZAI
# 
# Before running this script:
# 1. Create a GitHub Personal Access Token (PAT) with 'write:packages' permission
# 2. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
# 3. Generate new token with 'write:packages' and 'read:packages' scopes
#
# Usage:
# .\upload_to_ghcr.ps1

Write-Host "=== ALZAI Docker Images Upload to GitHub Container Registry ===" -ForegroundColor Cyan
Write-Host ""

# Check if images exist
Write-Host "Checking if images are built..." -ForegroundColor Yellow
try {
    docker image inspect ghcr.io/moshezvili/alzai-training:latest | Out-Null
    Write-Host "✅ Training image found" -ForegroundColor Green
} catch {
    Write-Host "❌ Training image not found. Building now..." -ForegroundColor Red
    docker build -f docker/Dockerfile.training -t ghcr.io/moshezvili/alzai-training:latest .
}

try {
    docker image inspect ghcr.io/moshezvili/alzai-serving:latest | Out-Null
    Write-Host "✅ Serving image found" -ForegroundColor Green
} catch {
    Write-Host "❌ Serving image not found. Building now..." -ForegroundColor Red
    docker build -f docker/Dockerfile.serving -t ghcr.io/moshezvili/alzai-serving:latest .
}

Write-Host ""
Write-Host "=== Authentication ===" -ForegroundColor Cyan
Write-Host "Please enter your GitHub Personal Access Token when prompted for password:" -ForegroundColor Yellow
Write-Host "Login to GitHub Container Registry..." -ForegroundColor Yellow
docker login ghcr.io -u moshezvili

Write-Host ""
Write-Host "=== Pushing Images ===" -ForegroundColor Cyan
Write-Host "Pushing training image..." -ForegroundColor Yellow
docker push ghcr.io/moshezvili/alzai-training:latest

Write-Host "Pushing serving image..." -ForegroundColor Yellow
docker push ghcr.io/moshezvili/alzai-serving:latest

Write-Host ""
Write-Host "=== Upload Complete ===" -ForegroundColor Green
Write-Host "Your images are now available at:" -ForegroundColor Green
Write-Host "📦 Training: ghcr.io/moshezvili/alzai-training:latest" -ForegroundColor Cyan
Write-Host "📦 Serving:  ghcr.io/moshezvili/alzai-serving:latest" -ForegroundColor Cyan
Write-Host ""
Write-Host "To use these images:" -ForegroundColor Green
Write-Host "docker pull ghcr.io/moshezvili/alzai-training:latest" -ForegroundColor White
Write-Host "docker pull ghcr.io/moshezvili/alzai-serving:latest" -ForegroundColor White
