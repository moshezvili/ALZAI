#!/bin/bash
# GitHub Container Registry Upload Script for ALZAI
# 
# Before running this script:
# 1. Create a GitHub Personal Access Token (PAT) with 'write:packages' permission
# 2. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
# 3. Generate new token with 'write:packages' and 'read:packages' scopes
#
# Usage:
# ./upload_to_ghcr.sh

echo "=== ALZAI Docker Images Upload to GitHub Container Registry ==="
echo ""

# Check if images exist
echo "Checking if images are built..."
if ! docker image inspect ghcr.io/moshezvili/alzai-training:latest > /dev/null 2>&1; then
    echo "❌ Training image not found. Building now..."
    docker build -f docker/Dockerfile.training -t ghcr.io/moshezvili/alzai-training:latest .
else
    echo "✅ Training image found"
fi

if ! docker image inspect ghcr.io/moshezvili/alzai-serving:latest > /dev/null 2>&1; then
    echo "❌ Serving image not found. Building now..."
    docker build -f docker/Dockerfile.serving -t ghcr.io/moshezvili/alzai-serving:latest .
else
    echo "✅ Serving image found"
fi

echo ""
echo "=== Authentication ==="
echo "Please enter your GitHub Personal Access Token when prompted for password:"
echo "Login to GitHub Container Registry..."
docker login ghcr.io -u moshezvili

echo ""
echo "=== Pushing Images ==="
echo "Pushing training image..."
docker push ghcr.io/moshezvili/alzai-training:latest

echo "Pushing serving image..."
docker push ghcr.io/moshezvili/alzai-serving:latest

echo ""
echo "=== Upload Complete ==="
echo "Your images are now available at:"
echo "📦 Training: ghcr.io/moshezvili/alzai-training:latest"
echo "📦 Serving:  ghcr.io/moshezvili/alzai-serving:latest"
echo ""
echo "To use these images:"
echo "docker pull ghcr.io/moshezvili/alzai-training:latest"
echo "docker pull ghcr.io/moshezvili/alzai-serving:latest"
