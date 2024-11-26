#!/bin/bash
set -e

# Get the current package version
VERSION=$(node -p "require('./package.json').version")

# Configure Docker image details
IMAGE_NAME="ghcr.io/kaiserruben/tokenlense"
LATEST_TAG="$IMAGE_NAME:latest"
VERSION_TAG="$IMAGE_NAME:v$VERSION"

# Build the Docker image
echo "🏗️  Building Docker image..."
docker build -t $LATEST_TAG -t $VERSION_TAG .

# Push the Docker image
echo "⬆️  Pushing Docker image..."
docker push $LATEST_TAG
docker push $VERSION_TAG

echo "✅ Successfully built and pushed Docker image"
echo "   Latest tag: $LATEST_TAG"
echo "   Version tag: $VERSION_TAG"