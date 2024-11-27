#!/bin/bash
set -e

# Get the current package version
VERSION=$(node -p "require('./package.json').version")

# Configure Docker image details
IMAGE_NAME="ghcr.io/kaiserruben/tokenlense"
LATEST_TAG="$IMAGE_NAME:latest"
VERSION_TAG="$IMAGE_NAME:$VERSION"

# Build ARM64
echo "🏗️  Building Docker image for ARM64..."
docker buildx build --platform linux/arm64 \
    -t "$IMAGE_NAME:${VERSION}-arm64" \
    -t "$IMAGE_NAME:latest-arm64" \
    --push .

# Build AMD64
echo "🏗️  Building Docker image for AMD64..."
docker buildx build --platform linux/amd64 \
    -t "$IMAGE_NAME:${VERSION}-amd64" \
    -t "$IMAGE_NAME:latest-amd64" \
    --push .

# Create and push the combined manifest
echo "📦 Creating and pushing multi-arch manifests..."
docker buildx imagetools create -t $LATEST_TAG \
    "$IMAGE_NAME:latest-arm64" \
    "$IMAGE_NAME:latest-amd64"

docker buildx imagetools create -t $VERSION_TAG \
    "$IMAGE_NAME:${VERSION}-arm64" \
    "$IMAGE_NAME:${VERSION}-amd64"

echo "✅ Successfully built and pushed Docker images"
echo "   Latest tag: $LATEST_TAG"
echo "   Version tag: $VERSION_TAG"
