#!/bin/bash
set -e

APP_NAME="ollama-one"
IMAGE_TAG="${APP_NAME}:latest"
TAR_FILE="${APP_NAME}-docker.tar.gz"

echo "=========================================="
echo " Building Ollama-One (Stateless)"
echo "=========================================="

# 1. Build local Go binary
echo "==> Building Go binary (${APP_NAME})..."
CGO_ENABLED=0 go build -o "${APP_NAME}" .
chmod +x "${APP_NAME}"
echo "✓ Built binary: ./${APP_NAME}"

# 2. Build Docker image if not skipped
if [[ "$1" == "--no-docker" ]]; then
    echo "Skipping Docker image build (--no-docker flag passed)."
else
    echo "==> Building stateless Docker image (${IMAGE_TAG})..."
    docker build -t "${IMAGE_TAG}" .

    echo "==> Exporting Docker image to ${TAR_FILE}..."
    docker save "${IMAGE_TAG}" | gzip > "${TAR_FILE}"
    echo "✓ Produced Docker archive: ${TAR_FILE} ($(du -h "${TAR_FILE}" | cut -f1))"
    echo ""
    echo "To load this image on any Docker host:"
    echo "  docker load < ${TAR_FILE}"
    echo ""
    echo "To run statelessly (accepts API keys per request):"
    echo "  docker run -d -p 11434:11434 --name ${APP_NAME} ${IMAGE_TAG}"
fi

# 3. Optional --run flag to execute binary
if [[ "$1" == "--run" ]] || [[ "$2" == "--run" ]]; then
    echo ""
    echo "==> Starting ${APP_NAME}..."
    ./"${APP_NAME}"
fi