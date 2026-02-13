#!/bin/bash
set -e

# FlagCX RPM package build script
# Usage: ./build-flagcx-rpm.sh <backend> [base_image_version]
# Supported backends: nvidia, metax, ascend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKEND="${1:-}"
BASE_IMAGE_VERSION="${2:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Show usage
if [ -z "$BACKEND" ]; then
    log_error "No backend specified"
    echo ""
    echo "Usage: $0 <backend> [base_image_version]"
    echo ""
    echo "Supported backends:"
    echo "  nvidia  - Build RPM packages for NVIDIA GPUs"
    echo "  metax   - Build RPM packages for MetaX accelerators"
    echo "  ascend  - Build RPM packages for Ascend NPUs"
    echo ""
    echo "Examples:"
    echo "  $0 nvidia"
    echo "  $0 ascend 8.5.0-910-openeuler24.03-py3.11"
    exit 1
fi

# Validate backend and set base image
case "$BACKEND" in
    nvidia)
        BASE_IMAGE="nvcr.io/nvidia/cuda"
        [ -z "$BASE_IMAGE_VERSION" ] && BASE_IMAGE_VERSION="12.4.1-devel-rockylinux8"
        ;;
    metax)
        BASE_IMAGE="harbor.baai.ac.cn/flagbase/flagbase-metax"
        [ -z "$BASE_IMAGE_VERSION" ] && BASE_IMAGE_VERSION="latest"
        log_warn "MetaX RPM build may require custom base image with RPM tools"
        ;;
    ascend)
        BASE_IMAGE="ascendai/cann"
        [ -z "$BASE_IMAGE_VERSION" ] && BASE_IMAGE_VERSION="8.5.0-910-openeuler24.03-py3.11"
        ;;
    *)
        log_error "Invalid backend: $BACKEND"
        echo "Supported backends: nvidia, metax, ascend"
        exit 1
        ;;
esac

log_info "Building FlagCX RPM packages for $BACKEND backend"
log_info "Using base image: ${BASE_IMAGE}:${BASE_IMAGE_VERSION}"

# Sync changelog from CHANGELOG.md
log_step "Synchronizing changelog..."
if [ -f "${PROJECT_DIR}/packaging/sync-changelog.py" ]; then
    python3 "${PROJECT_DIR}/packaging/sync-changelog.py" || log_warn "Failed to sync changelog"
else
    log_warn "sync-changelog.py not found, skipping changelog sync"
fi

# Build Docker image
log_step "Building Docker image..."
docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg BASE_IMAGE_VERSION="${BASE_IMAGE_VERSION}" \
    -f "${SCRIPT_DIR}/dockerfiles/Dockerfile.${BACKEND}" \
    -t "flagcx-rpm-${BACKEND}:${BASE_IMAGE_VERSION}" \
    "${PROJECT_DIR}"

# Extract RPM packages
log_step "Extracting RPM packages..."
OUTPUT_DIR="${PROJECT_DIR}/rpm-packages/${BACKEND}"
mkdir -p "${OUTPUT_DIR}"

CONTAINER_ID=$(docker create "flagcx-rpm-${BACKEND}:${BASE_IMAGE_VERSION}")
docker cp "${CONTAINER_ID}:/root/rpmbuild/RPMS/" "${OUTPUT_DIR}/" 2>/dev/null || true
docker cp "${CONTAINER_ID}:/root/rpmbuild/SRPMS/" "${OUTPUT_DIR}/" 2>/dev/null || true
docker rm "${CONTAINER_ID}"

log_info "✓ Packages built successfully for ${BACKEND}:"
echo ""
find "${OUTPUT_DIR}" -name "*.rpm" -exec ls -lh {} \;

log_info "Build complete! Packages in: ${OUTPUT_DIR}"
