#!/bin/bash
set -e

# Unified FlagCX Debian package build script
# Usage: ./packaging/debian/build-helpers/build-flagcx.sh <backend> [base_image_version]
# Supported backends: nvidia, metax

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
BACKEND="${1:-}"
BASE_IMAGE_VERSION="${2:-latest}"

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
    echo "  nvidia  - Build packages for NVIDIA GPUs"
    echo "  metax   - Build packages for MetaX accelerators"
    echo ""
    echo "Optional arguments:"
    echo "  base_image_version - Base image version tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 nvidia"
    echo "  $0 metax"
    echo "  $0 nvidia v1.2.3"
    exit 1
fi

# Validate backend and set base image
case "$BACKEND" in
    nvidia)
        BASE_IMAGE="harbor.baai.ac.cn/flagbase/flagbase-nvidia"
        VENDOR="nvidia"
        ;;
    metax)
        BASE_IMAGE="harbor.baai.ac.cn/flagbase/flagbase-metax"
        VENDOR="metax"
        ;;
    *)
        log_error "Invalid backend: $BACKEND"
        echo "Supported backends: nvidia, metax"
        exit 1
        ;;
esac

log_info "Building FlagCX Debian packages for $BACKEND backend"
log_info "Using base image: ${BASE_IMAGE}:${BASE_IMAGE_VERSION}"

# Sync changelog from CHANGELOG.md
log_step "Synchronizing changelog..."
if [ -f "${PROJECT_DIR}/packaging/sync-changelog.py" ]; then
    python3 "${PROJECT_DIR}/packaging/sync-changelog.py" || log_warn "Failed to sync changelog"
else
    log_warn "sync-changelog.py not found, skipping changelog sync"
fi

DOCKERFILE="${SCRIPT_DIR}/Dockerfile.deb"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    log_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

log_step "Using unified Dockerfile: $DOCKERFILE"

# Build container image
IMAGE_TAG="flagcx-deb-${BACKEND}:${BASE_IMAGE_VERSION}"
log_step "Building container image: $IMAGE_TAG"

if ! docker build \
    -f "$DOCKERFILE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg BASE_IMAGE_VERSION="$BASE_IMAGE_VERSION" \
    --build-arg VENDOR="$VENDOR" \
    --target output \
    -t "$IMAGE_TAG" \
    "$PROJECT_DIR"; then
    log_error "Docker build failed for $BACKEND"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${PROJECT_DIR}/debian-packages/${BACKEND}"
mkdir -p "$OUTPUT_DIR"

log_step "Extracting .deb packages to: $OUTPUT_DIR"

# Extract .deb files
CONTAINER_NAME="flagcx-deb-${BACKEND}-tmp-$$"
if docker create --name "$CONTAINER_NAME" "$IMAGE_TAG" 2>/dev/null; then
    docker cp "${CONTAINER_NAME}:/output/." "$OUTPUT_DIR/" || log_warn "No .deb files found"
    docker rm "$CONTAINER_NAME" >/dev/null
else
    log_error "Failed to create temporary container"
    exit 1
fi

# Verify packages were created
if ls "$OUTPUT_DIR"/*.deb 1> /dev/null 2>&1; then
    echo ""
    log_info "✓ Packages built successfully for $BACKEND:"
    echo ""
    ls -lh "$OUTPUT_DIR"/*.deb | while read -r line; do
        echo "  $line"
    done
    echo ""

    # Run lintian if available
    if command -v lintian >/dev/null 2>&1; then
        log_step "Running lintian checks..."
        echo ""
        for deb in "$OUTPUT_DIR"/*.deb; do
            [ -f "$deb" ] || continue
            echo "Checking $(basename "$deb")..."
            if lintian "$deb" 2>&1; then
                log_info "✓ Lintian check passed for $(basename "$deb")"
            else
                log_warn "⚠ Lintian found issues in $(basename "$deb") (non-fatal)"
            fi
            echo ""
        done
    else
        log_warn "lintian not found, skipping package validation"
        log_warn "Install with: sudo apt-get install lintian"
    fi

    log_info "Build complete! Packages in: $OUTPUT_DIR/"
else
    log_error "No .deb files were created"
    exit 1
fi
