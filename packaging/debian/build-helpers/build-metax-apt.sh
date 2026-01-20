#!/bin/bash
set -e

# Script to build FlagCX MetaX packages using public APT repository
# This avoids the need for MetaX container registry authentication

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR is now packaging/debian/build-helpers/, go up three levels to get project root
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_info "Building FlagCX MetaX packages using public APT repository"
log_info "No container registry authentication required!"

# Build using the Dockerfile (in build-helpers/)
if ! docker build -f "${SCRIPT_DIR}/Dockerfile.metax-apt" --target output -t "flagcx-deb-metax-apt" "$PROJECT_DIR"; then
    log_error "Docker build failed"
    exit 1
fi

# Create output directory
mkdir -p "${PROJECT_DIR}/debian-packages/metax"

# Extract .deb files
if docker create --name "flagcx-deb-metax-tmp" "flagcx-deb-metax-apt" 2>/dev/null; then
    docker cp "flagcx-deb-metax-tmp:/output/." "${PROJECT_DIR}/debian-packages/metax/" || log_warn "No .deb files found"
    docker rm "flagcx-deb-metax-tmp"
else
    log_error "Failed to create container"
    exit 1
fi

if ls "${PROJECT_DIR}/debian-packages/metax"/*.deb 1> /dev/null 2>&1; then
    log_info "Packages built successfully:"
    ls -lh "${PROJECT_DIR}/debian-packages/metax"/*.deb
else
    log_error "No .deb files were created"
    exit 1
fi

log_info "Build complete! Packages available in: ${PROJECT_DIR}/debian-packages/metax/"
