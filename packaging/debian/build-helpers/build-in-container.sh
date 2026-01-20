#!/bin/bash
set -e

# Script to build FlagCX Debian packages using vendor-specific containers
# Usage: ./packaging/debian/build-helpers/build-in-container.sh [nvidia|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR is now packaging/debian/build-helpers/, go up three levels to get project root
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
BUILD_TYPE="${1:-all}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

build_debian_package() {
    local backend=$1
    local containerfile="${PROJECT_DIR}/container/containerfile.${backend}"

    log_info "Building Debian package for ${backend} backend using containerfile: ${containerfile}"

    # Check if containerfile exists
    if [ ! -f "$containerfile" ]; then
        log_error "Containerfile not found: ${containerfile}"
        return 1
    fi

    # Create a temporary Dockerfile based on the vendor's containerfile
    local temp_dockerfile="${PROJECT_DIR}/.debian-build-${backend}.dockerfile"

    # Read the original containerfile and modify it for Debian packaging
    # 1. Keep everything up to the build step
    # 2. Add Debian packaging tools installation
    # 3. Replace the final make command with dpkg-buildpackage

    # Extract everything except the final RUN make line
    grep -v "^RUN make USE_" "$containerfile" > "$temp_dockerfile"

    # Rename the builder stage for multi-stage build
    sed -i '1s/FROM /FROM /' "$temp_dockerfile"
    sed -i '1s/$/ as builder/' "$temp_dockerfile"

    # Add backend environment variable
    echo "" >> "$temp_dockerfile"
    echo "# Set backend to build" >> "$temp_dockerfile"
    echo "ENV FLAGCX_BUILD_BACKEND=${backend}" >> "$temp_dockerfile"

    # Add Debian packaging tools
    cat >> "$temp_dockerfile" << 'DEBTOOLS'

# Install Debian packaging tools
RUN apt-get update && apt-get install -y \
    debhelper \
    devscripts \
    dpkg-dev \
    fakeroot \
    lsb-release \
    chrpath \
    patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Initialize git submodules (needed for nlohmann/json and googletest)
RUN git submodule update --init --recursive --depth 1 || true

# Change to the project directory for Debian build
WORKDIR /workspace/FlagCX

# Copy packaging/debian to debian/ for dpkg-buildpackage
RUN if [ -d "packaging/debian" ]; then \
        cp -r packaging/debian debian; \
    fi
DEBTOOLS

    # Add the dpkg-buildpackage command
    cat >> "$temp_dockerfile" << EOF

# Build the Debian packages
RUN if [ "\${FLAGCX_BUILD_BACKEND}" = "metax" ]; then \\
        DEB_BUILD_PROFILES="pkg.flagcx.metax-only" dpkg-buildpackage -us -uc -b -Pnocheck,pkg.flagcx.metax-only; \\
    elif [ "\${FLAGCX_BUILD_BACKEND}" = "nvidia" ]; then \\
        DEB_BUILD_PROFILES="pkg.flagcx.nvidia-only" dpkg-buildpackage -us -uc -b -Pnocheck,pkg.flagcx.nvidia-only; \\
    else \\
        dpkg-buildpackage -us -uc -b -Pnocheck; \\
    fi || { echo "Build failed, checking logs..."; find /workspace -name "*.log" -exec echo "=== {} ===" \\; -exec cat {} \\; 2>/dev/null || true; exit 1; }

# Collect the built .deb files
RUN mkdir -p /output && \\
    (cp /workspace/*${backend}*.deb /output/ 2>/dev/null || true) && \\
    ls -la /output/

# Output stage - collect the .deb files
FROM alpine:latest as output
COPY --from=builder /output/*.deb /output/
EOF

    log_info "Building container and extracting .deb packages..."

    # Build and extract packages
    if ! docker build -f "$temp_dockerfile" --target output -t "flagcx-deb-${backend}" "$PROJECT_DIR"; then
        log_error "Docker build failed for ${backend}"
        rm -f "$temp_dockerfile"
        return 1
    fi

    # Create output directory
    mkdir -p "${PROJECT_DIR}/debian-packages/${backend}"

    # Extract .deb files
    if docker create --name "flagcx-deb-${backend}-tmp" "flagcx-deb-${backend}" 2>/dev/null; then
        docker cp "flagcx-deb-${backend}-tmp:/output/." "${PROJECT_DIR}/debian-packages/${backend}/" || log_warn "No .deb files found to extract"
        docker rm "flagcx-deb-${backend}-tmp"
    else
        log_warn "No packages produced for ${backend}"
    fi

    # Cleanup
    rm -f "$temp_dockerfile"

    if ls "${PROJECT_DIR}/debian-packages/${backend}"/*.deb 1> /dev/null 2>&1; then
        log_info "Packages built successfully for ${backend}:"
        ls -lh "${PROJECT_DIR}/debian-packages/${backend}"/*.deb
    else
        log_error "No .deb files were created for ${backend}"
        return 1
    fi
}

# Main execution
case "$BUILD_TYPE" in
    nvidia|all)
        log_info "Building NVIDIA packages using container/containerfile.nvidia..."
        log_info "For MetaX packages, use: ./debian/build-helpers/build-metax-apt.sh (no container auth needed)"
        build_debian_package "nvidia"
        ;;
    metax)
        log_error "MetaX builds via containerfile require authentication."
        log_info "Use the APT-based build instead (no authentication needed):"
        log_info "  ./debian/build-helpers/build-metax-apt.sh"
        exit 1
        ;;
    *)
        log_error "Invalid build type: $BUILD_TYPE"
        echo "Usage: $0 [nvidia|all]"
        echo "For MetaX: ./debian/build-helpers/build-metax-apt.sh"
        exit 1
        ;;
esac

log_info "Build complete! Packages available in: ${PROJECT_DIR}/debian-packages/"
