#!/bin/bash
set -e

# Local script to test Nexus APT repository upload
# Usage: ./test-nexus-upload.sh [nvidia|metax|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
BACKEND="${1:-all}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Check for required environment variables
if [ -z "$NEXUS_USERNAME" ] || [ -z "$NEXUS_PASSWORD" ]; then
    log_error "Missing required environment variables"
    echo ""
    echo "Please set the following environment variables:"
    echo "  export NEXUS_USERNAME='your_username'"
    echo "  export NEXUS_PASSWORD='your_password'"
    echo ""
    echo "Then run this script again:"
    echo "  $0 [nvidia|metax|all]"
    exit 1
fi

NEXUS_REPO_URL="https://resource.flagos.net/repository/flagos-apt-hosted"

# Function to upload a deb package to Nexus APT hosted repository
upload_deb() {
    local deb_file="$1"
    local backend="$2"
    local filename=$(basename "$deb_file")

    # Extract package metadata for logging
    local package=$(dpkg-deb -f "$deb_file" Package)
    local version=$(dpkg-deb -f "$deb_file" Version)
    local arch=$(dpkg-deb -f "$deb_file" Architecture)

    echo ""
    log_step "Uploading: $filename"
    echo "  Package: $package"
    echo "  Version: $version"
    echo "  Architecture: $arch"
    echo "  Backend: $backend"

    # Upload to Nexus APT hosted repository
    local upload_url="${NEXUS_REPO_URL}/${filename}"

    echo "  URL: $upload_url"

    if curl -f -u "${NEXUS_USERNAME}:${NEXUS_PASSWORD}" \
         --upload-file "$deb_file" \
         "$upload_url"; then
        log_info "✓ Successfully uploaded $filename"
        return 0
    else
        log_error "✗ Failed to upload $filename"
        return 1
    fi
}

# Main upload logic
upload_backend() {
    local backend="$1"
    local package_dir="${PROJECT_DIR}/debian-packages/${backend}"

    if [ ! -d "$package_dir" ]; then
        log_warn "Package directory not found: $package_dir"
        log_warn "Run build-flagcx.sh first to build packages"
        return 1
    fi

    if ! ls "$package_dir"/*.deb 1> /dev/null 2>&1; then
        log_warn "No .deb files found in: $package_dir"
        return 1
    fi

    echo ""
    log_step "=== Uploading $backend packages ==="

    local upload_count=0
    local fail_count=0

    for deb in "$package_dir"/*.deb; do
        [ -f "$deb" ] || continue
        if upload_deb "$deb" "$backend"; then
            ((upload_count++))
        else
            ((fail_count++))
        fi
    done

    echo ""
    if [ $fail_count -eq 0 ]; then
        log_info "✓ All $backend packages uploaded successfully ($upload_count files)"
    else
        log_error "✗ Some uploads failed: $upload_count succeeded, $fail_count failed"
        return 1
    fi
}

# Execute uploads based on backend selection
case "$BACKEND" in
    nvidia)
        upload_backend "nvidia"
        ;;
    metax)
        upload_backend "metax"
        ;;
    all)
        upload_backend "nvidia" || true
        upload_backend "metax" || true
        ;;
    *)
        log_error "Invalid backend: $BACKEND"
        echo "Usage: $0 [nvidia|metax|all]"
        exit 1
        ;;
esac

echo ""
log_info "Upload test complete!"
echo ""
echo "To use the repository, add to /etc/apt/sources.list.d/flagcx.list:"
echo "deb https://resource.flagos.net/repository/flagos-apt-hosted/ flagos-apt-hosted main"
