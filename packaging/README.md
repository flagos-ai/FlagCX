# FlagCX Packaging

This directory contains packaging configurations for various Linux distributions.

## Directory Structure

```
packaging/
├── debian/              # Debian/Ubuntu packaging
│   ├── control         # Package metadata (with build profiles)
│   ├── rules           # Build rules
│   ├── changelog       # Version history
│   ├── copyright       # License information
│   └── build-helpers/  # Build scripts and Dockerfiles
│       ├── build-flagcx.sh          # Unified build script
│       ├── Dockerfile.deb           # Unified build configuration
│       └── test-nexus-upload.sh     # Local Nexus upload test script
└── rpm/                # Future: RPM packaging for RHEL/Fedora/etc.
```

## Why `packaging/` Instead of Top-Level `/debian`?

Following [Debian UpstreamGuide](https://wiki.debian.org/UpstreamGuide) recommendations:

> Upstream projects should NOT include a top-level `/debian` directory.
> Use `contrib/debian/` or `packaging/debian/` instead.

**Benefits:**
- Avoids conflicts with distribution maintainers' packaging
- Clearly indicates upstream-maintained packaging
- Allows multi-format support (Debian + RPM + others)
- Industry standard (see [Miniflux](https://github.com/miniflux/v2/tree/main/packaging), etc.)

## Building Debian Packages

Use the unified build script to build packages for any vendor/backend:

### Usage

```bash
./packaging/debian/build-helpers/build-flagcx.sh <vendor> [base_image_version]
```

**Parameters:**
- `<vendor>` - Hardware vendor/backend (e.g., `nvidia`, `metax`)
- `[base_image_version]` - Optional base image version tag (default: `latest`)

**Output:** `debian-packages/<vendor>/*.deb`

### Examples

**Build for NVIDIA:**
```bash
./packaging/debian/build-helpers/build-flagcx.sh nvidia
# Output: debian-packages/nvidia/*.deb
```

**Build for MetaX:**
```bash
./packaging/debian/build-helpers/build-flagcx.sh metax
# Output: debian-packages/metax/*.deb
```

**Specify custom base image version:**
```bash
./packaging/debian/build-helpers/build-flagcx.sh nvidia v1.2.3
./packaging/debian/build-helpers/build-flagcx.sh metax latest
```

### Base Images

The build script uses upstream base images from `harbor.baai.ac.cn/flagbase/`:
- NVIDIA: `flagbase-nvidia:<version>`
- MetaX: `flagbase-metax:<version>`

To add support for a new vendor, ensure a corresponding base image exists at:
`harbor.baai.ac.cn/flagbase/flagbase-<vendor>:<version>`

### Quality Checks

The build script automatically runs `lintian` to validate the generated packages if available:

```bash
# Install lintian (optional but recommended)
sudo apt-get install lintian

# Build packages - lintian runs automatically
./packaging/debian/build-helpers/build-flagcx.sh nvidia
```

Lintian checks are non-fatal and won't stop the build if issues are found.

## Installation

Install packages for your hardware vendor:

```bash
# General syntax
sudo dpkg -i debian-packages/<vendor>/*.deb

# Example: NVIDIA
sudo dpkg -i debian-packages/nvidia/*.deb

# Example: MetaX
sudo dpkg -i debian-packages/metax/*.deb
```

## CI/CD

Automated builds are triggered by:
- Push to `main` branch (when packaging files change)
- Pull requests to `main`
- Manual workflow dispatch

See `.github/workflows/build-deb.yml` for details.

### Publishing to Nexus APT Repository

Packages are uploaded to Nexus when:
- A version tag is pushed: `git tag v1.0.0 && git push origin v1.0.0`
- Manual workflow dispatch via GitHub Actions

After upload, users can install packages from the APT repository:

```bash
# Add the FlagOS APT repository
echo "deb https://resource.flagos.net/repository/flagos-apt-hosted/ flagos-apt-hosted main" | \
  sudo tee /etc/apt/sources.list.d/flagcx.list

# Update package list
sudo apt-get update

# Install packages for your vendor
sudo apt-get install libflagcx-<vendor>        # Runtime library
sudo apt-get install libflagcx-<vendor>-dev    # Development files

# Examples:
sudo apt-get install libflagcx-nvidia libflagcx-nvidia-dev
sudo apt-get install libflagcx-metax libflagcx-metax-dev
```

## Architecture

The build process uses a **unified multi-stage Dockerfile** with build profiles:

### Build Profiles Support

The `debian/control` file defines build profiles to support multiple backends:
- `pkg.flagcx.nvidia-only` - Build only NVIDIA packages
- `pkg.flagcx.metax-only` - Build only MetaX packages

### Unified Dockerfile

A single `Dockerfile.deb` builds packages for all backends using build arguments:
- `BASE_IMAGE` - Upstream base image (e.g., `flagbase-nvidia`, `flagbase-metax`)
- `BASE_IMAGE_VERSION` - Image version tag (default: `latest`)
- `VENDOR` - Backend vendor name (used for build profile selection)

### Build Stages

1. **Builder stage**: Based on upstream flagbase images
   - Contains all necessary build dependencies (CUDA/NCCL or MACA SDK)
   - Installs Debian packaging tools (`debhelper`, `dpkg-dev`, etc.)
   - Runs `dpkg-buildpackage` with `DEB_BUILD_PROFILES=pkg.flagcx.${VENDOR}-only`
   - Only builds packages for the specified vendor

2. **Output stage**: Minimal Alpine image
   - Only contains the built `.deb` files
   - Used to extract packages to the host

This approach ensures:
- ✓ Reproducible builds using official base images
- ✓ Single Dockerfile for all backends (DRY principle)
- ✓ Backend selection via build profiles
- ✓ No custom Docker images to maintain
- ✓ Clean separation of build environment and outputs

## Future Plans

- [ ] Add RPM packaging in `packaging/rpm/`
- [ ] Add Arch Linux packaging
- [x] Add APT repository hosting (Nexus)
