# FlagCX Packaging

This directory contains packaging configurations for various Linux distributions.

## Directory Structure

```
packaging/
├── debian/              # Debian/Ubuntu packaging
│   ├── control         # Package metadata (with build profiles)
│   ├── rules           # Build rules
│   ├── changelog       # Version history (auto-generated)
│   ├── copyright       # License information
│   └── build-helpers/  # Build scripts and Dockerfiles
│       ├── build-flagcx.sh          # Unified build script
│       ├── Dockerfile.deb           # Unified build configuration
│       └── test-nexus-upload.sh     # Local Nexus upload test script
├── rpm/                 # RPM packaging for RHEL/Rocky/OpenEuler
│   ├── specs/
│   │   └── flagcx.spec            # RPM spec file
│   ├── dockerfiles/
│   │   ├── Dockerfile.nvidia      # NVIDIA backend build environment
│   │   ├── Dockerfile.metax       # MetaX backend build environment
│   │   └── Dockerfile.ascend      # Ascend backend build environment
│   └── build-flagcx-rpm.sh        # Build script
├── sync-changelog.py    # Changelog sync (docs/CHANGELOG.md -> deb/rpm)
├── CHANGELOG-MANAGEMENT.md
└── README.md            # This file
```

## Why `packaging/` Instead of Top-Level `/debian`?

Following [Debian UpstreamGuide](https://wiki.debian.org/UpstreamGuide) recommendations:

> Upstream projects should NOT include a top-level `/debian` directory.
> Use `contrib/debian/` or `packaging/debian/` instead.

Benefits:
- Avoids conflicts with distribution maintainers' packaging
- Clearly indicates upstream-maintained packaging
- Allows multi-format support (Debian + RPM + others)
- Industry standard (see [Miniflux](https://github.com/miniflux/v2/tree/main/packaging), etc.)

## Supported Backends and Architectures

| Backend | DEB (Debian/Ubuntu) | RPM (RHEL/Rocky/OpenEuler) |
|---------|--------------------|-----------------------------|
| NVIDIA  | amd64              | x86_64 (Rocky Linux 8/9)    |
| MetaX   | amd64              | x86_64 (TBD)                |
| Ascend  | amd64, arm64       | x86_64, aarch64 (OpenEuler 24.03) |

## Building Packages

### Prerequisites

- Docker
- Docker Buildx (for multi-architecture builds)
- Python 3 (for changelog sync)

### Debian Packages

```bash
./packaging/debian/build-helpers/build-flagcx.sh <backend> [base_image_version]

# Examples
./packaging/debian/build-helpers/build-flagcx.sh nvidia
./packaging/debian/build-helpers/build-flagcx.sh metax
./packaging/debian/build-helpers/build-flagcx.sh nvidia v1.2.3
```

Output: `debian-packages/<backend>/*.deb`

Base images are from `harbor.baai.ac.cn/flagbase/`:
- NVIDIA: `flagbase-nvidia:<version>`
- MetaX: `flagbase-metax:<version>`

The build script automatically runs `lintian` for quality checks if available.

### RPM Packages

```bash
./packaging/rpm/build-flagcx-rpm.sh <backend> [base_image_version]

# Examples
./packaging/rpm/build-flagcx-rpm.sh nvidia
./packaging/rpm/build-flagcx-rpm.sh ascend
./packaging/rpm/build-flagcx-rpm.sh ascend 8.5.0-910-openeuler24.03-py3.11
```

Output: `rpm-packages/<backend>/RPMS/<arch>/*.rpm`

Default base images:
- NVIDIA: `nvcr.io/nvidia/cuda:12.4.1-devel-rockylinux8`
- Ascend: `ascendai/cann:8.5.0-910-openeuler24.03-py3.11`

## Installation

### Debian/Ubuntu

```bash
# From local packages
sudo dpkg -i debian-packages/<backend>/*.deb

# From APT repository
echo "deb https://resource.flagos.net/repository/flagos-apt-hosted/ flagos-apt-hosted main" | \
  sudo tee /etc/apt/sources.list.d/flagcx.list
sudo apt-get update
sudo apt-get install libflagcx-<backend> libflagcx-<backend>-dev
```

### RHEL/Rocky/OpenEuler

```bash
sudo yum install libflagcx-nvidia-0.8.0-1.el8.x86_64.rpm
sudo yum install libflagcx-nvidia-devel-0.8.0-1.el8.x86_64.rpm

# Or on OpenEuler
sudo dnf install libflagcx-ascend-0.8.0-1.oe2403.aarch64.rpm
```

## Package Contents

### Runtime Package (libflagcx-{backend})
- DEB: `/usr/lib/<multiarch>/libflagcx.so.*`
- RPM: `/usr/lib64/libflagcx.so.*`

### Development Package (libflagcx-{backend}-dev / -devel)
- Headers: `/usr/include/flagcx/`
- DEB: `/usr/lib/<multiarch>/libflagcx.so`
- RPM: `/usr/lib64/libflagcx.so`

## Changelog Management

Changelogs are managed from a single source: `docs/CHANGELOG.md`.

The sync script automatically converts it to both Debian and RPM formats:

```bash
python3 packaging/sync-changelog.py
```

See [CHANGELOG-MANAGEMENT.md](CHANGELOG-MANAGEMENT.md) for details.

## CI/CD

Automated builds are triggered by:
- Version tag push (`v*`)
- Pull requests to `main` (when packaging files change)
- Manual workflow dispatch

Workflows:
- `.github/workflows/build-deb.yml` - Debian packages
- `.github/workflows/build-rpm.yml` - RPM packages

## Architecture

### DEB Build

Uses a unified multi-stage Dockerfile with build profiles:
- `pkg.flagcx.nvidia-only` / `pkg.flagcx.metax-only` / `pkg.flagcx.ascend-only`
- Single `Dockerfile.deb` for all backends via build arguments
- Builder stage (flagbase image) -> Output stage (Alpine, .deb files only)

### RPM Build

Uses per-backend Dockerfiles with native RPM distributions:
- NVIDIA: Rocky Linux 8 with CUDA toolkit
- Ascend: OpenEuler 24.03 with CANN toolkit
- Builds via `rpmbuild` with `--define 'backend <name>'`

### Why Separate Build Environments?

RPM packages must be built on RPM-based distributions:
- Different file system layouts (`/usr/lib64` vs `/usr/lib/<multiarch>`)
- Different dependency resolution (yum/dnf vs apt)
- System library version mismatches

## Key Differences: DEB vs RPM

| Aspect | Debian/Ubuntu | RPM (RHEL/Rocky/OpenEuler) |
|--------|---------------|----------------------------|
| Package Tool | dpkg-buildpackage | rpmbuild |
| Lib Directory | /usr/lib/x86_64-linux-gnu/ | /usr/lib64/ |
| Build Deps | Build-Depends in control | BuildRequires in spec |
| Runtime Deps | Depends in control | Requires in spec |
| Profiles | Build-Profiles | RPM macros (--define) |
| Quality Check | lintian | rpmlint |

## Troubleshooting

### Missing dependencies during build
Ensure the base image includes all required SDKs (CUDA, MACA, CANN).

### Wrong library path (RPM)
RPM uses `/usr/lib64` on x86_64, not `/usr/lib/x86_64-linux-gnu`.

### SONAME conflicts (RPM)
Use `patchelf --set-soname` to fix SONAME in the spec file.

## References

- [Debian UpstreamGuide](https://wiki.debian.org/UpstreamGuide)
- [RPM Packaging Guide](https://rpm-packaging-guide.github.io/)
- [Fedora Packaging Guidelines](https://docs.fedoraproject.org/en-US/packaging-guidelines/)
- [OpenEuler Packaging](https://docs.openeuler.org/en/docs/22.03_LTS/docs/ApplicationDev/packaging-software.html)
