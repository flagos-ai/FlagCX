#!/bin/bash
set -euo pipefail

BACKEND="${1:-}"

if [ -z "${BACKEND}" ]; then
    echo "ERROR: backend is required" >&2
    exit 1
fi

case "${BACKEND}" in
    nvidia|metax|ascend)
        ;;
    *)
        echo "ERROR: unsupported backend: ${BACKEND}" >&2
        exit 1
        ;;
esac

PKG_MANAGER="$(command -v dnf || command -v yum || true)"
if [ -z "${PKG_MANAGER}" ]; then
    echo "ERROR: neither dnf nor yum is available in the base image" >&2
    exit 1
fi

"${PKG_MANAGER}" install -y epel-release || \
    echo "EPEL not available for this base image, continuing without it"

"${PKG_MANAGER}" install -y \
    rpm-build \
    rpmdevtools \
    gcc-c++ \
    make \
    cmake \
    patchelf

"${PKG_MANAGER}" install -y json-devel 2>/dev/null \
    || "${PKG_MANAGER}" install -y nlohmann-json-devel 2>/dev/null \
    || { echo "ERROR: neither json-devel nor nlohmann-json-devel is available; rpmbuild requires nlohmann::json headers" >&2; exit 1; }

"${PKG_MANAGER}" clean all

rpmdev-setuptree

SPEC_VERSION="$(awk '/^Version:/ {print $2; exit}' /workspace/packaging/rpm/specs/flagcx.spec)"
tar czf "/root/rpmbuild/SOURCES/flagcx-${SPEC_VERSION}.tar.gz" \
    --transform "s,^\.,flagcx-${SPEC_VERSION}," \
    --exclude='.git' \
    --exclude='build' \
    --exclude='debian-packages' \
    --exclude='rpm-packages' \
    .

RPMBUILD_ARGS=(
    -ba
    --define "backend ${BACKEND}"
)

if [ -n "${RPM_DIST_TAG:-}" ]; then
    RPMBUILD_ARGS+=(--define "dist ${RPM_DIST_TAG}")
fi
if [ "${RPM_EXTERNAL_VENDOR_RUNTIME:-0}" = "1" ]; then
    RPMBUILD_ARGS+=(--define "external_vendor_runtime 1")
fi
if [ -n "${DEVICE_HOME:-}" ]; then
    RPMBUILD_ARGS+=(--define "device_home ${DEVICE_HOME}")
fi
if [ -n "${CCL_HOME:-}" ]; then
    RPMBUILD_ARGS+=(--define "ccl_home ${CCL_HOME}")
fi

rpmbuild "${RPMBUILD_ARGS[@]}" \
    /workspace/packaging/rpm/specs/flagcx.spec

ls -lh /root/rpmbuild/RPMS/*/*.rpm
