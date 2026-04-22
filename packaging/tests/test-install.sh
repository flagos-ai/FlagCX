#!/bin/bash
set -euo pipefail

# Usage: test-install.sh <deb|rpm> <backend> <package-dir>
# Verifies FlagCX packages install correctly and files are in place.

PKG_TYPE="${1:?Usage: test-install.sh <deb|rpm> <backend> <package-dir>}"
BACKEND="${2:?Missing backend (nvidia|metax|ascend)}"
PKG_DIR="${3:?Missing package directory}"

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== FlagCX Package Installation Test ==="
echo "Type:    $PKG_TYPE"
echo "Backend: $BACKEND"
echo "Packages: $PKG_DIR"
echo ""

# --- Install packages ---
echo "--- Installing packages ---"
if [ "$PKG_TYPE" = "deb" ]; then
    # Pick the newest version of each package (avoid multi-version conflicts)
    RT_PKG=$(ls -v "${PKG_DIR}"/libflagcx-"${BACKEND}"_*.deb 2>/dev/null | tail -1)
    DEV_PKG=$(ls -v "${PKG_DIR}"/libflagcx-"${BACKEND}"-dev_*.deb 2>/dev/null | tail -1)
    echo "  Runtime: $(basename "$RT_PKG")"
    echo "  Dev:     $(basename "$DEV_PKG")"
    apt-get update -qq
    # Use --force-depends: vendor libs (libnccl, libnvidia-compute, etc.)
    # are not available in plain distro containers
    dpkg -i --force-depends "$RT_PKG" "$DEV_PKG"
elif [ "$PKG_TYPE" = "rpm" ]; then
    # Detect package manager
    if command -v dnf >/dev/null 2>&1; then
        PM="dnf"
    elif command -v yum >/dev/null 2>&1; then
        PM="yum"
    else
        echo "ERROR: No supported package manager (dnf/yum) found"
        exit 1
    fi
    # Pick newest runtime + devel RPMs
    RT_PKG=$(ls -v "${PKG_DIR}"/libflagcx-"${BACKEND}"-[0-9]*.rpm 2>/dev/null | tail -1)
    DEV_PKG=$(ls -v "${PKG_DIR}"/libflagcx-"${BACKEND}"-devel-*.rpm 2>/dev/null | tail -1)
    echo "  Runtime: $(basename "$RT_PKG")"
    echo "  Devel:   $(basename "$DEV_PKG")"
    # Use --nodeps: vendor libs are not available in plain distro containers
    rpm -ivh --nodeps "$RT_PKG" "$DEV_PKG"
else
    echo "ERROR: Unknown package type: $PKG_TYPE (expected deb or rpm)"
    exit 1
fi

echo ""
echo "--- Verifying installation ---"

# --- Runtime package checks ---
echo "[Runtime: libflagcx-${BACKEND}]"

check "libflagcx.so.0 exists" test -f /usr/lib/libflagcx.so.0 -o -f /usr/lib64/libflagcx.so.0

# Run ldconfig and verify library is indexed
ldconfig 2>/dev/null || true
check "ldconfig indexes libflagcx.so.0" bash -c "ldconfig -p 2>/dev/null | grep -q libflagcx"

check "libflagcx.so symlink exists" test -e /usr/lib/libflagcx.so -o -e /usr/lib64/libflagcx.so

# Verify SONAME (requires binutils - optional)
SO_PATH=$(find /usr/lib /usr/lib64 -name "libflagcx.so.0" 2>/dev/null | head -1)
if [ -n "$SO_PATH" ]; then
    if command -v objdump >/dev/null 2>&1; then
        check "SONAME is libflagcx.so.0" bash -c "objdump -p '$SO_PATH' | grep -q 'SONAME.*libflagcx.so.0'"
    elif command -v readelf >/dev/null 2>&1; then
        check "SONAME is libflagcx.so.0" bash -c "readelf -d '$SO_PATH' | grep -q 'SONAME.*libflagcx.so.0'"
    else
        echo "  SKIP: SONAME check (no objdump/readelf)"
    fi
else
    echo "  FAIL: libflagcx.so.0 not found for SONAME check"
    FAIL=$((FAIL + 1))
fi

# --- Development package checks ---
echo "[Development: libflagcx-${BACKEND}-dev(el)]"

check "include dir exists" test -d /usr/include/flagcx
check "flagcx.h header exists" bash -c "find /usr/include/flagcx -name flagcx.h 2>/dev/null | grep -q flagcx.h"

# Count headers
HEADER_COUNT=$(find /usr/include/flagcx -name "*.h" 2>/dev/null | wc -l)
check "headers installed (found: ${HEADER_COUNT})" test "$HEADER_COUNT" -gt 0

# --- Package metadata checks ---
echo "[Package metadata]"

if [ "$PKG_TYPE" = "deb" ]; then
    check "runtime pkg is installed" bash -c "dpkg -s 'libflagcx-${BACKEND}' 2>/dev/null | grep -q 'Status: install ok installed'"
    check "dev pkg is installed" bash -c "dpkg -s 'libflagcx-${BACKEND}-dev' 2>/dev/null | grep -q 'Status: install ok installed'"
elif [ "$PKG_TYPE" = "rpm" ]; then
    check "runtime pkg is installed" rpm -q "libflagcx-${BACKEND}"
    check "devel pkg is installed" rpm -q "libflagcx-${BACKEND}-devel"
fi

# --- License check ---
echo "[License]"
if [ "$PKG_TYPE" = "deb" ]; then
    check "copyright file exists" test -f "/usr/share/doc/libflagcx-${BACKEND}/copyright" -o -f "/usr/share/doc/libflagcx-${BACKEND}/changelog.Debian.gz"
elif [ "$PKG_TYPE" = "rpm" ]; then
    check "LICENSE file exists" test -f "/usr/share/licenses/libflagcx-${BACKEND}/LICENSE"
fi

# --- Summary ---
echo ""
echo "=== Results ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    echo "STATUS: FAILED"
    exit 1
else
    echo "STATUS: ALL PASSED"
    exit 0
fi
