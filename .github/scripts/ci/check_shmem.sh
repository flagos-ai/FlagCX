#!/usr/bin/env bash

# Locate a cann-shmem (aclshmem) install for the Ascend shmem leg (F4).
#
# Prints the usable SHMEM_HOME on stdout and exits 0, or prints nothing and
# exits 1.  A candidate must carry <shmem.h> (what the adaptor and the test
# kernels compile against) and libshmem (the link + rpath target).  The
# version file, when present, goes to stderr so a CI log pins what actually
# ran without polluting the stdout contract.

set -euo pipefail

for dir in "${SHMEM_HOME:-}" \
           /usr/local/Ascend/shmem/latest \
           /data/shmem/install/shmem \
           /flagos/lib/python3.12/site-packages/shmem; do
  [[ -n "$dir" && -d "$dir" ]] || continue
  [[ -r "$dir/include/shmem.h" ]] || continue
  # The device headers (gm2gm/...) are what the test kernels compile
  # against; a host-only tree cannot build the shmem leg.
  [[ -d "$dir/include/device" ]] || continue
  if [[ -e "$dir/lib/libshmem.so" || -e "$dir/libshmem.so" ]]; then
    if [[ -r "$dir/VERSION" ]]; then
      echo "cann-shmem VERSION: $(head -n 1 "$dir/VERSION")" >&2
    fi
    echo "$dir"
    exit 0
  fi
done

exit 1
