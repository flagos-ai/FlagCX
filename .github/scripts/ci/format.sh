#!/usr/bin/env bash

# Copyright (c) 2026 BAAI. All rights reserved.
#
# See LICENSE for license information.

# Run the repository's pre-commit hooks locally.
# Usage: bash .github/scripts/ci/format.sh

set -euo pipefail

FLAGCX_PATH=${FLAGCX_PATH:-${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}}

cd "$FLAGCX_PATH"

python3 -m pip install pre-commit clang-format==14.0.6
clang-format --version
python3 -m pre_commit run --all-files
