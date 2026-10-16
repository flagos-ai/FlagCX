#!/usr/bin/env bash
set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
python_bin=${PYTHON_BIN:-python3}
cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/plugin/torch${PYTHONPATH:+:$PYTHONPATH}"
export COVERAGE_RCFILE="$project_root/.github/configs/coverage.ini"

status=0
FLAGCX_CI_TEST_LABEL="Python Torch unit tests" \
  bash .github/scripts/ci/run_with_timeout.sh \
  "$python_bin" -m coverage run -m unittest discover \
  -s plugin/torch/tests -p 'test_*.py' || status=$?

# Preserve failure coverage without changing the test's exit status.
report_status=0
"$python_bin" -m coverage json -o coverage-cuda-gpu-torch.json || report_status=$?
if ((status != 0)); then
  exit "$status"
fi
exit "$report_status"
