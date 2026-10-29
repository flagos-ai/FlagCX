# FlagCX CI

`unit-test.yml` is the automatic controller for PRs to `main`/`devops`, pushes
to those branches, and the daily schedule. It runs the selected platforms'
unit, performance, and Torch API tests, plus common checks. Platform-specific
changes use `paths_watch`; any shared or unknown changed path selects every
enabled platform. Renames include both their old and new paths.

PR labels `platform:cuda`, `platform:hygon`, `platform:metax`,
`platform:kunlunxin`, and `platform:ppu` override path selection. Multiple
platform labels select their union;
`platform:all` selects every enabled platform. Invalid platform labels fail
matrix generation. Add/remove label events rerun the controller.

Maintainers with `write`, `maintain`, or `admin` repository permission can post
one of these exact commands on a PR:

```text
/test
/test cuda
/test hygon
/test metax
/test kunlunxin
/test ppu
/test all
```

`/test` uses the PR's labels and changed paths. An explicit platform overrides
that selection. Manual dispatch accepts `all` or comma-separated platform names.
Each request tests the resolved PR head repository and immutable commit SHA.
Comment requests have separate concurrency groups and cannot cancel PR checks.
GitHub only enables `issue_comment` workflows present on the default branch;
comment-triggered runs belong to that branch's check suite, so the automatic
PR run remains the merge gate.

## Required Check

After the first successful run, configure the repository's branch protection
or ruleset to require only **ci-complete** for these test workflows. YAML adds
the check but cannot change repository protection settings. Packaging workflows
keep their existing independent triggers.

The gate fails on matrix/configuration errors, failed or cancelled tests, or
unexpected skips of selected platforms. Empty selections skip hardware jobs
while still running common checks.

## Retry Logs

Each test invocation has a timeout and at most one retry. Cancellation and
missing/non-executable commands do not retry. Builds and container orchestration
are not retried. Recovered tests emit a flaky warning and Step Summary entry.
The `attempts-*` artifacts preserve both attempts for seven days, including
tests executed in explicitly managed Docker containers.

## Python Coverage

The CUDA Torch job measures Python code in `src/flagcx` and the Torch build
helpers using `plugin/torch/tests/test_*.py`. It does not measure C++ or MPI code.
The report and artifact are `coverage-cuda-gpu-torch.json`. Coverage runs inside
the configured CUDA container, using the built plugin. The report is persisted
through the attempt-log mount and copied to the runner for upload. Test failure
still fails CI, even when a partial coverage report is generated.

Repository configuration:

- Secret `FLAGCICD_API_TOKEN`: bearer token for the metrics endpoint.
- Endpoint: `https://flagcicd.flagos.net/flagcicd/api/v1/cicd/metrics/`.

The token is explicitly forwarded only for source in the same repository.
Fork PRs, including comment-triggered runs, generate/archive reports but skip
FlagCICD uploads. Missing reports or tokens also skip uploading. Upload errors
and the two-minute upload timeout do not change test conclusions.

The upload action is pinned to verified commit
`5bd0e138a6ae482b1a1cc809ea6705ecfd6eb410`, which removes `user_id` and requires
`api_token`.

Verify an authorized manual run on a runner with access to FlagCICD: the upload
log must show HTTP 2xx and the FlagCICD UI must display that run's Python
coverage. HTTP 401/403 indicates a token issue; curl exit 6 indicates DNS
failure; exit 7/28 indicates connectivity or timeout. Green CI alone does not
prove upload success.
