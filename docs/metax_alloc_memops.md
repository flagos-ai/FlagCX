# MetaX allocation and stream-memory diagnostic

Run in the same MetaX image and driver environment as the failing adaptor/RMA
tests. No FlagCX library, MPI, GoogleTest or device-kernel build is required:

```sh
python3 test/diagnostics/metax_alloc_memops.py \
  --sdk /opt/maca --output-dir /tmp/metax-alloc-memops
```

The MetaX `adaptor` CI suite runs this automatically in its preparation hook,
before RDMA preflight and builds. Reports and logs go into
`$FLAGCX_CI_LOG_DIR/metax-alloc-memops/` and are uploaded by the existing
always-run attempt-log artifact step. Diagnostic failures produce a warning;
the existing adaptor tests still determine their own results.

Coverage is **9 allocation routes × 2 sizes × 2 widths × 3 operations = 108
cases**. `mcMalloc` is the baseline. All eight named `mcExtMallocWithFlags`
options are checked against the actual SDK declarations, including Default,
Finegrained, SignalMemory, WriteCoherence, both MapPcie modes and both FixedMem
modes. The supervisor does not invent numeric flags. An option absent from
the SDK is reported explicitly; it is never replaced by another allocator.

Sizes are 8 bytes and the host page size. Widths are 32 and 64 bits, with
nonzero high bits in the 64-bit test value. Operations run in separate native
processes:

- `write`: initialize, enqueue write, synchronize, read back and compare.
- `wait`: initialize to the expected value, enqueue a satisfied wait,
  synchronize and read back. This checks native wait support independently of
  stream-write support; it does not test waiting for a remote publication.
- `write_wait`: initialize to zero, enqueue write then GEQ wait on the same
  stream, synchronize and compare. A rejected write never queues a wait.

Streams use `mcStreamNonBlocking`, matching the failing adaptor tests. An
explicit device synchronization completes initialization before the stream
accesses the word.

Every native process, including allocation, metadata and cleanup, has a
5-second deadline. Compiler checks/builds have a 15-second deadline, and
compiler-version collection has a 5-second deadline. Each timed-out process
group is killed, with at most 0.25 seconds spent waiting for reaping. Logs use
files instead of inherited output pipes, so descendants cannot hold pipe EOF
open. A driver task that cannot be reaped is recorded with `reaped=false`;
it does not cause an unbounded wait in the supervisor. With the defaults,
the sequential subprocess budgets total less than 15 minutes, plus report
and scheduling overhead. External job cancellation, host failure or an
unresponsive filesystem cannot be prevented by a user-space diagnostic.

`--case-timeout` and `--compile-timeout` accept finite values in `(0,60]`.
Increase the case deadline for a slow cold-start environment before treating
a timeout as an API compatibility result. Tests run on logical device 0;
select another using `--device`.

Results are checkpointed after each case:

- `summary.md`: every strategy/size/width and all three operation outcomes.
- `results.json`: raw API codes/messages, pointer attributes, values, build
  checks, runtime/driver/GPU metadata, process exits and deadlines.
- Per-case `.log` files: unbuffered output, including the last completed API
  before a crash or timeout. `OP_RESULT` precedes cleanup; `RESULT` follows it.
- `sdk-header-excerpts.txt`: allocation/memops declarations and nearby SDK
  comments, with relevant header paths and SHA-256 hashes in `results.json`.

Exit codes are 0 only when all cases pass, 1 when the matrix finishes with
unsupported/error/timeout outcomes, 2 for setup/build failure, and 130 for
interruption. All 108 entries exist even on setup failure; unexecuted entries
are labelled `NOT_RUN_*`, not passed. Availability-check compiler errors and
timeouts are distinguished from absent SDK enum declarations.

An allocation mode may legitimately reject an 8-byte request. Keep that
result; consult the installed SDK comments before adding a larger/aligned
case. A diagnostic PASS establishes neither IPC cache coherence nor RDMA
atomic support. Preserve the existing peer visibility and GDR validation when
choosing a production allocator. Inter signal transport changes are deferred.

Official references:

- [Runtime programming guide](https://developer.metax-tech.com/api/client/document/preview/567/C500_RuntimeAPIProgrammingGuide_CN.html).
- [Official support reply](https://developer.metax-tech.com/forum/t/l2-cachehuan-cun-shi-xiao-qiang-zhi-hbmdu-qu-fang-fa/352/)
  points to `/opt/maca/include/mcr/mc_runtime_api.h` for
  `mcExtMallocWithFlags` documentation. Use the CI image's actual SDK version.

Hardware-free supervisor regression tests (these are not MACA results):

```sh
python3 -m unittest discover -s test/script -p test_metax_alloc_memops.py -v
```
