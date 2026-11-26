// SDK-only record-count controls for a same-stream heterogeneous group.
// Record the first 0..3 group/recv/send completion events after the callback.
// No FlagCX, native collective library, device kernels or stream-value calls.
#include <cuda_runtime.h>
#include <mpi.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const size_t chunkBytes = 1024 * 1024;
static const size_t bytes = 2 * chunkBytes;
static FILE *rankLog;
static int rank;

static double now() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void logStage(const char *format, ...) {
  // One lock keeps callback/proxy output readable in both destinations.
  flockfile(stdout);
  fprintf(stdout, "rank=%d time=%.6f ", rank, now());
  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
  fputc('\n', stdout);
  fflush(stdout);
  if (rankLog != NULL) {
    fprintf(rankLog, "rank=%d time=%.6f ", rank, now());
    va_start(args, format);
    vfprintf(rankLog, format, args);
    va_end(args);
    fputc('\n', rankLog);
    fflush(rankLog);
  }
  funlockfile(stdout);
}

static void fail(const char *stage) {
  logStage("FAIL stage=%s; retain live SDK resources until job termination",
           stage);
  // Never free exports/imports while another rank or copy may still use them.
  // The outer MPI watchdog also bounds SDK calls and MPI_Abort itself.
  MPI_Abort(MPI_COMM_WORLD, 1);
  _Exit(1);
}

#define SDK(call)                                                              \
  do {                                                                         \
    logStage("begin %s", #call);                                               \
    cudaError_t result = (call);                                               \
    logStage("end %s result=%d", #call, (int)result);                          \
    if (result != cudaSuccess)                                                 \
      fail(#call);                                                             \
  } while (0)

#define MPI_CHECK(call)                                                        \
  do {                                                                         \
    logStage("begin %s", #call);                                               \
    int result = (call);                                                       \
    logStage("end %s result=%d", #call, result);                               \
    if (result != MPI_SUCCESS)                                                 \
      fail(#call);                                                             \
  } while (0)

static int waitEvent(cudaEvent_t event, double deadline, const char *stage) {
  double started = now();
  logStage("begin %s event=%p", stage, (void *)event);
  cudaError_t result;
  do {
    result = cudaEventQuery(event);
    if (result != cudaErrorNotReady)
      break;
    usleep(1000);
  } while (now() < deadline);
  logStage("end %s result=%d elapsed=%.6f", stage, (int)result,
           now() - started);
  return result == cudaSuccess;
}

struct CopyState {
  int device;
  int started;
  int done;
  int expired;
  int workerReady;
  int workerFailed;
  void *source;
  void *destination;
  cudaMemcpyKind copyKind;
  cudaStream_t copyStream;
  cudaEvent_t copied[2];
};

static void CUDART_CB callback(void *argument) {
  struct CopyState *state = (struct CopyState *)argument;
  logStage("callback enter");
  __atomic_store_n(&state->started, 1, __ATOMIC_RELEASE);
  double deadline = now() + 8;
  while (!__atomic_load_n(&state->done, __ATOMIC_ACQUIRE)) {
    if (now() >= deadline) {
      __atomic_store_n(&state->expired, 1, __ATOMIC_RELEASE);
      break;
    }
    usleep(1000);
  }
  logStage("callback exit expired=%d",
           __atomic_load_n(&state->expired, __ATOMIC_ACQUIRE));
}

static void *proxy(void *argument) {
  struct CopyState *state = (struct CopyState *)argument;
  // Match the real proxy: initialize its device before the callback blocks.
  SDK(cudaSetDevice(state->device));
  __atomic_store_n(&state->workerReady, 1, __ATOMIC_RELEASE);
  double deadline = now() + 8;
  while (!__atomic_load_n(&state->started, __ATOMIC_ACQUIRE)) {
    if (now() >= deadline) {
      state->workerFailed = 1;
      __atomic_store_n(&state->done, 1, __ATOMIC_RELEASE);
      return NULL;
    }
    usleep(1000);
  }
  // Callback entry proves that preceding user-stream work has completed.
  // Match the production signalStart/pollStart handoff above rather than
  // introducing an additional GPU wait on the copy stream.
  for (int i = 0; i < 2; ++i) {
    logStage("copy submit chunk=%d bytes=%zu", i, chunkBytes);
    SDK(cudaMemcpyAsync((char *)state->destination + i * chunkBytes,
                        (char *)state->source + i * chunkBytes, chunkBytes,
                        state->copyKind, state->copyStream));
    SDK(cudaEventRecord(state->copied[i], state->copyStream));
  }
  deadline = now() + 5;
  for (int i = 0; i < 2; ++i) {
    if (!waitEvent(state->copied[i], deadline, "copy completion")) {
      state->workerFailed = 1;
      break;
    }
  }
  // Release the callback even on failure; the job still fails, without freeing
  // buffers whose asynchronous completion has not been established.
  __atomic_store_n(&state->done, 1, __ATOMIC_RELEASE);
  return NULL;
}

static unsigned char pattern(size_t i, int sourceRank) {
  return (unsigned char)(1 + (i + sourceRank * 71) % 251);
}

static void validate(const unsigned char *data, int sourceRank,
                     const char *stage) {
  for (size_t i = 0; i < bytes; ++i) {
    if (data[i] != pattern(i, sourceRank)) {
      logStage("data mismatch stage=%s offset=%zu actual=%u expected=%u", stage,
               i, data[i], pattern(i, sourceRank));
      fail(stage);
    }
  }
  logStage("data valid stage=%s bytes=%zu sourceRank=%d", stage, bytes,
           sourceRank);
}

int main(int argc, char **argv) {
  if (argc != 4 ||
      (strcmp(argv[1], "local") && strcmp(argv[1], "ipc") &&
       strcmp(argv[1], "staging")) ||
      (argv[2][0] < '0' || argv[2][0] > '3' || argv[2][1] != '\0')) {
    fprintf(stderr, "Usage: %s local|ipc|staging 0|1|2|3 result-directory\n",
            argv[0]);
    return 2;
  }
  int provided;
  // Normal MPI operations stay on the main thread. Request MULTIPLE because a
  // proxy SDK failure may terminate the job with MPI_Abort.
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) !=
      MPI_SUCCESS)
    return 1;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    fail("MPI_Comm_rank");
  char logPath[4096];
  snprintf(logPath, sizeof(logPath), "%s/rank-%d.log", argv[3], rank);
  rankLog = fopen(logPath, "w");
  if (rankLog == NULL)
    fail("open rank log");
  int ranks;
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  int ipc = strcmp(argv[1], "ipc") == 0;
  int staging = strcmp(argv[1], "staging") == 0;
  int recordCount = argv[2][0] - '0';
  if (ranks != (ipc ? 2 : 1) || provided < MPI_THREAD_MULTIPLE)
    fail("MPI rank count or thread support");
  struct CopyState state = {};
  state.device = ipc && rank == 1 ? 2 : 0;
  logStage("case kind=%s record_count=%d device=%d ranks=%d wait=stream",
           argv[1], recordCount, state.device, ranks);
  int devices;
  SDK(cudaGetDeviceCount(&devices));
  if (devices <= state.device)
    fail("required device unavailable");
  SDK(cudaSetDevice(state.device));
  cudaStream_t userStream;
  cudaEvent_t readyRecv, readySend, groupDone, recvDone, sendDone;
  SDK(cudaStreamCreateWithFlags(&userStream, cudaStreamNonBlocking));
  SDK(cudaStreamCreateWithFlags(&state.copyStream, cudaStreamNonBlocking));
  SDK(cudaEventCreateWithFlags(&readyRecv, cudaEventDisableTiming));
  SDK(cudaEventCreateWithFlags(&readySend, cudaEventDisableTiming));
  SDK(cudaEventCreateWithFlags(&state.copied[0], cudaEventDisableTiming));
  SDK(cudaEventCreateWithFlags(&state.copied[1], cudaEventDisableTiming));
  // Allocate identical events for every count; only their Records differ.
  SDK(cudaEventCreateWithFlags(&groupDone, cudaEventDisableTiming));
  SDK(cudaEventCreateWithFlags(&recvDone, cudaEventDisableTiming));
  SDK(cudaEventCreateWithFlags(&sendDone, cudaEventDisableTiming));
  cudaEvent_t completionEvents[3] = {groupDone, recvDone, sendDone};
  const char *completionStages[3] = {"group completion", "recv completion",
                                     "send completion"};
  logStage("streams user=%p copy=%p events readyRecv=%p readySend=%p "
           "groupDone=%p recvDone=%p sendDone=%p",
           (void *)userStream, (void *)state.copyStream, (void *)readyRecv,
           (void *)readySend, (void *)groupDone, (void *)recvDone,
           (void *)sendDone);
  void *localDestination;
  unsigned char *input, *output;
  SDK(cudaMallocHost((void **)&input, bytes));
  SDK(cudaMallocHost((void **)&output, bytes));
  SDK(cudaMalloc(&state.source, bytes));
  if (staging)
    SDK(cudaMallocHost(&localDestination, bytes));
  else
    SDK(cudaMalloc(&localDestination, bytes));
  for (size_t i = 0; i < bytes; ++i)
    input[i] = pattern(i, rank);
  SDK(cudaMemcpyAsync(state.source, input, bytes, cudaMemcpyHostToDevice,
                      state.copyStream));
  state.copyKind = staging ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
  SDK(cudaMemcpyAsync(localDestination, state.source, bytes, state.copyKind,
                      state.copyStream));
  SDK(cudaEventRecord(state.copied[0], state.copyStream));
  if (!waitEvent(state.copied[0], now() + 5, "baseline copy"))
    fail("baseline copy");
  if (staging) {
    validate((unsigned char *)localDestination, rank, "baseline");
    memset(localDestination, 0, bytes);
  } else {
    SDK(cudaMemcpyAsync(output, localDestination, bytes, cudaMemcpyDeviceToHost,
                        state.copyStream));
    SDK(cudaMemsetAsync(localDestination, 0, bytes, state.copyStream));
    SDK(cudaStreamSynchronize(state.copyStream));
    validate(output, rank, "baseline");
  }
  state.destination = localDestination;
  if (ipc) {
    cudaIpcMemHandle_t exported, imported;
    SDK(cudaIpcGetMemHandle(&exported, localDestination));
    MPI_CHECK(MPI_Sendrecv(&exported, sizeof(exported), MPI_BYTE, 1 - rank, 0,
                           &imported, sizeof(imported), MPI_BYTE, 1 - rank, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    SDK(cudaIpcOpenMemHandle(&state.destination, imported,
                             cudaIpcMemLazyEnablePeerAccess));
  }
  logStage("buffers source=%p copyDestination=%p localDestination=%p",
           state.source, state.destination, localDestination);
  pthread_t worker;
  if (pthread_create(&worker, NULL, proxy, &state) != 0)
    fail("pthread_create");
  double deadline = now() + 5;
  while (!__atomic_load_n(&state.workerReady, __ATOMIC_ACQUIRE)) {
    if (now() >= deadline)
      fail("proxy initialization");
    usleep(1000);
  }
  // groupPrepareOne records recv then send readiness. With one operation
  // stream, its second readiness wait stays on that same stream.
  SDK(cudaEventRecord(readyRecv, userStream));
  SDK(cudaEventRecord(readySend, userStream));
  SDK(cudaStreamWaitEvent(userStream, readySend, 0));
  SDK(cudaLaunchHostFunc(userStream, callback, &state));
  // The proxy may proceed as soon as callback entry publishes started, just
  // as pollStart does in production; do not gate it on the main thread.
  for (int i = 0; i < recordCount; ++i) {
    logStage("begin completion record index=%d event=%p stream=%p", i + 1,
             (void *)completionEvents[i], (void *)userStream);
    cudaError_t recorded = cudaEventRecord(completionEvents[i], userStream);
    logStage("end completion record index=%d event=%p stream=%p result=%d",
             i + 1, (void *)completionEvents[i], (void *)userStream,
             (int)recorded);
    if (recorded != cudaSuccess)
      fail("completion event record");
  }
  // Match the sequential C2C planner: synchronize while the proxy is live,
  // before any host join or completion-event query can wait out its copies.
  SDK(cudaStreamSynchronize(userStream));
  if (__atomic_load_n(&state.expired, __ATOMIC_ACQUIRE))
    fail("callback deadline");
  logStage("begin pthread_join");
  int joined = pthread_join(worker, NULL);
  logStage("end pthread_join result=%d", joined);
  if (joined != 0)
    fail("pthread_join");
  if (state.workerFailed || __atomic_load_n(&state.expired, __ATOMIC_ACQUIRE))
    fail("proxy or callback completion");
  deadline = now() + 5;
  for (int i = 0; i < recordCount; ++i) {
    if (!waitEvent(completionEvents[i], deadline, completionStages[i]))
      fail("group completion");
  }
  // Outside the measured interval, confirm peer writes before validation.
  // The probe has no FIFO receive progress to provide this confirmation.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (staging) {
    validate((unsigned char *)localDestination, rank, "callback copy");
  } else {
    SDK(cudaMemcpyAsync(output, localDestination, bytes, cudaMemcpyDeviceToHost,
                        userStream));
    SDK(cudaStreamSynchronize(userStream));
    validate(output, ipc ? 1 - rank : rank, "callback copy");
  }
  if (ipc)
    SDK(cudaIpcCloseMemHandle(state.destination));
  // No owner frees its export until every peer has closed its import.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  SDK(cudaEventDestroy(groupDone));
  SDK(cudaEventDestroy(recvDone));
  SDK(cudaEventDestroy(sendDone));
  SDK(cudaEventDestroy(readyRecv));
  SDK(cudaEventDestroy(readySend));
  SDK(cudaEventDestroy(state.copied[0]));
  SDK(cudaEventDestroy(state.copied[1]));
  SDK(cudaStreamDestroy(userStream));
  SDK(cudaStreamDestroy(state.copyStream));
  if (staging)
    SDK(cudaFreeHost(localDestination));
  else
    SDK(cudaFree(localDestination));
  SDK(cudaFree(state.source));
  SDK(cudaFreeHost(input));
  SDK(cudaFreeHost(output));
  MPI_CHECK(MPI_Finalize());
  char reportPath[4096];
  snprintf(reportPath, sizeof(reportPath), "%s/rank-%d.result", argv[3], rank);
  FILE *report = fopen(reportPath, "w");
  if (report == NULL) {
    logStage("FAIL open rank result");
    return 1;
  }
  int written = fprintf(
      report, "ILUVATAR_COPY_PROBE_PASS kind=%s record_count=%d rank=%d\n",
      argv[1], recordCount, rank);
  int closed = fclose(report);
  if (written < 0 || closed != 0)
    return 1;
  logStage("ILUVATAR_COPY_PROBE_PASS kind=%s record_count=%d rank=%d", argv[1],
           recordCount, rank);
  fclose(rankLog);
  return 0;
}
