// SDK-only B/C control for the blocking callback used by heterogeneous groups.
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
  logStage("begin %s event=%p", stage, (void *)event);
  cudaError_t result;
  do {
    result = cudaEventQuery(event);
    if (result != cudaErrorNotReady)
      break;
    usleep(1000);
  } while (now() < deadline);
  logStage("end %s result=%d", stage, (int)result);
  return result == cudaSuccess;
}

struct CopyState {
  int device;
  int started;
  int gate;
  int done;
  int expired;
  int workerReady;
  int workerFailed;
  void *source;
  void *destination;
  cudaMemcpyKind copyKind;
  cudaStream_t copyStream;
  cudaEvent_t ready;
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
  while (!__atomic_load_n(&state->started, __ATOMIC_ACQUIRE) ||
         !__atomic_load_n(&state->gate, __ATOMIC_ACQUIRE)) {
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
      (strcmp(argv[2], "B") && strcmp(argv[2], "C"))) {
    fprintf(stderr, "Usage: %s local|ipc|staging B|C result-directory\n",
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
  int withCompletion = strcmp(argv[2], "C") == 0;
  if (ranks != (ipc ? 2 : 1) || provided < MPI_THREAD_MULTIPLE)
    fail("MPI rank count or thread support");
  struct CopyState state = {};
  state.device = ipc && rank == 1 ? 2 : 0;
  logStage("case kind=%s control=%s device=%d ranks=%d", argv[1], argv[2],
           state.device, ranks);
  int devices;
  SDK(cudaGetDeviceCount(&devices));
  if (devices <= state.device)
    fail("required device unavailable");
  SDK(cudaSetDevice(state.device));
  cudaStream_t userStream;
  cudaEvent_t groupDone;
  SDK(cudaStreamCreateWithFlags(&userStream, cudaStreamNonBlocking));
  SDK(cudaStreamCreateWithFlags(&state.copyStream, cudaStreamNonBlocking));
  SDK(cudaEventCreateWithFlags(&state.ready, cudaEventDisableTiming));
  SDK(cudaEventCreateWithFlags(&state.copied[0], cudaEventDisableTiming));
  SDK(cudaEventCreateWithFlags(&state.copied[1], cudaEventDisableTiming));
  // Allocate this event in both controls; only its Record differs.
  SDK(cudaEventCreateWithFlags(&groupDone, cudaEventDisableTiming));
  logStage("streams user=%p copy=%p events ready=%p groupDone=%p",
           (void *)userStream, (void *)state.copyStream, (void *)state.ready,
           (void *)groupDone);
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
  SDK(cudaEventRecord(state.ready, userStream));
  SDK(cudaLaunchHostFunc(userStream, callback, &state));
  if (withCompletion)
    SDK(cudaEventRecord(groupDone, userStream));
  __atomic_store_n(&state.gate, 1, __ATOMIC_RELEASE);
  if (pthread_join(worker, NULL) != 0)
    fail("pthread_join");
  if (state.workerFailed || __atomic_load_n(&state.expired, __ATOMIC_ACQUIRE))
    fail("proxy or callback completion");
  if (withCompletion && !waitEvent(groupDone, now() + 5, "group completion"))
    fail("group completion");
  SDK(cudaStreamSynchronize(userStream));
  if (__atomic_load_n(&state.expired, __ATOMIC_ACQUIRE))
    fail("callback deadline");
  // Every importing rank has completed its writes before the owner reads.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (staging) {
    validate((unsigned char *)localDestination, rank, "callback copy");
  } else {
    SDK(cudaMemcpyAsync(output, localDestination, bytes, cudaMemcpyDeviceToHost,
                        state.copyStream));
    SDK(cudaStreamSynchronize(state.copyStream));
    validate(output, ipc ? 1 - rank : rank, "callback copy");
  }
  if (ipc)
    SDK(cudaIpcCloseMemHandle(state.destination));
  // No owner frees its export until every peer has closed its import.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  SDK(cudaEventDestroy(groupDone));
  SDK(cudaEventDestroy(state.ready));
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
  int written =
      fprintf(report, "ILUVATAR_COPY_PROBE_PASS kind=%s case=%s rank=%d\n",
              argv[1], argv[2], rank);
  int closed = fclose(report);
  if (written < 0 || closed != 0)
    return 1;
  logStage("ILUVATAR_COPY_PROBE_PASS kind=%s case=%s rank=%d", argv[1], argv[2],
           rank);
  fclose(rankLog);
  return 0;
}
