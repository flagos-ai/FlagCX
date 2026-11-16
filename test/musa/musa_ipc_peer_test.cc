/*************************************************************************
 * Standalone cross-process MUSA IPC peer test.
 *
 * Two MPI ranks, two physical GPUs (default device 6 and 7), no FlagCX
 * collective stack. The only FlagCX code exercised is the MUSA device adaptor
 * IPC path exported through flagcxDeviceHandle:
 *
 *   ipcMemHandleCreate / Get / Open / Close / Free
 *   deviceMalloc / deviceMemcpy / deviceFree / deviceSynchronize
 *
 * The handle is exchanged over MPI (same mechanism as
 * test/unittest/adaptor/coll_ipc_mem_handle.cpp), not over FlagCX.
 *
 * Cases (each prints exactly one CASE=<name> PASS/FAIL line per rank):
 *   1. handle_roundtrip       create -> get -> peer open -> close -> free
 *   2. peer_read_after_write  owner fills + deviceSynchronize, peer reads
 *   3. peer_write_visible     peer writes mapped memory, owner reads it back
 *   4. reopen_cycles          >= 8 open/close cycles on one handle
 *   5. invalid_handle         null / zeroed handle must fail cleanly
 *
 * Output contract (grepped by run_musa_ipc_peer.sh):
 *   IPC_PEER rank=<r> dev=<d> CASE=<name> PASS
 *   IPC_PEER rank=<r> dev=<d> RESULT=PASS checks=5
 * A failing run prints CASE=... FAIL and RESULT=FAIL, and exits non-zero.
 *
 * Exit codes: 0 all cases passed, 1 a case failed, 2 bad invocation/config.
 ************************************************************************/

#include <mpi.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "flagcx.h"
#include "musa_adaptor.h" /* requires -DUSE_MUSA_ADAPTOR */

/* Implemented in musa_ipc_peer_kernels.mu (compiled by mcc). */
extern "C" int flagcxIpcPeerFill(void *ptr, unsigned long long n,
                                 unsigned int seed);
extern "C" const char *flagcxIpcPeerErrorString(int code);

namespace {

constexpr int kNumCases = 5;
constexpr int kElems = 1 << 16; /* 256 KiB of uint32 */
constexpr size_t kBytes = static_cast<size_t>(kElems) * sizeof(uint32_t);
constexpr size_t kRoundTripBytes = 4096;

int g_rank = 0;
int g_dev = -1;
int g_fail = 0;

uint32_t hostPattern(uint32_t seed, uint64_t i) {
  return seed ^ static_cast<uint32_t>(i * 16777619ULL);
}

const char *resName(flagcxResult_t r) {
  switch (r) {
    case flagcxSuccess:
      return "flagcxSuccess";
    case flagcxUnhandledDeviceError:
      return "flagcxUnhandledDeviceError";
    case flagcxSystemError:
      return "flagcxSystemError";
    case flagcxInternalError:
      return "flagcxInternalError";
    case flagcxInvalidArgument:
      return "flagcxInvalidArgument";
    case flagcxInvalidUsage:
      return "flagcxInvalidUsage";
    case flagcxRemoteError:
      return "flagcxRemoteError";
    case flagcxInProgress:
      return "flagcxInProgress";
    case flagcxNumResults:
      return "flagcxNumResults";
    case flagcxNotSupported:
      return "flagcxNotSupported";
    default:
      return "flagcxUnknownResult";
  }
}

void info(const char *fmt, ...) {
  std::printf("IPC_PEER rank=%d dev=%d INFO ", g_rank, g_dev);
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
  std::fflush(stdout);
}

void casePass(const char *name, const char *extra = "") {
  std::printf("IPC_PEER rank=%d dev=%d CASE=%s PASS%s\n", g_rank, g_dev, name,
              extra);
  std::fflush(stdout);
}

void caseFail(const char *name, const char *fmt, ...) {
  ++g_fail;
  std::printf("IPC_PEER rank=%d dev=%d CASE=%s FAIL detail=", g_rank, g_dev,
              name);
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
  std::fflush(stdout);
}

int envInt(const char *name, int dflt) {
  const char *v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') {
    return dflt;
  }
  char *end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v || (end != nullptr && *end != '\0')) {
    return dflt;
  }
  return static_cast<int>(parsed);
}

/* Exchange one 64-byte MUSA IPC handle with the peer. Both ranks always call
 * this so the collective cannot desynchronise, even if local setup failed. */
void exchangeHandle(MPI_Comm comm, int peer, flagcxIpcMemHandle_t mine,
                    flagcxIpcMemHandle_t theirs, int tag) {
  unsigned char sendbuf[sizeof(musaIpcMemHandle_t)];
  unsigned char recvbuf[sizeof(musaIpcMemHandle_t)];
  std::memset(sendbuf, 0, sizeof(sendbuf));
  std::memset(recvbuf, 0, sizeof(recvbuf));
  if (mine != nullptr) {
    std::memcpy(sendbuf, &mine->base, sizeof(sendbuf));
  }
  MPI_Sendrecv(sendbuf, static_cast<int>(sizeof(sendbuf)), MPI_BYTE, peer, tag,
               recvbuf, static_cast<int>(sizeof(recvbuf)), MPI_BYTE, peer, tag,
               comm, MPI_STATUS_IGNORE);
  if (theirs != nullptr) {
    std::memcpy(&theirs->base, recvbuf, sizeof(recvbuf));
  }
}

bool verifyPattern(flagcxDeviceHandle_t h, void *devPtr, uint32_t seed,
                   const char *label) {
  std::vector<uint32_t> got(kElems, 0);
  flagcxResult_t r = h->deviceMemcpy(got.data(), devPtr, kBytes,
                                     flagcxMemcpyDeviceToHost, nullptr);
  if (r != flagcxSuccess) {
    info("%s: deviceMemcpy D2H failed (%s)", label, resName(r));
    return false;
  }
  if (h->deviceSynchronize() != flagcxSuccess) {
    info("%s: deviceSynchronize failed", label);
    return false;
  }
  for (size_t i = 0; i < static_cast<size_t>(kElems); ++i) {
    uint32_t expected = hostPattern(seed, static_cast<uint64_t>(i));
    if (got[i] != expected) {
      info("%s: mismatch at %zu got=0x%08x expected=0x%08x", label, i, got[i],
           expected);
      return false;
    }
  }
  return true;
}

/* Case 1: full create/get/open/close/free round trip. */
bool caseHandleRoundTrip(flagcxDeviceHandle_t h, MPI_Comm comm, int peer,
                         int tag) {
  bool ok = true;
  void *buf = nullptr;
  flagcxIpcMemHandle_t mine = nullptr;
  flagcxIpcMemHandle_t theirs = nullptr;
  size_t hsize = 0;

  flagcxResult_t r =
      h->deviceMalloc(&buf, kRoundTripBytes, flagcxMemDevice, nullptr);
  if (r != flagcxSuccess || buf == nullptr) {
    info("handle_roundtrip: deviceMalloc failed (%s)", resName(r));
    ok = false;
  }

  if (ok) {
    r = h->ipcMemHandleCreate(&mine, &hsize);
    if (r != flagcxSuccess || mine == nullptr) {
      info("handle_roundtrip: ipcMemHandleCreate failed (%s)", resName(r));
      ok = false;
    } else if (hsize != sizeof(musaIpcMemHandle_t)) {
      info("handle_roundtrip: handle size %zu != %zu", hsize,
           sizeof(musaIpcMemHandle_t));
      ok = false;
    }
  }

  if (ok) {
    r = h->ipcMemHandleGet(mine, buf);
    if (r != flagcxSuccess) {
      info("handle_roundtrip: ipcMemHandleGet failed (%s)", resName(r));
      ok = false;
    }
  }

  /* The peer receives a handle through its own opaque handle object, exactly
   * like the real cross-process hand-off. */
  if (ok) {
    r = h->ipcMemHandleCreate(&theirs, nullptr);
    if (r != flagcxSuccess || theirs == nullptr) {
      info("handle_roundtrip: peer handle alloc failed (%s)", resName(r));
      ok = false;
    }
  }

  int preLocal = ok ? 1 : 0;
  int preAll = 0;
  MPI_Allreduce(&preLocal, &preAll, 1, MPI_INT, MPI_MIN, comm);
  if (preAll == 0) {
    info("handle_roundtrip: local setup failed on at least one rank");
    if (theirs != nullptr) {
      h->ipcMemHandleFree(theirs);
    }
    if (mine != nullptr) {
      h->ipcMemHandleFree(mine);
    }
    if (buf != nullptr) {
      h->deviceFree(buf, flagcxMemDevice, nullptr);
    }
    caseFail("handle_roundtrip", "create/get failed");
    return false;
  }

  exchangeHandle(comm, peer, mine, theirs, tag);

  void *mapped = nullptr;
  r = h->ipcMemHandleOpen(theirs, &mapped);
  if (r != flagcxSuccess || mapped == nullptr) {
    info("handle_roundtrip: ipcMemHandleOpen failed (%s)", resName(r));
    ok = false;
  }
  if (mapped != nullptr) {
    /* An open that hands back this rank's own allocation would make every
     * later peer case meaningless, so reject it explicitly. */
    if (mapped == buf) {
      info("handle_roundtrip: ipcMemHandleOpen returned the local buffer");
      ok = false;
    }
    r = h->ipcMemHandleClose(mapped);
    if (r != flagcxSuccess) {
      info("handle_roundtrip: ipcMemHandleClose failed (%s)", resName(r));
      ok = false;
    }
  }

  int postLocal = ok ? 1 : 0;
  int postAll = 0;
  MPI_Allreduce(&postLocal, &postAll, 1, MPI_INT, MPI_MIN, comm);

  if (theirs != nullptr) {
    h->ipcMemHandleFree(theirs);
  }
  if (mine != nullptr) {
    h->ipcMemHandleFree(mine);
  }
  if (buf != nullptr) {
    h->deviceFree(buf, flagcxMemDevice, nullptr);
  }

  if (postAll != 0) {
    casePass("handle_roundtrip");
    return true;
  }
  caseFail("handle_roundtrip", "open/close failed");
  return false;
}

/* Cases 2-4 share one pair of exported buffers. */
bool caseDataExchange(flagcxDeviceHandle_t h, MPI_Comm comm, int peer,
                      int cycles) {
  bool setup = true;
  void *own = nullptr;
  void *scratch = nullptr;
  void *peerView = nullptr;
  flagcxIpcMemHandle_t mine = nullptr;
  flagcxIpcMemHandle_t theirs = nullptr;

  flagcxResult_t r = h->deviceMalloc(&own, kBytes, flagcxMemDevice, nullptr);
  if (r != flagcxSuccess || own == nullptr) {
    info("data_exchange: own deviceMalloc failed (%s)", resName(r));
    setup = false;
  }
  r = h->deviceMalloc(&scratch, kBytes, flagcxMemDevice, nullptr);
  if (r != flagcxSuccess || scratch == nullptr) {
    info("data_exchange: scratch deviceMalloc failed (%s)", resName(r));
    setup = false;
  }

  if (setup) {
    r = h->ipcMemHandleCreate(&mine, nullptr);
    if (r != flagcxSuccess || mine == nullptr) {
      info("data_exchange: ipcMemHandleCreate failed (%s)", resName(r));
      setup = false;
    }
  }
  if (setup) {
    int err =
        flagcxIpcPeerFill(own, kElems, 1000u + static_cast<unsigned>(g_rank));
    if (err != 0) {
      info("data_exchange: initial fill failed (%s)",
           flagcxIpcPeerErrorString(err));
      setup = false;
    }
  }
  if (setup) {
    r = h->ipcMemHandleGet(mine, own);
    if (r != flagcxSuccess) {
      info("data_exchange: ipcMemHandleGet failed (%s)", resName(r));
      setup = false;
    }
  }
  if (setup) {
    r = h->ipcMemHandleCreate(&theirs, nullptr);
    if (r != flagcxSuccess || theirs == nullptr) {
      info("data_exchange: peer handle alloc failed (%s)", resName(r));
      setup = false;
    }
  }

  auto cleanup = [&]() {
    if (peerView != nullptr) {
      h->ipcMemHandleClose(peerView);
    }
    if (theirs != nullptr) {
      h->ipcMemHandleFree(theirs);
    }
    if (mine != nullptr) {
      h->ipcMemHandleFree(mine);
    }
    if (own != nullptr) {
      h->deviceFree(own, flagcxMemDevice, nullptr);
    }
    if (scratch != nullptr) {
      h->deviceFree(scratch, flagcxMemDevice, nullptr);
    }
  };

  int setupLocal = setup ? 1 : 0;
  int setupAll = 0;
  MPI_Allreduce(&setupLocal, &setupAll, 1, MPI_INT, MPI_MIN, comm);

  if (setupAll == 0) {
    caseFail("peer_read_after_write", "shared setup failed");
    caseFail("peer_write_visible", "shared setup failed");
    caseFail("reopen_cycles", "shared setup failed");
    cleanup();
    return false;
  }

  /* Exchange only once both ranks have a valid exported buffer, so a failed
   * rank never feeds a zeroed handle into ipcMemHandleOpen. */
  exchangeHandle(comm, peer, mine, theirs, 200);

  r = h->ipcMemHandleOpen(theirs, &peerView);
  if (r != flagcxSuccess || peerView == nullptr) {
    info("data_exchange: ipcMemHandleOpen failed (%s)", resName(r));
    setup = false;
  }
  int openLocal = setup ? 1 : 0;
  int openAll = 0;
  MPI_Allreduce(&openLocal, &openAll, 1, MPI_INT, MPI_MIN, comm);
  if (openAll == 0) {
    caseFail("peer_read_after_write", "peer handle open failed");
    caseFail("peer_write_visible", "peer handle open failed");
    caseFail("reopen_cycles", "peer handle open failed");
    cleanup();
    return false;
  }

  /* Case 2: peer read-after-write. The owner filled and synchronised before
   * sending the handle; this rank copies through the IPC mapping and checks
   * the owner's pattern. */
  bool c2 = true;
  r = h->deviceMemcpy(scratch, peerView, kBytes, flagcxMemcpyDeviceToDevice,
                      nullptr);
  if (r != flagcxSuccess) {
    info("peer_read_after_write: D2D copy failed (%s)", resName(r));
    c2 = false;
  }
  if (c2 && h->deviceSynchronize() != flagcxSuccess) {
    info("peer_read_after_write: deviceSynchronize failed");
    c2 = false;
  }
  if (c2) {
    c2 = verifyPattern(h, scratch, 1000u + static_cast<unsigned>(1 - g_rank),
                       "peer_read_after_write");
  }
  int c2Local = c2 ? 1 : 0;
  int c2All = 0;
  MPI_Allreduce(&c2Local, &c2All, 1, MPI_INT, MPI_MIN, comm);
  if (c2All != 0) {
    casePass("peer_read_after_write");
  } else {
    caseFail("peer_read_after_write", "local=%s peer=%s", c2 ? "ok" : "failed",
             c2All ? "ok" : "failed");
  }

  /* Case 3: peer write visible on owner, both directions at once. Each rank
   * writes a new pattern into the peer's buffer through the IPC mapping. */
  bool c3 = true;
  int err = flagcxIpcPeerFill(peerView, kElems,
                              2000u + static_cast<unsigned>(g_rank));
  if (err != 0) {
    info("peer_write_visible: fill into peer mapping failed (%s)",
         flagcxIpcPeerErrorString(err));
    c3 = false;
  }
  MPI_Barrier(comm); /* both peer writes are host-synchronised before reads */
  if (c3) {
    c3 = verifyPattern(h, own, 2000u + static_cast<unsigned>(1 - g_rank),
                       "peer_write_visible");
  }
  int c3Local = c3 ? 1 : 0;
  int c3All = 0;
  MPI_Allreduce(&c3Local, &c3All, 1, MPI_INT, MPI_MIN, comm);
  if (c3All != 0) {
    casePass("peer_write_visible");
  } else {
    caseFail("peer_write_visible", "local=%s peer=%s", c3 ? "ok" : "failed",
             c3All ? "ok" : "failed");
  }

  /* Case 4: repeated open/close on the same peer handle. Each open must yield
   * a distinct live pointer and each close must succeed; >= 8 cycles proves
   * there is no handle leak or hang. The last open is also dereferenced: case
   * 3 left this rank's own pattern (2000+rank) in the peer buffer, so copying
   * through a fresh mapping and verifying that seed proves the reopened handle
   * still maps the peer allocation instead of a local alias or a stale page. */
  bool c4 = true;
  if (peerView != nullptr) {
    if (h->ipcMemHandleClose(peerView) != flagcxSuccess) {
      info("reopen_cycles: initial close failed");
      c4 = false;
    }
    peerView = nullptr;
  }
  for (int i = 0; i < cycles && c4; ++i) {
    void *mapped = nullptr;
    r = h->ipcMemHandleOpen(theirs, &mapped);
    if (r != flagcxSuccess || mapped == nullptr) {
      info("reopen_cycles: open %d failed (%s)", i, resName(r));
      c4 = false;
      break;
    }
    if (mapped == own || mapped == scratch) {
      info("reopen_cycles: open %d aliased a local allocation", i);
      h->ipcMemHandleClose(mapped);
      c4 = false;
      break;
    }
    r = h->ipcMemHandleClose(mapped);
    if (r != flagcxSuccess) {
      info("reopen_cycles: close %d failed (%s)", i, resName(r));
      c4 = false;
      break;
    }
  }
  if (c4) {
    void *mapped = nullptr;
    r = h->ipcMemHandleOpen(theirs, &mapped);
    if (r != flagcxSuccess || mapped == nullptr) {
      info("reopen_cycles: final reopen failed (%s)", resName(r));
      c4 = false;
    } else if (mapped == own || mapped == scratch) {
      info("reopen_cycles: final reopen aliased a local allocation");
      c4 = false;
    } else {
      r = h->deviceMemcpy(scratch, mapped, kBytes, flagcxMemcpyDeviceToDevice,
                          nullptr);
      if (r != flagcxSuccess) {
        info("reopen_cycles: final D2D copy failed (%s)", resName(r));
        c4 = false;
      } else if (h->deviceSynchronize() != flagcxSuccess) {
        info("reopen_cycles: final deviceSynchronize failed");
        c4 = false;
      } else {
        c4 = verifyPattern(h, scratch, 2000u + static_cast<unsigned>(g_rank),
                           "reopen_cycles");
      }
    }
    if (mapped != nullptr) {
      h->ipcMemHandleClose(mapped);
    }
  }
  int c4Local = c4 ? 1 : 0;
  int c4All = 0;
  MPI_Allreduce(&c4Local, &c4All, 1, MPI_INT, MPI_MIN, comm);
  if (c4All != 0) {
    char extra[64];
    std::snprintf(extra, sizeof(extra), " cycles=%d", cycles);
    casePass("reopen_cycles", extra);
  } else {
    caseFail("reopen_cycles", "local=%s peer=%s cycles=%d",
             c4 ? "ok" : "failed", c4All ? "ok" : "failed", cycles);
  }

  if (peerView != nullptr) {
    h->ipcMemHandleClose(peerView);
  }
  if (theirs != nullptr) {
    h->ipcMemHandleFree(theirs);
  }
  if (mine != nullptr) {
    h->ipcMemHandleFree(mine);
  }
  if (own != nullptr) {
    h->deviceFree(own, flagcxMemDevice, nullptr);
  }
  if (scratch != nullptr) {
    h->deviceFree(scratch, flagcxMemDevice, nullptr);
  }
  return (c2All != 0) && (c3All != 0) && (c4All != 0);
}

/* Case 5: invalid input must fail cleanly, without hanging or crashing. */
void caseInvalidHandle(flagcxDeviceHandle_t h) {
  bool ok = true;

  void *mapped = nullptr;
  flagcxResult_t r = h->ipcMemHandleOpen(nullptr, &mapped);
  if (r != flagcxInvalidArgument || mapped != nullptr) {
    info("invalid_handle: null handle returned %s (mapped=%p)", resName(r),
         mapped);
    ok = false;
  }

  if (h->ipcMemHandleClose(nullptr) != flagcxInvalidArgument) {
    info("invalid_handle: close(nullptr) did not return flagcxInvalidArgument");
    ok = false;
  }

  flagcxIpcMemHandle_t bad = nullptr;
  size_t badSize = 0;
  r = h->ipcMemHandleCreate(&bad, &badSize);
  if (r != flagcxSuccess || bad == nullptr) {
    info("invalid_handle: could not allocate a scratch handle (%s)",
         resName(r));
    ok = false;
  } else {
    if (badSize != sizeof(musaIpcMemHandle_t)) {
      info("invalid_handle: scratch handle size %zu != %zu", badSize,
           sizeof(musaIpcMemHandle_t));
      ok = false;
    }
    std::memset(&bad->base, 0, sizeof(bad->base));
    void *out = nullptr;
    r = h->ipcMemHandleOpen(bad, &out);
    if (r == flagcxSuccess) {
      /* Accepting a zeroed handle would mean the driver hands out a bogus
       * pointer; that is a gate failure, not a pass. */
      info("invalid_handle: zeroed handle unexpectedly opened (%p)", out);
      if (out != nullptr) {
        h->ipcMemHandleClose(out);
      }
      ok = false;
    } else if (out != nullptr) {
      info("invalid_handle: failed open left non-null pointer %p", out);
      h->ipcMemHandleClose(out);
      ok = false;
    } else {
      info("invalid_handle: zeroed handle rejected cleanly (%s)", resName(r));
    }
    h->ipcMemHandleFree(bad);
  }

  if (ok) {
    casePass("invalid_handle");
  } else {
    caseFail("invalid_handle", "expected a clean error return");
  }
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  g_rank = rank;

  const int devA = envInt("TEST_DEV_A", 6);
  const int devB = envInt("TEST_DEV_B", 7);
  g_dev = (rank == 0) ? devA : devB;

  if (size != 2 || devA == devB || devA < 0 || devB < 0) {
    if (rank == 0) {
      std::printf("IPC_PEER ERROR need exactly 2 ranks and 2 distinct "
                  "non-negative devices (size=%d devA=%d devB=%d)\n",
                  size, devA, devB);
      std::fflush(stdout);
    }
    MPI_Finalize();
    return 2;
  }

  int cycles = envInt("IPC_PEER_CYCLES", 8);
  if (cycles < 8) {
    cycles = 8;
  }
  if (cycles > 64) {
    cycles = 64;
  }

  flagcxDeviceHandle_t h = nullptr;
  bool ok = true;
  flagcxResult_t r = flagcxDeviceHandleInit(&h);
  if (r != flagcxSuccess || h == nullptr) {
    info("flagcxDeviceHandleInit failed (%s)", resName(r));
    ok = false;
  }

  int gotDev = -1;
  int devCount = 0;
  if (ok) {
    r = h->setDevice(g_dev);
    if (r != flagcxSuccess) {
      info("setDevice(%d) failed (%s)", g_dev, resName(r));
      ok = false;
    }
  }
  if (ok) {
    r = h->getDevice(&gotDev);
    if (r != flagcxSuccess || gotDev != g_dev) {
      info("getDevice returned %d, expected %d (%s)", gotDev, g_dev,
           resName(r));
      ok = false;
    }
  }
  if (ok) {
    r = h->getDeviceCount(&devCount);
    if (r != flagcxSuccess) {
      info("getDeviceCount failed (%s)", resName(r));
      ok = false;
    }
  }

  /* Distinct physical devices, proven at the hardware level. getDevice() only
   * confirms the logical index we asked for; two indices could still alias one
   * physical device, which would let every peer case succeed inside a single
   * GPU. Exchange the PCI bus ids and require them to differ. */
  char pciId[64] = {0};
  if (ok) {
    if (musaDeviceGetPCIBusId(pciId, static_cast<int>(sizeof(pciId)), g_dev) !=
        musaSuccess) {
      info("musaDeviceGetPCIBusId(%d) failed", g_dev);
      ok = false;
    }
  }
  char peerPciId[64] = {0};
  MPI_Sendrecv(pciId, 64, MPI_CHAR, 1 - rank, 77, peerPciId, 64, MPI_CHAR,
               1 - rank, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (ok && std::strcmp(pciId, peerPciId) == 0) {
    info("both ranks resolved to the same physical device (%s)", pciId);
    ok = false;
  }

  const int need = (devA > devB ? devA : devB) + 1;
  int minCount = 0;
  int maxNeed = 0;
  MPI_Allreduce(&devCount, &minCount, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&need, &maxNeed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (ok && minCount < maxNeed) {
    info("only %d visible devices, need at least %d", minCount, maxNeed);
    ok = false;
  }

  int setupLocal = ok ? 1 : 0;
  int setupAll = 0;
  MPI_Allreduce(&setupLocal, &setupAll, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (setupAll == 0) {
    std::printf("IPC_PEER rank=%d dev=%d RESULT=FAIL checks=0 failures=1 "
                "detail=setup\n",
                rank, g_dev);
    std::fflush(stdout);
    if (h != nullptr) {
      flagcxDeviceHandleFree(h);
    }
    MPI_Finalize();
    return 1;
  }

  info("visible=%d cycles=%d", devCount, cycles);

  caseHandleRoundTrip(h, MPI_COMM_WORLD, 1 - rank, 100);
  caseDataExchange(h, MPI_COMM_WORLD, 1 - rank, cycles);
  caseInvalidHandle(h);

  int globalFail = 0;
  MPI_Allreduce(&g_fail, &globalFail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (globalFail == 0) {
    std::printf("IPC_PEER rank=%d dev=%d RESULT=PASS checks=%d\n", rank, g_dev,
                kNumCases);
  } else {
    std::printf("IPC_PEER rank=%d dev=%d RESULT=FAIL checks=%d failures=%d\n",
                rank, g_dev, kNumCases, globalFail);
  }
  std::fflush(stdout);

  if (h != nullptr) {
    flagcxDeviceHandleFree(h);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return globalFail == 0 ? 0 : 1;
}
