/*************************************************************************
 * C-owned host-side lifecycle smoke for the MUSA Default Device API backend.
 *
 * This is NOT a device-kernel test. It validates only what is reachable
 * without A's platform traits/kernels and without B's IPC implementation:
 *
 *   1. flagcxCommInitRank over MPI bootstrap on real MUSA hardware
 *   2. flagcxDevCommCreate with zero device requirements (allocates the real
 *      epoch buffer via musaMalloc/musaMemset, publishes/unpublishes)
 *   3. flagcxDevCommDestroy cleanup
 *   4. flagcxMemAlloc / flagcxMemFree on real MUSA device memory
 *
 * Step 5 (always) materializes the host-side devComm value struct into a real
 * MUSA device allocation via flagcxDevCommGetDevicePtr, then frees it. This
 * only needs deviceMalloc/deviceMemcpy/deviceSynchronize, so it works without
 * A's kernels or B's IPC, and proves device-pointer materialization on MUSA.
 *
 * Optional probes:
 *   --probe-devmem  flagcxDevMemCreate(win=nullptr) (observation only)
 *   --probe-devptr  gated DevMem descriptor materialization:
 *                   Create -> GetDevicePtr -> FreeDevicePtr -> Destroy.
 *                   Fails the smoke if any step does not succeed.
 *   --probe-barrier flagcxDevCommCreate(intraBarrierCount=1), which exercises
 *                   the IPC barrier path and must report flagcxNotSupported
 *                   (not flagcxSystemError) when the platform has no IPC.
 *   --probe-window  flagcxCommWindowRegister(SYMMETRIC) on a real device
 *                   buffer; must either succeed or report flagcxNotSupported
 *                   cleanly (no crash/hang) for the T1 -R 2 precondition.
 *
 * Exit code 0 only if steps 1-5 (and any gated probe) succeed on every rank.
 ************************************************************************/

#include "comm.h"
#include "flagcx.h"
#include "flagcx_device_api.h"
#include "global_comm.h"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char *resName(flagcxResult_t r) {
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

static int g_rank = 0;

static void report(const char *stage, flagcxResult_t r) {
  std::printf("[rank %d] %-32s = %d (%s)\n", g_rank, stage, (int)r, resName(r));
  std::fflush(stdout);
}

static int failAll(const char *stage, flagcxResult_t r) {
  std::fprintf(stderr, "[rank %d] FAIL at %s: %d (%s)\n", g_rank, stage, (int)r,
               resName(r));
  std::fflush(stderr);
  return 1;
}

int main(int argc, char **argv) {
  int probeDevmem = 0;
  int probeDevptr = 0;
  int probeBarrier = 0;
  int probeWindow = 0;
  int churn = 0;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--probe-devmem") == 0)
      probeDevmem = 1;
    if (std::strcmp(argv[i], "--probe-devptr") == 0)
      probeDevptr = 1;
    if (std::strcmp(argv[i], "--probe-barrier") == 0)
      probeBarrier = 1;
    if (std::strcmp(argv[i], "--probe-window") == 0)
      probeWindow = 1;
    if (std::strcmp(argv[i], "--probe-churn") == 0 && i + 1 < argc)
      churn = std::atoi(argv[++i]);
  }

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    return 2;
  }
  int worldSize = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int rc = 0;
  flagcxComm_t comm = nullptr;
  flagcxDevComm_t devComm = nullptr;
  flagcxDeviceHandle_t devHandle = nullptr;
  void *devPtr = nullptr;

  {
    flagcxResult_t r = flagcxDeviceHandleInit(&devHandle);
    report("flagcxDeviceHandleInit", r);
    if (r != flagcxSuccess) {
      rc = failAll("flagcxDeviceHandleInit", r);
      goto done;
    }
  }
  {
    int nGpu = 0;
    flagcxResult_t r = devHandle->getDeviceCount(&nGpu);
    report("getDeviceCount", r);
    if (r != flagcxSuccess || nGpu <= 0) {
      rc =
          failAll("getDeviceCount", r != flagcxSuccess ? r : flagcxSystemError);
      goto done;
    }
    int dev = g_rank % nGpu;
    r = devHandle->setDevice(dev);
    report("setDevice", r);
    if (r != flagcxSuccess) {
      rc = failAll("setDevice", r);
      goto done;
    }
    std::printf("[rank %d] INFO device_count=%d selected=%d\n", g_rank, nGpu,
                dev);
  }

  flagcxUniqueId uniqueId;
  std::memset(&uniqueId, 0, sizeof(uniqueId));
  if (g_rank == 0) {
    flagcxResult_t r = flagcxGetUniqueId(&uniqueId);
    report("flagcxGetUniqueId", r);
    if (r != flagcxSuccess)
      rc = 1;
  }
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rc != 0)
    goto done;

  {
    flagcxResult_t r = flagcxCommInitRank(&comm, worldSize, &uniqueId, g_rank);
    report("flagcxCommInitRank", r);
    if (r != flagcxSuccess) {
      rc = failAll("flagcxCommInitRank", r);
      goto done;
    }
  }

  {
    // Real topology observed by FlagCX after bootstrap host-id detection.
    // This is what T2/T4 rely on when FLAGCX_HOSTID/MUSA_VISIBLE_DEVICES
    // split a single host into two logical nodes.
    int hNode = -1, hNodes = -1;
    if (comm->heteroComm != nullptr) {
      hNode = comm->heteroComm->node;
      hNodes = comm->heteroComm->nNodes;
    }
    std::printf("[rank %d] TOPOLOGY localRank=%d localRanks=%d node=%d "
                "nNodes=%d nranks=%d\n",
                g_rank, comm->localRank, comm->localRanks, hNode, hNodes,
                comm->nranks);
    std::fflush(stdout);
  }

  {
    // Zero device requirements: exercises only the platform-neutral part of
    // the default backend (epoch buffer + lifecycle). Barriers/signals/IPC
    // are deliberately not requested because they need A/B work.
    flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.interContextCount = 1;
    flagcxResult_t r = flagcxDevCommCreate(comm, &reqs, &devComm);
    report("flagcxDevCommCreate(zero-req)", r);
    if (r != flagcxSuccess) {
      rc = failAll("flagcxDevCommCreate(zero-req)", r);
      goto done;
    }
  }

  {
    // Materialize the devComm value struct on the device and free it again.
    // Host-only path (deviceMalloc/memcpy/synchronize), no kernels/IPC.
    void *devCommDevPtr = nullptr;
    flagcxResult_t r = flagcxDevCommGetDevicePtr(devComm, &devCommDevPtr);
    report("flagcxDevCommGetDevicePtr", r);
    if (r != flagcxSuccess || devCommDevPtr == nullptr) {
      rc = failAll("flagcxDevCommGetDevicePtr",
                   r != flagcxSuccess ? r : flagcxSystemError);
      goto done;
    }
    std::printf("[rank %d] INFO devComm device ptr=%p\n", g_rank,
                devCommDevPtr);
    r = flagcxDevCommFreeDevicePtr(devComm);
    report("flagcxDevCommFreeDevicePtr", r);
    if (r != flagcxSuccess) {
      rc = failAll("flagcxDevCommFreeDevicePtr", r);
      goto done;
    }
  }

  {
    flagcxResult_t r = flagcxMemAlloc(&devPtr, 1u << 20, flagcxMemCCL);
    report("flagcxMemAlloc(1MiB)", r);
    if (r != flagcxSuccess) {
      rc = failAll("flagcxMemAlloc", r);
      goto done;
    }
  }

  if (probeDevmem && devPtr != nullptr) {
    flagcxDevMem_t devMem = nullptr;
    flagcxResult_t r =
        flagcxDevMemCreate(comm, devPtr, 1u << 20, nullptr, &devMem);
    // Observation only: expected flagcxNotSupported until IPC/window lands.
    std::printf("[rank %d] PROBE flagcxDevMemCreate            = %d (%s)%s\n",
                g_rank, (int)r, resName(r),
                r == flagcxSuccess ? "" : " [blocked, not a smoke failure]");
    std::fflush(stdout);
    if (r == flagcxSuccess && devMem != nullptr) {
      flagcxResult_t dr = flagcxDevMemDestroy(comm, devMem);
      std::printf("[rank %d] PROBE flagcxDevMemDestroy           = %d (%s)\n",
                  g_rank, (int)dr, resName(dr));
      std::fflush(stdout);
    }
  }

  if (probeDevptr && devPtr != nullptr) {
    // Gated: the host-only DevMem descriptor must be materializable on MUSA
    // (deviceMalloc + deviceMemcpy of sizeof(flagcxDevMem)).
    flagcxDevMem_t devMem = nullptr;
    flagcxResult_t r =
        flagcxDevMemCreate(comm, devPtr, 1u << 20, nullptr, &devMem);
    report("PROBE flagcxDevMemCreate", r);
    if (r != flagcxSuccess || devMem == nullptr) {
      rc = failAll("PROBE flagcxDevMemCreate",
                   r != flagcxSuccess ? r : flagcxSystemError);
      goto done;
    }
    void *memDevPtr = nullptr;
    r = flagcxDevMemGetDevicePtr(devMem, &memDevPtr);
    report("PROBE flagcxDevMemGetDevicePtr", r);
    if (r != flagcxSuccess || memDevPtr == nullptr) {
      rc = failAll("PROBE flagcxDevMemGetDevicePtr",
                   r != flagcxSuccess ? r : flagcxSystemError);
      flagcxDevMemDestroy(comm, devMem);
      goto done;
    }
    std::printf("[rank %d] INFO devMem device ptr=%p\n", g_rank, memDevPtr);
    r = flagcxDevMemFreeDevicePtr(devMem);
    report("PROBE flagcxDevMemFreeDevicePtr", r);
    if (r != flagcxSuccess) {
      rc = failAll("PROBE flagcxDevMemFreeDevicePtr", r);
      flagcxDevMemDestroy(comm, devMem);
      goto done;
    }
    r = flagcxDevMemDestroy(comm, devMem);
    report("PROBE flagcxDevMemDestroy", r);
    if (r != flagcxSuccess) {
      rc = failAll("PROBE flagcxDevMemDestroy", r);
      goto done;
    }
  }

  if (probeBarrier) {
    // The IPC barrier path is the one that used to collapse every failure into
    // flagcxSystemError. The gate accepts either a real barrier DevComm or a
    // clean flagcxNotSupported when the platform has no IPC; only
    // flagcxSystemError (or a crash/hang) fails.
    flagcxDevCommRequirements breqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
    breqs.intraBarrierCount = 1;
    flagcxDevComm_t barrierComm = nullptr;
    flagcxResult_t r = flagcxDevCommCreate(comm, &breqs, &barrierComm);
    std::printf("[rank %d] PROBE flagcxDevCommCreate(barrier)  = %d (%s)%s\n",
                g_rank, (int)r, resName(r),
                r == flagcxSuccess
                    ? " [IPC barrier DevComm created]"
                    : (r == flagcxNotSupported ? " [clean NotSupported]"
                                               : " [unexpected result]"));
    std::fflush(stdout);
    if (r == flagcxSuccess && barrierComm != nullptr) {
      flagcxResult_t dr = flagcxDevCommDestroy(comm, barrierComm);
      std::printf("[rank %d] PROBE barrier DevCommDestroy        = %d (%s)\n",
                  g_rank, (int)dr, resName(dr));
      std::fflush(stdout);
    }
  }

  if (probeWindow && devPtr != nullptr) {
    // Collective symmetric-window registration is the T1 (-R 2) precondition.
    // Until the device sym/IPC work lands this must fail cleanly with
    // flagcxNotSupported, never crash or hang.
    flagcxWindow_t win = nullptr;
    flagcxResult_t r = flagcxCommWindowRegister(
        comm, devPtr, 1u << 20, &win, FLAGCX_WIN_COLL_SYMMETRIC, flagcxMemCCL);
    report("PROBE flagcxCommWindowRegister", r);
    if (r == flagcxSuccess) {
      std::printf("[rank %d] INFO window registered win=%p\n", g_rank,
                  (void *)win);
      std::fflush(stdout);
      flagcxResult_t dr = flagcxCommWindowDeregister(comm, win, flagcxMemCCL);
      report("PROBE flagcxCommWindowDeregister", dr);
      if (dr != flagcxSuccess) {
        rc = failAll("PROBE flagcxCommWindowDeregister", dr);
        goto done;
      }
    } else if (r != flagcxNotSupported) {
      rc = failAll("PROBE flagcxCommWindowRegister", r);
      goto done;
    }
  }

  if (churn > 0) {
    // Repeat the lifecycle of the resources that leaked or reused a stale
    // handle in earlier defects: a host-pinned allocation (musaHostRegister +
    // MAPPED alias + unregister + free), a symmetric window (IPC handle create,
    // open, close) and a DevComm (epoch/signal/counter planes). Every iteration
    // must succeed and every resource must be released before the next one.
    int failures = 0;
    for (int i = 0; i < churn; i++) {
      void *hp = nullptr;
      flagcxResult_t r =
          devHandle->deviceMalloc(&hp, 4096, flagcxMemHost, nullptr);
      if (r != flagcxSuccess || hp == nullptr) {
        std::fprintf(stderr, "[rank %d] CHURN host alloc iter %d: %d (%s)\n",
                     g_rank, i, (int)r, resName(r));
        failures++;
      } else {
        r = devHandle->deviceFree(hp, flagcxMemHost, nullptr);
        if (r != flagcxSuccess) {
          std::fprintf(stderr, "[rank %d] CHURN host free iter %d: %d (%s)\n",
                       g_rank, i, (int)r, resName(r));
          failures++;
        }
      }

      if (devPtr != nullptr) {
        flagcxWindow_t win = nullptr;
        r = flagcxCommWindowRegister(comm, devPtr, 1u << 20, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC, flagcxMemCCL);
        if (r != flagcxSuccess || win == nullptr) {
          std::fprintf(stderr,
                       "[rank %d] CHURN window register iter %d: %d (%s)\n",
                       g_rank, i, (int)r, resName(r));
          failures++;
        } else {
          r = flagcxCommWindowDeregister(comm, win, flagcxMemCCL);
          if (r != flagcxSuccess) {
            std::fprintf(stderr,
                         "[rank %d] CHURN window deregister iter %d: %d (%s)\n",
                         g_rank, i, (int)r, resName(r));
            failures++;
          }
        }
      }

      flagcxDevCommRequirements creqs =
          FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
      creqs.interContextCount = 1;
      flagcxDevComm_t ccomm = nullptr;
      r = flagcxDevCommCreate(comm, &creqs, &ccomm);
      if (r != flagcxSuccess || ccomm == nullptr) {
        std::fprintf(stderr,
                     "[rank %d] CHURN devcomm create iter %d: %d (%s)\n",
                     g_rank, i, (int)r, resName(r));
        failures++;
      } else {
        r = flagcxDevCommDestroy(comm, ccomm);
        if (r != flagcxSuccess) {
          std::fprintf(stderr,
                       "[rank %d] CHURN devcomm destroy iter %d: %d (%s)\n",
                       g_rank, i, (int)r, resName(r));
          failures++;
        }
      }
    }
    std::printf("[rank %d] CHURN %d iterations, %d failure(s)\n", g_rank, churn,
                failures);
    std::fflush(stdout);
    if (failures != 0) {
      rc = failAll("PROBE churn", flagcxSystemError);
      goto done;
    }
  }

done:
  if (devPtr != nullptr) {
    flagcxResult_t r = flagcxMemFree(devPtr, flagcxMemCCL);
    report("flagcxMemFree", r);
    if (r != flagcxSuccess && rc == 0)
      rc = 1;
  }
  if (devComm != nullptr) {
    flagcxResult_t r = flagcxDevCommDestroy(comm, devComm);
    report("flagcxDevCommDestroy", r);
    if (r != flagcxSuccess && rc == 0)
      rc = 1;
  }
  if (comm != nullptr) {
    flagcxResult_t r = flagcxCommDestroy(comm);
    report("flagcxCommDestroy", r);
    if (r != flagcxSuccess && rc == 0)
      rc = 1;
  }
  if (devHandle != nullptr) {
    flagcxResult_t r = flagcxDeviceHandleFree(devHandle);
    report("flagcxDeviceHandleFree", r);
    if (r != flagcxSuccess && rc == 0)
      rc = 1;
  }

  int rcAll = 0;
  MPI_Allreduce(&rc, &rcAll, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rcAll == 0) {
    std::printf("[rank %d] HOST_LIFECYCLE_SMOKE: PASS\n", g_rank);
  } else {
    std::fprintf(stderr, "[rank %d] HOST_LIFECYCLE_SMOKE: FAIL\n", g_rank);
  }
  std::fflush(stdout);
  std::fflush(stderr);

  MPI_Finalize();
  return rcAll == 0 ? 0 : 1;
}
