/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "alloc.h"
#include "coll_proxy_transport.h"
#include "flagcx_hetero.h"
#include "group.h"
#include "proxy.h"
#include "runner.h"
#include "runner_result.h"
#include "uni_runner_algorithm.h"
#include "uni_runner_impl.h"
// bootstrap.h wraps its dependencies in extern "C"; load C++ dependencies
// first.
#include "bootstrap.h"
#include <cerrno>
#include <cstdlib>

FLAGCX_PARAM(UniRunnerGroupSize, "UNIRUNNER_GROUPSIZE", 0);

flagcxResult_t flagcxReadUniRunnerAlgorithm(int64_t *value) {
  if (value == nullptr)
    return flagcxInvalidArgument;
  const char *configured = flagcxGetEnv("FLAGCX_UNIRUNNER_ALGO");
  if (configured == nullptr || configured[0] == '\0') {
    *value = -1;
    return flagcxSuccess;
  }
  errno = 0;
  char *end = nullptr;
  const long long parsed = strtoll(configured, &end, 0);
  if (errno != 0 || end == configured || *end != '\0' ||
      parsed < uniRunnerAlgorithmRingAR ||
      parsed > uniRunnerAlgorithmGemmReduceScatterDsl)
    return flagcxInvalidArgument;
  *value = parsed;
  return flagcxSuccess;
}

struct uniRunnerAlgorithmChoice {
  int64_t algorithm;
  int result;
  int64_t groupSize;
};

flagcxResult_t agreeUniRunnerAlgorithm(flagcxComm_t comm,
                                       UniRunnerAlgorithm selected,
                                       flagcxResult_t localResult,
                                       int64_t groupSize) {
  if (comm == nullptr || comm->bootstrap == nullptr || comm->nranks <= 0 ||
      comm->rank < 0 || comm->rank >= comm->nranks)
    return flagcxInvalidArgument;
  uniRunnerAlgorithmChoice *choices = nullptr;
  flagcxResult_t allocationResult =
      flagcxCalloc(&choices, static_cast<size_t>(comm->nranks));
  flagcxResult_t commonPrepare = flagcxSuccess;
  flagcxResult_t result =
      flagcxConvergeBootstrapStatus(comm, allocationResult, &commonPrepare);
  if (result != flagcxSuccess || commonPrepare != flagcxSuccess) {
    free(choices);
    return result != flagcxSuccess ? result : commonPrepare;
  }
  choices[comm->rank] = {static_cast<int64_t>(selected),
                         static_cast<int>(localResult), groupSize};
  result = bootstrapCollAllGather(comm->bootstrap, choices,
                                  sizeof(uniRunnerAlgorithmChoice));
  if (result == flagcxSuccess) {
    for (int rank = 0; rank < comm->nranks; ++rank) {
      if (choices[rank].result != flagcxSuccess) {
        result = static_cast<flagcxResult_t>(choices[rank].result);
        break;
      }
    }
  }
  if (result == flagcxSuccess) {
    for (int rank = 0; rank < comm->nranks; ++rank) {
      if (choices[rank].algorithm != choices[0].algorithm ||
          (selected == uniRunnerAlgorithmGroupedAR &&
           choices[rank].groupSize != choices[0].groupSize)) {
        result = flagcxInvalidUsage;
        break;
      }
    }
  }
  free(choices);
  return result;
}

static int resolveUniRunnerGroupedAGGroupSize(flagcxComm_t comm) {
  if (comm->nranks <= 0) {
    return 0;
  }

  int groupSize = flagcxParamUniRunnerGroupSize();
  if (groupSize <= 0) {
    groupSize = comm->localRanks > 1 ? comm->localRanks : comm->nranks;
  }
  if (groupSize <= 0 || groupSize > comm->nranks ||
      comm->nranks % groupSize != 0) {
    TRACE(FLAGCX_UNIRUNNER,
          "rank %d groupedAG groupSize %d invalid for nranks %d, fallback to "
          "nranks",
          comm->rank, groupSize, comm->nranks);
    groupSize = comm->nranks;
  }
  return groupSize;
}

struct uniRunnerAllGatherSelection {
  int result;
  int64_t algorithm;
  int groupedAgEnabled;
  int streamValueSupported;
  int groupSize;
};

static flagcxResult_t
agreeUniRunnerAllGatherSelection(flagcxComm_t comm, flagcxResult_t localResult,
                                 UniRunnerAlgorithm algorithm,
                                 bool localEnabled, bool localSupported,
                                 int localGroupSize, bool *useGroupedAg) {
  if (comm == nullptr || comm->bootstrap == nullptr || comm->nranks <= 0 ||
      comm->rank < 0 || comm->rank >= comm->nranks || useGroupedAg == nullptr)
    return flagcxInvalidArgument;

  *useGroupedAg = false;
  uniRunnerAllGatherSelection *selections = nullptr;
  flagcxResult_t allocationResult =
      flagcxCalloc(&selections, static_cast<size_t>(comm->nranks));
  flagcxResult_t commonPrepare = flagcxSuccess;
  flagcxResult_t result =
      flagcxConvergeBootstrapStatus(comm, allocationResult, &commonPrepare);
  if (result != flagcxSuccess || commonPrepare != flagcxSuccess) {
    free(selections);
    return result != flagcxSuccess ? result : commonPrepare;
  }
  selections[comm->rank] = {
      static_cast<int>(localResult), static_cast<int64_t>(algorithm),
      localEnabled ? 1 : 0, localSupported ? 1 : 0, localGroupSize};
  result = bootstrapCollAllGather(comm->bootstrap, selections,
                                  sizeof(uniRunnerAllGatherSelection));
  if (result != flagcxSuccess) {
    free(selections);
    return result;
  }

  bool allEnabledAndSupported = true;
  bool sameGroupSize = true;
  bool sameAlgorithm = true;
  const int groupSize = selections[0].groupSize;
  const int64_t selectedAlgorithm = selections[0].algorithm;
  flagcxResult_t commonResult = flagcxSuccess;
  for (int rank = 0; rank < comm->nranks; ++rank) {
    const auto &selection = selections[rank];
    if (selection.result != static_cast<int>(flagcxSuccess) &&
        commonResult == flagcxSuccess)
      commonResult = static_cast<flagcxResult_t>(selection.result);
    if (!selection.groupedAgEnabled || !selection.streamValueSupported)
      allEnabledAndSupported = false;
    if (selection.groupSize != groupSize)
      sameGroupSize = false;
    if (selection.algorithm != selectedAlgorithm)
      sameAlgorithm = false;
  }
  free(selections);
  if (commonResult != flagcxSuccess)
    return commonResult;
  if (!sameAlgorithm)
    return flagcxInvalidUsage;

  *useGroupedAg = allEnabledAndSupported && sameGroupSize;
  if (allEnabledAndSupported && !sameGroupSize)
    TRACE(FLAGCX_UNIRUNNER,
          "rank %d GroupedAG groupSize differs across ranks; use P2P AllGather",
          comm->rank);
  return flagcxSuccess;
}

static int resolveUniRunnerHierarchicalSlicedARGroupSize(flagcxComm_t comm) {
  if (comm->nranks <= 0) {
    return 0;
  }
  int64_t groupSize = flagcxParamUniRunnerGroupSize();
  if (groupSize <= 0) {
    groupSize = 8;
  }
  if (groupSize > comm->nranks || comm->nranks % groupSize != 0) {
    TRACE(FLAGCX_UNIRUNNER,
          "rank %d hierarchical SlicedAR groupSize %lld invalid for nranks %d, "
          "fallback to nranks",
          comm->rank, static_cast<long long>(groupSize), comm->nranks);
    groupSize = comm->nranks;
  }
  return static_cast<int>(groupSize);
}

flagcxResult_t uniRunnerReduce(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  bool initialized = false;
  size_t scratchBytes = 0;
  FLAGCXCHECK(validateUniRunnerReduceArgs(count, datatype, op));
  FLAGCXCHECK(validateUniRunnerLaunchConfig(flagcxCommOpReduce));
  FLAGCXCHECK(checkedUniRunnerTypeBytes(count, 2, datatype, &scratchBytes));
  res = deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes, flagcxMemDevice,
                                    stream);
  if (res != flagcxSuccess)
    return res;
  res = initUniRunner(comm, stream);
  if (res != flagcxSuccess)
    goto out;
  initialized = true;
  FLAGCXCHECKGOTO(initUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                            scratchbuff, count, datatype, op,
                                            root, comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  if (initialized)
    flagcxRunnerRecordFirstError(cleanupUniRunner(comm), &res);
  if (scratchbuff != nullptr)
    flagcxRunnerRecordFirstError(
        deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream), &res);
  return res;
}

flagcxResult_t uniRunnerGather(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  return flagcxRunnerRunGroup(
      []() { return flagcxHeteroGroupStart(); },
      [&]() {
        if (comm->rank == root) {
          for (int r = 0; r < comm->nranks; r++) {
            flagcxResult_t result = flagcxHeteroRecvOnChannel(
                static_cast<void *>(buffer + r * size), count, datatype, r,
                comm->heteroComm, stream,
                flagcxCollProxyChannelForEdge(r, root, 0));
            if (!flagcxRunnerResultAccepted(result))
              return result;
          }
        }
        return flagcxHeteroSendOnChannel(
            sendbuff, count, datatype, root, comm->heteroComm, stream,
            flagcxCollProxyChannelForEdge(comm->rank, root, 0));
      },
      []() { return flagcxHeteroGroupEnd(); });
}

flagcxResult_t uniRunnerScatter(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  return flagcxRunnerRunGroup(
      []() { return flagcxHeteroGroupStart(); },
      [&]() {
        if (comm->rank == root) {
          for (int r = 0; r < comm->nranks; r++) {
            flagcxResult_t result = flagcxHeteroSendOnChannel(
                static_cast<const void *>(buffer + r * size), count, datatype,
                r, comm->heteroComm, stream,
                flagcxCollProxyChannelForEdge(root, r, 0));
            if (!flagcxRunnerResultAccepted(result))
              return result;
          }
        }
        return flagcxHeteroRecvOnChannel(
            recvbuff, count, datatype, root, comm->heteroComm, stream,
            flagcxCollProxyChannelForEdge(root, comm->rank, 0));
      },
      []() { return flagcxHeteroGroupEnd(); });
}

flagcxResult_t uniRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  return flagcxRunnerRunGroup(
      []() { return flagcxHeteroGroupStart(); },
      [&]() {
        if (comm->rank == root) {
          for (int r = 0; r < comm->nranks; r++) {
            flagcxResult_t result = flagcxHeteroSendOnChannel(
                sendbuff, count, datatype, r, comm->heteroComm, stream,
                flagcxCollProxyChannelForEdge(root, r, 0));
            if (!flagcxRunnerResultAccepted(result))
              return result;
          }
        }
        return flagcxHeteroRecvOnChannel(
            recvbuff, count, datatype, root, comm->heteroComm, stream,
            flagcxCollProxyChannelForEdge(root, comm->rank, 0));
      },
      []() { return flagcxHeteroGroupEnd(); });
}

flagcxResult_t uniRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  size_t scratchBytes = 0;
  UniRunnerAlgorithm algorithm = uniRunnerAlgorithmRingAR;
  flagcxResult_t selectionResult = resolveUniRunnerAlgorithm(
      uniRunnerAlgorithmRingAR, uniRunnerAlgorithmRingAR,
      uniRunnerAlgorithmSlicedAR, &algorithm);
  if (selectionResult == flagcxSuccess)
    selectionResult = validateUniRunnerReduceArgs(count, datatype, op);
  if (selectionResult == flagcxSuccess &&
      algorithm != uniRunnerAlgorithmIpcRingAR)
    selectionResult = validateUniRunnerLaunchConfig(flagcxCommOpAllReduce);
  int64_t groupedArGroupSize = 0;
  if (selectionResult == flagcxSuccess &&
      algorithm == uniRunnerAlgorithmGroupedAR) {
    const int64_t requested = flagcxParamUniRunnerGroupSize();
    groupedArGroupSize = requested == 0 ? comm->localRanks : requested;
    if (groupedArGroupSize <= 0 || groupedArGroupSize > comm->nranks ||
        comm->nranks % groupedArGroupSize != 0) {
      WARN("GroupedAR requires GROUPSIZE to divide nranks (default: "
           "localRanks)");
      selectionResult = flagcxInvalidArgument;
    }
  }
  FLAGCXCHECK(agreeUniRunnerAlgorithm(comm, algorithm, selectionResult,
                                      groupedArGroupSize));
  res = initUniRunner(comm, stream);
  if (res != flagcxSuccess)
    return res;
  switch (algorithm) {
    case uniRunnerAlgorithmGroupedAR: {
      // Match grouped AG's contiguous groups; one rank/NIC per local lane.
      FLAGCXCHECKGOTO(initUniRunnerStateGroupedAR(
                          runnerState, sendbuff, recvbuff, count, datatype, op,
                          comm, static_cast<int>(groupedArGroupSize)),
                      res, out);
      break;
    }
    case uniRunnerAlgorithmIpcRingAR:
      /* Primitive-level Ring/Simple AllReduce over intra-node IPC memory. */
      FLAGCXCHECKGOTO(initUniRunnerStateIpcRingAR(runnerState, sendbuff,
                                                  recvbuff, count, datatype, op,
                                                  comm),
                      res, out);
      break;
    case uniRunnerAlgorithmLocRed:
      /* initialize uniRunnerState for reduce test */
      FLAGCXCHECKGOTO(initUniRunnerStateLocRed(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                      res, out);
      break;
    case uniRunnerAlgorithmRingAG:
      /* initialize uniRunnerState for p2p test */
      FLAGCXCHECKGOTO(initUniRunnerStateRingAG(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                      res, out);
      break;
    case uniRunnerAlgorithmHierarchicalSlicedAR: {
      const int groupSize = resolveUniRunnerHierarchicalSlicedARGroupSize(comm);
      FLAGCXCHECKGOTO(
          checkedUniRunnerTypeBytes(count, 1, datatype, &scratchBytes), res,
          out);
      FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes,
                                                  flagcxMemDevice, stream),
                      res, out);
      FLAGCXCHECKGOTO(initUniRunnerStateHierarchicalSlicedAR(
                          runnerState, sendbuff, recvbuff, scratchbuff, count,
                          datatype, op, comm, groupSize),
                      res, out);
      break;
    }
    case uniRunnerAlgorithmSlicedAR:
      /* initialize uniRunnerState for sliced AllReduce */
      FLAGCXCHECKGOTO(initUniRunnerStateSlicedAR(runnerState, sendbuff,
                                                 recvbuff, count, datatype, op,
                                                 comm),
                      res, out);
      break;
    case uniRunnerAlgorithmRingAR:
      /* initialize uniRunnerState for ring AllReduce */
      FLAGCXCHECKGOTO(initUniRunnerStateRingAR(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                      res, out);
      break;
    default:
      res = flagcxInvalidArgument;
      goto out;
  }
  res = runUniRunner(comm);
out:
  if (algorithm == uniRunnerAlgorithmIpcRingAR && res != flagcxSuccess) {
    runnerState->ipcRingResourcesDirty = true;
  }
  flagcxRunnerRecordFirstError(cleanupUniRunner(comm), &res);
  if (scratchbuff != nullptr) {
    flagcxRunnerRecordFirstError(
        deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream), &res);
  }
  return res;
}

flagcxResult_t uniRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                      size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxComm_t comm,
                                      flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  bool initialized = false;
  size_t scratchBytes = 0;
  FLAGCXCHECK(validateUniRunnerReduceArgs(recvcount, datatype, op));
  FLAGCXCHECK(validateUniRunnerLaunchConfig(flagcxCommOpReduceScatter));
  FLAGCXCHECK(checkedUniRunnerTypeBytes(recvcount, comm->nranks, datatype,
                                        &scratchBytes));
  res = deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes, flagcxMemDevice,
                                    stream);
  if (res != flagcxSuccess)
    return res;
  res = initUniRunner(comm, stream);
  if (res != flagcxSuccess)
    goto out;
  initialized = true;
  FLAGCXCHECKGOTO(initUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                           scratchbuff, recvcount, datatype, op,
                                           comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  if (initialized)
    flagcxRunnerRecordFirstError(cleanupUniRunner(comm), &res);
  if (scratchbuff != nullptr)
    flagcxRunnerRecordFirstError(
        deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream), &res);
  return res;
}

static flagcxResult_t
submitUniRunnerP2pAllGather(const void *sendbuff, void *recvbuff,
                            size_t sendcount, flagcxDataType_t datatype,
                            flagcxComm_t comm, flagcxStream_t stream,
                            size_t rankBytes) {
  char *bufferOut = static_cast<char *>(recvbuff);
  return flagcxRunnerRunGroup(
      []() { return flagcxHeteroGroupStart(); },
      [&]() {
        for (int r = 0; r < comm->nranks; r++) {
          flagcxResult_t result = flagcxHeteroSendOnChannel(
              sendbuff, sendcount, datatype, r, comm->heteroComm, stream,
              flagcxCollProxyChannelForEdge(comm->rank, r, 0));
          if (!flagcxRunnerResultAccepted(result))
            return result;
          result = flagcxHeteroRecvOnChannel(
              static_cast<void *>(bufferOut + r * rankBytes), sendcount,
              datatype, r, comm->heteroComm, stream,
              flagcxCollProxyChannelForEdge(r, comm->rank, 0));
          if (!flagcxRunnerResultAccepted(result))
            return result;
        }
        return flagcxSuccess;
      },
      []() { return flagcxHeteroGroupEnd(); });
}

flagcxResult_t uniRunnerAllGather(const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream) {
  if (comm == NULL || comm->heteroComm == NULL || comm->nranks <= 0) {
    return flagcxInvalidArgument;
  }
  size_t size = 0;
  size_t totalSize = 0;
  UniRunnerAlgorithm algorithm = uniRunnerAlgorithmGroupedAG;
  flagcxResult_t localResult = resolveUniRunnerAlgorithm(
      uniRunnerAlgorithmGroupedAG, uniRunnerAlgorithmGroupedAG,
      uniRunnerAlgorithmP2pAG, &algorithm);
  if (flagcxGroupDepth > 0) {
    // GroupedAG runs its own P2P groups. While an outer group is open, submit
    // the existing P2P AllGather tasks for that group's normal launch path.
    FLAGCXCHECK(localResult);
    FLAGCXCHECK(checkedUniRunnerTypeBytes(sendcount, 1, datatype, &size));
    FLAGCXCHECK(checkedUniRunnerTypeBytes(
        sendcount, static_cast<size_t>(comm->nranks), datatype, &totalSize));
    return submitUniRunnerP2pAllGather(sendbuff, recvbuff, sendcount, datatype,
                                       comm, stream, size);
  }
  if (localResult == flagcxSuccess)
    localResult = checkedUniRunnerTypeBytes(sendcount, 1, datatype, &size);
  if (localResult == flagcxSuccess)
    localResult = checkedUniRunnerTypeBytes(
        sendcount, static_cast<size_t>(comm->nranks), datatype, &totalSize);

  const bool localGroupedAgEnabled = algorithm == uniRunnerAlgorithmGroupedAG;
  bool streamValueSupported = false;
  if (localResult == flagcxSuccess && localGroupedAgEnabled)
    localResult = flagcxStreamValueSupported(&streamValueSupported);
  if (localResult == flagcxSuccess && streamValueSupported)
    localResult = validateUniRunnerLaunchConfig(flagcxCommOpAllGather);
  const int groupSize = resolveUniRunnerGroupedAGGroupSize(comm);
  bool useGroupedAg = false;
  FLAGCXCHECK(agreeUniRunnerAllGatherSelection(
      comm, localResult, algorithm, localGroupedAgEnabled, streamValueSupported,
      groupSize, &useGroupedAg));
  if (!useGroupedAg)
    return submitUniRunnerP2pAllGather(sendbuff, recvbuff, sendcount, datatype,
                                       comm, stream, size);

  (void)totalSize;

  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  res = initUniRunner(comm, stream);
  if (res != flagcxSuccess)
    return res;
  // The self-copy uses cpyStream, while the caller may still be producing
  // sendbuff on stream. Order both streams before submitting the DAG.
  FLAGCXCHECKGOTO(
      deviceAdaptor->eventCreate(&runnerState->groupedAgInputReadyEvent,
                                 flagcxEventDisableTiming),
      res, out);
  FLAGCXCHECKGOTO(
      deviceAdaptor->eventRecord(runnerState->groupedAgInputReadyEvent, stream),
      res, out);
  FLAGCXCHECKGOTO(
      deviceAdaptor->streamWaitEvent(runnerState->cpyStream,
                                     runnerState->groupedAgInputReadyEvent),
      res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateGroupedAG(runnerState, sendbuff, recvbuff,
                                              sendcount, datatype, comm,
                                              groupSize),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  flagcxRunnerRecordFirstError(cleanupUniRunner(comm), &res);
  return res;
}

flagcxResult_t uniRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxComm_t comm, flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  return flagcxRunnerRunGroup(
      []() { return flagcxHeteroGroupStart(); },
      [&]() {
        for (int r = 0; r < comm->nranks; r++) {
          flagcxResult_t result = flagcxHeteroSendOnChannel(
              static_cast<const void *>(bufferIn + r * size), count, datatype,
              r, comm->heteroComm, stream,
              flagcxCollProxyChannelForEdge(comm->rank, r, 0));
          if (!flagcxRunnerResultAccepted(result))
            return result;
          result = flagcxHeteroRecvOnChannel(
              static_cast<void *>(bufferOut + r * size), count, datatype, r,
              comm->heteroComm, stream,
              flagcxCollProxyChannelForEdge(r, comm->rank, 0));
          if (!flagcxRunnerResultAccepted(result))
            return result;
        }
        return flagcxSuccess;
      },
      []() { return flagcxHeteroGroupEnd(); });
}

flagcxResult_t uniRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                  size_t *sdispls, void *recvbuff,
                                  size_t *recvcounts, size_t *rdispls,
                                  flagcxDataType_t datatype, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  size_t size = getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  return flagcxRunnerRunGroup(
      []() { return flagcxHeteroGroupStart(); },
      [&]() {
        for (int r = 0; r < comm->nranks; r++) {
          if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
            flagcxResult_t result = flagcxHeteroSendOnChannel(
                static_cast<const void *>(bufferIn + sdispls[r] * size),
                sendcounts[r], datatype, r, comm->heteroComm, stream,
                flagcxCollProxyChannelForEdge(comm->rank, r, 0));
            if (!flagcxRunnerResultAccepted(result))
              return result;
          }
          if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
            flagcxResult_t result = flagcxHeteroRecvOnChannel(
                static_cast<void *>(bufferOut + rdispls[r] * size),
                recvcounts[r], datatype, r, comm->heteroComm, stream,
                flagcxCollProxyChannelForEdge(r, comm->rank, 0));
            if (!flagcxRunnerResultAccepted(result))
              return result;
          }
        }
        return flagcxSuccess;
      },
      []() { return flagcxHeteroGroupEnd(); });
}

flagcxResult_t uniRunnerSend(const void *sendbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerRecv(void *recvbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupStart() {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupEnd() {
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

struct flagcxRunner uniRunner = {
    // Communication functions
    uniRunnerReduce, uniRunnerGather, uniRunnerScatter, uniRunnerBroadcast,
    uniRunnerAllReduce, uniRunnerReduceScatter, uniRunnerAllGather,
    uniRunnerAlltoAll, uniRunnerAlltoAllv, uniRunnerSend, uniRunnerRecv,
    // Group semantics
    uniRunnerGroupStart, uniRunnerGroupEnd};
