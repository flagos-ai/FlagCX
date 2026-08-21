/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device Scalar IR API — Struct-free entry points for Triton/LLVM.
 *
 * This header declares extern "C" device functions that take ONLY scalars
 * and opaque pointers. No struct instantiation is required by the caller.
 *
 * Design:
 *   - CoopAny/Team replaced by flagcxCoopKind_t / flagcxTeamKind_t enums
 *   - Net (transport) obtained via flagcxDevNetGetFromCommS (pre-allocated)
 *
 * When compiled to LLVM bitcode (clang -x cuda --cuda-device-only),
 * these functions can be linked by Triton without knowing any struct layout.
 *
 * The old struct-based API (flagcx_device_wrapper.h) remains for native
 * CUDA kernels. Both paths share the same underlying implementation.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_SCALAR_IR_H_
#define FLAGCX_DEVICE_SCALAR_IR_H_

#include "comm_traits.h" /* flagcxDevSignal_t, flagcxDevCounter_t,
                                flagcxDevNetFenceLevel, flagcxDeviceMemoryOrder_t */
#include "device_utils.h"
#include "flagcx.h" /* flagcxDataType_t, flagcxResult_t */
#include "flagcx_device_enums.h"

/* Deprecation macro for APIs superseded by unified flagcxDev* */
#ifndef FLAGCX_DEPRECATED
#if defined(__GNUC__) || defined(__clang__)
#define FLAGCX_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define FLAGCX_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define FLAGCX_DEPRECATED(msg)
#endif
#endif

/* ================================================================
 * Category 1: Comm Queries (4)
 *
 * Same as wrapper.h — included here for completeness so that scalar
 * IR users need only this single header.
 *
 * @param comm  Opaque pointer to flagcxDevComm (device communicator).
 * ================================================================ */

/** @brief Get this rank's global index within the communicator. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetRank(const void *comm);
/** @brief Get total number of ranks in the communicator. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetSize(const void *comm);
/** @brief Get this rank's index within its intra-node group. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetIntraRank(const void *comm);
/** @brief Get number of ranks in this rank's intra-node group. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevCommGetIntraSize(const void *comm);

/* ================================================================
 * Category 2: Scalar Cooperative Group (6)
 *
 * Replace struct-based CoopAny with enum dispatch.
 * Basic variants (thread/warp/block) need no extra params.
 * Extended variants (tile_span, lanes) take up to 3 params.
 *
 * @param kind    Cooperation kind: THREAD/WARP/BLOCK (basic) or
 *                TILE_SPAN/LANES (extended).
 * @param param0  For TILE_SPAN: t0 (first tile). For LANES: laneMask.
 * @param param1  For TILE_SPAN: nTiles. Unused for LANES.
 * @param param2  For TILE_SPAN: id. Unused for LANES.
 * ================================================================ */

/** @brief Thread rank within the cooperative group (basic kinds only). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxCoopThreadRankS(flagcxCoopKind_t kind);

/** @brief Number of threads in the cooperative group (basic kinds only). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxCoopSizeS(flagcxCoopKind_t kind);

/** @brief Synchronize the cooperative group (basic kinds only). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxCoopSyncS(flagcxCoopKind_t kind);

/** @brief Thread rank within the cooperative group (extended kinds). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxCoopThreadRankExS(flagcxCoopKind_t kind, uint32_t param0, uint32_t param1,
                        uint32_t param2);

/** @brief Group size for extended kinds. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxCoopSizeExS(flagcxCoopKind_t kind, uint32_t param0, uint32_t param1,
                  uint32_t param2);

/** @brief Synchronize the cooperative group (extended kinds). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxCoopSyncExS(flagcxCoopKind_t kind, uint32_t param0, uint32_t param1,
                  uint32_t param2);

/* ================================================================
 * Category 3: Scalar Team (2)
 *
 * Replace struct-based Team with enum dispatch.
 * Team info is derived from comm + team kind internally.
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param teamKind  Team topology: INTRA, INTER, or WORLD.
 * @param rank      Team-local rank to convert.
 * ================================================================ */

/** @brief Convert a team-local rank to a world rank. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxTeamRankToWorldS(const void *comm, flagcxTeamKind_t teamKind, int rank);

/** @brief Convert a team-local rank to an intra-node rank. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxTeamRankToIntraS(const void *comm, flagcxTeamKind_t teamKind, int rank);

/* ================================================================
 * Category 4: Pointer Access (4)
 *
 * Pointer functions that use teamKind instead of team struct.
 *
 * @param mem       Opaque pointer to flagcxDevMem (memory descriptor).
 * @param offset    Byte offset into the memory region.
 * @param comm      Opaque pointer to flagcxDevComm (for multicast).
 * @param teamKind  Team topology selector (for peer pointer).
 * @param peer      Peer rank within the team (for peer/intra pointer).
 * ================================================================ */

/** @brief Get a pointer to a peer's memory region (team-relative addressing).
 */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetPeerPointerS(const void *mem, size_t offset, const void *comm,
                      flagcxTeamKind_t teamKind, int peer);

/** @brief Get a pointer to the local memory region at the given offset. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetLocalPointerS(const void *mem, size_t offset);

/** @brief Get a pointer to an intra-node peer's memory region. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetIntraPointerS(const void *mem, size_t offset, int peer);

/** @brief Get a multicast pointer spanning all ranks in the comm. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void *
flagcxGetMulticastPointerS(const void *mem, size_t offset, const void *comm);

/* ================================================================
 * Category 5: Utility (1)
 * ================================================================ */

/** @brief Get byte size of a flagcxDataType_t element.
 *  @param dt  Data type enum value. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR size_t
flagcxDataTypeSizeDevice(flagcxDataType_t dt);

/* ================================================================
 * Category 6: Scalar Barrier — Intra-Node (3)
 *
 * Split arrive/wait: ArriveS signals peers, WaitS waits for peers.
 * SyncS combines both into a single call.
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param coopKind  Cooperation level for the barrier operation.
 * @param index     Barrier channel index (typically blockIdx.x).
 * @param multimem  Whether to use multicast memory barrier variant.
 * @param order     Memory ordering semantics (acquire/release/relaxed).
 * ================================================================ */

/** @brief Signal arrival at intra-node barrier. */
FLAGCX_DEPRECATED("use flagcxDevBarrierArrive with FLAGCX_TEAM_INTRA")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxIntraBarrierArriveS(const void *comm, flagcxCoopKind_t coopKind,
                          uint32_t index, bool multimem,
                          flagcxDeviceMemoryOrder_t order);

/** @brief Wait for all peers at intra-node barrier. */
FLAGCX_DEPRECATED("use flagcxDevBarrierWait with FLAGCX_TEAM_INTRA")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxIntraBarrierWaitS(const void *comm, flagcxCoopKind_t coopKind,
                        uint32_t index, bool multimem,
                        flagcxDeviceMemoryOrder_t order);

/** @brief Arrive + wait (full sync) at intra-node barrier. */
FLAGCX_DEPRECATED("use flagcxDevBarrierSync with FLAGCX_TEAM_INTRA")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxIntraBarrierSyncS(const void *comm, flagcxCoopKind_t coopKind,
                        uint32_t index, bool multimem,
                        flagcxDeviceMemoryOrder_t order);

/* ================================================================
 * Category 7: Scalar Barrier — Inter-Node (3)
 *
 * @param net       Opaque pointer to flagcxDevNet (transport handle).
 * @param coopKind  Cooperation level for the barrier operation.
 * @param index     Barrier channel index.
 * @param order     Memory ordering semantics.
 * @param fence     Network fence level (controls DMA visibility).
 * ================================================================ */

/** @brief Signal arrival at inter-node barrier. */
FLAGCX_DEPRECATED("use flagcxDevBarrierArrive with FLAGCX_TEAM_INTER")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxInterBarrierArriveS(const void *net, flagcxCoopKind_t coopKind,
                          uint32_t index, flagcxDeviceMemoryOrder_t order,
                          flagcxDevNetFenceLevel fence);

/** @brief Wait for all inter-node peers at barrier. */
FLAGCX_DEPRECATED("use flagcxDevBarrierWait with FLAGCX_TEAM_INTER")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxInterBarrierWaitS(const void *net, flagcxCoopKind_t coopKind,
                        uint32_t index, flagcxDeviceMemoryOrder_t order,
                        flagcxDevNetFenceLevel fence);

/** @brief Arrive + wait (full sync) at inter-node barrier. */
FLAGCX_DEPRECATED("use flagcxDevBarrierSync with FLAGCX_TEAM_INTER")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxInterBarrierSyncS(const void *net, flagcxCoopKind_t coopKind,
                        uint32_t index, flagcxDeviceMemoryOrder_t order,
                        flagcxDevNetFenceLevel fence);

/* ================================================================
 * Category 8: Scalar Barrier — World (3)
 *
 * Combines intra-node + inter-node barrier into a single world-level
 * synchronization.
 *
 * @param net       Opaque pointer to flagcxDevNet (transport handle).
 * @param coopKind  Cooperation level for the barrier operation.
 * @param index     Barrier channel index.
 * @param multimem  Whether to use multicast memory for intra-node phase.
 * @param order     Memory ordering semantics.
 * @param fence     Network fence level (controls DMA visibility).
 * ================================================================ */

/** @brief Signal arrival at world barrier (intra + inter). */
FLAGCX_DEPRECATED("use flagcxDevBarrierArrive with FLAGCX_TEAM_WORLD")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxWorldBarrierArriveS(
    const void *net, flagcxCoopKind_t coopKind, uint32_t index, bool multimem,
    flagcxDeviceMemoryOrder_t order, flagcxDevNetFenceLevel fence);

/** @brief Wait for all peers at world barrier (intra + inter). */
FLAGCX_DEPRECATED("use flagcxDevBarrierWait with FLAGCX_TEAM_WORLD")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxWorldBarrierWaitS(
    const void *net, flagcxCoopKind_t coopKind, uint32_t index, bool multimem,
    flagcxDeviceMemoryOrder_t order, flagcxDevNetFenceLevel fence);

/** @brief Arrive + wait (full sync) at world barrier (intra + inter). */
FLAGCX_DEPRECATED("use flagcxDevBarrierSync with FLAGCX_TEAM_WORLD")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxWorldBarrierSyncS(
    const void *net, flagcxCoopKind_t coopKind, uint32_t index, bool multimem,
    flagcxDeviceMemoryOrder_t order, flagcxDevNetFenceLevel fence);

/* ================================================================
 * Category 9: Net — Obtain Pre-Allocated Transport (1)
 *
 * Returns a pointer to the pre-allocated flagcxDevNet object at
 * context index `idx`. Replaces flagcxDevNetInitC + caller alloca.
 *
 * @param comm  Opaque pointer to flagcxDevComm.
 * @param idx   Context index (0 .. contextCount-1).
 * @return      Opaque const pointer to the flagcxDevNet for that context.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR const void *
flagcxDevNetGetFromCommS(const void *comm, int idx);

/* ================================================================
 * Category 10: Net — Signal / Counter / Flush (scalar coop) (7+3)
 *
 * Same operations as wrapper.h Category 9, but with coopKind enum
 * instead of const void *coop struct pointer.
 *
 * @param net       Opaque pointer to flagcxDevNet (transport handle).
 * @param coopKind  Cooperation level for wait/flush operations.
 * @param signalId  Signal slot identifier.
 * @param counterId Counter slot identifier.
 * @param least     Minimum value to wait for (inclusive).
 * @param bits      Bit width for the comparison (32 or 64).
 * @param order     Memory ordering semantics.
 * @param slot      Signal/counter slot (for reset/shadow operations).
 * @param delta     Value to add to the signal shadow.
 * ================================================================ */

/** @brief Read a signal value (non-blocking). */
FLAGCX_DEPRECATED("use flagcxDevReadSignal")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR uint64_t
flagcxDevNetReadSignalS(const void *net, flagcxDevSignal_t signalId, int bits,
                        flagcxDeviceMemoryOrder_t order);

/** @brief Spin-wait until signal >= least. */
FLAGCX_DEPRECATED("use flagcxDevWaitSignal")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetWaitSignalS(const void *net, flagcxCoopKind_t coopKind,
                        flagcxDevSignal_t signalId, uint64_t least, int bits,
                        flagcxDeviceMemoryOrder_t order);

/** @brief Spin-wait until signal meets its shadow value. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetWaitSignalMeetShadowS(const void *net, flagcxCoopKind_t coopKind,
                                  flagcxDevSignal_t signalId, int bits,
                                  flagcxDeviceMemoryOrder_t order);

/** @brief Read a counter value (non-blocking). */
FLAGCX_DEPRECATED("use flagcxDevReadCounter")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR uint64_t
flagcxDevNetReadCounterS(const void *net, flagcxDevCounter_t counterId,
                         int bits, flagcxDeviceMemoryOrder_t order);

/** @brief Spin-wait until counter >= least. */
FLAGCX_DEPRECATED("use flagcxDevWaitCounter")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetWaitCounterS(const void *net, flagcxCoopKind_t coopKind,
                         flagcxDevCounter_t counterId, uint64_t least, int bits,
                         flagcxDeviceMemoryOrder_t order);

/** @brief Flush pending RDMA/network writes. */
FLAGCX_DEPRECATED("use flagcxDevFlush")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetFlushS(const void *net, flagcxCoopKind_t coopKind,
                   flagcxDeviceMemoryOrder_t order);

/* Reset / Shadow (no coop needed — shared with C API) */

/** @brief Reset a signal slot and its local shadow to zero. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetResetSignal(const void *net, flagcxDevSignal_t slot);
/** @brief Reset a counter slot to zero. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetResetCounter(const void *net, flagcxDevCounter_t slot);
/** @brief Increase the local shadow for a signal (for MeetShadow waits). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetIncreaseSignalShadow(const void *net, flagcxDevSignal_t slot,
                                 uint64_t delta);

/* ================================================================
 * Category 11: Net — Two-Sided (scalar coop) (4)
 *
 * @param net       Opaque pointer to flagcxDevNet (transport handle).
 * @param coopKind  Cooperation level.
 * @param mem       Opaque pointer to flagcxDevMem (buffer descriptor).
 * @param offset    Byte offset into the buffer.
 * @param count     Number of elements to transfer.
 * @param datatype  Element data type.
 * @param peer      Remote rank for the transfer.
 * @return          flagcxResult_t cast to int (0 = success).
 * ================================================================ */

/** @brief Initiate a send operation. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetSendS(const void *net, flagcxCoopKind_t coopKind, const void *mem,
                  size_t offset, size_t count, flagcxDataType_t datatype,
                  int peer);

/** @brief Initiate a receive operation. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetRecvS(const void *net, flagcxCoopKind_t coopKind, const void *mem,
                  size_t offset, size_t count, flagcxDataType_t datatype,
                  int peer);

/** @brief Wait for all pending two-sided operations to complete. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetWaitS(const void *net, flagcxCoopKind_t coopKind);

/** @brief Terminate the transport session (release resources). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR int
flagcxDevNetTermS(const void *net, flagcxCoopKind_t coopKind);

/* ================================================================
 * Category 12: Net — One-Sided put (scalar coop + team kind) (16)
 *
 * Naming: flagcxDevNetPutS[_R<remote>][_L<local>]
 * Actions: None, SigInc, SigAdd, CtrInc
 *
 * @param net            Opaque pointer to flagcxDevNet.
 * @param comm           Opaque pointer to flagcxDevComm.
 * @param teamKind       Team topology selector (INTRA/INTER/WORLD).
 * @param peer           Destination rank within the team.
 * @param dst            Opaque pointer to destination flagcxDevMem.
 * @param dstOffset      Byte offset into destination memory.
 * @param src            Opaque pointer to source flagcxDevMem.
 * @param srcOffset      Byte offset into source memory.
 * @param bytes          Number of bytes to transfer.
 * @param coopKind       Cooperation level for the operation.
 * @param remoteSignal   Remote signal slot to increment/add on completion.
 * @param remoteValue    Value to add to remote signal (SigAdd variants).
 * @param remoteCounter  Remote counter slot to increment on completion.
 * @param localSignal    Local signal slot to increment/add on completion.
 * @param localValue     Value to add to local signal (LSigAdd variants).
 * @param localCounter   Local counter slot to increment on completion.
 * ================================================================ */

/* (None, None) */
FLAGCX_DEPRECATED("use flagcxDevPut")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS(const void *net, const void *comm, flagcxTeamKind_t teamKind,
                 int peer, const void *dst, size_t dstOffset, const void *src,
                 size_t srcOffset, size_t bytes, flagcxCoopKind_t coopKind);

/* (SigInc, None) */
FLAGCX_DEPRECATED("use flagcxDevPut_RSigInc")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutS_RSigInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal);

/* (SigAdd, None) */
FLAGCX_DEPRECATED("use flagcxDevPut_RSigAdd")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RSigAdd(const void *net, const void *comm,
                         flagcxTeamKind_t teamKind, int peer, const void *dst,
                         size_t dstOffset, const void *src, size_t srcOffset,
                         size_t bytes, flagcxCoopKind_t coopKind,
                         flagcxDevSignal_t remoteSignal, uint64_t remoteValue);

/* Deprecated compatibility variants retained for one deprecation cycle. */

/* (CtrInc, None) */
FLAGCX_DEPRECATED("migrate to flagcxDev* unified one-sided operations")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutS_RCtrInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevCounter_t remoteCounter);

/* (None, SigInc) */
FLAGCX_DEPRECATED("migrate to local-counter completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutS_LSigInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t localSignal);

/* (SigInc, SigInc) */
FLAGCX_DEPRECATED("migrate to local-counter completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RSigInc_LSigInc(const void *net, const void *comm,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevSignal_t remoteSignal,
                                 flagcxDevSignal_t localSignal);

/* (SigAdd, SigInc) */
FLAGCX_DEPRECATED("migrate to local-counter completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LSigInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    uint64_t remoteValue, flagcxDevSignal_t localSignal);

/* (CtrInc, SigInc) */
FLAGCX_DEPRECATED("migrate to flagcxDev* unified one-sided operations")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LSigInc(const void *net, const void *comm,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevCounter_t remoteCounter,
                                 flagcxDevSignal_t localSignal);

/* (None, SigAdd) */
FLAGCX_DEPRECATED("migrate to local-counter completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_LSigAdd(const void *net, const void *comm,
                         flagcxTeamKind_t teamKind, int peer, const void *dst,
                         size_t dstOffset, const void *src, size_t srcOffset,
                         size_t bytes, flagcxCoopKind_t coopKind,
                         flagcxDevSignal_t localSignal, uint64_t localValue);

/* (SigInc, SigAdd) */
FLAGCX_DEPRECATED("migrate to local-counter completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RSigInc_LSigAdd(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    flagcxDevSignal_t localSignal, uint64_t localValue);

/* (SigAdd, SigAdd) */
FLAGCX_DEPRECATED("migrate to local-counter completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LSigAdd(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    uint64_t remoteValue, flagcxDevSignal_t localSignal, uint64_t localValue);

/* (CtrInc, SigAdd) */
FLAGCX_DEPRECATED("migrate to flagcxDev* unified one-sided operations")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LSigAdd(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevCounter_t remoteCounter,
    flagcxDevSignal_t localSignal, uint64_t localValue);

/* (None, CtrInc) */
FLAGCX_DEPRECATED("use flagcxDevPut_LCtrInc")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutS_LCtrInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevCounter_t localCounter);

/* (CtrInc, CtrInc) */
FLAGCX_DEPRECATED("migrate to flagcxDev* unified one-sided operations")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LCtrInc(const void *net, const void *comm,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevCounter_t remoteCounter,
                                 flagcxDevCounter_t localCounter);

/* (SigAdd, CtrInc) */
FLAGCX_DEPRECATED("use flagcxDevPut_RSigAdd")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LCtrInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    uint64_t remoteValue, flagcxDevCounter_t localCounter);

/* (SigInc, CtrInc) */
FLAGCX_DEPRECATED("use flagcxDevPut_RSigInc")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutS_RSigInc_LCtrInc(const void *net, const void *comm,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevSignal_t remoteSignal,
                                 flagcxDevCounter_t localCounter);

/* ================================================================
 * Category 13: Net — One-Sided signal (scalar) (3)
 *
 * Send a signal/counter update to a remote peer without data transfer.
 *
 * @param net       Opaque pointer to flagcxDevNet.
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param teamKind  Team topology selector.
 * @param peer      Destination rank within the team.
 * @param coopKind  Cooperation level.
 * @param signal    Remote signal slot to increment/add.
 * @param counter   Remote counter slot to increment.
 * @param value     Value to add (SigAdd variant only).
 * ================================================================ */

/** @brief Increment a remote peer's signal slot by 1. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetSignalSigIncS(const void *net, const void *comm,
                          flagcxTeamKind_t teamKind, int peer,
                          flagcxCoopKind_t coopKind, flagcxDevSignal_t signal);

/** @brief Add a value to a remote peer's signal slot. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetSignalSigAddS(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    flagcxCoopKind_t coopKind, flagcxDevSignal_t signal, uint64_t value);

/** @brief Increment a remote counter (deprecated compatibility entry point). */
FLAGCX_DEPRECATED("migrate to signal completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetSignalCtrIncS(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    flagcxCoopKind_t coopKind, flagcxDevCounter_t counter);

/* ================================================================
 * Category 14: Net — One-Sided putValue<uint64_t> (scalar) (4)
 *
 * Write a scalar uint64_t value to a remote peer's memory, optionally
 * with a remote signal/counter action on completion.
 *
 * @param net            Opaque pointer to flagcxDevNet.
 * @param comm           Opaque pointer to flagcxDevComm.
 * @param teamKind       Team topology selector.
 * @param peer           Destination rank within the team.
 * @param dst            Opaque pointer to destination flagcxDevMem.
 * @param dstOffset      Byte offset into destination memory.
 * @param value          The uint64_t value to write.
 * @param coopKind       Cooperation level.
 * @param remoteSignal   Remote signal slot (SigInc/SigAdd variants).
 * @param remoteAddValue Value to add to remote signal (SigAdd variant).
 * @param remoteCounter  Remote counter slot (CtrInc variant).
 * ================================================================ */

/** @brief Put a scalar value to a remote peer (no completion action). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetPutValueS(const void *net, const void *comm,
                      flagcxTeamKind_t teamKind, int peer, const void *dst,
                      size_t dstOffset, uint64_t value,
                      flagcxCoopKind_t coopKind);

/** @brief Put a scalar value + increment remote signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutValueS_RSigInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, uint64_t value,
    flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal);

/** @brief Put a scalar value + add to remote signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutValueS_RSigAdd(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, uint64_t value,
    flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    uint64_t remoteAddValue);

/** @brief Put a value + increment remote counter (deprecated compatibility). */
FLAGCX_DEPRECATED("migrate to signal completion in the Unified IR API")
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevNetPutValueS_RCtrInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, uint64_t value,
    flagcxCoopKind_t coopKind, flagcxDevCounter_t remoteCounter);

/* ================================================================
 * Category 15: Net — One-Sided get (scalar) (1)
 *
 * @param net        Opaque pointer to flagcxDevNet.
 * @param comm       Opaque pointer to flagcxDevComm.
 * @param teamKind   Team topology selector.
 * @param peer       Source rank within the team.
 * @param src        Opaque pointer to source flagcxDevMem (remote).
 * @param srcOffset  Byte offset into source memory.
 * @param dst        Opaque pointer to destination flagcxDevMem (local).
 * @param dstOffset  Byte offset into destination memory.
 * @param bytes      Number of bytes to transfer.
 * @param coopKind   Cooperation level.
 * ================================================================ */

/** @brief One-sided get: read from a remote peer's memory into local. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevNetGetS(const void *net, const void *comm, flagcxTeamKind_t teamKind,
                 int peer, const void *src, size_t srcOffset, const void *dst,
                 size_t dstOffset, size_t bytes, flagcxCoopKind_t coopKind);

#endif /* FLAGCX_DEVICE_SCALAR_IR_H_ */
