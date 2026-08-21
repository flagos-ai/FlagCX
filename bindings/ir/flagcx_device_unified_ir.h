/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Unified One-Sided IR — Transport-Transparent Device API.
 *
 * These functions auto-dispatch between P2P (NVLink/IPC direct stores)
 * and Net (FIFO/GIN/RDMA) paths based on peer reachability.
 *
 * All functions are extern "C", using only void* and scalar parameters.
 * Naming: flagcxDev* (no "Net" — transport-transparent).
 *
 * Design:
 *   - Transport dispatch: flagcxGetPeerPointer() != null → P2P, else → Net
 *   - Scope/Order: mapped to PTX fences for P2P, implicit for Net
 *   - Signal/Wait: managed through net layer (signal buffers)
 *   - Barrier: teamKind dispatch (INTRA→IPC, INTER→Net, WORLD→combined)
 ************************************************************************/
#ifndef FLAGCX_DEVICE_UNIFIED_IR_H_
#define FLAGCX_DEVICE_UNIFIED_IR_H_

#include "comm_traits.h"
#include "device_utils.h"
#include "flagcx_device_enums.h"

/* ================================================================
 * Category U1: Unified Put (6)
 *
 * Transport-transparent put. Checks peer reachability via P2P;
 * falls back to Net path if unreachable.
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param dst       Opaque pointer to destination flagcxDevMem.
 * @param dstOffset Byte offset into destination.
 * @param src       Opaque pointer to source flagcxDevMem.
 * @param srcOffset Byte offset into source.
 * @param bytes     Number of bytes to transfer.
 * @param teamKind  Team topology: INTRA, INTER, or WORLD.
 * @param peer      Destination rank within the team.
 * @param contextId Context identifier.
 * @param coopKind  Cooperation level (THREAD/WARP/BLOCK).
 * @param scope     Memory fence scope for P2P path.
 * @param order     Memory ordering semantics.
 * ================================================================ */

/** @brief Basic put (no completion action). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevPut(const void *comm, const void *dst, size_t dstOffset,
             const void *src, size_t srcOffset, size_t bytes,
             flagcxDevTeamKind_t teamKind, int peer,
             flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
             flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order);

/** @brief Put + remote signal increment on completion. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevPut_RSigInc(const void *comm, const void *dst, size_t dstOffset,
                     const void *src, size_t srcOffset, size_t bytes,
                     flagcxDevTeamKind_t teamKind, int peer,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
                     flagcxDevSignal_t remoteSignal);

/** @brief Put + remote signal add on completion. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevPut_RSigAdd(const void *comm, const void *dst, size_t dstOffset,
                     const void *src, size_t srcOffset, size_t bytes,
                     flagcxDevTeamKind_t teamKind, int peer,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
                     flagcxDevSignal_t remoteSignal, uint64_t signalValue);

/** @brief Put + local counter increment on completion (sender-side tracking).
 */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevPut_LCtrInc(const void *comm, const void *dst, size_t dstOffset,
                     const void *src, size_t srcOffset, size_t bytes,
                     flagcxDevTeamKind_t teamKind, int peer,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
                     flagcxDevCounter_t localCounter);

/** @brief Put + remote signal increment + local counter increment. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevPut_RSigInc_LCtrInc(
    const void *comm, const void *dst, size_t dstOffset, const void *src,
    size_t srcOffset, size_t bytes, flagcxDevTeamKind_t teamKind, int peer,
    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
    flagcxDevSignal_t remoteSignal, flagcxDevCounter_t localCounter);

/** @brief Put + remote signal add + local counter increment. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevPut_RSigAdd_LCtrInc(
    const void *comm, const void *dst, size_t dstOffset, const void *src,
    size_t srcOffset, size_t bytes, flagcxDevTeamKind_t teamKind, int peer,
    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
    flagcxDevSignal_t remoteSignal, uint64_t signalValue,
    flagcxDevCounter_t localCounter);

/* ================================================================
 * Category U2: Unified Get (1)
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param src       Opaque pointer to remote source flagcxDevMem.
 * @param srcOffset Byte offset into remote source.
 * @param dst       Opaque pointer to local destination flagcxDevMem.
 * @param dstOffset Byte offset into local destination.
 * @param bytes     Number of bytes to transfer.
 * @param teamKind  Team topology selector.
 * @param peer      Source rank within the team.
 * @param contextId Context identifier.
 * @param coopKind  Cooperation level.
 * @param scope     Memory fence scope for P2P path.
 * @param order     Memory ordering semantics.
 * ================================================================ */

/** @brief Transport-transparent get from remote peer. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevGet(const void *comm, const void *src, size_t srcOffset,
             const void *dst, size_t dstOffset, size_t bytes,
             flagcxDevTeamKind_t teamKind, int peer,
             flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
             flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order);

/* ================================================================
 * Category U3: Unified PutValue (3)
 *
 * Scalar value write to remote peer.
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param dst       Opaque pointer to destination flagcxDevMem.
 * @param dstOffset Byte offset into destination.
 * @param value     64-bit scalar value to write.
 * @param teamKind  Team topology selector.
 * @param peer      Destination rank within the team.
 * @param contextId Context identifier.
 * @param coopKind  Cooperation level.
 * @param scope     Memory fence scope for P2P path.
 * @param order     Memory ordering semantics.
 * ================================================================ */

/** @brief Write a scalar value to remote peer. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevPutValue(const void *comm, const void *dst, size_t dstOffset,
                  uint64_t value, flagcxDevTeamKind_t teamKind, int peer,
                  flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                  flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order);

/** @brief Write a scalar value + remote signal increment. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevPutValue_RSigInc(
    const void *comm, const void *dst, size_t dstOffset, uint64_t value,
    flagcxDevTeamKind_t teamKind, int peer, flagcxDevContext_t contextId,
    flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope,
    flagcxDevMemoryOrder_t order, flagcxDevSignal_t remoteSignal);

/** @brief Write a scalar value + remote signal add. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevPutValue_RSigAdd(
    const void *comm, const void *dst, size_t dstOffset, uint64_t value,
    flagcxDevTeamKind_t teamKind, int peer, flagcxDevContext_t contextId,
    flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope,
    flagcxDevMemoryOrder_t order, flagcxDevSignal_t remoteSignal,
    uint64_t signalValue);

/* ================================================================
 * Category U4: Unified Signal (2)
 *
 * Send signal notifications to remote peers.
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param teamKind  Team topology selector.
 * @param peer      Target rank within the team.
 * @param signal    Signal slot identifier.
 * @param value     Value to add (SignalAdd only).
 * @param contextId Context identifier.
 * @param coopKind  Cooperation level.
 * @param scope     Memory fence scope.
 * ================================================================ */

/** @brief Increment remote signal by 1. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevSignalInc(const void *comm, flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, flagcxDevContext_t contextId,
                   flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope);

/** @brief Add value to remote signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevSignalAdd(const void *comm, flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, uint64_t value,
                   flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                   flagcxDevMemoryScope_t scope);

/* ================================================================
 * Category U5: Unified Wait (2)
 *
 * Spin-wait for local signal/counter to reach threshold.
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param signal    Signal slot to wait on.
 * @param counter   Counter slot to wait on.
 * @param least     Minimum value (inclusive) to wait for.
 * @param bits      Bit width for comparison (32 or 64).
 * @param contextId Context identifier.
 * @param coopKind  Cooperation level.
 * @param order     Memory ordering on completion.
 * ================================================================ */

/** @brief Wait until signal >= least. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevWaitSignal(const void *comm, flagcxDevSignal_t signal, uint64_t least,
                    int bits, flagcxDevContext_t contextId,
                    flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order);

/** @brief Wait until counter >= least. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevWaitCounter(const void *comm, flagcxDevCounter_t counter,
                     uint64_t least, int bits, flagcxDevContext_t contextId,
                     flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryOrder_t order);

/* ================================================================
 * Category U6: Unified Read (2)
 *
 * Non-blocking read of signal/counter value.
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param signal    Signal slot to read.
 * @param counter   Counter slot to read.
 * @param bits      Bit width (32 or 64).
 * @param contextId Context identifier.
 * @param order     Memory ordering for the load.
 * @return          Current value of the signal/counter.
 * ================================================================ */

/** @brief Read signal value (non-blocking). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR uint64_t
flagcxDevReadSignal(const void *comm, flagcxDevSignal_t signal, int bits,
                    flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order);

/** @brief Read counter value (non-blocking). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR uint64_t flagcxDevReadCounter(
    const void *comm, flagcxDevCounter_t counter, int bits,
    flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order);

/* ================================================================
 * Category U7: Unified Flush / Reset / Shadow (4)
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param contextId Context identifier.
 * @param coopKind  Cooperation level (Flush only).
 * @param order     Memory ordering (Flush only).
 * @param slot      Signal/counter slot identifier.
 * @param delta     Value to add to shadow.
 * ================================================================ */

/** @brief Flush pending network writes. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevFlush(const void *comm, flagcxDevContext_t contextId,
               flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order);

/** @brief Reset a signal slot and its local shadow to zero. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevResetSignal(const void *comm, flagcxDevContext_t contextId,
                     flagcxDevSignal_t slot);

/** @brief Reset a counter slot to zero. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevResetCounter(const void *comm, flagcxDevContext_t contextId,
                      flagcxDevCounter_t slot);

/** @brief Increase the local shadow for a signal. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevIncreaseSignalShadow(const void *comm, flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, uint64_t delta);

/** @brief Spin-wait until signal meets its local shadow value. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevWaitSignalMeetShadow(
    const void *comm, flagcxDevContext_t contextId, flagcxDevSignal_t slot,
    int bits, flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order);

/* ================================================================
 * Category U8: Unified Barrier (3)
 *
 * Transport-transparent barrier. Dispatches based on teamKind:
 *   INTRA → P2P IPC atomic barrier
 *   INTER → Net signal barrier
 *   WORLD → Intra arrive → Inter arrive → Inter wait → Intra wait
 *
 * @param comm      Opaque pointer to flagcxDevComm.
 * @param teamKind  Barrier scope (INTRA/INTER/WORLD).
 * @param index     Barrier channel index (typically blockIdx.x).
 * @param contextId Context identifier.
 * @param coopKind  Cooperation level.
 * @param order     Memory ordering semantics.
 * @param scope     Memory fence scope.
 * ================================================================ */

/** @brief Signal arrival at barrier. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void flagcxDevBarrierArrive(
    const void *comm, flagcxDevTeamKind_t teamKind, uint32_t index,
    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryOrder_t order, flagcxDevMemoryScope_t scope);

/** @brief Wait for all peers at barrier. */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevBarrierWait(const void *comm, flagcxDevTeamKind_t teamKind,
                     uint32_t index, flagcxDevContext_t contextId,
                     flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order,
                     flagcxDevMemoryScope_t scope);

/** @brief Arrive + wait (full synchronization). */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_DECORATOR void
flagcxDevBarrierSync(const void *comm, flagcxDevTeamKind_t teamKind,
                     uint32_t index, flagcxDevContext_t contextId,
                     flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order,
                     flagcxDevMemoryScope_t scope);

#endif // FLAGCX_DEVICE_UNIFIED_IR_H_
