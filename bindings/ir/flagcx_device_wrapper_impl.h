/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API — C wrapper implementations for LLVM IR generation.
 *
 * This file is compiled by clang with -emit-llvm to produce LLVM bitcode.
 * All functions use FLAGCX_IR_EXTERN_C (= extern "C" under
 * __clang_llvm_bitcode_lib__) to ensure stable, unmangled symbol names.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_WRAPPER_IMPL_H_
#define FLAGCX_DEVICE_WRAPPER_IMPL_H_

#include "flagcx_device_wrapper.h"
#include <new>

#if FLAGCX_CHECK_DEVICE_CC

/* ================================================================
 * Category 1: Comm Queries (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetRank(const void *commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetSize(const void *commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getSize();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetIntraRank(const void *commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getIntraRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevCommGetIntraSize(const void *commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getIntraSize();
}

/* ================================================================
 * Category 2: Cooperative Group — Init / Query / Sync (8)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitBlock(void *coopOpaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coopOpaque;
  ::new (coop) flagcxCoopAny(flagcxCoopBlock());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitWarp(void *coopOpaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coopOpaque;
  ::new (coop) flagcxCoopAny(flagcxCoopWarp());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitThread(void *coopOpaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coopOpaque;
  ::new (coop) flagcxCoopAny(flagcxCoopThread());
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitTileSpan(void *coopOpaque, int t0, int nTiles, int id) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coopOpaque;
  ::new (coop) flagcxCoopAny(flagcxCoopTileSpan(t0, nTiles, id));
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopAnyInitLanes(void *coopOpaque, uint32_t laneMask) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coopOpaque;
  ::new (coop) flagcxCoopAny(flagcxCoopLanes(laneMask));
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopThreadRankC(const void *coopOpaque) {
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  return coop->threadRank();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxCoopSizeC(const void *coopOpaque) {
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  return coop->size();
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopSyncC(void *coopOpaque) {
  flagcxCoopAny *coop = (flagcxCoopAny *)coopOpaque;
  coop->sync();
}

/* ================================================================
 * Category 3: Team Functions (5)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxGetTeamIntra(const void *commOpaque, void *teamOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam *out = (flagcxTeam *)teamOpaque;
  *out = flagcxTeamIntra(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxGetTeamWorld(const void *commOpaque, void *teamOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam *out = (flagcxTeam *)teamOpaque;
  *out = flagcxTeamWorld(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxGetTeamInter(const void *commOpaque, void *teamOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam *out = (flagcxTeam *)teamOpaque;
  *out = flagcxTeamInter(*comm);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToWorldC(const void *commOpaque, const void *teamOpaque,
                       int rank) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  return flagcxTeamRankToWorld(*comm, *team, rank);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToIntraC(const void *commOpaque, const void *teamOpaque,
                       int rank) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  return flagcxTeamRankToIntra(*comm, *team, rank);
}

/* ================================================================
 * Category 4: Pointer Access (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointerC(const void *memOpaque, size_t offset,
                      const void *teamOpaque, int peer) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  return flagcxGetPeerPointer(*mem, offset, *team, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointerC(const void *memOpaque, size_t offset) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  return flagcxGetLocalPointer(*mem, offset);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetIntraPointerC(const void *memOpaque, size_t offset, int peer) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  return flagcxGetIntraPointer(*mem, offset, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointerC(const void *memOpaque, size_t offset,
                           const void *commOpaque) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return flagcxGetMulticastPointer(*mem, offset, *comm);
}

/* ================================================================
 * Category 5: Utility (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR size_t
flagcxDataTypeSizeDevice(flagcxDataType_t dt) {
  switch (dt) {
    case flagcxChar:
      return sizeof(char);
    case flagcxUint8:
      return sizeof(unsigned char);
    case flagcxInt:
      return sizeof(int);
    case flagcxUint32:
      return sizeof(unsigned int);
    case flagcxInt64:
      return sizeof(long long);
    case flagcxUint64:
      return sizeof(unsigned long long);
    case flagcxHalf:
      return 2;
    case flagcxFloat:
      return sizeof(float);
    case flagcxDouble:
      return sizeof(double);
    case flagcxBfloat16:
      return 2;
    default:
      return 0;
  }
}

/* ================================================================
 * Category 6: Intra-Node Barrier Session (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionInit(void *sessionOpaque, const void *coopOpaque,
                              const void *commOpaque, const void *teamOpaque,
                              uint32_t index, bool multimem) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)sessionOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopAny>(
      *coop, *comm, *team, index, multimem);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionArrive(void *sessionOpaque,
                                flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)sessionOpaque;
  session->bar.arrive(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionWait(void *sessionOpaque,
                              flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)sessionOpaque;
  session->bar.wait(order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierSessionSync(void *sessionOpaque,
                              flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierSession_C *session =
      (flagcxIntraBarrierSession_C *)sessionOpaque;
  session->bar.sync(order);
}

/* ================================================================
 * Category 7: Inter-Node Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierSessionInit(void *sessionOpaque, const void *coopOpaque,
                              const void *transOpaque, const void *teamOpaque,
                              uint32_t index) {
  flagcxInterBarrierSession_C *session =
      (flagcxInterBarrierSession_C *)sessionOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopAny>(
      *coop, *trans, *team, index);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxInterBarrierSessionSync(void *sessionOpaque,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxDevNetFenceLevel fence) {
  flagcxInterBarrierSession_C *session =
      (flagcxInterBarrierSession_C *)sessionOpaque;
  session->bar.sync(order, fence);
}

/* ================================================================
 * Category 8: World Barrier Session (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierSessionInit(void *sessionOpaque, const void *coopOpaque,
                              flagcxTeamTagWorld tag, const void *transOpaque,
                              uint32_t index, bool multimem) {
  flagcxBarrierSession_C *session = (flagcxBarrierSession_C *)sessionOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  ::new (&(session->bar)) flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopAny>(
      *coop, tag, *trans, index, multimem);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxWorldBarrierSessionSync(void *sessionOpaque,
                              flagcxDeviceMemoryOrder_t order,
                              flagcxDevNetFenceLevel fence) {
  flagcxBarrierSession_C *session = (flagcxBarrierSession_C *)sessionOpaque;
  session->bar.sync(order, fence);
}

/* ================================================================
 * Category 9: Transport — Init / Signal Read / Wait / Counter / Flush (7)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetInitC(void *transOpaque, const void *commOpaque, int idx) {
  flagcxDevNet *trans = (flagcxDevNet *)transOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  ::new (trans) flagcxDevNet(*comm, idx);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevNetReadSignal(const void *transOpaque, flagcxDevSignal_t signalId,
                       int bits, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  return trans->readSignal(signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitSignal(const void *transOpaque, const void *coopOpaque,
                       flagcxDevSignal_t signalId, uint64_t least, int bits,
                       flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->waitSignal(*coop, signalId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitSignalMeetShadow(const void *transOpaque,
                                 const void *coopOpaque,
                                 flagcxDevSignal_t signalId, int bits,
                                 flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->waitSignalMeetShadow(*coop, signalId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxDevNetReadCounter(const void *transOpaque, flagcxDevCounter_t counterId,
                        int bits, flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  return trans->readCounter(counterId, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetWaitCounter(const void *transOpaque, const void *coopOpaque,
                        flagcxDevCounter_t counterId, uint64_t least, int bits,
                        flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->waitCounter(*coop, counterId, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetFlush(const void *transOpaque, const void *coopOpaque,
                  flagcxDeviceMemoryOrder_t order) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->flush(*coop, order);
}

/* ================================================================
 * Category 9b: Net — Reset / Shadow (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetResetSignal(const void *netOpaque, flagcxDevSignal_t slot) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  net->resetSignal(slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetResetCounter(const void *netOpaque, flagcxDevCounter_t slot) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  net->resetCounter(slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetIncreaseSignalShadow(const void *netOpaque, flagcxDevSignal_t slot,
                                 uint64_t delta) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  net->increaseSignalShadow(slot, delta);
}

/* ================================================================
 * Category 10: Transport — Two-Sided (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetSend(const void *transOpaque, const void *coopOpaque,
                 const void *memOpaque, size_t offset, size_t count,
                 flagcxDataType_t datatype, int peer) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  return (int)trans->send(*coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetRecv(const void *transOpaque, const void *coopOpaque,
                 const void *memOpaque, size_t offset, size_t count,
                 flagcxDataType_t datatype, int peer) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  return (int)trans->recv(*coop, *mem, offset, count, datatype, peer);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetWait(const void *transOpaque, const void *coopOpaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  return (int)trans->wait(*coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxDevNetTerm(const void *transOpaque, const void *coopOpaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  return (int)trans->term(*coop);
}

/* ================================================================
 * Category 11: Transport — One-Sided put
 * (6 supported + 10 deprecated compatibility variants)
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut(const void *transOpaque, const void *teamOpaque, int peer,
                const void *dstOpaque, size_t dstOffset, const void *srcOpaque,
                size_t srcOffset, size_t bytes, const void *coopOpaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_None{}, *coop);
}

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc(const void *transOpaque, const void *teamOpaque,
                        int peer, const void *dstOpaque, size_t dstOffset,
                        const void *srcOpaque, size_t srcOffset, size_t bytes,
                        const void *coopOpaque,
                        flagcxDevSignal_t remoteSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal}, flagcxDevNet_None{}, *coop);
}

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigAdd(const void *transOpaque, const void *teamOpaque,
                        int peer, const void *dstOpaque, size_t dstOffset,
                        const void *srcOpaque, size_t srcOffset, size_t bytes,
                        const void *coopOpaque, flagcxDevSignal_t remoteSignal,
                        uint64_t remoteValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
             flagcxDevNet_None{}, *coop);
}

/* (None, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_LCtrInc(const void *transOpaque, const void *teamOpaque,
                        int peer, const void *dstOpaque, size_t dstOffset,
                        const void *srcOpaque, size_t srcOffset, size_t bytes,
                        const void *coopOpaque,
                        flagcxDevCounter_t localCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_CounterInc{localCounter}, *coop);
}

/* (SigInc, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc_LCtrInc(const void *transOpaque, const void *teamOpaque,
                                int peer, const void *dstOpaque,
                                size_t dstOffset, const void *srcOpaque,
                                size_t srcOffset, size_t bytes,
                                const void *coopOpaque,
                                flagcxDevSignal_t remoteSignal,
                                flagcxDevCounter_t localCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal},
             flagcxDevNet_CounterInc{localCounter}, *coop);
}

/* (SigAdd, CtrInc) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigAdd_LCtrInc(const void *transOpaque, const void *teamOpaque,
                                int peer, const void *dstOpaque,
                                size_t dstOffset, const void *srcOpaque,
                                size_t srcOffset, size_t bytes,
                                const void *coopOpaque,
                                flagcxDevSignal_t remoteSignal,
                                uint64_t remoteValue,
                                flagcxDevCounter_t localCounter) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
             flagcxDevNet_CounterInc{localCounter}, *coop);
}

/* ================================================================
 * Category 12: Transport — One-Sided signal (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalSigInc(const void *transOpaque, const void *teamOpaque,
                         int peer, const void *coopOpaque,
                         flagcxDevSignal_t signal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->signal(*team, peer, flagcxDevNet_SignalInc{signal}, *coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalSigAdd(const void *transOpaque, const void *teamOpaque,
                         int peer, const void *coopOpaque,
                         flagcxDevSignal_t signal, uint64_t value) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->signal(*team, peer, flagcxDevNet_SignalAdd{signal, value}, *coop);
}

/* ================================================================
 * Category 13: Transport — One-Sided putValue<uint64_t> (4)
 * ================================================================ */

/* (None, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue(const void *transOpaque, const void *teamOpaque, int peer,
                     const void *dstOpaque, size_t dstOffset, uint64_t value,
                     const void *coopOpaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->putValue(*team, peer, *dst, dstOffset, value, flagcxDevNet_None{},
                  *coop);
}

/* (SigInc, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue_RSigInc(const void *transOpaque, const void *teamOpaque,
                             int peer, const void *dstOpaque, size_t dstOffset,
                             uint64_t value, const void *coopOpaque,
                             flagcxDevSignal_t remoteSignal) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->putValue(*team, peer, *dst, dstOffset, value,
                  flagcxDevNet_SignalInc{remoteSignal}, *coop);
}

/* (SigAdd, None) */
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue_RSigAdd(const void *transOpaque, const void *teamOpaque,
                             int peer, const void *dstOpaque, size_t dstOffset,
                             uint64_t value, const void *coopOpaque,
                             flagcxDevSignal_t remoteSignal,
                             uint64_t remoteAddValue) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->putValue(*team, peer, *dst, dstOffset, value,
                  flagcxDevNet_SignalAdd{remoteSignal, remoteAddValue}, *coop);
}

/* ================================================================
 * Category 14: Transport — One-Sided get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetGet(const void *transOpaque, const void *teamOpaque, int peer,
                const void *srcOpaque, size_t srcOffset, const void *dstOpaque,
                size_t dstOffset, size_t bytes, const void *coopOpaque) {
  const flagcxDevNet *trans = (const flagcxDevNet *)transOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  trans->get(*team, peer, *src, srcOffset, *dst, dstOffset, bytes, *coop);
}

// Include scalar IR implementations (struct-free API for Triton/LLVM)
#include "flagcx_device_scalar_ir_impl.h"

// Keep deprecated pre-Unified-IR one-sided symbols linkable.
#include "flagcx_device_compat_ir_impl.h"

#endif /* FLAGCX_CHECK_DEVICE_CC */
#endif /* FLAGCX_DEVICE_WRAPPER_IMPL_H_ */
