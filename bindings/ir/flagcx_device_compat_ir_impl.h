/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Deprecated Device IR compatibility entry points.
 *
 * Keep the pre-Unified-IR one-sided symbol set linkable for one deprecation
 * cycle. New code should use the transport-transparent flagcxDev* API.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_COMPAT_IR_IMPL_H_
#define FLAGCX_DEVICE_COMPAT_IR_IMPL_H_

template <typename RemoteAction, typename LocalAction>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCompatNetPutC(const void *netOpaque, const void *teamOpaque, int peer,
                    const void *dstOpaque, size_t dstOffset,
                    const void *srcOpaque, size_t srcOffset, size_t bytes,
                    const void *coopOpaque, RemoteAction remoteAction,
                    LocalAction localAction) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  net->put(*team, peer, *dst, dstOffset, *src, srcOffset, bytes, remoteAction,
           localAction, *coop);
}

template <typename RemoteAction, typename LocalAction>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCompatNetPutS(const void *netOpaque, const void *commOpaque,
                    flagcxTeamKind_t teamKind, int peer, const void *dstOpaque,
                    size_t dstOffset, const void *srcOpaque, size_t srcOffset,
                    size_t bytes, flagcxCoopKind_t coopKind,
                    RemoteAction remoteAction, LocalAction localAction) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes, remoteAction,
           localAction, coop);
}

template <typename Action>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCompatNetSignalC(const void *netOpaque, const void *teamOpaque, int peer,
                       const void *coopOpaque, Action action) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  net->signal(*team, peer, action, *coop);
}

template <typename Action>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCompatNetSignalS(const void *netOpaque, const void *commOpaque,
                       flagcxTeamKind_t teamKind, int peer,
                       flagcxCoopKind_t coopKind, Action action) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->signal(team, peer, action, coop);
}

template <typename Action>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCompatNetPutValueC(const void *netOpaque, const void *teamOpaque,
                         int peer, const void *dstOpaque, size_t dstOffset,
                         uint64_t value, const void *coopOpaque,
                         Action action) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxTeam *team = (const flagcxTeam *)teamOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxCoopAny *coop = (const flagcxCoopAny *)coopOpaque;
  net->putValue(*team, peer, *dst, dstOffset, value, action, *coop);
}

template <typename Action>
static FLAGCX_DEVICE_INLINE_DECORATOR void flagcxCompatNetPutValueS(
    const void *netOpaque, const void *commOpaque, flagcxTeamKind_t teamKind,
    int peer, const void *dstOpaque, size_t dstOffset, uint64_t value,
    flagcxCoopKind_t coopKind, Action action) {
  const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  net->putValue(team, peer, *dst, dstOffset, value, action, coop);
}

/* Struct-based compatibility entry points. */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc(const void *net, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevCounter_t remoteCounter) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_None{});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_LSigInc(const void *net, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_None{},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc_LSigInc(const void *net, const void *team, int peer,
                                const void *dst, size_t dstOffset,
                                const void *src, size_t srcOffset, size_t bytes,
                                const void *coop,
                                flagcxDevSignal_t remoteSignal,
                                flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_SignalInc{remoteSignal},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigAdd_LSigInc(const void *net, const void *team, int peer,
                                const void *dst, size_t dstOffset,
                                const void *src, size_t srcOffset, size_t bytes,
                                const void *coop,
                                flagcxDevSignal_t remoteSignal,
                                uint64_t remoteValue,
                                flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc_LSigInc(const void *net, const void *team, int peer,
                                const void *dst, size_t dstOffset,
                                const void *src, size_t srcOffset, size_t bytes,
                                const void *coop,
                                flagcxDevCounter_t remoteCounter,
                                flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_LSigAdd(const void *net, const void *team, int peer,
                        const void *dst, size_t dstOffset, const void *src,
                        size_t srcOffset, size_t bytes, const void *coop,
                        flagcxDevSignal_t localSignal, uint64_t localValue) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_None{},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigInc_LSigAdd(const void *net, const void *team, int peer,
                                const void *dst, size_t dstOffset,
                                const void *src, size_t srcOffset, size_t bytes,
                                const void *coop,
                                flagcxDevSignal_t remoteSignal,
                                flagcxDevSignal_t localSignal,
                                uint64_t localValue) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_SignalInc{remoteSignal},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RSigAdd_LSigAdd(
    const void *net, const void *team, int peer, const void *dst,
    size_t dstOffset, const void *src, size_t srcOffset, size_t bytes,
    const void *coop, flagcxDevSignal_t remoteSignal, uint64_t remoteValue,
    flagcxDevSignal_t localSignal, uint64_t localValue) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc_LSigAdd(const void *net, const void *team, int peer,
                                const void *dst, size_t dstOffset,
                                const void *src, size_t srcOffset, size_t bytes,
                                const void *coop,
                                flagcxDevCounter_t remoteCounter,
                                flagcxDevSignal_t localSignal,
                                uint64_t localValue) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPut_RCtrInc_LCtrInc(const void *net, const void *team, int peer,
                                const void *dst, size_t dstOffset,
                                const void *src, size_t srcOffset, size_t bytes,
                                const void *coop,
                                flagcxDevCounter_t remoteCounter,
                                flagcxDevCounter_t localCounter) {
  flagcxCompatNetPutC(net, team, peer, dst, dstOffset, src, srcOffset, bytes,
                      coop, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_CounterInc{localCounter});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalCtrInc(const void *net, const void *team, int peer,
                         const void *coop, flagcxDevCounter_t counter) {
  flagcxCompatNetSignalC(net, team, peer, coop,
                         flagcxDevNet_CounterInc{counter});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValue_RCtrInc(const void *net, const void *team, int peer,
                             const void *dst, size_t dstOffset, uint64_t value,
                             const void *coop,
                             flagcxDevCounter_t remoteCounter) {
  flagcxCompatNetPutValueC(net, team, peer, dst, dstOffset, value, coop,
                           flagcxDevNet_CounterInc{remoteCounter});
}

/* Scalar compatibility entry points. */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_RCtrInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevCounter_t remoteCounter) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_None{});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevNetPutS_LSigInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_None{},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigInc_LSigInc(const void *net, const void *comm,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevSignal_t remoteSignal,
                                 flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_SignalInc{remoteSignal},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LSigInc(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    uint64_t remoteValue, flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind,
                      flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LSigInc(const void *net, const void *comm,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevCounter_t remoteCounter,
                                 flagcxDevSignal_t localSignal) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_SignalInc{localSignal});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_LSigAdd(const void *net, const void *comm,
                         flagcxTeamKind_t teamKind, int peer, const void *dst,
                         size_t dstOffset, const void *src, size_t srcOffset,
                         size_t bytes, flagcxCoopKind_t coopKind,
                         flagcxDevSignal_t localSignal, uint64_t localValue) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_None{},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigInc_LSigAdd(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    flagcxDevSignal_t localSignal, uint64_t localValue) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_SignalInc{remoteSignal},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RSigAdd_LSigAdd(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevSignal_t remoteSignal,
    uint64_t remoteValue, flagcxDevSignal_t localSignal, uint64_t localValue) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind,
                      flagcxDevNet_SignalAdd{remoteSignal, remoteValue},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LSigAdd(
    const void *net, const void *comm, flagcxTeamKind_t teamKind, int peer,
    const void *dst, size_t dstOffset, const void *src, size_t srcOffset,
    size_t bytes, flagcxCoopKind_t coopKind, flagcxDevCounter_t remoteCounter,
    flagcxDevSignal_t localSignal, uint64_t localValue) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_SignalAdd{localSignal, localValue});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutS_RCtrInc_LCtrInc(const void *net, const void *comm,
                                 flagcxTeamKind_t teamKind, int peer,
                                 const void *dst, size_t dstOffset,
                                 const void *src, size_t srcOffset,
                                 size_t bytes, flagcxCoopKind_t coopKind,
                                 flagcxDevCounter_t remoteCounter,
                                 flagcxDevCounter_t localCounter) {
  flagcxCompatNetPutS(net, comm, teamKind, peer, dst, dstOffset, src, srcOffset,
                      bytes, coopKind, flagcxDevNet_CounterInc{remoteCounter},
                      flagcxDevNet_CounterInc{localCounter});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetSignalCtrIncS(const void *net, const void *comm,
                          flagcxTeamKind_t teamKind, int peer,
                          flagcxCoopKind_t coopKind,
                          flagcxDevCounter_t counter) {
  flagcxCompatNetSignalS(net, comm, teamKind, peer, coopKind,
                         flagcxDevNet_CounterInc{counter});
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevNetPutValueS_RCtrInc(const void *net, const void *comm,
                              flagcxTeamKind_t teamKind, int peer,
                              const void *dst, size_t dstOffset, uint64_t value,
                              flagcxCoopKind_t coopKind,
                              flagcxDevCounter_t remoteCounter) {
  flagcxCompatNetPutValueS(net, comm, teamKind, peer, dst, dstOffset, value,
                           coopKind, flagcxDevNet_CounterInc{remoteCounter});
}

#endif /* FLAGCX_DEVICE_COMPAT_IR_IMPL_H_ */
