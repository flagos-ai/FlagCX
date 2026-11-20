// Minimal synchronous transport for compiling/executing the real IR test
// bodies. It deliberately has no grid barrier and no implicit reset
// synchronization.
constexpr int kCompletionBits = 64;
using flagcxDevContext_t = int;
using flagcxDevCoopKind_t = int;
constexpr int FLAGCX_COOP_THREAD = 0, FLAGCX_COOP_WARP = 1,
              FLAGCX_COOP_BLOCK = 2;
constexpr int FLAGCX_TEAM_WORLD = 0, FLAGCX_TEAM_INTRA = 1,
              FLAGCX_TEAM_INTER = 2;
constexpr int flagcxDeviceMemoryOrderAcquire = 1,
              flagcxDeviceMemoryOrderRelease = 2;
constexpr int flagcxDeviceMemoryOrderAcqRel = 3,
              flagcxDeviceMemoryOrderSeqCst = 4;
constexpr int flagcxDeviceScopeSystem = 0;
constexpr int FLAGCX_UNIFIED_IR_SIGNAL_INC = 0,
              FLAGCX_UNIFIED_IR_SIGNAL_ADD = 1;
constexpr int FLAGCX_UNIFIED_IR_PUT_LCTR_INC = 0;
constexpr int FLAGCX_UNIFIED_IR_PUT_RSIG_INC_LCTR_INC = 1;
constexpr int FLAGCX_UNIFIED_IR_PUT_RSIG_ADD_LCTR_INC = 2;
enum class flagcxDevNetFenceLevel { Relaxed };
int flagcxDevCommGetRank(const void *) { return simRank; }
int flagcxDevCommGetSize(const void *) { return ranks; }
int flagcxDevCommGetIntraRank(const void *) { return simRank % 2; }
int flagcxDevCommGetIntraSize(const void *) { return 2; }
void flagcxCoopSyncS(int) {}
bool flagcxUnifiedIrTestCoopActive(int) { return true; }
int worldPeer(int team, int peer) {
  return team == FLAGCX_TEAM_WORLD   ? peer
         : team == FLAGCX_TEAM_INTRA ? simRank - simRank % 2 + peer
                                     : 2 * peer + simRank % 2;
}
Channel teamChannels[3][ranks][ctas];
void flagcxDevBarrierSync(const void *, int team, int index, int, int, int,
                          int) {
  int group = team == FLAGCX_TEAM_WORLD   ? 0
              : team == FLAGCX_TEAM_INTRA ? simRank / 2
                                          : simRank % 2;
  int participants = team == FLAGCX_TEAM_WORLD   ? ranks
                     : team == FLAGCX_TEAM_INTRA ? 2
                                                 : ranks / 2;
  teamChannels[team][group][index].sync(participants);
}
flagcxDevNet contextNet(int ctx) { return flagcxDevNet(flagcxDevComm{}, ctx); }
void flagcxDevResetSignal(const void *, int ctx, unsigned sig) {
  contextNet(ctx).resetSignal(sig);
}
void flagcxDevResetCounter(const void *, int ctx, unsigned ctr) {
  contextNet(ctx).resetCounter(ctr);
}
uint64_t flagcxDevReadSignal(const void *, unsigned sig, int, int ctx, int) {
  return contextNet(ctx).readSignal(sig);
}
uint64_t flagcxDevReadCounter(const void *, unsigned ctr, int, int ctx, int) {
  return contextNet(ctx).readCounter(ctr);
}
void flagcxDevWaitSignal(const void *, unsigned sig, uint64_t target, int,
                         int ctx, int, int) {
  contextNet(ctx).waitSignal(flagcxCoopBlock{}, sig, target);
}
void flagcxDevWaitCounter(const void *, unsigned ctr, uint64_t target, int,
                          int ctx, int, int) {
  contextNet(ctx).waitCounter(flagcxCoopBlock{}, ctr, target);
}
void flagcxDevSignalInc(const void *, int team, int peer, unsigned sig, int ctx,
                        int, int) {
  contextNet(ctx).remote(worldPeer(team, peer), flagcxDevNet_SignalInc{sig});
}
void flagcxDevSignalAdd(const void *, int team, int peer, unsigned sig,
                        uint64_t delta, int ctx, int, int) {
  contextNet(ctx).remote(worldPeer(team, peer),
                         flagcxDevNet_SignalAdd{sig, delta});
}
void flagcxDevIncreaseSignalShadow(const void *, int ctx, unsigned sig,
                                   uint64_t delta) {
  *contextNet(ctx).getSignalShadowPtr(sig) += delta;
}
void flagcxDevWaitSignalMeetShadow(const void *, int ctx, unsigned sig, int,
                                   int, int) {
  auto net = contextNet(ctx);
  net.waitSignal(flagcxCoopBlock{}, sig, *net.getSignalShadowPtr(sig));
}
void flagcxDevFlush(const void *, int, int, int) {}
#define PUT_ARGUMENTS                                                          \
  const void *, const void *dst, size_t doff, const void *src, size_t soff,    \
      size_t bytes, int team, int peer, int ctx, int, int, int
#define PUT_CALL(R, L)                                                         \
  contextNet(ctx).put(                                                         \
      flagcxTeam{false}, worldPeer(team, peer), *(const flagcxDevMem *)dst,    \
      doff, *(const flagcxDevMem *)src, soff, bytes, R, L, flagcxCoopThread{})
void flagcxDevPut(PUT_ARGUMENTS) {
  PUT_CALL(flagcxDevNet_None{}, flagcxDevNet_None{});
}
void flagcxDevPut_RSigInc(PUT_ARGUMENTS, unsigned sig) {
  PUT_CALL(flagcxDevNet_SignalInc{sig}, flagcxDevNet_None{});
}
void flagcxDevPut_RSigAdd(PUT_ARGUMENTS, unsigned sig, uint64_t value) {
  auto remote = flagcxDevNet_SignalAdd{sig, value};
  PUT_CALL(remote, flagcxDevNet_None{});
}
void flagcxDevPut_LCtrInc(PUT_ARGUMENTS, unsigned ctr) {
  PUT_CALL(flagcxDevNet_None{}, flagcxDevNet_CounterInc{ctr});
}
void flagcxDevPut_RSigInc_LCtrInc(PUT_ARGUMENTS, unsigned sig, unsigned ctr) {
  PUT_CALL(flagcxDevNet_SignalInc{sig}, flagcxDevNet_CounterInc{ctr});
}
void flagcxDevPut_RSigAdd_LCtrInc(PUT_ARGUMENTS, unsigned sig, uint64_t value,
                                  unsigned ctr) {
  auto remote = flagcxDevNet_SignalAdd{sig, value};
  PUT_CALL(remote, flagcxDevNet_CounterInc{ctr});
}
#undef PUT_CALL
#undef PUT_ARGUMENTS
void flagcxDevPutValue_RSigInc(const void *, const void *dst, size_t off,
                               uint64_t value, int team, int peer, int ctx, int,
                               int, int, unsigned sig) {
  contextNet(ctx).putValue(flagcxTeam{false}, worldPeer(team, peer),
                           *(const flagcxDevMem *)dst, off, value,
                           flagcxDevNet_SignalInc{sig}, flagcxCoopThread{});
}
void flagcxDevPutValue_RSigAdd(const void *, const void *dst, size_t off,
                               uint64_t value, int team, int peer, int ctx, int,
                               int, int, unsigned sig, uint64_t delta) {
  contextNet(ctx).putValue(
      flagcxTeam{false}, worldPeer(team, peer), *(const flagcxDevMem *)dst, off,
      value, flagcxDevNet_SignalAdd{sig, delta}, flagcxCoopThread{});
}
flagcxDevNet &netFromPtr(const void *ptr) { return *(flagcxDevNet *)ptr; }
const void *flagcxDevNetGetFromCommS(const void *, int ctx) {
  thread_local flagcxDevNet net(flagcxDevComm{}, 0);
  net.ctx = ctx;
  return &net;
}
void flagcxWorldBarrierSyncS(const void *, int, int index, bool, int,
                             flagcxDevNetFenceLevel) {
  channels[index].sync();
}
void flagcxDevNetResetSignal(const void *net, unsigned sig) {
  netFromPtr(net).resetSignal(sig);
}
void flagcxDevNetResetCounter(const void *net, unsigned ctr) {
  netFromPtr(net).resetCounter(ctr);
}
uint64_t flagcxDevNetReadSignalS(const void *net, unsigned sig, int, int) {
  return netFromPtr(net).readSignal(sig);
}
uint64_t flagcxDevNetReadCounterS(const void *net, unsigned ctr, int, int) {
  return netFromPtr(net).readCounter(ctr);
}
void flagcxDevNetIncreaseSignalShadow(const void *net, unsigned sig,
                                      uint64_t delta) {
  *netFromPtr(net).getSignalShadowPtr(sig) += delta;
}
void flagcxDevNetWaitSignalS(const void *net, int, unsigned sig,
                             uint64_t target, int, int) {
  netFromPtr(net).waitSignal(flagcxCoopBlock{}, sig, target);
}
void flagcxDevNetWaitCounterS(const void *net, int, unsigned ctr,
                              uint64_t target, int, int) {
  netFromPtr(net).waitCounter(flagcxCoopBlock{}, ctr, target);
}
void flagcxDevNetFlushS(const void *, int, int) {}
void *flagcxGetLocalPointerS(const void *mem, size_t off) {
  return flagcxGetLocalPointer(*(const flagcxDevMem *)mem, off);
}
#define SCALAR_PUT_ARGUMENTS                                                   \
  const void *net, const void *, int, int peer, const void *dst, size_t doff,  \
      const void *src, size_t soff, size_t bytes, int
#define SCALAR_PUT_CALL(R, L)                                                  \
  netFromPtr(net).put(flagcxTeam{false}, peer, *(const flagcxDevMem *)dst,     \
                      doff, *(const flagcxDevMem *)src, soff, bytes, R, L,     \
                      flagcxCoopThread{})
void flagcxDevNetPutS(SCALAR_PUT_ARGUMENTS) {
  SCALAR_PUT_CALL(flagcxDevNet_None{}, flagcxDevNet_None{});
}
void flagcxDevNetPutS_RSigInc(SCALAR_PUT_ARGUMENTS, unsigned sig) {
  SCALAR_PUT_CALL(flagcxDevNet_SignalInc{sig}, flagcxDevNet_None{});
}
void flagcxDevNetPutS_RSigInc_LCtrInc(SCALAR_PUT_ARGUMENTS, unsigned sig,
                                      unsigned ctr) {
  SCALAR_PUT_CALL(flagcxDevNet_SignalInc{sig}, flagcxDevNet_CounterInc{ctr});
}
#undef SCALAR_PUT_ARGUMENTS
#undef SCALAR_PUT_CALL
void flagcxDevNetSignalSigIncS(const void *net, const void *, int, int peer,
                               int, unsigned sig) {
  netFromPtr(net).remote(peer, flagcxDevNet_SignalInc{sig});
}
void flagcxDevNetSignalSigAddS(const void *net, const void *, int, int peer,
                               int, unsigned sig, uint64_t value) {
  netFromPtr(net).remote(peer, flagcxDevNet_SignalAdd{sig, value});
}
void flagcxDevNetPutValueS_RSigInc(const void *net, const void *, int, int peer,
                                   const void *dst, size_t off, uint64_t value,
                                   int, unsigned sig) {
  netFromPtr(net).putValue(flagcxTeam{false}, peer, *(const flagcxDevMem *)dst,
                           off, value, flagcxDevNet_SignalInc{sig},
                           flagcxCoopThread{});
}
void runIrTests(flagcxDevComm &, flagcxDevMem, flagcxDevMem);
