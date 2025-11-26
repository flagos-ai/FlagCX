#include "tuner_util.h"

#ifdef USE_METAX_ADAPTOR
static EnvVar algo("MCCL_ALGO", {"ring", "tree"}, "ring");

static EnvVar proto("MCCL_PROTO", {"LL", "LL128", "Simple"}, "Simple");

static EnvVar minChannels("MCCL_MIN_NCHANNELS", {"2", "4", "8", "16", "32"},
                          "2");

static EnvVar maxChannels("MCCL_MAX_NCHANNELS", {"8", "16", "32", "64"}, "32");

static EnvVar rwkBuffer("MCCL_DISABLE_CACHEABLE_BUFFER", {"0", "1"}, "0");

static EnvVar minP2PChannels("MCCL_MIN_P2P_NCHANNELS", {"1", "4", "8", "16"},
                             "1");

static EnvVar maxP2PChannels("MCCL_MAX_P2P_NCHANNELS", {"16", "32", "64"},
                             "64");

static EnvVar cacheFastWriteBack("MCCL_FAST_WRITE_BACK", {"-2", "1"}, "-2");

static EnvVar cacheEarlyWriteBack("MCCL_EARLY_WRITE_BACK",
                                  {"-2", "1", "4", "8", "15"}, "-2");

std::vector<EnvVar> mcclTunerVars = {algo,
                                     proto,
                                     minChannels,
                                     maxChannels,
                                     rwkBuffer,
                                     minP2PChannels,
                                     maxP2PChannels,
                                     cacheFastWriteBack,
                                     cacheEarlyWriteBack};
#endif