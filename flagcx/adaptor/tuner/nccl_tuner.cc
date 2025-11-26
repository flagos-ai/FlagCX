#include "tuner_util.h"

#ifdef USE_NVIDIA_ADAPTOR

static EnvVar algo("NCCL_ALGO", {"ring", "tree"}, "ring");

static EnvVar proto("NCCL_PROTO", {"LL", "LL128", "Simple"}, "Simple");

static EnvVar thread("NCCL_NTHREADS", {"128", "256"}, "256");

static EnvVar minChannel("NCCL_MIN_NCHANNELS", {"16", "32"}, "16");

static EnvVar chunkSize("NCCL_P2P_NVL_CHUNKSIZE", {"1024", "2048"}, "1024");

std::vector<EnvVar> ncclTunerVars = {algo, proto, thread, minChannel,
                                     chunkSize};

#endif