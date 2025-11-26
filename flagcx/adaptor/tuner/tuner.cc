#include "tuner_util.h"

#ifdef USE_NVIDIA_ADAPTOR
std::vector<EnvVar> &vars = ncclTunerVars;
#elif USE_METAX_ADAPTOR
std::vector<EnvVar> &vars = mcclTunerVars;
#elif USE_KUNLUNXIN_ADAPTOR
std::vector<EnvVar> &vars = xcclTunerVars;
#else
std::vector<EnvVar> emptyVars = {};
std::vector<EnvVar> &vars = emptyVars;
#endif