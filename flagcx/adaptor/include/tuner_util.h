#ifndef FLAGCX_TUNER_UTIL_H_
#define FLAGCX_TUNER_UTIL_H_

#include "tuner.h" // struct flagcxEnvConfig
#include <string>
#include <vector>

// This is a demonstration function that provide a way to load all config list
// for a specific GPU.

struct EnvVar {
  std::string name;
  std::vector<std::string> choices;
  std::string defaultValue;
  EnvVar(std::string n = "") : name(std::move(n)) {}
  EnvVar(std::string n, std::vector<std::string> c, std::string d = "")
      : name(std::move(n)), choices(std::move(c)), defaultValue(std::move(d)) {}
};

flagcxResult_t generateCandidate(std::vector<struct flagcxEnvConfig> &cfgList);
static void safeStrCopy(char *dst, size_t dstSize, const std::string &src);

extern std::vector<EnvVar> ncclTunerVars;
extern std::vector<EnvVar> mcclTunerVars;
extern std::vector<EnvVar> xcclTunerVars;
extern std::vector<EnvVar> &vars;

#endif // end include guard