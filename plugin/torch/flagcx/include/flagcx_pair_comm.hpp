#pragma once

#include "flagcx.h"
#include <map>
#include <utility>

namespace c10d {
// Each process owns one rank, but may participate in many pairs. Native/IPC
// shutdown can wait for the other endpoint, so all processes use this common
// (low rank, high rank) order regardless of pair creation/insertion history.
using FlagcxPairCommMap = std::map<std::pair<int, int>, flagcxComm_t>;
} // namespace c10d
