#pragma once

#include "flagcx.h"
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <optional>
#include <string>

// Macro to throw on a non-successful Flagcx return value.
#define C10D_FLAGCX_CHECK(cmd, failureReason)                                  \
  do {                                                                         \
    flagcxResult_t result = cmd;                                               \
    if (result != flagcxSuccess) {                                             \
      std::string err = "FLAGCX error in: " + std::string(__FILE__) + ":" +    \
                        std::to_string(__LINE__) +                             \
                        ", " /*+ flagcxGetErrorWithVersion(result)*/ + "\n" +  \
                        getFlagcxErrorDetailStr(result, failureReason);        \
      TORCH_CHECK_WITH(DistBackendError, false, err);                          \
    }                                                                          \
  } while (0)

namespace c10d {

std::string getFlagcxVersion();

std::string flagcxGetErrorWithVersion(flagcxResult_t error);

// Provides additional detail into Flagcx error codes based on when these are
// thrown in the Flagcx codebase.
std::string getFlagcxErrorDetailStr(
    flagcxResult_t error,
    std::optional<std::string> processGroupFailureReason = std::nullopt);

// RAII helper class to manage Flagcx group API.
//
// flagcxGroupEnd() is where the grouped operations are submitted, so its
// result is the result of the batch, not a teardown detail. Call close() to
// end the group wherever that result has to be acted on; the destructor then
// does nothing. A scope left without close() - an exception on the way out of
// the group - still ends the group in the destructor, so the group is never
// left open, and the result is reported there instead of thrown, since the
// destructor may run while another exception is in flight.
struct flagcxGroupGuard final {
  flagcxGroupGuard(flagcxComm_t comm) : comm_(comm) {
    C10D_FLAGCX_CHECK(flagcxGroupStart(comm_), std::nullopt);
  }
  flagcxGroupGuard(const flagcxGroupGuard &) = delete;
  flagcxGroupGuard &operator=(const flagcxGroupGuard &) = delete;
  flagcxGroupGuard(flagcxGroupGuard &&) = delete;
  flagcxGroupGuard &operator=(flagcxGroupGuard &&) = delete;
  ~flagcxGroupGuard() noexcept {
    try {
      if (close() != flagcxSuccess) {
        TORCH_WARN("flagcxGroupEnd failed while ending a group that was left "
                   "open; the operations grouped in it may not have been "
                   "submitted");
      }
    } catch (...) {
      // Preserve the exception that caused the group to be left open.
    }
  }
  // Ends the group and returns the result of submitting it. Reports success
  // without calling into Flagcx again if the group is already closed. The
  // result is the result of the batch, so dropping it is the defect this
  // guard is meant to prevent.
  [[nodiscard]] flagcxResult_t close() {
    if (closed_) {
      return flagcxSuccess;
    }
    closed_ = true;
    return flagcxGroupEnd(comm_);
  }
  flagcxComm_t comm_ = nullptr;
  bool closed_ = false;
};

} // namespace c10d
