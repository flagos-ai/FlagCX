/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Compile-time completion/control word selection for host-side FIFO code.
 ************************************************************************/

#ifndef FLAGCX_COMPLETION_WORD_H_
#define FLAGCX_COMPLETION_WORD_H_

#include "comm_traits.h"
#include "flagcx_kernel_core.h"
#include <cstdint>

template <typename...>
using flagcxVoidT = void;

template <typename API, typename = void>
struct flagcxBackendCompletionWord {
  using type = uint64_t;
};

template <typename API>
struct flagcxBackendCompletionWord<API,
                                   flagcxVoidT<typename API::CompletionWord>> {
  using type = typename API::CompletionWord;
};

using flagcxCompletionWord_t =
    typename flagcxBackendCompletionWord<DeviceAPI>::type;

static inline flagcxCompletionWord_t *
flagcxFifoControlPtr(uint64_t *buffer, flagcxFifoIndex index) {
  return reinterpret_cast<flagcxCompletionWord_t *>(buffer + index);
}

static inline volatile flagcxCompletionWord_t *
flagcxFifoControlPtr(volatile uint64_t *buffer, flagcxFifoIndex index) {
  return reinterpret_cast<volatile flagcxCompletionWord_t *>(buffer + index);
}

#endif // FLAGCX_COMPLETION_WORD_H_
