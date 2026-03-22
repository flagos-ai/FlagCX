#ifdef USE_GLOO_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "check.h"
#include "comm.h"
#include "flagcx.h"
#include "utils.h"

#include "gloo/algorithm.h"
#include "gloo/allgather.h"
#include "gloo/allgatherv.h"
#include "gloo/allreduce.h"
#include "gloo/alltoall.h"
#include "gloo/alltoallv.h"
#include "gloo/barrier.h"
#include "gloo/broadcast.h"
#include "gloo/context.h"
#include "gloo/gather.h"
#include "gloo/reduce.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/rendezvous/store.h"
#include "gloo/scatter.h"
#include "gloo/transport/context.h"
#include "gloo/transport/device.h"
#include "gloo/transport/ibverbs/device.h"
#include "gloo/transport/tcp/device.h"

#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#define GLOO_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8 * 1024 * 1024) // 8MB
using bufferPtr = std::unique_ptr<::gloo::transport::UnboundBuffer>;
struct stagedBuffer {
  int offset;
  int size = GLOO_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
  int cnt;
  void *buffer;
  gloo::transport::UnboundBuffer *unboundBuffer;
};
typedef stagedBuffer *stagedBuffer_t;

#define GENERATE_GLOO_TYPES(type, func, args...)                               \
  switch (type) {                                                              \
    case flagcxChar:                                                           \
      func<char>(args);                                                        \
      break;                                                                   \
    case flagcxUint8:                                                          \
      func<uint8_t>(args);                                                     \
      break;                                                                   \
    case flagcxInt:                                                            \
      func<int>(args);                                                         \
      break;                                                                   \
    case flagcxUint32:                                                         \
      func<uint32_t>(args);                                                    \
      break;                                                                   \
    case flagcxInt64:                                                          \
      func<int64_t>(args);                                                     \
      break;                                                                   \
    case flagcxUint64:                                                         \
      func<uint64_t>(args);                                                    \
      break;                                                                   \
    case flagcxHalf:                                                           \
      func<::gloo::float16>(args);                                             \
      break;                                                                   \
    case flagcxFloat:                                                          \
      func<float>(args);                                                       \
      break;                                                                   \
    case flagcxDouble:                                                         \
      func<double>(args);                                                      \
      break;                                                                   \
    case flagcxBfloat16:                                                       \
      printf("Invalid data type");                                             \
      break;                                                                   \
    default:                                                                   \
      printf("Invalid data type");                                             \
      break;                                                                   \
  }

typedef void (*flagcxGlooReduceFunc)(void *, const void *, const void *,
                                     size_t);
template <typename T>
flagcxGlooReduceFunc getGlooReduceFunc(flagcxRedOp_t op) {
  switch (op) {
    case flagcxSum:
      return flagcxGlooReduceFunc(&::gloo::sum<T>);
    case flagcxProd:
      return flagcxGlooReduceFunc(&::gloo::product<T>);
    case flagcxMax:
      return flagcxGlooReduceFunc(&::gloo::max<T>);
    case flagcxMin:
      return flagcxGlooReduceFunc(&::gloo::min<T>);
    case flagcxAvg:
      printf("Gloo backend does not support flagcxAvg Redop\n");
      return nullptr;
    default:
      return nullptr;
  }
}

template <typename T, typename F>
void getFunction(F &fn, flagcxRedOp_t op) {
  fn = getGlooReduceFunc<T>(op);
}

template <typename F>
F getFunction(flagcxDataType_t dtype, flagcxRedOp_t op) {
  F fn;
  GENERATE_GLOO_TYPES(dtype, getFunction, fn, op);
  return fn;
}

template <typename T, typename O>
void setInput(O &opts, void *ptr, size_t count) {
  opts.setInput(static_cast<T *>(ptr), count);
}

template <typename T, typename O>
void setInput(O &opts, void *ptr, std::vector<int64_t> vec) {
  opts.setInput(static_cast<T *>(ptr), vec);
}

template <typename T, typename O>
void setInputs(O &opts, void **ptrs, size_t len, size_t count) {
  opts.setInputs(reinterpret_cast<T **>(ptrs), len, count);
}

template <typename T, typename O>
void setOutput(O &opts, void *ptr, size_t count) {
  opts.setOutput(static_cast<T *>(ptr), count);
}

template <typename T, typename O>
void setOutput(O &opts, void *ptr, std::vector<int64_t> vec) {
  opts.setOutput(static_cast<T *>(ptr), vec);
}

struct MaxLengthData {
  unsigned long maxLength;
};

class flagcxGlooContext : public ::gloo::Context {
public:
  flagcxGlooContext(int rank, int nranks, bootstrapState *bootstrap)
      : ::gloo::Context(rank, nranks) {
    bootstrap_ = bootstrap;
  }

  ~flagcxGlooContext() {}

  void connectFullMesh(std::shared_ptr<::gloo::transport::Device> &dev) {
    unsigned long maxLength = 0;
    std::vector<std::vector<char>> addresses(size);
    auto transportContext = dev->createContext(rank, size);
    // transportContext->setTimeout(getTimeout());

    for (int i = 0; i < size; i++) {
      if (i == rank)
        continue;

      auto &pair = transportContext->createPair(i);

      // Store address for pair for this rank
      auto address = pair->address().bytes();
      maxLength = std::max(maxLength, address.size());
      addresses[i] = std::move(address);
    }

    // bootstrap allgather to get max length
    MaxLengthData *maxLengthData;
    flagcxCalloc(&maxLengthData, size);
    maxLengthData[rank].maxLength = maxLength;
    bootstrapAllGather(bootstrap_, (void *)maxLengthData,
                       sizeof(MaxLengthData));
    bootstrapBarrier(bootstrap_, rank, size, 0);
    for (int i = 0; i < size; ++i) {
      maxLength = std::max(maxLength, maxLengthData[i].maxLength);
    }

    // Prepare input and output
    std::vector<char> addressData(size * size * maxLength);
    for (int i = 0; i < size; ++i) {
      if (i == rank) {
        continue;
      }

      auto offset = (rank * size + i) * maxLength;
      auto &address = addresses[i];
      memcpy(addressData.data() + offset, address.data(), address.size());
    }

    // bootstrap allgather to get all addresses
    bootstrapAllGather(bootstrap_, (void *)addressData.data(),
                       size * maxLength);
    bootstrapBarrier(bootstrap_, rank, size, 0);

    // Connect every pair
    for (int i = 0; i < size; ++i) {
      if (i == rank) {
        continue;
      }

      auto offset = (rank + i * size) * maxLength;
      std::vector<char> address(maxLength);
      memcpy(address.data(), addressData.data() + offset, maxLength);
      transportContext->getPair(i)->connect(address);
    }

    device_ = dev;
    transportContext_ = std::move(transportContext);
  }

public:
  bootstrapState *bootstrap_;
};

struct flagcxInnerComm {
  std::shared_ptr<flagcxGlooContext> base;
};

#endif // USE_GLOO_ADAPTOR