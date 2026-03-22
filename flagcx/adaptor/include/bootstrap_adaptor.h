#ifdef USE_BOOTSTRAP_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "check.h"
#include "comm.h"
#include "flagcx.h"
#include "utils.h"

#define BOOTSTRAP_ADAPTOR_SEND_RECV_TAG -6767
#define BOOTSTRAP_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8 * 1024 * 1024) // 8MB
struct stagedBuffer {
  int offset;
  int size;
  void *buffer;
};
typedef struct stagedBuffer *stagedBuffer_t;

struct flagcxInnerComm {
  bootstrapState *base;
};

#endif // USE_BOOTSTRAP_ADAPTOR