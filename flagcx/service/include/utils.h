/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_UTILS_H_
#define FLAGCX_UTILS_H_

#include "check.h"
#include "global_comm.h"
#include "pthread.h"
#include "type.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <new>
#include <sched.h>
#include <stdint.h>
#include <time.h>

#define LOADAPI(struct, api, ptr)                                              \
  api:                                                                         \
  (typeof(struct ::api))ptr

// PCI Bus ID <-> int64 conversion functions
flagcxResult_t int64ToBusId(int64_t id, char *busId);
flagcxResult_t busIdToInt64(const char *busId, int64_t *id);

flagcxResult_t getBusId(int cudaDev, int64_t *busId);

flagcxResult_t getHostName(char *hostname, int maxlen, const char delim);
uint64_t getHash(const char *string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
flagcxResult_t getRandomData(void *buffer, size_t bytes);

const char *flagcxOpToString(flagcxRedOp_t op);
const char *flagcxDatatypeToString(flagcxDataType_t type);
const char *flagcxAlgoToString(int algo);
const char *flagcxProtoToString(int proto);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char *string, struct netIf *ifList, int maxList);
bool matchIfList(const char *string, int port, struct netIf *ifList,
                 int listSize, bool matchExact);

static long log2i(long n) {
  long l = 0;
  while (n >>= 1)
    l++;
  return l;
}

inline uint64_t clockNano() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000 * 1000 * 1000 + ts.tv_nsec;
}

/* get any bytes of random data from /dev/urandom, return 0 if it succeeds; else
 * return -1 */
inline flagcxResult_t getRandomData(void *buffer, size_t bytes) {
  flagcxResult_t ret = flagcxSuccess;
  if (bytes > 0) {
    const size_t one = 1UL;
    FILE *fp = fopen("/dev/urandom", "r");
    if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one)
      ret = flagcxSystemError;
    if (fp)
      fclose(fp);
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////

template <typename Int>
inline void flagcxAtomicRefCountIncrement(Int *refs) {
  __atomic_fetch_add(refs, 1, __ATOMIC_RELAXED);
}

template <typename Int>
inline Int flagcxAtomicRefCountDecrement(Int *refs) {
  return __atomic_sub_fetch(refs, 1, __ATOMIC_ACQ_REL);
}

////////////////////////////////////////////////////////////////////////////////
/* flagcxMemoryStack: Pools memory for fast LIFO ordered allocation. Note that
 * granularity of LIFO is not per object, instead frames containing many objects
 * are pushed and popped. Therefor deallocation is extremely cheap since its
 * done at the frame granularity.
 *
 * The initial state of the stack is with one frame, the "nil" frame, which
 * cannot be popped. Therefor objects allocated in the nil frame cannot be
 * deallocated sooner than stack destruction.
 */
struct flagcxMemoryStack;

void flagcxMemoryStackConstruct(struct flagcxMemoryStack *me);
void flagcxMemoryStackDestruct(struct flagcxMemoryStack *me);
void flagcxMemoryStackPush(struct flagcxMemoryStack *me);
void flagcxMemoryStackPop(struct flagcxMemoryStack *me);
template <typename T>
T *flagcxMemoryStackAlloc(struct flagcxMemoryStack *me, size_t n = 1);

////////////////////////////////////////////////////////////////////////////////
/* flagcxMemoryPool: A free-list of same-sized allocations. It is an invalid for
 * a pool instance to ever hold objects whose type have differing
 * (sizeof(T), alignof(T)) pairs. The underlying memory is supplied by
 * a backing `flagcxMemoryStack` passed during Alloc(). If memory
 * backing any currently held object is deallocated then it is an error to do
 * anything other than reconstruct it, after which it is a valid empty pool.
 */
struct flagcxMemoryPool;

// Equivalent to zero-initialization
void flagcxMemoryPoolConstruct(struct flagcxMemoryPool *me);
template <typename T>
T *flagcxMemoryPoolAlloc(struct flagcxMemoryPool *me,
                         struct flagcxMemoryStack *backing);
template <typename T>
void flagcxMemoryPoolFree(struct flagcxMemoryPool *me, T *obj);
void flagcxMemoryPoolTakeAll(struct flagcxMemoryPool *me,
                             struct flagcxMemoryPool *from);

////////////////////////////////////////////////////////////////////////////////
/* flagcxIntruQueue: A singly-linked list queue where the per-object next
 * pointer field is given via the `next` template argument.
 *
 * Example:
 *   struct Foo {
 *     struct Foo *next1, *next2; // can be a member of two lists at once
 *   };
 *   flagcxIntruQueue<Foo, &Foo::next1> list1;
 *   flagcxIntruQueue<Foo, &Foo::next2> list2;
 */
template <typename T, T *T::*next>
struct flagcxIntruQueue;

template <typename T, T *T::*next>
void flagcxIntruQueueConstruct(flagcxIntruQueue<T, next> *me);
template <typename T, T *T::*next>
bool flagcxIntruQueueEmpty(flagcxIntruQueue<T, next> *me);
template <typename T, T *T::*next>
T *flagcxIntruQueueHead(flagcxIntruQueue<T, next> *me);
template <typename T, T *T::*next>
void flagcxIntruQueueEnqueue(flagcxIntruQueue<T, next> *me, T *x);
template <typename T, T *T::*next>
T *flagcxIntruQueueDequeue(flagcxIntruQueue<T, next> *me);
template <typename T, T *T::*next>
T *flagcxIntruQueueTryDequeue(flagcxIntruQueue<T, next> *me);
template <typename T, T *T::*next>
void flagcxIntruQueueFreeAll(flagcxIntruQueue<T, next> *me,
                             flagcxMemoryPool *memPool);

////////////////////////////////////////////////////////////////////////////////
/* flagcxThreadSignal: Couples a pthread mutex and cond together. The "mutex"
 * and "cond" fields are part of the public interface.
 */
struct flagcxThreadSignal {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

// returns {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER}
constexpr flagcxThreadSignal flagcxThreadSignalStaticInitializer();

void flagcxThreadSignalConstruct(struct flagcxThreadSignal *me);
void flagcxThreadSignalDestruct(struct flagcxThreadSignal *me);

// A convenience instance per-thread.
extern __thread struct flagcxThreadSignal flagcxThreadSignalLocalInstance;

////////////////////////////////////////////////////////////////////////////////

template <typename T, T *T::*next>
struct flagcxIntruQueueMpsc;

template <typename T, T *T::*next>
void flagcxIntruQueueMpscConstruct(struct flagcxIntruQueueMpsc<T, next> *me);
template <typename T, T *T::*next>
bool flagcxIntruQueueMpscEmpty(struct flagcxIntruQueueMpsc<T, next> *me);
// Enqueue element. Returns true if queue is not abandoned. Even if queue is
// abandoned the element enqueued, so the caller needs to make arrangements for
// the queue to be tended.
template <typename T, T *T::*next>
bool flagcxIntruQueueMpscEnqueue(struct flagcxIntruQueueMpsc<T, next> *me,
                                 T *x);
// Dequeue all elements at a glance. If there aren't any and `waitSome` is
// true then this call will wait until it can return a non empty list.
template <typename T, T *T::*next>
T *flagcxIntruQueueMpscDequeueAll(struct flagcxIntruQueueMpsc<T, next> *me,
                                  bool waitSome);
// Dequeue all elements and set queue to abandoned state.
template <typename T, T *T::*next>
T *flagcxIntruQueueMpscAbandon(struct flagcxIntruQueueMpsc<T, next> *me);

////////////////////////////////////////////////////////////////////////////////

// Function helps init single homo cluster.
// return homoComm via homoComm parameter.
flagcxResult_t flagcxHomoCommInit(flagcxUniqueId_t commId,
                                  flagcxUniqueId *uniqueIdData,
                                  struct bootstrapState *state,
                                  flagcxComm_t comm,
                                  flagcxInnerComm_t *homoComm /*out*/);
////////////////////////////////////////////////////////////////////////////////

struct flagcxMemoryStack {
  struct Hunk {
    struct Hunk *above; // reverse stack pointer
    size_t size; // size of this allocation (including this header struct)
  };
  struct Unhunk { // proxy header for objects allocated out-of-hunk
    struct Unhunk *next;
    void *obj;
  };
  struct Frame {
    struct Hunk *hunk;     // top of non-empty hunks
    uintptr_t bumper, end; // points into top hunk
    struct Unhunk *unhunks;
    struct Frame *below;
  };

  static void *allocateSpilled(struct flagcxMemoryStack *me, size_t size,
                               size_t align);
  static void *allocate(struct flagcxMemoryStack *me, size_t size,
                        size_t align);

  struct Hunk stub;
  struct Frame topFrame;
};

inline void flagcxMemoryStackConstruct(struct flagcxMemoryStack *me) {
  me->stub.above = nullptr;
  me->stub.size = 0;
  me->topFrame.hunk = &me->stub;
  me->topFrame.bumper = 0;
  me->topFrame.end = 0;
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = nullptr;
}

inline void *flagcxMemoryStack::allocate(struct flagcxMemoryStack *me,
                                         size_t size, size_t align) {
  uintptr_t o = (me->topFrame.bumper + align - 1) & -uintptr_t(align);
  void *obj;
  if (__builtin_expect(o + size <= me->topFrame.end, true)) {
    me->topFrame.bumper = o + size;
    obj = reinterpret_cast<void *>(o);
  } else {
    obj = allocateSpilled(me, size, align);
  }
  return obj;
}

template <typename T>
inline T *flagcxMemoryStackAlloc(struct flagcxMemoryStack *me, size_t n) {
  void *obj = flagcxMemoryStack::allocate(me, n * sizeof(T), alignof(T));
  memset(obj, 0, n * sizeof(T));
  return (T *)obj;
}

inline void flagcxMemoryStackPush(struct flagcxMemoryStack *me) {
  using Frame = flagcxMemoryStack::Frame;
  Frame tmp = me->topFrame;
  Frame *snapshot =
      (Frame *)flagcxMemoryStack::allocate(me, sizeof(Frame), alignof(Frame));
  *snapshot = tmp; // C++ struct assignment
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = snapshot;
}

inline void flagcxMemoryStackPop(struct flagcxMemoryStack *me) {
  flagcxMemoryStack::Unhunk *un = me->topFrame.unhunks;
  while (un != nullptr) {
    free(un->obj);
    un = un->next;
  }
  me->topFrame = *me->topFrame.below; // C++ struct assignment
}

////////////////////////////////////////////////////////////////////////////////

struct flagcxMemoryPool {
  struct Cell {
    Cell *next;
  };
  struct Cell *head;
  struct Cell *tail; // meaningful only when head != nullptr
};

inline void flagcxMemoryPoolConstruct(struct flagcxMemoryPool *me) {
  me->head = nullptr;
}

template <typename T>
inline T *flagcxMemoryPoolAlloc(struct flagcxMemoryPool *me,
                                struct flagcxMemoryStack *backing) {
  using Cell = flagcxMemoryPool::Cell;
  Cell *cell;
  if (__builtin_expect(me->head != nullptr, true)) {
    cell = me->head;
    me->head = cell->next;
  } else {
    // Use the internal allocate() since it doesn't memset to 0 yet.
    size_t cellSize = std::max(sizeof(Cell), sizeof(T));
    size_t cellAlign = std::max(alignof(Cell), alignof(T));
    cell = (Cell *)flagcxMemoryStack::allocate(backing, cellSize, cellAlign);
  }
  memset(cell, 0, sizeof(T));
  return reinterpret_cast<T *>(cell);
}

template <typename T>
inline void flagcxMemoryPoolFree(struct flagcxMemoryPool *me, T *obj) {
  using Cell = flagcxMemoryPool::Cell;
  Cell *cell = reinterpret_cast<Cell *>(obj);
  cell->next = me->head;
  if (me->head == nullptr)
    me->tail = cell;
  me->head = cell;
}

inline void flagcxMemoryPoolTakeAll(struct flagcxMemoryPool *me,
                                    struct flagcxMemoryPool *from) {
  if (from->head != nullptr) {
    from->tail->next = me->head;
    if (me->head == nullptr)
      me->tail = from->tail;
    me->head = from->head;
    from->head = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, T *T::*next>
struct flagcxIntruQueue {
  T *head, *tail;
};

template <typename T, T *T::*next>
inline void flagcxIntruQueueConstruct(flagcxIntruQueue<T, next> *me) {
  me->head = nullptr;
  me->tail = nullptr;
}

template <typename T, T *T::*next>
inline bool flagcxIntruQueueEmpty(flagcxIntruQueue<T, next> *me) {
  return me->head == nullptr;
}

template <typename T, T *T::*next>
inline T *flagcxIntruQueueHead(flagcxIntruQueue<T, next> *me) {
  return me->head;
}

template <typename T, T *T::*next>
inline T *flagcxIntruQueueTail(flagcxIntruQueue<T, next> *me) {
  return me->tail;
}

template <typename T, T *T::*next>
inline void flagcxIntruQueueEnqueue(flagcxIntruQueue<T, next> *me, T *x) {
  x->*next = nullptr;
  (me->head ? me->tail->*next : me->head) = x;
  me->tail = x;
}

template <typename T, T *T::*next>
inline T *flagcxIntruQueueDequeue(flagcxIntruQueue<T, next> *me) {
  T *ans = me->head;
  me->head = ans->*next;
  if (me->head == nullptr)
    me->tail = nullptr;
  return ans;
}

template <typename T, T *T::*next>
inline bool flagcxIntruQueueDelete(flagcxIntruQueue<T, next> *me, T *x) {
  T *prev = nullptr;
  T *cur = me->head;
  bool found = false;

  while (cur) {
    if (cur == x) {
      found = true;
      break;
    }
    prev = cur;
    cur = cur->*next;
  }

  if (found) {
    if (prev == nullptr)
      me->head = cur->*next;
    else
      prev->*next = cur->*next;
    if (cur == me->tail)
      me->tail = prev;
  }
  return found;
}

template <typename T, T *T::*next>
inline T *flagcxIntruQueueTryDequeue(flagcxIntruQueue<T, next> *me) {
  T *ans = me->head;
  if (ans != nullptr) {
    me->head = ans->*next;
    if (me->head == nullptr)
      me->tail = nullptr;
  }
  return ans;
}

template <typename T, T *T::*next>
void flagcxIntruQueueFreeAll(flagcxIntruQueue<T, next> *me,
                             flagcxMemoryPool *pool) {
  T *head = me->head;
  me->head = nullptr;
  me->tail = nullptr;
  while (head != nullptr) {
    T *tmp = head->*next;
    flagcxMemoryPoolFree(pool, tmp);
    head = tmp;
  }
}

/* cmp function determines the sequence of objects in the queue. If cmp returns
 * value >= 0, it means a > b, and we should put a before b; otherwise, b should
 * be put ahead of a. */
template <typename T, T *T::*next>
inline void flagcxIntruQueueSortEnqueue(flagcxIntruQueue<T, next> *me, T *x,
                                        int (*cmp)(T *a, T *b)) {
  T *cur = me->head;
  T *prev = NULL;

  if (cur == NULL) {
    x->*next = nullptr;
    me->tail = me->head = x;
  } else {
    while (cur) {
      if (cmp(cur, x) > 0) {
        prev = cur;
        cur = cur->next;
      } else {
        break;
      }
    }

    x->*next = cur;
    if (prev) {
      prev->*next = x;
      if (cur == NULL)
        me->tail = x;
    } else {
      me->head = x;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

#define FLAGCX_TEMPLETELIST_DEFINE(listName, type, prev, next)                 \
  inline void flagcx##listName##ListEnList(type **head, type *x) {             \
    if (x->next == reinterpret_cast<type *>(0x1)) {                            \
      x->next = *head;                                                         \
      x->prev = NULL;                                                          \
      if (*head)                                                               \
        (*head)->prev = x;                                                     \
      *head = x;                                                               \
    }                                                                          \
  }                                                                            \
  inline type *flagcx##listName##ListDeList(type **head) {                     \
    type *ret = *head;                                                         \
    if ((*head)->next)                                                         \
      (*head)->next->prev = NULL;                                              \
    *head = (*head)->next;                                                     \
    ret->next = reinterpret_cast<type *>(0x1);                                 \
    return ret;                                                                \
  }                                                                            \
  inline void flagcx##listName##ListDelete(type **head, type *x) {             \
    if (x->prev)                                                               \
      x->prev->next = x->next;                                                 \
    if (x->next)                                                               \
      x->next->prev = x->prev;                                                 \
    if (*head == x)                                                            \
      *head = x->next;                                                         \
    x->next = reinterpret_cast<type *>(0x1);                                   \
  }                                                                            \
  inline bool flagcx##listName##ListEmpty(type *head) { return head == NULL; }

////////////////////////////////////////////////////////////////////////////////

constexpr flagcxThreadSignal flagcxThreadSignalStaticInitializer() {
  return {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};
}

inline void flagcxThreadSignalConstruct(struct flagcxThreadSignal *me) {
  pthread_mutex_init(&me->mutex, nullptr);
  pthread_cond_init(&me->cond, nullptr);
}

inline void flagcxThreadSignalDestruct(struct flagcxThreadSignal *me) {
  pthread_mutex_destroy(&me->mutex);
  pthread_cond_destroy(&me->cond);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T, T *T::*next>
struct flagcxIntruQueueMpsc {
  T *head;
  uintptr_t tail;
  struct flagcxThreadSignal *waiting;
};

template <typename T, T *T::*next>
void flagcxIntruQueueMpscConstruct(struct flagcxIntruQueueMpsc<T, next> *me) {
  me->head = nullptr;
  me->tail = 0x0;
  me->waiting = nullptr;
}

template <typename T, T *T::*next>
bool flagcxIntruQueueMpscEmpty(struct flagcxIntruQueueMpsc<T, next> *me) {
  return __atomic_load_n(&me->tail, __ATOMIC_RELAXED) <= 0x2;
}

template <typename T, T *T::*next>
bool flagcxIntruQueueMpscEnqueue(flagcxIntruQueueMpsc<T, next> *me, T *x) {
  __atomic_store_n(&(x->*next), nullptr, __ATOMIC_RELAXED);
  uintptr_t utail = __atomic_exchange_n(
      &me->tail, reinterpret_cast<uintptr_t>(x), __ATOMIC_ACQ_REL);
  T *prev = reinterpret_cast<T *>(utail);
  T **prevNext = utail <= 0x2 ? &me->head : &(prev->*next);
  __atomic_store_n(prevNext, x, __ATOMIC_RELAXED);
  if (utail == 0x1) {                        // waiting
    __atomic_thread_fence(__ATOMIC_ACQUIRE); // to see me->waiting
    // This lock/unlock is essential to ensure we don't race ahead of the
    // consumer and signal the cond before they begin waiting on it.
    struct flagcxThreadSignal *waiting = me->waiting;
    pthread_mutex_lock(&waiting->mutex);
    pthread_mutex_unlock(&waiting->mutex);
    pthread_cond_broadcast(&waiting->cond);
  }
  return utail != 0x2; // not abandoned
}

template <typename T, T *T::*next>
T *flagcxIntruQueueMpscDequeueAll(flagcxIntruQueueMpsc<T, next> *me,
                                  bool waitSome) {
  T *head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
  if (head == nullptr) {
    if (!waitSome)
      return nullptr;
    uint64_t t0 = clockNano();
    bool sleeping = false;
    do {
      if (clockNano() - t0 >= 10 * 1000) { // spin for first 10us
        struct flagcxThreadSignal *waitSignal =
            &flagcxThreadSignalLocalInstance;
        pthread_mutex_lock(&waitSignal->mutex);
        uintptr_t expected = sleeping ? 0x1 : 0x0;
        uintptr_t desired = 0x1;
        me->waiting = waitSignal; // release done by successful compare exchange
        if (__atomic_compare_exchange_n(&me->tail, &expected, desired,
                                        /*weak=*/true, __ATOMIC_RELEASE,
                                        __ATOMIC_RELAXED)) {
          sleeping = true;
          pthread_cond_wait(&waitSignal->cond, &waitSignal->mutex);
        }
        pthread_mutex_unlock(&waitSignal->mutex);
      }
      head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
    } while (head == nullptr);
  }

  __atomic_store_n(&me->head, nullptr, __ATOMIC_RELAXED);
  uintptr_t utail = __atomic_exchange_n(&me->tail, 0x0, __ATOMIC_ACQ_REL);
  T *tail = utail <= 0x2 ? nullptr : reinterpret_cast<T *>(utail);
  T *x = head;
  while (x != tail) {
    T *x1;
    int spins = 0;
    while (true) {
      x1 = __atomic_load_n(&(x->*next), __ATOMIC_RELAXED);
      if (x1 != nullptr)
        break;
      if (++spins == 1024) {
        spins = 1024 - 1;
        sched_yield();
      }
    }
    x = x1;
  }
  return head;
}

template <typename T, T *T::*next>
T *flagcxIntruQueueMpscAbandon(flagcxIntruQueueMpsc<T, next> *me) {
  uintptr_t expected = 0x0;
  if (__atomic_compare_exchange_n(&me->tail, &expected, /*desired=*/0x2,
                                  /*weak=*/true, __ATOMIC_RELAXED,
                                  __ATOMIC_RELAXED)) {
    return nullptr;
  } else {
    int spins = 0;
    T *head;
    while (true) {
      head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
      if (head != nullptr)
        break;
      if (++spins == 1024) {
        spins = 1024 - 1;
        sched_yield();
      }
    }
    __atomic_store_n(&me->head, nullptr, __ATOMIC_RELAXED);
    uintptr_t utail = __atomic_exchange_n(&me->tail, 0x2, __ATOMIC_ACQ_REL);
    T *tail = utail <= 0x2 ? nullptr : reinterpret_cast<T *>(utail);
    T *x = head;
    while (x != tail) {
      T *x1;
      spins = 0;
      while (true) {
        x1 = __atomic_load_n(&(x->*next), __ATOMIC_RELAXED);
        if (x1 != nullptr)
          break;
        if (++spins == 1024) {
          spins = 1024 - 1;
          sched_yield();
        }
      }
      x = x1;
    }
    return head;
  }
}

#define GENERATE_ALL_TYPES(type, func, args...)                                \
  switch (type) {                                                              \
    case flagcxInt8:                                                           \
      func<char>(args);                                                        \
      break;                                                                   \
    case flagcxUint8:                                                          \
      func<unsigned char>(args);                                               \
      break;                                                                   \
    case flagcxInt32:                                                          \
      func<int32_t>(args);                                                     \
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
    case flagcxFloat:                                                          \
      func<float>(args);                                                       \
      break;                                                                   \
    case flagcxFloat64:                                                        \
      func<double>(args);                                                      \
      break;                                                                   \
    case flagcxFloat16:                                                        \
    case flagcxBfloat16:                                                       \
    default:                                                                   \
      WARN("Unsupported data type %d", type);                                  \
      return flagcxInvalidArgument;                                            \
  }

template <typename T>
void sum(void *res, const void *op1, const void *op2, size_t n) {
  const T *a = static_cast<const T *>(op1);
  const T *b = static_cast<const T *>(op2);
  T *c = static_cast<T *>(res);
  for (auto i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

template <typename T>
void min(void *res, const void *op1, const void *op2, size_t n) {
  const T *a = static_cast<const T *>(op1);
  const T *b = static_cast<const T *>(op2);
  T *c = static_cast<T *>(res);
  for (auto i = 0; i < n; i++) {
    c[i] = std::min(a[i], b[i]);
  }
}

template <typename T>
void max(void *res, const void *op1, const void *op2, size_t n) {
  const T *a = static_cast<const T *>(op1);
  const T *b = static_cast<const T *>(op2);
  T *c = static_cast<T *>(res);
  for (auto i = 0; i < n; i++) {
    c[i] = std::max(a[i], b[i]);
  }
}

void *flagcxOpenLib(const char *path, int flags,
                    void (*error_handler)(const char *, int, const char *));
#endif
