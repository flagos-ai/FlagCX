/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Unified MR Registry — shared address-range registry for all subsystems
 * (P2P Engine, Collective Proxy, RMA One-sided).
 *
 * Provides O(log n) containment lookup via sorted flat array + binary search,
 * with pthread_rwlock for concurrent readers on the data path.
 ************************************************************************/

#ifndef FLAGCX_MR_REGISTRY_H_
#define FLAGCX_MR_REGISTRY_H_

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "flagcx.h" // flagcxResult_t

#ifdef __cplusplus
extern "C" {
#endif

/* ───── Owner bitmask constants ───── */

#define FLAGCX_MR_OWNER_NONE 0x00
#define FLAGCX_MR_OWNER_P2P 0x01
#define FLAGCX_MR_OWNER_COLL 0x02
#define FLAGCX_MR_OWNER_RMA 0x04

#define FLAGCX_MR_OWNER_COUNT 3
#define FLAGCX_MR_OWNER_IDX_P2P 0
#define FLAGCX_MR_OWNER_IDX_COLL 1
#define FLAGCX_MR_OWNER_IDX_RMA 2

/* ───── Subsystem extension structs ───── */

#define FLAGCX_MR_IPC_HANDLE_BYTES 64

struct flagcxMrP2pExt {
  uint64_t mrId;
  bool hasIpc;
  uint32_t ipcHandleSize;
  char ipcHandle[FLAGCX_MR_IPC_HANDLE_BYTES];
};

struct flagcxMrCollExt {
  void *proxyConn;
  int channelId;
};

struct flagcxMrRmaExt {
  int oneSideHandleIdx;
};

/* ───── Tagged extension output ───── */

/*
 * Safe by-value output for extension data from self-locking lookup APIs.
 * The `type` field indicates which union member is valid:
 *   FLAGCX_MR_OWNER_P2P  → .p2p
 *   FLAGCX_MR_OWNER_COLL → .coll
 *   FLAGCX_MR_OWNER_RMA  → .rma
 *   FLAGCX_MR_OWNER_NONE → no data (extension not present on entry)
 */
struct flagcxMrExtension {
  uint32_t type;
  union {
    struct flagcxMrP2pExt p2p;
    struct flagcxMrCollExt coll;
    struct flagcxMrRmaExt rma;
  };
};

/* ───── Core entry ───── */

struct flagcxMrEntry {
  uintptr_t baseAddr;
  size_t size;
  int ibDevN;
  int ptrType;
  uint32_t ownerMask;
  void *mhandles[FLAGCX_MR_OWNER_COUNT]; /* per-subsystem adaptor mhandle */

  struct flagcxMrP2pExt *p2p;   /* NULL if !(ownerMask & P2P) */
  struct flagcxMrCollExt *coll; /* NULL if !(ownerMask & COLL) */
  struct flagcxMrRmaExt *rma;   /* NULL if !(ownerMask & RMA) */
};

/* ───── mrId index entry (sorted by mrId for O(log n) lookup) ───── */

struct flagcxMrIdEntry {
  uint64_t mrId;
  uintptr_t baseAddr; /* cross-reference into main entries[] array */
};

/* ───── Registry container ───── */

struct flagcxMrRegistry {
  struct flagcxMrEntry *entries; /* sorted by baseAddr, contiguous */
  int count;
  int capacity;
  uint64_t nextId; /* monotonic ID generator for all subsystems */

  struct flagcxMrIdEntry *idIndex; /* sorted by mrId (ascending) */
  int idCount;
  int idCapacity;

  pthread_rwlock_t rwlock;
};

/* ───── Lifecycle ───── */

flagcxResult_t flagcxMrRegistryCreate(struct flagcxMrRegistry **reg);
flagcxResult_t flagcxMrRegistryDestroy(struct flagcxMrRegistry *reg);

/* ───── Registration (write-locks internally) ───── */

/*
 * Register a memory region or add ownership to an existing entry.
 *
 * - If exact (addr, size) match exists: adds ownerBit, stores mhandle/ext.
 * - If exact addr match but different size: returns flagcxInternalError.
 * - If overlap with a different entry: returns flagcxInternalError.
 * - If new: inserts into sorted array.
 *
 * ownerBit: one of FLAGCX_MR_OWNER_{P2P,COLL,RMA}
 * mhandle:  adaptor handle (stored in mhandles[ownerIdx])
 * ext:      subsystem extension struct pointer (ownership transferred to entry)
 * outId:    if non-NULL, receives a monotonic ID on first registration.
 *           For P2P: persisted in p2p->mrId, stable across repeated calls.
 *           For COLL/RMA: one-shot assignment (not stored on entry);
 *           re-registration of an already-owned entry sets *outId = 0.
 */
flagcxResult_t flagcxMrRegistryRegister(struct flagcxMrRegistry *reg,
                                        uintptr_t addr, size_t size, int ibDevN,
                                        int ptrType, uint32_t ownerBit,
                                        void *mhandle, void *ext,
                                        uint64_t *outId);

/*
 * Remove one owner from entry identified by baseAddr.
 * If ownerMask becomes 0, removes entry from array.
 *
 * outEntry: if non-NULL, populated with common fields before removal.
 *           Extension pointers (p2p, coll, rma) are NULLed to prevent UAF.
 * outExt:   if non-NULL, returns the subsystem extension pointer (caller
 *           frees). If NULL, the extension is freed internally.
 */
flagcxResult_t flagcxMrRegistryDeregister(struct flagcxMrRegistry *reg,
                                          uintptr_t addr, uint32_t ownerBit,
                                          struct flagcxMrEntry *outEntry,
                                          void **outExt);

/* ───── Lookup (read-locks internally) ───── */

/*
 * NOTE: Lookup functions NULL out extension pointers (p2p, coll, rma) in the
 * returned outEntry to prevent use-after-free under concurrent deregistration.
 * To safely obtain extension data, pass a non-NULL outExts[] array indexed by
 * FLAGCX_MR_OWNER_IDX_*. Non-NULL slots receive a by-value copy of the
 * extension while the read lock is held. Each slot is self-describing via its
 * .type field. Pass NULL for the entire array if extensions are not needed.
 */

/*
 * O(log n) containment lookup: find entry where baseAddr <= addr <
 * baseAddr+size. Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t flagcxMrRegistryLookup(struct flagcxMrRegistry *reg,
                                      uintptr_t addr,
                                      struct flagcxMrEntry *outEntry,
                                      struct flagcxMrExtension *outExts[]);

/*
 * O(log n) exact-match lookup by baseAddr.
 * Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t flagcxMrRegistryFindExact(struct flagcxMrRegistry *reg,
                                         uintptr_t addr,
                                         struct flagcxMrEntry *outEntry,
                                         struct flagcxMrExtension *outExts[]);

/*
 * O(log n) lookup by P2P mrId via secondary index.
 * Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t flagcxMrRegistryLookupById(struct flagcxMrRegistry *reg,
                                          uint64_t mrId,
                                          struct flagcxMrEntry *outEntry,
                                          struct flagcxMrExtension *outExts[]);

/*
 * O(n) lookup by mhandle pointer for a given owner index.
 * Useful for deregistration when only the handle is known.
 * Returns flagcxSuccess if found, flagcxInternalError if not.
 */
flagcxResult_t
flagcxMrRegistryFindByHandle(struct flagcxMrRegistry *reg, int ownerIdx,
                             void *mhandle, struct flagcxMrEntry *outEntry,
                             struct flagcxMrExtension *outExts[]);

/* ───── Iteration (external locking) ───── */

flagcxResult_t flagcxMrRegistryRdLock(struct flagcxMrRegistry *reg);
flagcxResult_t flagcxMrRegistryRdUnlock(struct flagcxMrRegistry *reg);
flagcxResult_t flagcxMrRegistryWrLock(struct flagcxMrRegistry *reg);
flagcxResult_t flagcxMrRegistryWrUnlock(struct flagcxMrRegistry *reg);

/* Access entries directly (only valid while holding lock).
 * flagcxMrRegistryEntries returns NULL if registry is empty (count == 0).
 * Callers must check count before dereferencing the returned pointer. */
int flagcxMrRegistryCount(struct flagcxMrRegistry *reg);
struct flagcxMrEntry *flagcxMrRegistryEntries(struct flagcxMrRegistry *reg);

/* ───── Global instance ───── */

extern struct flagcxMrRegistry *flagcxGlobalMrRegistry;

flagcxResult_t flagcxMrRegistryGlobalInit(void);

/*
 * Release one reference to the global registry. Destroys when refcount
 * reaches 0.
 *
 * Precondition: all data-path operations (Lookup, LookupById, PrepareDesc)
 * must be quiesced before the final release. The caller is responsible for
 * ensuring no concurrent registry access is in flight when the last
 * reference is dropped. Violating this is undefined behavior (use-after-free).
 */
flagcxResult_t flagcxMrRegistryGlobalRelease(void);

#ifdef __cplusplus
}
#endif

#endif /* FLAGCX_MR_REGISTRY_H_ */
