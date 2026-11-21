#ifndef FLAGCX_ASCEND_H_
#define FLAGCX_ASCEND_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Copy the FlagCX-managed HCCL communicator's name into the caller's buffer
 * for native operator interoperability. Only stable homogeneous HCCL
 * communicators with matching rank order are supported.
 * name must have capacity >= 128 bytes. The communicator remains owned by
 * FlagCX and must outlive all operations using the returned name.
 * Returns flagcxNotSupported if the selected CCL adaptor does not implement
 * name retrieval, or for heterogeneous routes, hybrid groups, or tuning.
 * On success, name is nonempty and NUL-terminated.
 * Callers must order external operations against FlagCX operations on streams.
 */
flagcxResult_t flagcxCommGetHcclCommName(flagcxComm_t comm, char *name,
                                         size_t capacity);

#ifdef __cplusplus
}
#endif

#endif // FLAGCX_ASCEND_H_
